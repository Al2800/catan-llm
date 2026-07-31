#!/usr/bin/env python3
"""Ticket 09 rental QLoRA smoke for Qwen/Qwen3.5-9B.

Runs on a 24–80GB GPU (HF Jobs / RunPod / etc.):
  1) resolve + record Hub revision SHA
  2) 4-bit QLoRA micro-train at max_seq_length>=4096 with peak VRAM
  3) assistant-mask check on that pin
  4) one game vs Random (parse/fallback logged)
  5) write outputs/hw_smoke/report.json

Usage (on rental):
  pip install -e '.[train]' bitsandbytes
  python scripts/rental_hw_smoke.py
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import torch
import yaml
from catanatron.models.player import Color, RandomPlayer
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.data.parser import format_assistant_target
from catan_llm.eval.arena import SeatSpec, run_match
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.sim.adapter import generate_trajectories
from catan_llm.training.masking import (
    IGNORE_INDEX,
    assistant_span_intact,
    build_assistant_only_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "qwen3.5-9b-qlora.yaml"
OUT_DIR = REPO_ROOT / "outputs" / "hw_smoke"
MODEL_ID = "Qwen/Qwen3.5-9B"
MAX_SEQ = 4096
MICRO_STEPS = 15


def _peak_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _resolve_revision() -> str:
    api = HfApi()
    info = api.model_info(MODEL_ID)
    return str(info.sha)


def _make_train_jsonl(path: Path, n_games: int = 4, seed: int = 7) -> Path:
    traj = OUT_DIR / "trajectories.jsonl"
    generate_trajectories(
        bot_names=["random", "weightedrandom", "valuefunction", "random"],
        num_games=n_games,
        seed=seed,
        out_path=traj,
        vps_to_win=8,
        workers=1,
        overwrite=True,
    )
    ds_dir = OUT_DIR / "dataset"
    build_chat_dataset(traj, ds_dir, name="hw-smoke", version="v0")
    train = ds_dir / "train.jsonl"
    if not train.exists():
        raise FileNotFoundError(f"missing {train}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(train.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _mask_check(tokenizer, revision: str) -> dict:
    assistant = format_assistant_target(3, "highest pip settlement")
    messages = [
        {"role": "system", "content": "You are an expert Catan player. RULES..."},
        {
            "role": "user",
            "content": (
                "AVAILABLE ACTIONS:\n  [0] ROLL\n  [3] BUILD_SETTLEMENT\nRespond JSON."
            ),
        },
        {"role": "assistant", "content": assistant},
    ]
    batch = build_assistant_only_labels(tokenizer, messages, max_seq_length=MAX_SEQ)
    intact = assistant_span_intact(batch, assistant, tokenizer)
    ok = (
        batch.assistant_token_count > 0
        and all(x == IGNORE_INDEX for x in batch.labels[: batch.prompt_token_count])
        and intact
        and not batch.truncated
    )
    return {
        "ok": ok,
        "revision": revision,
        "prompt_token_count": batch.prompt_token_count,
        "assistant_token_count": batch.assistant_token_count,
        "truncated": batch.truncated,
        "assistant_span_intact": intact,
    }


def _build_qlora(revision: str, cfg: dict):
    qcfg = (cfg.get("model") or {}).get("quantization") or {}
    lcfg = cfg.get("lora") or {}
    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        revision=revision,
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except (ValueError, OSError, KeyError) as exc:
        # Qwen3.5 ships as a multimodal ConditionalGeneration checkpoint.
        print(f"AutoModelForCausalLM failed ({exc}); trying AutoModel", flush=True)
        from transformers import AutoModel

        model = AutoModel.from_pretrained(MODEL_ID, **load_kwargs)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lcfg.get("r", 16)),
            lora_alpha=int(lcfg.get("lora_alpha", 32)),
            lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
            bias=str(lcfg.get("bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(
                lcfg.get(
                    "target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
            ),
        ),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    return tokenizer, model


def _messages_to_text(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _micro_train(tokenizer, model, train_jsonl: Path, steps: int = MICRO_STEPS) -> dict:
    rows = []
    with train_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("no train rows")
    # Cap samples but keep enough for MICRO_STEPS; pad by repeat if tiny.
    rows = rows[:64]
    while len(rows) < steps:
        rows.extend(rows[: max(1, steps - len(rows))])

    texts = [{"text": _messages_to_text(tokenizer, r["messages"])} for r in rows]
    dataset = Dataset.from_list(texts)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_dir = OUT_DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(train_dir),
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_steps=steps,
        bf16=True,
        fp16=False,
        report_to="none",
        max_length=MAX_SEQ,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    ckpt = train_dir / "adapter"
    trainer.save_model(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    peak = _peak_vram_gb()
    return {
        "checkpoint": str(ckpt),
        "max_steps": steps,
        "max_seq_length": MAX_SEQ,
        "train_runtime_s": round(time.time() - t0, 2),
        "train_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        "peak_vram_gb": peak,
        "metrics": dict(getattr(result, "metrics", {}) or {}),
    }


def _prepare_for_inference(model) -> None:
    """Disable train-only knobs so generate() can use KV cache."""
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    # PEFT wraps the base model — clear checkpointing there too.
    base = getattr(model, "get_base_model", lambda: None)()
    if base is not None and hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    for cfg in (getattr(model, "config", None), getattr(base, "config", None) if base else None):
        if cfg is not None:
            cfg.use_cache = True


def _complete_fn_from_peft(base_model, tokenizer, max_new_tokens: int = 96):
    device = next(base_model.parameters()).device

    def complete(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Cap prompt length for smoke latency; labels already validated at 4096.
        if inputs["input_ids"].shape[-1] > 2048:
            inputs["input_ids"] = inputs["input_ids"][:, -2048:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -2048:]
        with torch.inference_mode():
            out = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return complete


def _one_game(model, tokenizer, seed: int = 1007) -> dict:
    _prepare_for_inference(model)
    complete = _complete_fn_from_peft(model, tokenizer)
    llm = LLMPlayer(Color.RED, complete_fn=complete, model=MODEL_ID)
    seats = [
        SeatSpec(name="llm", kind="llm", player=llm),
        SeatSpec(name="random", kind="random", player=RandomPlayer(Color.BLUE)),
        SeatSpec(name="random2", kind="random", player=RandomPlayer(Color.ORANGE)),
        SeatSpec(name="random3", kind="random", player=RandomPlayer(Color.WHITE)),
    ]
    stats = run_match(
        seats,
        num_games=1,
        seed=seed,
        vps_to_win=8,
        candidate_name="llm",
        versus_name="random",
    )
    # Parse/fallback rates are absorbed into MatchStats via consume_eval_counters.
    return stats.summary(candidate="llm", versus="random")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "ticket": "09",
        "model": MODEL_ID,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "go_no_go": "no-go",
        "error": None,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for rental QLoRA smoke")

        cfg = _load_cfg()
        revision = _resolve_revision()
        report["revision"] = revision
        report["config_path"] = str(CONFIG_PATH.relative_to(REPO_ROOT))
        report["max_seq_length"] = MAX_SEQ
        print(f"Resolved revision: {revision}", flush=True)

        train_jsonl = _make_train_jsonl(OUT_DIR / "train.jsonl")
        report["data"] = {"train_jsonl": str(train_jsonl), "bytes": train_jsonl.stat().st_size}

        tokenizer, model = _build_qlora(revision, cfg)
        report["load_peak_vram_gb"] = _peak_vram_gb()
        print(f"Loaded 4-bit+LoRA; peak VRAM={report['load_peak_vram_gb']}GB", flush=True)

        report["mask"] = _mask_check(tokenizer, revision)
        print(f"Mask check: {report['mask']}", flush=True)
        if not report["mask"]["ok"]:
            raise RuntimeError("assistant-mask check failed")

        report["train"] = _micro_train(tokenizer, model, train_jsonl, steps=MICRO_STEPS)
        print(f"Micro-train done: {report['train']}", flush=True)

        report["eval"] = _one_game(model, tokenizer)
        print(f"One-game eval: {json.dumps(report['eval'], indent=2)}", flush=True)

        train_ok = report["train"].get("peak_vram_gb") is not None
        eval_ok = "win_rates" in report["eval"] and int(report["eval"].get("games", 0) or 0) >= 1
        report["go_no_go"] = "go" if (train_ok and report["mask"]["ok"] and eval_ok) else "no-go"
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["go_no_go"] = "no-go"
        print(report["traceback"], flush=True)

    out = OUT_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("===REPORT_JSON_BEGIN===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("===REPORT_JSON_END===", flush=True)
    print(f"Wrote {out}", flush=True)

    # Best-effort upload for retrieval from HF Jobs logs/UI.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            api = HfApi(token=token)
            repo_id = os.environ.get("HF_SMOKE_REPO", "AlCampbell/catan-llm-hw-smoke")
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
            api.upload_file(
                path_or_fileobj=str(out),
                path_in_repo="report.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded report to hf://datasets/{repo_id}/report.json", flush=True)
        except Exception as upload_exc:  # noqa: BLE001
            print(f"Report upload skipped: {upload_exc}", flush=True)

    return 0 if report.get("go_no_go") == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
