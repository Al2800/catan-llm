"""Production Qwen3.5-9B QLoRA SFT (ticket 15).

Driven by ``configs/qwen3.5-9b-qlora.yaml``. Requires CUDA + ``pip install -e '.[train]'``.
Local 16GB is no-go; run on rental ≥24GB (see ticket 09 L40S smoke).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from catan_llm.training.masking import (
    IGNORE_INDEX,
    assistant_span_intact,
    build_assistant_only_labels,
    load_qwen_config,
    qwen_max_seq_length,
    qwen_model_name,
    qwen_revision,
)
from catan_llm.training.sft import load_chat_jsonl

DEFAULT_CONFIG = Path("configs/qwen3.5-9b-qlora.yaml")


def _write_train_history_md(
    path: Path, report: dict[str, Any], log_history: list[dict[str, Any]]
) -> None:
    """Compact markdown chart of train loss for humans / PR artifacts."""
    losses = [
        (row.get("step"), row.get("loss"))
        for row in log_history
        if isinstance(row, dict) and row.get("loss") is not None
    ]
    lines = [
        "# QLoRA train history",
        "",
        f"- model: `{report.get('model')}`",
        f"- revision: `{report.get('revision')}`",
        f"- max_seq_length: {report.get('max_seq_length')}",
        f"- peak_vram_gb: {report.get('peak_vram_gb')}",
        f"- step_time_s: {report.get('step_time_s')}",
        f"- train_loss: {report.get('train_loss')}",
        "",
        "## Loss samples",
        "",
        "| step | loss |",
        "|---:|---:|",
    ]
    for step, loss in losses[:200]:
        lines.append(f"| {step} | {loss} |")
    if losses:
        # Tiny sparkline via hashes scaled to min/max.
        vals = [float(loss) for _, loss in losses]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        blocks = "▁▂▃▄▅▆▇█"
        spark = "".join(blocks[min(7, int((v - lo) / span * 7))] for v in vals[-64:])
        lines.extend(["", f"sparkline (last {min(64, len(vals))}): `{spark}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class QLoRATrainReport:
    config_path: str
    model: str
    revision: str | None
    train_file: str
    val_file: str | None
    output_dir: str
    max_seq_length: int
    assistant_only_loss: bool
    mask_check: dict[str, Any]
    peak_vram_gb: float | None
    train_runtime_s: float | None
    step_time_s: float | None
    train_loss: float | None
    checkpoint: str | None
    resumed_from: str | None
    metrics: dict[str, Any]
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticket": "15",
            "config_path": self.config_path,
            "model": self.model,
            "revision": self.revision,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "output_dir": self.output_dir,
            "max_seq_length": self.max_seq_length,
            "assistant_only_loss": self.assistant_only_loss,
            "mask_check": self.mask_check,
            "peak_vram_gb": self.peak_vram_gb,
            "train_runtime_s": self.train_runtime_s,
            "step_time_s": self.step_time_s,
            "train_loss": self.train_loss,
            "checkpoint": self.checkpoint,
            "resumed_from": self.resumed_from,
            "metrics": self.metrics,
            "dry_run": self.dry_run,
        }


def resolve_data_path(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve train/val paths relative to repo root; accept absolute paths."""
    p = Path(path)
    if p.is_file():
        return p.resolve()
    root = repo_root or Path.cwd()
    candidate = (root / p).resolve()
    if candidate.is_file():
        return candidate
    # Common Phase-1 layout alias used in older config drafts.
    alt = root / "data" / "phase1" / "processed" / "expert-v1" / p.name
    if alt.is_file():
        return alt.resolve()
    return candidate


def verify_assistant_only_mask(tokenizer, *, max_seq_length: int = 4096) -> dict[str, Any]:
    """Sanity-check assistant-only labels on the pinned chat template."""
    from catan_llm.data.parser import format_assistant_target

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
    batch = build_assistant_only_labels(
        tokenizer, messages, max_seq_length=max_seq_length
    )
    intact = assistant_span_intact(batch, assistant, tokenizer)
    prompt_masked = all(
        x == IGNORE_INDEX for x in batch.labels[: batch.prompt_token_count]
    )
    ok = (
        batch.assistant_token_count > 0
        and prompt_masked
        and intact
        and not batch.truncated
    )
    return {
        "ok": ok,
        "prompt_token_count": batch.prompt_token_count,
        "assistant_token_count": batch.assistant_token_count,
        "truncated": batch.truncated,
        "assistant_span_intact": intact,
        "prompt_masked": prompt_masked,
    }


def _peak_vram_gb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def _require_train_deps() -> None:
    try:
        import bitsandbytes  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
        from datasets import Dataset  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA training extras missing. Install with: pip install -e '.[train]'"
        ) from exc


def _load_model_and_tokenizer(cfg: dict[str, Any], revision: str | None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = str((cfg.get("model") or {}).get("name_or_path") or qwen_model_name())
    qcfg = (cfg.get("model") or {}).get("quantization") or {}
    lcfg = cfg.get("lora") or {}
    trust = bool((cfg.get("model") or {}).get("trust_remote_code", True))

    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust}
    if revision:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust,
        "quantization_config": bnb,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if revision:
        load_kwargs["revision"] = revision

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, OSError, KeyError) as exc:
        # Qwen3.5 may ship as a multimodal ConditionalGeneration checkpoint.
        print(f"AutoModelForCausalLM failed ({exc}); trying AutoModel", flush=True)
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_id, **load_kwargs)

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lcfg.get("r", 16)),
            lora_alpha=int(lcfg.get("lora_alpha", 32)),
            lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
            bias=str(lcfg.get("bias", "none")),
            task_type=str(lcfg.get("task_type", "CAUSAL_LM")),
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


def _rows_to_conversational(rows: list[dict]) -> list[dict]:
    """TRL conversational format: {messages: [...]} only."""
    out = []
    for r in rows:
        messages = r.get("messages")
        if not messages:
            raise ValueError("chat JSONL row missing messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError("chat JSONL messages must end with assistant")
        # Reject obvious non-v2 / empty prompts.
        if not any(m.get("role") == "system" for m in messages):
            raise ValueError("chat JSONL missing system message (schema v2 chat)")
        out.append({"messages": messages})
    return out


def run_qlora_sft(
    config_path: Path | None = None,
    *,
    train_file: Path | str | None = None,
    val_file: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
    resume_from: Path | str | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> QLoRATrainReport:
    """Run (or dry-run) production QLoRA SFT from the pinned YAML config."""
    root = repo_root or Path.cwd()
    cfg_path = Path(config_path) if config_path else root / DEFAULT_CONFIG
    if not cfg_path.is_file():
        cfg_path = root / "configs" / "qwen3.5-9b-qlora.yaml"
    cfg = load_qwen_config(cfg_path)

    data_cfg = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    model_name = qwen_model_name(cfg_path)
    revision = qwen_revision(cfg_path)
    max_seq = int(data_cfg.get("max_seq_length") or qwen_max_seq_length(cfg_path))
    if max_seq < 4096:
        raise ValueError(
            f"max_seq_length={max_seq} < 4096 — refuse to train (DATA_CONTRACT / ticket 09)"
        )

    train_path = resolve_data_path(
        train_file or data_cfg.get("train_file") or "data/phase1/processed/expert-v1/train.jsonl",
        repo_root=root,
    )
    val_raw = val_file if val_file is not None else data_cfg.get("val_file")
    val_path = resolve_data_path(val_raw, repo_root=root) if val_raw else None
    out_dir = Path(
        output_dir or train_cfg.get("output_dir") or "outputs/sft/qwen3.5-9b-qlora"
    )
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    assistant_only = bool(train_cfg.get("assistant_only_loss", True))
    resumed = str(resume_from) if resume_from else None

    if dry_run:
        if not train_path.is_file():
            raise FileNotFoundError(f"train_file not found: {train_path}")
        rows = load_chat_jsonl(train_path)
        if max_samples is not None:
            rows = rows[: max_samples]
        sample = _rows_to_conversational(rows[: min(8, len(rows))])
        # Mask check with a lightweight tokenizer when available.
        mask_check: dict[str, Any] = {"ok": False, "skipped": True}
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True
            )
            mask_check = verify_assistant_only_mask(tok, max_seq_length=max_seq)
            mask_check["tokenizer"] = "HuggingFaceTB/SmolLM2-135M-Instruct"
            mask_check["note"] = (
                "dry-run uses SmolLM for mask shape; rental train uses pinned Qwen"
            )
        except Exception as exc:  # pragma: no cover - offline CI
            mask_check = {"ok": False, "error": str(exc), "skipped": True}

        report = QLoRATrainReport(
            config_path=str(cfg_path),
            model=model_name,
            revision=revision,
            train_file=str(train_path),
            val_file=str(val_path) if val_path else None,
            output_dir=str(out_dir),
            max_seq_length=max_seq,
            assistant_only_loss=assistant_only,
            mask_check=mask_check,
            peak_vram_gb=None,
            train_runtime_s=None,
            step_time_s=None,
            train_loss=None,
            checkpoint=None,
            resumed_from=resumed,
            metrics={
                "num_train_rows_available": len(rows),
                "dry_run_sample_rows": len(sample),
                "would_max_steps": max_steps or train_cfg.get("num_train_epochs"),
            },
            dry_run=True,
        )
        (out_dir / "dry_run_report.json").write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        return report

    _require_train_deps()
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for production QLoRA (local 16GB is no-go; use rental ≥24GB)"
        )
    if not train_path.is_file():
        raise FileNotFoundError(f"train_file not found: {train_path}")

    rows = load_chat_jsonl(train_path)
    if max_samples is not None:
        rows = rows[: int(max_samples)]
    train_ds = Dataset.from_list(_rows_to_conversational(rows))

    eval_ds = None
    if val_path is not None and Path(val_path).is_file():
        val_rows = load_chat_jsonl(Path(val_path))
        if max_samples is not None:
            val_rows = val_rows[: max(32, int(max_samples) // 10)]
        eval_ds = Dataset.from_list(_rows_to_conversational(val_rows))

    tokenizer, model = _load_model_and_tokenizer(cfg, revision)
    mask_check = verify_assistant_only_mask(tokenizer, max_seq_length=max_seq)
    mask_check["revision"] = revision
    if not mask_check["ok"]:
        raise RuntimeError(f"assistant-only mask check failed: {mask_check}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    import inspect

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": int(
            train_cfg.get("per_device_train_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            train_cfg.get("gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-4)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0) or 0),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "bf16": bool(train_cfg.get("bf16", True)),
        "fp16": False,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "logging_steps": int(train_cfg.get("logging_steps", 20)),
        "save_steps": int(train_cfg.get("save_steps", 200)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "report_to": train_cfg.get("report_to") or "none",
        "max_length": max_seq,
        "packing": bool(data_cfg.get("packing", False)),
        "assistant_only_loss": assistant_only,
        "optim": "paged_adamw_8bit",
    }
    if max_steps is not None:
        sft_kwargs["max_steps"] = int(max_steps)
    else:
        sft_kwargs["num_train_epochs"] = float(train_cfg.get("num_train_epochs", 2))

    if eval_ds is not None:
        sft_kwargs["eval_strategy"] = str(train_cfg.get("eval_strategy", "steps"))
        sft_kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 200))
    else:
        sft_kwargs["eval_strategy"] = "no"

    # TRL/transformers versions differ on TrainingArguments field names.
    allowed = set(inspect.signature(SFTConfig.__init__).parameters)
    # Prefer ratio when supported; else approximate with warmup_steps if max_steps known.
    if "warmup_ratio" not in allowed and "warmup_steps" in allowed:
        sft_kwargs.pop("warmup_ratio", None)
        if not sft_kwargs.get("warmup_steps") and max_steps:
            sft_kwargs["warmup_steps"] = max(1, int(0.03 * int(max_steps)))
    filtered = {k: v for k, v in sft_kwargs.items() if k in allowed}
    dropped = sorted(set(sft_kwargs) - set(filtered))
    if dropped:
        print(f"SFTConfig dropping unsupported kwargs: {dropped}", flush=True)
    args = SFTConfig(**filtered)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    from catan_llm.training.hub_checkpoints import (
        HubCheckpointUploadCallback,
        resolve_hub_checkpoint_repo,
    )

    hub_repo = resolve_hub_checkpoint_repo(train_cfg.get("hub_checkpoint_repo"))
    hub_cb = None
    if hub_repo:
        hub_cb = HubCheckpointUploadCallback(hub_repo)
        trainer.add_callback(hub_cb.as_trainer_callback())
        print(f"hub-checkpoint: will upload saves → hf://{hub_repo}/checkpoints/", flush=True)

    train_result = trainer.train(
        resume_from_checkpoint=str(resume_from) if resume_from else None
    )
    ckpt = out_dir / "adapter"
    trainer.save_model(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))

    runtime = round(time.time() - t0, 2)
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    if hub_cb is not None:
        metrics["hub_checkpoint_uploads"] = list(hub_cb.uploaded)
    # Prefer trainer-reported step time; else derive from runtime / steps.
    step_time = metrics.get("train_steps_per_second")
    derived_step = None
    if step_time:
        derived_step = round(1.0 / float(step_time), 4)
    else:
        global_step = getattr(trainer.state, "global_step", None)
        if global_step and global_step > 0:
            derived_step = round(runtime / float(global_step), 4)

    report = QLoRATrainReport(
        config_path=str(cfg_path),
        model=model_name,
        revision=revision,
        train_file=str(train_path),
        val_file=str(val_path) if val_path else None,
        output_dir=str(out_dir),
        max_seq_length=max_seq,
        assistant_only_loss=assistant_only,
        mask_check=mask_check,
        peak_vram_gb=_peak_vram_gb(),
        train_runtime_s=runtime,
        step_time_s=derived_step,
        train_loss=float(getattr(train_result, "training_loss", 0.0) or 0.0),
        checkpoint=str(ckpt),
        resumed_from=resumed,
        metrics=metrics,
        dry_run=False,
    )
    (out_dir / "train_report.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )
    # Step telemetry for training visualization (ticket 17 / spectate-adjacent).
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    (out_dir / "train_history.json").write_text(
        json.dumps(log_history, indent=2), encoding="utf-8"
    )
    _write_train_history_md(out_dir / "train_history.md", report.as_dict(), log_history)
    # Persist a small copy of the resolved config for resume/audit.
    (out_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": cfg.get("model"),
                "lora": cfg.get("lora"),
                "data": {
                    **data_cfg,
                    "train_file": str(train_path),
                    "val_file": str(val_path) if val_path else None,
                },
                "train": train_cfg,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return report
