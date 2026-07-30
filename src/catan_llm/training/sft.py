"""Tiny SFT smoke loop (TRL) for Phase 0.

Default model is intentionally tiny so CPU CI / laptop smoke stays feasible.
Real Phase-2 training uses Qwen3-8B + QLoRA (see docs/SCOPE.md).
"""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_chat_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_sft_smoke(
    train_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_SMOKE_MODEL,
    max_steps: int = 30,
    max_seq_length: int = 512,
    batch_size: int = 1,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_samples: int = 256,
) -> Path:
    """Run a short SFT and save a HuggingFace checkpoint directory."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training extras missing. Install with: pip install -e '.[train]'"
        ) from exc

    rows = load_chat_jsonl(train_path)[:max_samples]
    if not rows:
        raise ValueError(f"No training rows in {train_path}")

    def to_text(example):
        # Prefer tokenizer chat template when available; fallback to plain concat.
        return {"messages": example["messages"]}

    dataset = Dataset.from_list([to_text(r) for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            texts.append(text)
        return texts

    args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=max_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        max_length=max_seq_length,
        dataset_text_field=None,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "checkpoint"))
    tokenizer.save_pretrained(str(output_dir / "checkpoint"))
    return output_dir / "checkpoint"


def local_complete_fn_from_checkpoint(checkpoint_dir: Path, max_new_tokens: int = 64):
    """Build a complete_fn(system, user) -> str using a local HF checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def complete(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"system: {system}\nuser: {user}\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return complete
