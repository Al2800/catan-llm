"""Load a QLoRA adapter for local generation (Gate B / play)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from catan_llm.training.masking import load_qwen_config, qwen_model_name, qwen_revision


def load_peft_generator(
    adapter_dir: Path | str,
    *,
    config_path: Path | str | None = None,
    max_new_tokens: int = 128,
    max_prompt_tokens: int = 3072,
) -> tuple[Any, Any, Callable[[str, str], str]]:
    """Return (model, tokenizer, complete_fn) for an adapter checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_dir = Path(adapter_dir)
    cfg = load_qwen_config(Path(config_path) if config_path else None)
    model_id = qwen_model_name(Path(config_path) if config_path else None)
    revision = qwen_revision(Path(config_path) if config_path else None)
    qcfg = (cfg.get("model") or {}).get("quantization") or {}
    trust = bool((cfg.get("model") or {}).get("trust_remote_code", True))

    bnb = BitsAndBytesConfig(
        load_in_4bit=bool(qcfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qcfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(qcfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust}
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust,
        "quantization_config": bnb,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if revision:
        tok_kwargs["revision"] = revision
        load_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, OSError, KeyError):
        from transformers import AutoModel

        base = AutoModel.from_pretrained(model_id, **load_kwargs)

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True

    device = next(model.parameters()).device

    def complete(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"system: {system}\nuser: {user}\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if inputs["input_ids"].shape[-1] > max_prompt_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -max_prompt_tokens:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_prompt_tokens:]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(gen, skip_special_tokens=True)

    return model, tokenizer, complete
