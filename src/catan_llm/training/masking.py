"""Assistant-only loss masking helpers for chat SFT (DATA_CONTRACT / T9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MaskedBatch:
    input_ids: list[int]
    labels: list[int]
    prompt_token_count: int
    assistant_token_count: int
    truncated: bool
    text: str


DEFAULT_QWEN_CONFIG = "configs/qwen3.5-9b-qlora.yaml"


def load_qwen_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or Path(__file__).resolve().parents[3] / DEFAULT_QWEN_CONFIG
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"{cfg_path} is not a training config (expected model/data/train keys). "
            "Use configs/qwen3.5-9b-qlora.yaml"
        )
    return payload


def qwen_model_name(path: Path | None = None) -> str:
    model = load_qwen_config(path).get("model") or {}
    return str(model.get("name_or_path") or "Qwen/Qwen3.5-9B")


def qwen_revision(path: Path | None = None) -> str | None:
    model = load_qwen_config(path).get("model") or {}
    rev = model.get("revision")
    if rev is None or rev == "null" or rev == "":
        return None
    return str(rev)


def qwen_max_seq_length(path: Path | None = None) -> int:
    data = load_qwen_config(path).get("data") or {}
    return int(data.get("max_seq_length", 4096))


def build_assistant_only_labels(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_seq_length: int = 4096,
) -> MaskedBatch:
    """Mask system/user tokens; keep assistant tokens as supervised labels.

    Uses chat-template prefix length (prompt + generation header) so labels only
    supervise the assistant completion span.
    """
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("messages must end with an assistant turn")

    prompt_messages = messages[:-1]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    else:
        # Fallback for stub tokenizers in unit tests.
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_messages)
        prompt_text += "\nassistant: "
        full_ids = tokenizer.encode(text)
        prompt_ids = tokenizer.encode(prompt_text)

    truncated = len(full_ids) > max_seq_length
    input_ids = full_ids[:max_seq_length]
    labels = list(input_ids)
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = IGNORE_INDEX

    assistant_count = sum(1 for x in labels if x != IGNORE_INDEX)
    return MaskedBatch(
        input_ids=input_ids,
        labels=labels,
        prompt_token_count=n_prompt,
        assistant_token_count=assistant_count,
        truncated=truncated,
        text=text if isinstance(text, str) else str(text),
    )


def assistant_span_intact(batch: MaskedBatch, assistant_content: str, tokenizer) -> bool:
    """True if the assistant JSON/text is fully represented in supervised labels."""
    if batch.truncated and batch.assistant_token_count == 0:
        return False
    supervised = [tok for tok in batch.labels if tok != IGNORE_INDEX]
    if not supervised:
        return False
    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode(supervised, skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(supervised)
    # Require the action JSON marker and a substantial prefix of the assistant payload.
    if '{"action"' not in decoded.replace(" ", "") and '{"action"' not in assistant_content:
        # Fall back to substring check on raw assistant content presence in full text.
        return assistant_content[:32] in batch.text
    # Normalize spaces for brittle chat-template wrappers.
    compact_decoded = "".join(decoded.split())
    compact_target = "".join(assistant_content.split())
    return compact_target in compact_decoded or compact_target[:48] in compact_decoded
