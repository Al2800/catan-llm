"""No-truncation gate: assistant JSON must fit within max_seq_length (ticket 08)."""

from __future__ import annotations

from catan_llm.data.dataset import decision_to_chat
from catan_llm.sim.adapter import play_one
from catan_llm.training.masking import (
    build_assistant_only_labels,
    qwen_max_seq_length,
)


class _CharTokenizer:
    chat_template = None

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(i) for i in ids)


def test_config_max_seq_length_at_least_4096():
    assert qwen_max_seq_length() >= 4096


def test_live_chat_assistant_not_truncated_under_budget():
    """Canonical short-game chats must keep the full assistant JSON under 4096."""
    max_len = qwen_max_seq_length()
    result = play_one(
        ["random", "random", "random", "random"],
        seed=21,
        vps_to_win=6,
    )
    # Prefer HF tokenizer when present (closer to training); else char stub.
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True
        )
    except Exception:
        tokenizer = _CharTokenizer()

    checked = 0
    for record in result.records[:40]:
        chat = decision_to_chat(record)
        messages = chat["messages"]
        assistant = messages[-1]["content"]
        assert '{"action"' in assistant
        batch = build_assistant_only_labels(
            tokenizer, messages, max_seq_length=max_len
        )
        assert not batch.truncated, (
            f"decision {record.decision_idx} exceeds max_seq_length={max_len} "
            f"(len={len(batch.input_ids)} would truncate assistant)"
        )
        assert batch.assistant_token_count > 0
        checked += 1
    assert checked > 0


def test_truncation_flag_when_budget_too_small():
    tok = _CharTokenizer()
    messages = [
        {"role": "system", "content": "S" * 100},
        {"role": "user", "content": "U" * 100},
        {"role": "assistant", "content": '{"action": 0, "reasoning": "x"}'},
    ]
    batch = build_assistant_only_labels(tok, messages, max_seq_length=50)
    assert batch.truncated
