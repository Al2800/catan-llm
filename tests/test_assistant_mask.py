"""Assistant-only loss masking (ticket 07 / T9).

If Qwen revision is not pinned yet (ticket 09), the Qwen one-batch test skips
with a clear message. A stub/SmolLM path still proves the mask helper in CI.
"""

from __future__ import annotations

import pytest

from catan_llm.data.parser import format_assistant_target
from catan_llm.training.masking import (
    IGNORE_INDEX,
    assistant_span_intact,
    build_assistant_only_labels,
    qwen_max_seq_length,
    qwen_model_name,
    qwen_revision,
)


class _CharTokenizer:
    """Minimal tokenizer for CI without downloading models."""

    chat_template = None

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(c) for c in text]

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(i) for i in ids)


def _sample_messages(action_index: int = 3) -> list[dict[str, str]]:
    assistant = format_assistant_target(action_index, "highest pip settlement")
    return [
        {"role": "system", "content": "You are an expert Catan player. RULES..."},
        {
            "role": "user",
            "content": "AVAILABLE ACTIONS:\n  [0] ROLL\n  [3] BUILD_SETTLEMENT\nRespond JSON.",
        },
        {"role": "assistant", "content": assistant},
    ]


def test_mask_helper_masks_system_user_keeps_assistant():
    tok = _CharTokenizer()
    messages = _sample_messages()
    batch = build_assistant_only_labels(tok, messages, max_seq_length=4096)
    assert batch.prompt_token_count > 0
    assert batch.assistant_token_count > 0
    assert all(x == IGNORE_INDEX for x in batch.labels[: batch.prompt_token_count])
    assert any(x != IGNORE_INDEX for x in batch.labels[batch.prompt_token_count :])
    assert not batch.truncated
    assert assistant_span_intact(batch, messages[-1]["content"], tok)
    assert '{"action"' in "".join(
        chr(i) for i in batch.labels if i != IGNORE_INDEX
    )


def test_qwen_config_max_seq_length_floor():
    assert qwen_max_seq_length() >= 4096


def test_qwen_tokenizer_mask_when_revision_pinned():
    """Pinned Qwen chat-template mask check (tokenizer only — CI-safe)."""
    revision = qwen_revision()
    model_name = qwen_model_name()
    if revision is None:
        pytest.skip(
            f"{model_name} model.revision is null in configs/qwen3.5-9b-qlora.yaml; "
            "pin it in ticket 09 before Phase 1 (assistant-mask must then pass)."
        )

    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    messages = _sample_messages()
    batch = build_assistant_only_labels(tokenizer, messages, max_seq_length=4096)
    assert batch.assistant_token_count > 0
    assert all(x == IGNORE_INDEX for x in batch.labels[: batch.prompt_token_count])
    assert assistant_span_intact(batch, messages[-1]["content"], tokenizer)
    assert not batch.truncated


def test_qwen_one_batch_loss_when_weights_cached():
    """Optional nonzero-loss proof on full Qwen weights (opt-in; heavy)."""
    import os

    if os.environ.get("CATAN_LLM_LOAD_QWEN") != "1":
        pytest.skip(
            "Set CATAN_LLM_LOAD_QWEN=1 to load full Qwen weights "
            "(tokenizer mask + rental smoke already cover ticket 09)."
        )

    revision = qwen_revision()
    model_name = qwen_model_name()
    if revision is None:
        pytest.skip("model.revision not pinned")

    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    messages = _sample_messages()
    batch = build_assistant_only_labels(tokenizer, messages, max_seq_length=4096)

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Pinned Qwen weights unavailable locally: {exc}")

    model.eval()
    input_ids = torch.tensor([batch.input_ids], dtype=torch.long)
    labels = torch.tensor([batch.labels], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    assert float(out.loss) > 0.0


def test_smol_one_batch_nonzero_loss_if_train_stack_present():
    """CI-friendly one-batch proof on SmolLM when transformers/torch are installed."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )
    model.eval()

    messages = _sample_messages()
    batch = build_assistant_only_labels(tokenizer, messages, max_seq_length=1024)
    assert batch.assistant_token_count > 0
    assert all(x == IGNORE_INDEX for x in batch.labels[: batch.prompt_token_count])
    assert assistant_span_intact(batch, messages[-1]["content"], tokenizer)

    input_ids = torch.tensor([batch.input_ids], dtype=torch.long)
    labels = torch.tensor([batch.labels], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    assert float(out.loss) > 0.0
