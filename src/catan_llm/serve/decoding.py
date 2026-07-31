"""Constrained / structured decoding helpers (ticket 16)."""

from __future__ import annotations

from typing import Any

# Assistant target shape from DATA_CONTRACT / parser.
ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "integer", "minimum": 0},
        "rationale": {"type": "string"},
    },
    "required": ["action"],
    "additionalProperties": True,
}


def completion_kwargs(
    *,
    structured: bool = True,
    max_tokens: int = 128,
    temperature: float = 0.0,
    backend: str = "openai",
) -> dict[str, Any]:
    """Extra chat.completions body fields for structured JSON when supported.

    ``backend``:
      - ``openai``: ``response_format`` json_object (widely ignored-or-accepted)
      - ``vllm``: also attach ``guided_json`` when structured (vLLM-specific)
      - ``none``: no constraints
    """
    body: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not structured or backend == "none":
        return body
    if backend in {"openai", "vllm", "auto"}:
        body["response_format"] = {"type": "json_object"}
    if backend == "vllm":
        # Older/newer vLLM builds differ; clients should degrade if rejected.
        body["guided_json"] = ACTION_JSON_SCHEMA
    return body
