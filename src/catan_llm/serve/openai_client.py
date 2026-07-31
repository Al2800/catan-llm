"""OpenAI-compatible chat client with optional structured-decoding degrade."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from catan_llm.serve.decoding import completion_kwargs


def chat_complete(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    api_key: str = "EMPTY",
    structured: bool = True,
    backend: str = "auto",
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> tuple[str, dict[str, Any]]:
    """Return (content, meta) where meta notes constrained-decoding usage."""
    url = base_url.rstrip("/") + "/chat/completions"
    extra = completion_kwargs(
        structured=structured,
        max_tokens=max_tokens,
        temperature=temperature,
        backend=backend,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **extra,
    }
    meta = {"structured_requested": structured, "structured_applied": False, "degraded": False}

    def _post(body: dict[str, Any]) -> str:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    try:
        text = _post(payload)
        meta["structured_applied"] = bool(structured and "response_format" in payload)
        return text, meta
    except urllib.error.HTTPError as exc:
        if not structured:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        # Graceful degrade: retry without constraints.
        plain = completion_kwargs(
            structured=False, max_tokens=max_tokens, temperature=temperature, backend="none"
        )
        payload = {
            "model": model,
            "messages": payload["messages"],
            **plain,
        }
        try:
            text = _post(payload)
            meta["degraded"] = True
            return text, meta
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError(f"LLM request failed after degrade: {exc2}") from exc2
