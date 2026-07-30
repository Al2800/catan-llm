"""Parse model outputs into legal Catanatron actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from catanatron.models.enums import Action

# Locked — SCOPE §12.12 / EVAL_PROTOCOL §1.
FALLBACK_POLICY = "first_legal"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ParseResult:
    ok: bool
    action: Action | None
    action_index: int | None
    reasoning: str = ""
    error: str | None = None
    raw: str = ""


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_action_response(text: str, playable_actions: list[Action]) -> ParseResult:
    """Parse `{"action": <index>, "reasoning": "..."}` into a legal action."""
    if not playable_actions:
        return ParseResult(ok=False, action=None, action_index=None, error="no_actions", raw=text)

    obj = extract_json_object(text)
    if obj is None:
        return ParseResult(
            ok=False, action=None, action_index=None, error="json_parse_failed", raw=text
        )

    if "action" not in obj:
        return ParseResult(
            ok=False, action=None, action_index=None, error="missing_action", raw=text
        )

    raw_idx = obj["action"]
    try:
        idx = int(raw_idx)
    except (TypeError, ValueError):
        return ParseResult(
            ok=False, action=None, action_index=None, error="action_not_int", raw=text
        )

    if idx < 0 or idx >= len(playable_actions):
        return ParseResult(
            ok=False,
            action=None,
            action_index=idx,
            error="action_out_of_range",
            raw=text,
            reasoning=str(obj.get("reasoning", "")),
        )

    return ParseResult(
        ok=True,
        action=playable_actions[idx],
        action_index=idx,
        reasoning=str(obj.get("reasoning", "")),
        raw=text,
    )


def fallback_action(playable_actions: list[Action]) -> Action:
    """Deterministic safe fallback: first legal action (engine-ordered)."""
    if not playable_actions:
        raise ValueError("No playable actions for fallback")
    return playable_actions[0]


def format_assistant_target(action_index: int, reasoning: str = "") -> str:
    """Canonical assistant completion used for SFT labels."""
    payload = {"action": action_index, "reasoning": reasoning}
    return json.dumps(payload, ensure_ascii=True)
