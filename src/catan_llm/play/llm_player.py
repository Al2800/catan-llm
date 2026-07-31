"""LLMPlayer — Catanatron Player that queries an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Callable

from catanatron.models.enums import ActionType
from catanatron.models.player import Player

from catan_llm.data.parser import FALLBACK_POLICY, fallback_action, parse_action_response
from catan_llm.data.renderer import render_system_prompt, render_user_prompt

AUTO_PLAY = {ActionType.ROLL}
# Re-export locked policy for eval reports.
assert FALLBACK_POLICY == "first_legal"


class LLMPlayer(Player):
    """Decide via rendered prompt → model completion → action parser."""

    def __init__(
        self,
        color,
        *,
        complete_fn: Callable[[str, str], str] | None = None,
        base_url: str | None = None,
        model: str = "local-model",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 128,
        is_bot: bool = True,
    ):
        super().__init__(color, is_bot=is_bot)
        self.complete_fn = complete_fn
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_cache: str | None = None
        self._parse_ok = 0
        self._parse_total = 0
        self._legal_ok = 0
        self._legal_total = 0
        self._fallback_count = 0
        self._error_hist: Counter[str] = Counter()
        self._phase_error_hist: Counter[str] = Counter()
        self.last_raw: str = ""
        self.last_error: str | None = None
        self.fallback_policy = FALLBACK_POLICY

    def reset_state(self):
        self._system_cache = None
        self._parse_ok = self._parse_total = 0
        self._legal_ok = self._legal_total = 0
        self._fallback_count = 0
        self._error_hist = Counter()
        self._phase_error_hist = Counter()
        self.last_raw = ""
        self.last_error = None

    def consume_eval_counters(self) -> dict[str, Any]:
        vals: dict[str, Any] = {
            "parse_ok": self._parse_ok,
            "parse_total": self._parse_total,
            "legal_ok": self._legal_ok,
            "legal_total": self._legal_total,
            "fallback_count": self._fallback_count,
            "action_error_hist": dict(self._error_hist),
            "phase_error_hist": dict(self._phase_error_hist),
        }
        self._parse_ok = self._parse_total = 0
        self._legal_ok = self._legal_total = 0
        self._fallback_count = 0
        self._error_hist = Counter()
        self._phase_error_hist = Counter()
        return vals

    def _record_failure(self, error: str | None, playable_actions) -> None:
        key = error or "unknown"
        self._error_hist[key] += 1
        # Bucket by dominant legal action type at the decision (context cue).
        if playable_actions:
            types = [
                a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                for a in playable_actions
            ]
            dominant = Counter(types).most_common(1)[0][0]
            self._phase_error_hist[f"{dominant}:{key}"] += 1

    def decide(self, game, playable_actions):
        playable_actions = list(playable_actions)
        if len(playable_actions) == 1:
            return playable_actions[0]
        # Cheap auto-play for pure ROLL prompts.
        if all(a.action_type in AUTO_PLAY for a in playable_actions):
            return playable_actions[0]

        if self._system_cache is None:
            self._system_cache = render_system_prompt(game, self.color)
        user = render_user_prompt(game, self.color, playable_actions)

        try:
            raw = self._complete(self._system_cache, user)
            self.last_raw = raw
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 — fallback must keep games alive
            self.last_error = str(exc)
            self._parse_total += 1
            self._legal_total += 1
            self._fallback_count += 1
            self._record_failure("request_failed", playable_actions)
            return fallback_action(playable_actions)

        result = parse_action_response(raw, playable_actions)
        self._parse_total += 1
        self._legal_total += 1
        if result.ok and result.action is not None:
            self._parse_ok += 1
            self._legal_ok += 1
            return result.action

        self.last_error = result.error
        self._fallback_count += 1
        self._record_failure(result.error, playable_actions)
        return fallback_action(playable_actions)

    def _complete(self, system: str, user: str) -> str:
        if self.complete_fn is not None:
            return self.complete_fn(system, user)
        if not self.base_url:
            raise RuntimeError("LLMPlayer requires complete_fn or base_url")
        return self._openai_chat(system, user)

    def _openai_chat(self, system: str, user: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]
