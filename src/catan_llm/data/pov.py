"""Teacher / learner POV helpers and Tier A leakage checks (SCOPE §5.1)."""

from __future__ import annotations

import re
from typing import Any

# Colors that may appear in prompts.
_COLORS = ("RED", "BLUE", "ORANGE", "WHITE")
_COLOR_ALT = "|".join(_COLORS)

# Opponent private-hand leakage patterns (resource / dev compositions).
# Card *counts* like "ORANGE (7 cards)" or "cards=7" are allowed.
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "BLUE hand=[W=2,...]" or "hand=[W=1" near a color
    re.compile(
        rf"\b({_COLOR_ALT})\b[^\n]{{0,40}}\bhand\s*=\s*\[",
        re.IGNORECASE,
    ),
    # "BLUE has W=2,B=1" / "ORANGE: W=3 H=2"
    re.compile(
        rf"\b({_COLOR_ALT})\b[^\n]{{0,60}}\b([WBSHO]|WOOD|BRICK|SHEEP|WHEAT|ORE)\s*=\s*\d",
        re.IGNORECASE,
    ),
    # Explicit "opponent hand" compositions
    re.compile(
        r"\bopponent[^\n]{0,40}\bhand\b[^\n]{0,40}\b([WBSHO]|WOOD|BRICK|SHEEP|WHEAT|ORE)\s*=\s*\d",
        re.IGNORECASE,
    ),
    # Dev-card private literals for opponents: "BLUE KNIGHT=2"
    re.compile(
        rf"\b({_COLOR_ALT})\b[^\n]{{0,40}}\b(KNIGHT|MONOPOLY|YEAR_OF_PLENTY|ROAD_BUILDING|VICTORY_POINT)\s*=\s*\d",
        re.IGNORECASE,
    ),
)


def find_tier_a_leaks(text: str) -> list[str]:
    """Return matched leakage snippets in Tier A / rationale text."""
    hits: list[str] = []
    for pattern in _LEAK_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(match.group(0))
    return hits


def assert_tier_a_pov_safe(text: str, *, context: str = "tier_a") -> None:
    leaks = find_tier_a_leaks(text)
    if leaks:
        raise AssertionError(
            f"{context}: Tier A text leaks opponent private-hand literals: {leaks!r} "
            f"in {text!r}"
        )


def assert_state_pov_safe(state: dict[str, Any], ego_color: str, *, context: str = "state") -> None:
    """Non-ego players must not expose resource/dev hand maps."""
    for player in state.get("players", []):
        color = player.get("color")
        if color == ego_color:
            continue
        if "hand" in player:
            raise AssertionError(
                f"{context}: opponent {color} has private 'hand' in state: {player['hand']!r}"
            )
        if isinstance(player.get("dev_cards"), dict):
            raise AssertionError(
                f"{context}: opponent {color} has private 'dev_cards' map in state"
            )


def assert_user_prompt_pov_safe(user_prompt: str, ego_color: str, *, context: str = "user") -> None:
    """Opponent lines in the canonical user prompt must not show hand=[...]."""
    for color in _COLORS:
        if color == ego_color:
            continue
        # Renderer format: "  BLUE: VP=... cards=N" — never "hand=[" for opponents.
        for line in user_prompt.splitlines():
            if not line.strip().startswith(color):
                continue
            if "hand=[" in line.replace(" ", ""):
                raise AssertionError(
                    f"{context}: opponent line leaks hand composition: {line!r}"
                )
            # Also catch "BLUE ... W=2" style on the status line
            assert_tier_a_pov_safe(line, context=f"{context}:{color}")
