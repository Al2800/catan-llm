"""Dataset filtering + quality stats (ticket 12 / DATA_CONTRACT §7, §9)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from catan_llm.data.parser import format_assistant_target
from catan_llm.data.schema import DecisionRecord
from catan_llm.data.tier_a import render_tier_a_rationale
from catan_llm.training.masking import build_assistant_only_labels, qwen_max_seq_length

RARE_ACTION_TYPES = frozenset(
    {
        "PLAY_MONOPOLY",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_ROAD_BUILDING",
        "PLAY_KNIGHT_CARD",
        "BUY_DEVELOPMENT_CARD",
        "MOVE_ROBBER",
        "MARITIME_TRADE",
    }
)


@dataclass
class FilterStats:
    raw_decisions: int = 0
    unfinished_dropped: int = 0
    illegal_dropped: int = 0
    truncated_dropped: int = 0
    kept: int = 0
    truncation_check: str = "skipped"
    action_type_hist: dict[str, int] = field(default_factory=dict)
    phase_hist: dict[str, int] = field(default_factory=dict)
    rare_action_hist: dict[str, int] = field(default_factory=dict)
    turn_tercile_hist: dict[str, int] = field(default_factory=dict)

    def as_quality_dict(self) -> dict[str, Any]:
        return {
            "unfinished_dropped": self.unfinished_dropped,
            "illegal_dropped": self.illegal_dropped,
            "truncated_dropped": self.truncated_dropped,
            "raw_decisions": self.raw_decisions,
            "kept": self.kept,
            "truncation_check": self.truncation_check,
            "action_type_hist": dict(sorted(self.action_type_hist.items())),
            "phase_hist": dict(sorted(self.phase_hist.items())),
            "rare_action_hist": dict(sorted(self.rare_action_hist.items())),
            "turn_tercile_hist": dict(sorted(self.turn_tercile_hist.items())),
        }


def _unfinished_game_keys(records: list[DecisionRecord]) -> set[str]:
    unfinished: set[str] = set()
    for r in records:
        if r.outcome is not None and r.outcome.finished is False:
            unfinished.add(r.game_key)
    return unfinished


def _turn_tercile(turn: int, max_turn: int) -> str:
    if max_turn <= 0:
        return "early"
    # 1-indexed-ish buckets over [0, max_turn]
    t = max(0, min(turn, max_turn))
    third = max(1, (max_turn + 1) // 3)
    if t < third:
        return "early"
    if t < 2 * third:
        return "mid"
    return "late"


def _assistant_truncated(
    record: DecisionRecord,
    tokenizer,
    *,
    max_seq_length: int,
    include_rationale: bool,
) -> bool:
    reasoning = render_tier_a_rationale(record) if include_rationale else ""
    assistant = format_assistant_target(record.action_index, reasoning)
    messages = [
        {"role": "system", "content": record.system_prompt},
        {"role": "user", "content": record.user_prompt},
        {"role": "assistant", "content": assistant},
    ]
    batch = build_assistant_only_labels(
        tokenizer, messages, max_seq_length=max_seq_length
    )
    return batch.truncated


def filter_decision_records(
    records: list[DecisionRecord],
    *,
    max_seq_length: int | None = None,
    include_rationale: bool = True,
    tokenizer=None,
    check_truncation: bool = True,
) -> tuple[list[DecisionRecord], FilterStats]:
    """Apply DATA_CONTRACT §7 drops and collect quality histograms."""
    stats = FilterStats(raw_decisions=len(records))
    max_len = int(max_seq_length if max_seq_length is not None else qwen_max_seq_length())
    unfinished = _unfinished_game_keys(records)

    if check_truncation and tokenizer is not None:
        stats.truncation_check = "enforced"
    elif check_truncation:
        stats.truncation_check = "skipped_no_tokenizer"
    else:
        stats.truncation_check = "disabled"

    kept: list[DecisionRecord] = []
    for r in records:
        if r.game_key in unfinished:
            stats.unfinished_dropped += 1
            continue
        if r.action_index < 0:
            stats.illegal_dropped += 1
            continue
        if not r.system_prompt or not r.user_prompt:
            stats.illegal_dropped += 1
            continue
        # Legality: action_index must address a listed valid action.
        if r.action_index >= len(r.valid_actions):
            stats.illegal_dropped += 1
            continue
        listed = r.valid_actions[r.action_index]
        if listed.action_type != r.action_taken.action_type:
            stats.illegal_dropped += 1
            continue
        if (
            check_truncation
            and tokenizer is not None
            and _assistant_truncated(
                r, tokenizer, max_seq_length=max_len, include_rationale=include_rationale
            )
        ):
            stats.truncated_dropped += 1
            continue
        kept.append(r)

    stats.kept = len(kept)
    action_hist: Counter[str] = Counter()
    phase_hist: Counter[str] = Counter()
    rare_hist: Counter[str] = Counter()
    max_turn_by_game: dict[str, int] = {}
    for r in kept:
        max_turn_by_game[r.game_key] = max(max_turn_by_game.get(r.game_key, 0), r.turn)
    tercile_hist: Counter[str] = Counter()
    for r in kept:
        at = r.action_taken.action_type
        action_hist[at] += 1
        phase_hist[r.phase or "unknown"] += 1
        if at in RARE_ACTION_TYPES or "TRADE" in at:
            rare_hist[at] += 1
        tercile_hist[_turn_tercile(r.turn, max_turn_by_game.get(r.game_key, 0))] += 1

    stats.action_type_hist = dict(action_hist)
    stats.phase_hist = dict(phase_hist)
    stats.rare_action_hist = dict(rare_hist)
    stats.turn_tercile_hist = dict(tercile_hist)
    return kept, stats


def try_load_truncation_tokenizer(*, prefer_qwen: bool = False):
    """Best-effort tokenizer for truncation checks (optional in CI).

    Default uses SmolLM (CI-friendly). Set prefer_qwen=True (or env
    CATAN_LLM_QWEN_TOKENIZER=1) to load the pinned Qwen tokenizer.
    """
    import os

    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    use_qwen = prefer_qwen or os.environ.get("CATAN_LLM_QWEN_TOKENIZER") == "1"
    if use_qwen:
        try:
            from catan_llm.training.masking import qwen_model_name, qwen_revision

            return AutoTokenizer.from_pretrained(
                qwen_model_name(),
                revision=qwen_revision(),
                trust_remote_code=True,
            )
        except Exception:
            pass
    try:
        return AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct", trust_remote_code=True
        )
    except Exception:
        return None
