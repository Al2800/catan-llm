"""Eval metrics: win rates, confidence intervals, Gate-B fields."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from catan_llm.data.parser import FALLBACK_POLICY


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class MatchStats:
    games: int = 0
    wins: dict[str, int] | None = None
    unfinished: int = 0
    total_turns: int = 0
    legality_ok: int = 0
    legality_total: int = 0
    parse_ok: int = 0
    parse_total: int = 0
    fallback_count: int = 0
    model_calls: int = 0
    seat_names: list[str] = field(default_factory=list)
    vp_margins: list[float] = field(default_factory=list)
    action_error_hist: Counter[str] = field(default_factory=Counter)
    phase_error_hist: Counter[str] = field(default_factory=Counter)

    def __post_init__(self):
        if self.wins is None:
            self.wins = defaultdict(int)

    def register_seats(self, names: list[str]) -> None:
        """Ensure zero-win seats still appear in win_rates."""
        self.seat_names = list(names)
        assert self.wins is not None
        for name in names:
            self.wins.setdefault(name, 0)

    def add_game(
        self,
        winner: str | None,
        turns: int,
        *,
        vp_margin: float | None = None,
    ):
        self.games += 1
        self.total_turns += turns
        if winner is None:
            self.unfinished += 1
        else:
            assert self.wins is not None
            self.wins[winner] += 1
        if vp_margin is not None and winner is not None:
            self.vp_margins.append(vp_margin)

    def add_model_call(self, *, parsed: bool, legal: bool, fallback: bool) -> None:
        self.model_calls += 1
        self.parse_total += 1
        self.legality_total += 1
        if parsed:
            self.parse_ok += 1
        if legal:
            self.legality_ok += 1
        if fallback:
            self.fallback_count += 1

    def absorb_llm_counters(
        self,
        parse_ok: int,
        parse_total: int,
        legal_ok: int,
        legal_total: int,
        fallback_count: int = 0,
        action_error_hist: dict[str, int] | None = None,
        phase_error_hist: dict[str, int] | None = None,
    ) -> None:
        """Bulk-add per-game LLMPlayer counters (Gate-B accounting)."""
        self.parse_ok += parse_ok
        self.parse_total += parse_total
        self.legality_ok += legal_ok
        self.legality_total += legal_total
        self.fallback_count += fallback_count
        self.model_calls += parse_total
        if action_error_hist:
            self.action_error_hist.update(action_error_hist)
        if phase_error_hist:
            self.phase_error_hist.update(phase_error_hist)

    def add_decision(self, *, parsed: bool | None = None, legal: bool | None = None):
        """Legacy helper — prefer ``absorb_llm_counters`` for Gate-B accounting."""
        if parsed is not None:
            self.parse_total += 1
            if parsed:
                self.parse_ok += 1
        if legal is not None:
            self.legality_total += 1
            if legal:
                self.legality_ok += 1

    def win_rate(self, name: str) -> float:
        assert self.wins is not None
        finished = self.games - self.unfinished
        if finished <= 0:
            return 0.0
        return self.wins.get(name, 0) / finished

    def summary(
        self,
        *,
        candidate: str = "candidate",
        versus: str = "weightedrandom",
    ) -> dict:
        assert self.wins is not None
        finished = self.games - self.unfinished
        names = list(self.seat_names) if self.seat_names else sorted(self.wins)
        for name in self.wins:
            if name not in names:
                names.append(name)

        win_rates = {}
        for name in names:
            wins = int(self.wins.get(name, 0))
            lo, hi = wilson_interval(wins, finished)
            win_rates[name] = {
                "wins": wins,
                "rate": wins / finished if finished else 0.0,
                "wilson95": [lo, hi],
            }

        gap = self.win_rate(candidate) - self.win_rate(versus)
        vp_margin = (
            sum(self.vp_margins) / len(self.vp_margins) if self.vp_margins else None
        )
        model_n = self.model_calls or self.parse_total
        return {
            "games": self.games,
            "finished": finished,
            "unfinished": self.unfinished,
            "avg_turns": (self.total_turns / self.games) if self.games else 0.0,
            "win_rates": win_rates,
            "win_share_gap": {
                f"{candidate},{versus}": gap,
                f"{candidate}_vs_{versus}": gap,
            },
            "vp_margin": vp_margin,
            "parse_rate_model": (self.parse_ok / model_n) if model_n else None,
            "legality_rate_model": (self.legality_ok / model_n) if model_n else None,
            "fallback_rate": (self.fallback_count / model_n) if model_n else None,
            "fallback_policy": FALLBACK_POLICY,
            "action_error_hist": dict(sorted(self.action_error_hist.items())),
            "phase_error_hist": dict(sorted(self.phase_error_hist.items())),
            # Legacy aliases used by older smoke tests / reports.
            "parse_rate": (self.parse_ok / self.parse_total) if self.parse_total else None,
            "legality_rate": (
                self.legality_ok / self.legality_total if self.legality_total else None
            ),
        }
