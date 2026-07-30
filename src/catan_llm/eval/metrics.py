"""Eval metrics: win rates, confidence intervals, legality."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


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

    def __post_init__(self):
        if self.wins is None:
            self.wins = defaultdict(int)

    def add_game(self, winner: str | None, turns: int):
        self.games += 1
        self.total_turns += turns
        if winner is None:
            self.unfinished += 1
        else:
            assert self.wins is not None
            self.wins[winner] += 1

    def add_decision(self, *, parsed: bool | None = None, legal: bool | None = None):
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

    def summary(self) -> dict:
        assert self.wins is not None
        finished = self.games - self.unfinished
        win_rates = {}
        for name, wins in self.wins.items():
            lo, hi = wilson_interval(wins, finished)
            win_rates[name] = {
                "wins": wins,
                "rate": wins / finished if finished else 0.0,
                "wilson95": [lo, hi],
            }
        return {
            "games": self.games,
            "finished": finished,
            "unfinished": self.unfinished,
            "avg_turns": (self.total_turns / self.games) if self.games else 0.0,
            "win_rates": win_rates,
            "parse_rate": (self.parse_ok / self.parse_total) if self.parse_total else None,
            "legality_rate": (
                self.legality_ok / self.legality_total if self.legality_total else None
            ),
        }
