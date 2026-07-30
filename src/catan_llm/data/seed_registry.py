"""Seed ranges loaded from docs/SEED_REGISTRY.md (single source of truth)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class SeedRange:
    name: str
    start: int
    count: int
    purpose: str
    status: str = "reserved"

    @property
    def end(self) -> int:
        """Exclusive end seed."""
        return self.start + self.count

    def clamp_games(self, num_games: int, *, start: int | None = None) -> tuple[int, int]:
        """Return (base_seed, clamped_num_games) within this range."""
        base = self.start if start is None else start
        if base < self.start or base >= self.end:
            raise ValueError(
                f"seed {base} outside range {self.name} [{self.start}, {self.end})"
            )
        max_games = self.end - base
        if num_games < 1:
            raise ValueError("num_games must be >= 1")
        return base, min(num_games, max_games)


_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*([\d_]+)\s*\|\s*([\d_]+)\s*\|\s*([\d_]+)\s*\|\s*(.*?)\s*\|\s*(\w+)\s*\|$"
)


def _registry_path() -> Path:
    # src/catan_llm/data/seed_registry.py → repo root
    return Path(__file__).resolve().parents[3] / "docs" / "SEED_REGISTRY.md"


def parse_seed_registry(text: str) -> dict[str, SeedRange]:
    ranges: dict[str, SeedRange] = {}
    for line in text.splitlines():
        match = _ROW_RE.match(line.strip())
        if not match:
            continue
        name, start_s, count_s, end_s, purpose, status = match.groups()
        start = int(start_s.replace("_", ""))
        count = int(count_s.replace("_", ""))
        end = int(end_s.replace("_", ""))
        if start + count != end:
            raise ValueError(
                f"SEED_REGISTRY inconsistent for {name}: start+count={start + count} end={end}"
            )
        ranges[name] = SeedRange(
            name=name,
            start=start,
            count=count,
            purpose=purpose.strip(),
            status=status.strip(),
        )
    if not ranges:
        raise ValueError("No seed ranges parsed from SEED_REGISTRY.md")
    return ranges


@lru_cache(maxsize=1)
def load_seed_registry(path: str | None = None) -> dict[str, SeedRange]:
    registry_path = Path(path) if path else _registry_path()
    return parse_seed_registry(registry_path.read_text(encoding="utf-8"))


def get_seed_range(name: str, *, path: str | None = None) -> SeedRange:
    ranges = load_seed_registry(path)
    if name not in ranges:
        known = ", ".join(sorted(ranges))
        raise KeyError(f"Unknown seed range {name!r}. Known: {known}")
    return ranges[name]


def resolve_generation_seeds(
    *,
    num_games: int,
    seed: int | None = None,
    seed_range_name: str | None = None,
) -> tuple[int, int, SeedRange | None]:
    """Resolve (base_seed, num_games, range_or_none) for trajectory generation.

    Cohort guidance (SCOPE §5.2): named train ranges are reservations — stop when
    filtered decision targets are met; do not blindly burn the full count.
    """
    if seed_range_name is None:
        base = 0 if seed is None else seed
        return base, num_games, None
    rng = get_seed_range(seed_range_name)
    base, clamped = rng.clamp_games(num_games, start=seed)
    return base, clamped, rng
