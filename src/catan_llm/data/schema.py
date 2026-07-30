"""Trajectory schema v1 — structured decision records.

See docs/SCOPE.md §4.2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExpertPolicy(str, Enum):
    ALPHABETA = "alphabeta"
    VALUEFUNCTION = "valuefunction"
    WEIGHTEDRANDOM = "weightedrandom"
    RANDOM = "random"
    LLM = "llm"
    OTHER = "other"


class GameOutcome(BaseModel):
    winner: str | None
    vps: dict[str, int]
    turns: int
    finished: bool = True


class ActionRecord(BaseModel):
    """Compact, JSON-stable action representation."""

    color: str
    action_type: str
    value: Any = None


class DecisionRecord(BaseModel):
    """One decision point in a game trajectory (schema v1)."""

    schema_version: str = "v1"
    game_id: str
    decision_idx: int
    seed: int
    map_type: str = "BASE"
    player_color: str
    turn: int
    phase: str
    state: dict[str, Any]
    valid_actions: list[ActionRecord]
    action_taken: ActionRecord
    action_index: int
    expert_policy: ExpertPolicy
    outcome: GameOutcome | None = None  # filled at game end / dataset finalize


class DatasetManifest(BaseModel):
    """Versioned dataset manifest (generator metadata + checksums)."""

    name: str
    version: str
    schema_version: str = "v1"
    created_at: str
    generator_versions: dict[str, str]
    bot_mix: list[str]
    seeds: list[int]
    num_games: int
    num_decisions: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
