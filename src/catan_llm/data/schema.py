"""Trajectory schema v2 — structured decision records.

See docs/DATA_CONTRACT.md.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from catan_llm.data.identity import (
    CATANATRON_COMMIT,
    KNOWN_PROMPT_VERSIONS,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)


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
    """One decision point in a game trajectory (schema v2)."""

    schema_version: str = SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
    game_key: str
    game_id: str
    decision_idx: int
    seed: int
    map_type: str = "BASE"
    map_hash: str
    bot_config: list[dict[str, Any]]
    bot_config_hash: str
    catanatron_commit: str = CATANATRON_COMMIT
    source_commit: str = "unknown"
    player_color: str
    turn: int
    phase: str
    board: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any]
    # Canonical prompts captured via live renderer at decision time (train/play parity).
    system_prompt: str = ""
    user_prompt: str = ""
    valid_actions: list[ActionRecord]
    action_taken: ActionRecord
    action_index: int
    expert_policy: ExpertPolicy
    outcome: GameOutcome | None = None  # filled at game end / dataset finalize

    @field_validator("schema_version")
    @classmethod
    def _schema_must_be_v2(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {value!r}")
        return value


class DatasetManifest(BaseModel):
    """Versioned dataset manifest (generator metadata + checksums)."""

    name: str
    version: str
    schema_version: str = SCHEMA_VERSION
    prompt_version: str = PROMPT_VERSION
    created_at: str
    source_commit: str = "unknown"
    catanatron_commit: str = CATANATRON_COMMIT
    generator_versions: dict[str, str]
    bot_mix: list[str]
    bot_config: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int]
    seed_range: dict[str, Any] | None = None
    map_type: str = "BASE"
    num_games: int
    num_decisions: int
    split_counts: dict[str, int] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    # Ticket 13: eval holdout must never enter training.
    immutable: bool = False
    role: str | None = None  # e.g. "train", "eval_holdout"
    # DATA_CONTRACT §9
    max_seq_length: int = 4096
    quality: dict[str, Any] = Field(default_factory=dict)


def require_schema_v2(records: list[DecisionRecord], *, context: str = "dataset") -> None:
    """Reject non-v2 / unknown-prompt rows on Phase-1 dataset paths."""
    for i, record in enumerate(records):
        if record.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"{context}: row {i} has schema_version={record.schema_version!r}; "
                f"Phase-1 paths require {SCHEMA_VERSION!r}"
            )
        if record.prompt_version not in KNOWN_PROMPT_VERSIONS:
            raise ValueError(
                f"{context}: row {i} has unknown prompt_version={record.prompt_version!r}; "
                f"known={sorted(KNOWN_PROMPT_VERSIONS)}"
            )
        if not record.game_key or not record.map_hash or not record.bot_config_hash:
            raise ValueError(
                f"{context}: row {i} missing game_key/map_hash/bot_config_hash"
            )
        if record.bot_config is None:
            raise ValueError(f"{context}: row {i} missing bot_config")
        if not record.system_prompt or not record.user_prompt:
            raise ValueError(
                f"{context}: row {i} missing system_prompt/user_prompt "
                "(required for train/play parity)"
            )
