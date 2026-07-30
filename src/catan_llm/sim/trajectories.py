"""Trajectory recording via Catanatron GameAccumulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catanatron.game import Game, GameAccumulator
from catanatron.state_functions import get_actual_victory_points

from catan_llm.data.actions import action_to_record
from catan_llm.data.identity import (
    CATANATRON_COMMIT,
    compact_board_dict,
    resolve_source_commit,
)
from catan_llm.data.renderer import (
    PROMPT_VERSION,
    compact_state_dict,
    render_system_prompt,
    render_user_prompt,
)
from catan_llm.data.schema import SCHEMA_VERSION, DecisionRecord, ExpertPolicy, GameOutcome
from catan_llm.sim.players import policy_for_player


class TrajectoryAccumulator(GameAccumulator):
    """Records one DecisionRecord per engine decision."""

    def __init__(
        self,
        *,
        seed: int,
        map_type: str = "BASE",
        map_hash: str,
        bot_config: list[dict[str, Any]],
        bot_config_hash: str,
        game_key: str,
        policy_by_color: dict[str, ExpertPolicy] | None = None,
        catanatron_commit: str = CATANATRON_COMMIT,
        source_commit: str | None = None,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.seed = seed
        self.map_type = map_type
        self.map_hash = map_hash
        self.bot_config = bot_config
        self.bot_config_hash = bot_config_hash
        self.game_key = game_key
        self.policy_by_color = policy_by_color or {}
        self.catanatron_commit = catanatron_commit
        self.source_commit = source_commit if source_commit is not None else resolve_source_commit()
        self.prompt_version = prompt_version
        self.records: list[DecisionRecord] = []
        self.game_id: str | None = None
        self.outcome: GameOutcome | None = None
        self._board: dict[str, Any] = {}

    def before(self, game: Game):
        self.game_id = game.id
        self.records.clear()
        self.outcome = None
        self._board = compact_board_dict(game)

    def step(self, game_before_action: Game, action):
        playable = list(game_before_action.playable_actions)
        try:
            action_index = playable.index(action)
        except ValueError:
            # Rare: engine accepts some actions not listed (e.g. domestic trade).
            action_index = -1

        color = action.color
        policy = self.policy_by_color.get(
            color.value, policy_for_player(game_before_action.state.current_player())
        )
        # Live renderer only — same functions as LLMPlayer (DATA_CONTRACT §4).
        system_prompt = render_system_prompt(game_before_action, color)
        user_prompt = render_user_prompt(game_before_action, color, playable)

        record = DecisionRecord(
            schema_version=SCHEMA_VERSION,
            prompt_version=self.prompt_version,
            game_key=self.game_key,
            game_id=self.game_id or game_before_action.id,
            decision_idx=len(self.records),
            seed=self.seed,
            map_type=self.map_type,
            map_hash=self.map_hash,
            bot_config=self.bot_config,
            bot_config_hash=self.bot_config_hash,
            catanatron_commit=self.catanatron_commit,
            source_commit=self.source_commit,
            player_color=color.value,
            turn=game_before_action.state.num_turns,
            phase=game_before_action.state.current_prompt.value,
            board=self._board,
            state=compact_state_dict(game_before_action, color),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            valid_actions=[action_to_record(a) for a in playable],
            action_taken=action_to_record(action),
            action_index=action_index,
            expert_policy=policy,
        )
        self.records.append(record)

    def after(self, game: Game):
        winner = game.winning_color()
        vps = {
            color.value: get_actual_victory_points(game.state, color)
            for color in game.state.colors
        }
        self.outcome = GameOutcome(
            winner=winner.value if winner else None,
            vps=vps,
            turns=game.state.num_turns,
            finished=winner is not None,
        )
        for record in self.records:
            record.outcome = self.outcome


def write_jsonl(path: Path, records: list[DecisionRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    return len(records)


def read_jsonl(path: Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(DecisionRecord.model_validate_json(line))
    return records


def append_jsonl(path: Path, records: list[DecisionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
        fh.flush()


def dump_outcome_summary(path: Path, outcomes: list[GameOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [o.model_dump() for o in outcomes]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def journal_path_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".journal")


def load_completed_game_keys(journal_path: Path) -> set[str]:
    if not journal_path.exists():
        return set()
    keys: set[str] = set()
    with journal_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                keys.add(payload["game_key"])
            except (json.JSONDecodeError, KeyError, TypeError):
                # Plain game_key lines are also accepted.
                keys.add(line)
    return keys


def append_journal(journal_path: Path, *, game_key: str, seed: int) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"game_key": game_key, "seed": seed}, sort_keys=True) + "\n")
        fh.flush()
