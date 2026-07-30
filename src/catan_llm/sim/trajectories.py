"""Trajectory recording via Catanatron GameAccumulator."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.game import Game, GameAccumulator
from catanatron.state_functions import get_actual_victory_points

from catan_llm.data.actions import action_to_record
from catan_llm.data.renderer import compact_state_dict
from catan_llm.data.schema import DecisionRecord, ExpertPolicy, GameOutcome
from catan_llm.sim.players import policy_for_player


class TrajectoryAccumulator(GameAccumulator):
    """Records one DecisionRecord per engine decision."""

    def __init__(
        self,
        *,
        seed: int,
        map_type: str = "BASE",
        policy_by_color: dict[str, ExpertPolicy] | None = None,
    ):
        self.seed = seed
        self.map_type = map_type
        self.policy_by_color = policy_by_color or {}
        self.records: list[DecisionRecord] = []
        self.game_id: str | None = None
        self.outcome: GameOutcome | None = None

    def before(self, game: Game):
        self.game_id = game.id
        self.records.clear()
        self.outcome = None

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

        record = DecisionRecord(
            game_id=self.game_id or game_before_action.id,
            decision_idx=len(self.records),
            seed=self.seed,
            map_type=self.map_type,
            player_color=color.value,
            turn=game_before_action.state.num_turns,
            phase=game_before_action.state.current_prompt.value,
            state=compact_state_dict(game_before_action, color),
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


def dump_outcome_summary(path: Path, outcomes: list[GameOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [o.model_dump() for o in outcomes]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
