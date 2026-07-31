"""Ticket 12: dataset filters + DATA_CONTRACT §9 quality fields."""

from __future__ import annotations

from pathlib import Path

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.data.identity import PROMPT_VERSION
from catan_llm.data.quality import RARE_ACTION_TYPES, filter_decision_records
from catan_llm.data.schema import (
    ActionRecord,
    DecisionRecord,
    ExpertPolicy,
    GameOutcome,
)
from catan_llm.sim.adapter import generate_trajectories


def _record(
    *,
    game_key: str = "g1",
    action_index: int = 0,
    finished: bool = True,
    action_type: str = "ROLL",
    turn: int = 1,
) -> DecisionRecord:
    valid = [
        ActionRecord(color="RED", action_type=action_type, value=None),
        ActionRecord(color="RED", action_type="END_TURN", value=None),
    ]
    if action_index >= 0:
        taken = valid[action_index]
    else:
        taken = ActionRecord(color="RED", action_type="UNKNOWN", value=None)
    return DecisionRecord(
        schema_version="v2",
        prompt_version=PROMPT_VERSION,
        game_key=game_key,
        game_id="id-" + game_key,
        decision_idx=0,
        seed=1,
        map_type="BASE",
        map_hash="m",
        bot_config=[{"name": "random"}],
        bot_config_hash="b",
        player_color="RED",
        turn=turn,
        phase="PLAY",
        state={},
        system_prompt="sys",
        user_prompt="user AVAILABLE ACTIONS",
        valid_actions=valid,
        action_taken=taken,
        action_index=action_index,
        expert_policy=ExpertPolicy.RANDOM,
        outcome=GameOutcome(
            winner="RED" if finished else None,
            vps={"RED": 10 if finished else 2},
            turns=10,
            finished=finished,
        ),
    )


def test_filter_drops_unfinished_and_illegal():
    rows = [
        _record(game_key="ok", action_index=0, finished=True),
        _record(game_key="bad", action_index=0, finished=False),
        _record(game_key="ok2", action_index=-1, finished=True),
    ]
    kept, stats = filter_decision_records(rows, check_truncation=False)
    assert len(kept) == 1
    assert stats.unfinished_dropped == 1
    assert stats.illegal_dropped == 1
    assert stats.kept == 1


def test_build_chat_dataset_writes_quality_manifest(tmp_path: Path):
    traj = tmp_path / "traj.jsonl"
    generate_trajectories(
        bot_names=["random", "weightedrandom", "valuefunction", "random"],
        num_games=2,
        seed=900_000,
        seed_range_name="hw_smoke",
        out_path=traj,
        vps_to_win=6,
        overwrite=True,
    )
    manifest = build_chat_dataset(
        traj,
        tmp_path / "ds",
        name="quality-smoke",
        version="v0",
        check_truncation=False,
        seed_range={"name": "hw_smoke"},
    )
    assert manifest.max_seq_length >= 4096
    assert "unfinished_dropped" in manifest.quality
    assert "action_type_hist" in manifest.quality
    assert "rare_action_hist" in manifest.quality
    assert "turn_tercile_hist" in manifest.quality
    assert manifest.num_decisions == manifest.quality["kept"]
    assert (tmp_path / "ds" / "quality.json").exists()
    assert set(manifest.split_counts) == {"train", "val", "test"}
    assert all(manifest.checksums.values())


def test_rare_action_constant_covers_contract_buckets():
    for name in (
        "PLAY_MONOPOLY",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_ROAD_BUILDING",
        "PLAY_KNIGHT_CARD",
        "BUY_DEVELOPMENT_CARD",
        "MOVE_ROBBER",
    ):
        assert name in RARE_ACTION_TYPES
