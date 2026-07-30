from pathlib import Path

import pytest

from catan_llm.data.dataset import split_by_game_key
from catan_llm.data.identity import bot_config_hash, make_game_key
from catan_llm.data.schema import DecisionRecord, ExpertPolicy
from catan_llm.data.seed_registry import (
    get_seed_range,
    load_seed_registry,
    resolve_generation_seeds,
)
from catan_llm.sim.adapter import generate_trajectories


def test_seed_registry_loads_train_main():
    ranges = load_seed_registry()
    assert "train_main" in ranges
    tm = ranges["train_main"]
    assert tm.start == 0
    assert tm.count == 50_000
    assert tm.end == 50_000


def test_resolve_seed_range_clamps_games():
    base, n, rng = resolve_generation_seeds(
        num_games=999, seed=None, seed_range_name="hw_smoke"
    )
    assert rng is not None
    assert base == 900_000
    assert n == 100  # full range is only 100


def test_generate_with_seed_range_name(tmp_path: Path):
    out = tmp_path / "t.jsonl"
    summary = generate_trajectories(
        bot_names=["random", "random", "random", "random"],
        num_games=2,
        seed_range_name="hw_smoke",
        out_path=out,
        vps_to_win=6,
        overwrite=True,
    )
    assert summary["base_seed"] == 900_000
    assert summary["seed_range"]["name"] == "hw_smoke"
    assert summary["num_games"] == 2
    assert "cohort_note" in summary


def test_split_by_game_key_deterministic_not_uuid():
    bot_config = [{"name": "random", "params": {}}]
    cfg_hash = bot_config_hash(bot_config)

    def make_rec(seed: int, game_id: str) -> DecisionRecord:
        map_hash = f"{seed:064d}"[:64]
        return DecisionRecord(
            game_key=make_game_key(seed, map_hash, cfg_hash),
            game_id=game_id,
            decision_idx=0,
            seed=seed,
            map_hash=map_hash,
            bot_config=bot_config,
            bot_config_hash=cfg_hash,
            player_color="RED",
            turn=0,
            phase="ROLL",
            state={},
            system_prompt="S",
            user_prompt="U",
            valid_actions=[{"color": "RED", "action_type": "ROLL", "value": None}],
            action_taken={"color": "RED", "action_type": "ROLL", "value": None},
            action_index=0,
            expert_policy=ExpertPolicy.RANDOM,
        )

    # Same game_keys, different UUIDs → same split assignment.
    records_a = [make_rec(i, f"uuid-a-{i}") for i in range(10)]
    records_b = [make_rec(i, f"uuid-b-{i}") for i in range(10)]
    split_a = split_by_game_key(records_a)
    split_b = split_by_game_key(records_b)
    keys_a = {r.game_key for r in split_a["train"]}
    keys_b = {r.game_key for r in split_b["train"]}
    assert keys_a == keys_b
    # All rows for a game_key stay together.
    train_keys = {r.game_key for r in split_a["train"]}
    val_keys = {r.game_key for r in split_a["val"]}
    test_keys = {r.game_key for r in split_a["test"]}
    assert train_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(test_keys)


def test_unknown_seed_range():
    with pytest.raises(KeyError):
        get_seed_range("not_a_real_range")
