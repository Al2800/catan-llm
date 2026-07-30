from pathlib import Path

import pytest

from catan_llm.data.identity import PROMPT_VERSION, SCHEMA_VERSION, hash_catan_map
from catan_llm.sim.adapter import (
    build_catan_map,
    generate_trajectories,
    map_for_seed,
    play_one,
)
from catan_llm.sim.trajectories import journal_path_for, load_completed_game_keys, read_jsonl


def test_play_one_records_v2_decisions():
    result = play_one(
        ["random", "random", "weightedrandom", "random"],
        seed=123,
        vps_to_win=6,
    )
    assert result.num_decisions > 0
    assert result.outcome.turns > 0
    assert result.game_key
    rec = result.records[0]
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.prompt_version == PROMPT_VERSION
    assert rec.game_key == result.game_key
    assert rec.map_hash == result.map_hash
    assert rec.bot_config
    assert rec.bot_config_hash
    assert rec.catanatron_commit
    assert rec.source_commit
    assert rec.outcome is not None
    assert result.records[-1].action_index >= 0


def test_generate_trajectories_jsonl(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    summary = generate_trajectories(
        bot_names=["random", "random", "random", "random"],
        num_games=2,
        seed=0,
        out_path=out,
        vps_to_win=6,
        workers=1,
        overwrite=True,
    )
    assert summary["num_games"] == 2
    assert summary["num_decisions"] > 0
    records = read_jsonl(out)
    assert len(records) == summary["num_decisions"]
    assert records[0].valid_actions
    assert records[0].schema_version == "v2"
    assert journal_path_for(out).exists()


def test_mini_map_fail_loud_and_differs_from_base():
    mini = build_catan_map("MINI")
    base = build_catan_map("BASE")
    assert len(mini.land_tiles) != len(base.land_tiles)
    assert hash_catan_map(mini) != hash_catan_map(base)

    mini_s = map_for_seed("MINI", seed=7, n_players=4)
    base_s = map_for_seed("BASE", seed=7, n_players=4)
    assert hash_catan_map(mini_s) != hash_catan_map(base_s)

    with pytest.raises(ValueError, match="Unsupported map_type"):
        build_catan_map("TOURNAMENT")


def test_play_one_mini():
    result = play_one(
        ["random", "random", "random", "random"],
        seed=5,
        map_type="MINI",
        vps_to_win=6,
    )
    assert result.num_decisions > 0
    assert result.records[0].map_type == "MINI"
    assert len(result.records[0].board.get("tiles", [])) == 7


def test_resume_safe_skips_completed(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    first = generate_trajectories(
        bot_names=["random", "random", "random", "random"],
        num_games=2,
        seed=10,
        out_path=out,
        vps_to_win=6,
        workers=1,
        overwrite=True,
    )
    assert first["num_games"] == 2
    n_after_first = len(read_jsonl(out))
    keys = load_completed_game_keys(journal_path_for(out))
    assert len(keys) == 2

    second = generate_trajectories(
        bot_names=["random", "random", "random", "random"],
        num_games=2,
        seed=10,
        out_path=out,
        vps_to_win=6,
        workers=1,
        resume=True,
    )
    assert second["num_games"] == 0
    assert second["skipped_games"] == 2
    assert len(read_jsonl(out)) == n_after_first


def test_no_resume_refuses_clobber(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    generate_trajectories(
        bot_names=["random", "random", "random", "random"],
        num_games=1,
        seed=0,
        out_path=out,
        vps_to_win=6,
        overwrite=True,
    )
    with pytest.raises(FileExistsError):
        generate_trajectories(
            bot_names=["random", "random", "random", "random"],
            num_games=1,
            seed=1,
            out_path=out,
            vps_to_win=6,
            resume=False,
            overwrite=False,
        )
