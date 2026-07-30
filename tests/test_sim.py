from pathlib import Path

from catan_llm.sim.adapter import generate_trajectories, play_one
from catan_llm.sim.trajectories import read_jsonl


def test_play_one_records_decisions():
    result = play_one(
        ["random", "random", "weightedrandom", "random"],
        seed=123,
        vps_to_win=6,
    )
    assert result.num_decisions > 0
    assert result.outcome.turns > 0
    assert result.records[0].schema_version == "v1"
    assert result.records[0].outcome is not None
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
    )
    assert summary["num_games"] == 2
    assert summary["num_decisions"] > 0
    records = read_jsonl(out)
    assert len(records) == summary["num_decisions"]
    assert records[0].valid_actions
