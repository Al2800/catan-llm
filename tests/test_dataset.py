from pathlib import Path

from catan_llm.data.dataset import build_chat_dataset, decision_to_chat
from catan_llm.sim.adapter import play_one


def test_build_chat_dataset(tmp_path: Path):
    result = play_one(["random", "random", "random", "random"], seed=9, vps_to_win=6)
    traj = tmp_path / "t.jsonl"
    from catan_llm.sim.trajectories import write_jsonl

    write_jsonl(traj, result.records)
    out = tmp_path / "ds"
    manifest = build_chat_dataset(traj, out, name="unit", version="v0")
    assert manifest.num_decisions > 0
    assert (out / "train.jsonl").exists()
    assert (out / "manifest.json").exists()

    chat = decision_to_chat(result.records[0])
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][-1]["role"] == "assistant"
    assert '"action"' in chat["messages"][-1]["content"]
