from pathlib import Path

import pytest

from catan_llm.data.dataset import build_chat_dataset, decision_to_chat
from catan_llm.data.identity import PROMPT_VERSION
from catan_llm.data.schema import DecisionRecord
from catan_llm.sim.adapter import play_one
from catan_llm.sim.trajectories import write_jsonl


def test_build_chat_dataset(tmp_path: Path):
    result = play_one(["random", "random", "random", "random"], seed=9, vps_to_win=6)
    traj = tmp_path / "t.jsonl"
    write_jsonl(traj, result.records)
    out = tmp_path / "ds"
    manifest = build_chat_dataset(traj, out, name="unit", version="v0")
    assert manifest.num_decisions > 0
    assert manifest.schema_version == "v2"
    assert manifest.prompt_version == PROMPT_VERSION
    assert (out / "train.jsonl").exists()
    assert (out / "manifest.json").exists()

    chat = decision_to_chat(result.records[0])
    assert chat["messages"][0]["role"] == "system"
    assert chat["messages"][-1]["role"] == "assistant"
    assert '"action"' in chat["messages"][-1]["content"]


def test_build_chat_dataset_rejects_unknown_prompt(tmp_path: Path):
    result = play_one(["random", "random", "random", "random"], seed=3, vps_to_win=6)
    bad = result.records[0].model_copy(update={"prompt_version": "legacy-compact"})
    traj = tmp_path / "bad.jsonl"
    write_jsonl(traj, [bad])
    with pytest.raises(ValueError, match="unknown prompt_version"):
        build_chat_dataset(traj, tmp_path / "ds")


def test_decision_record_v2_fields_present():
    result = play_one(["alphabeta", "random", "random", "random"], seed=1, vps_to_win=6)
    rec: DecisionRecord = result.records[0]
    assert rec.schema_version == "v2"
    assert rec.game_key
    assert rec.map_hash
    assert rec.bot_config[0]["name"] == "alphabeta"
    assert "depth" in rec.bot_config[0]["params"]
