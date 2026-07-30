"""Feature-aware Tier A rationales (ticket 10 / SCOPE §7.4)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from catan_llm.data.dataset import decision_to_chat, tier_a_rationale
from catan_llm.data.pov import assert_tier_a_pov_safe
from catan_llm.data.tier_a import is_feature_aware, render_tier_a_rationale
from catan_llm.sim.adapter import generate_trajectories, play_one
from catan_llm.sim.trajectories import read_jsonl

SAMPLES_JSONL = (
    Path(__file__).resolve().parents[1] / "docs" / "samples" / "tier_a_rationales.jsonl"
)
SAMPLES_MD = SAMPLES_JSONL.with_suffix(".md")


def test_feature_aware_covers_key_action_types():
    # Play a few seeds to collect settlement/road/robber/trade samples.
    by_type: dict[str, list[str]] = defaultdict(list)
    for seed in range(8):
        result = play_one(
            ["alphabeta", "weightedrandom", "random", "valuefunction"],
            seed=seed,
            vps_to_win=6,
        )
        for record in result.records:
            text = tier_a_rationale(record)
            assert_tier_a_pov_safe(text)
            assert is_feature_aware(text), text
            by_type[record.action_taken.action_type].append(text)

    assert by_type["BUILD_SETTLEMENT"]
    assert any("pips=" in t for t in by_type["BUILD_SETTLEMENT"])
    assert by_type["BUILD_ROAD"]
    assert any("longest_road" in t or "toward open" in t for t in by_type["BUILD_ROAD"])
    if by_type["MOVE_ROBBER"]:
        assert any("pip" in t or "steal from" in t for t in by_type["MOVE_ROBBER"])
    if by_type["MARITIME_TRADE"]:
        assert any("maritime" in t for t in by_type["MARITIME_TRADE"])


def test_value_delta_appended_when_present():
    result = play_one(["random", "random", "random", "random"], seed=2, vps_to_win=6)
    record = result.records[0]
    record.state = {**record.state, "value_delta": 0.12}
    text = render_tier_a_rationale(record)
    assert "valueΔ=+0.12" in text
    assert_tier_a_pov_safe(text)


def test_rejects_legacy_restatement_style():
    assert not is_feature_aware("alphabeta expands board position with BUILD_SETTLEMENT")
    assert not is_feature_aware("policy selects BUILD_ROAD")
    assert is_feature_aware("settlement node=12; pips=13 (H+O); port 2:1 O")


def test_small_generated_shard_is_feature_aware(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    summary = generate_trajectories(
        bot_names=["alphabeta", "random", "weightedrandom", "valuefunction"],
        num_games=3,
        seed=100,
        out_path=out,
        vps_to_win=6,
        overwrite=True,
    )
    assert summary["num_decisions"] > 0
    interesting = 0
    for record in read_jsonl(out):
        if record.action_taken.action_type in {"ROLL", "END_TURN"}:
            continue
        chat = decision_to_chat(record)
        reasoning = json.loads(chat["messages"][-1]["content"]).get("reasoning", "")
        assert_tier_a_pov_safe(reasoning)
        assert is_feature_aware(reasoning), reasoning
        interesting += 1
    assert interesting >= 8


def test_committed_sample_shard_for_review():
    assert SAMPLES_JSONL.exists(), "missing docs/samples/tier_a_rationales.jsonl"
    assert SAMPLES_MD.exists(), "missing docs/samples/tier_a_rationales.md"
    rows = []
    with SAMPLES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    assert len(rows) >= 8
    for row in rows:
        assert_tier_a_pov_safe(row["reasoning"])
        assert is_feature_aware(row["reasoning"])
