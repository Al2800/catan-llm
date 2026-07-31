"""Tickets 11 + 13: stop-at-target generation and immutable holdout."""

from __future__ import annotations

from pathlib import Path

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.data.seed_registry import get_seed_range, load_seed_registry
from catan_llm.sim.adapter import (
    count_filtered_decisions,
    generate_trajectories,
    rotate_bot_names,
)


def test_rotate_bot_names_deterministic():
    base = ["alphabeta", "valuefunction", "weightedrandom", "random"]
    assert rotate_bot_names(base, 0) == base
    assert rotate_bot_names(base, 1) == [
        "valuefunction",
        "weightedrandom",
        "random",
        "alphabeta",
    ]
    assert rotate_bot_names(base, 4) == base


def test_target_decisions_stops_early(tmp_path: Path):
    out = tmp_path / "traj.jsonl"
    summary = generate_trajectories(
        bot_names=["random", "weightedrandom", "valuefunction", "random"],
        num_games=50,
        seed=900_000,
        seed_range_name="hw_smoke",
        out_path=out,
        vps_to_win=6,
        workers=1,
        overwrite=True,
        target_decisions=80,
        rotate_seats=False,
    )
    assert summary["stopped_early"] is True
    assert summary["num_filtered_decisions"] >= 80
    assert summary["num_games"] < 50
    assert count_filtered_decisions(out) == summary["num_filtered_decisions"]


def test_train_and_holdout_seed_ranges_disjoint():
    ranges = load_seed_registry()
    hold = ranges["eval_holdout"]
    for name, other in ranges.items():
        if name == "eval_holdout":
            continue
        if not (name.startswith("train_") or name == "val_split_pool"):
            continue
        overlap = other.start < hold.end and hold.start < other.end
        assert not overlap, f"{name} overlaps eval_holdout"


def test_holdout_manifest_immutable(tmp_path: Path):
    raw = tmp_path / "holdout.jsonl"
    summary = generate_trajectories(
        bot_names=["random", "weightedrandom", "valuefunction", "random"],
        num_games=2,
        seed_range_name="eval_holdout",
        out_path=raw,
        vps_to_win=6,
        workers=1,
        overwrite=True,
    )
    assert summary["seed_range"]["name"] == "eval_holdout"
    hold = get_seed_range("eval_holdout")
    # Seeds must lie in holdout range.
    for line in raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json

        seed = int(json.loads(line)["seed"])
        assert hold.start <= seed < hold.end

    manifest = build_chat_dataset(
        raw,
        tmp_path / "ds",
        name="eval-holdout-test",
        version="v0",
        seed_range=summary["seed_range"],
        immutable=True,
        role="eval_holdout",
        split=False,
    )
    assert manifest.immutable is True
    assert manifest.role == "eval_holdout"
    assert (tmp_path / "ds" / "holdout.jsonl").exists()
    assert "holdout.jsonl" in manifest.checksums
