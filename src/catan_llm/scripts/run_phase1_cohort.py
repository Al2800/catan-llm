"""Phase-1 cohort orchestration (tickets 11 + 13).

SCOPE §5.2:
  - train_main (+ optional mini): stop at ≥100k filtered train decisions
  - eval_holdout: 5k games, immutable, never trained on
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.data.seed_registry import get_seed_range, load_seed_registry
from catan_llm.sim.adapter import count_filtered_decisions, generate_trajectories

DEFAULT_LADDER = ["alphabeta", "valuefunction", "weightedrandom", "random"]


def _range_meta(name: str) -> dict:
    rng = get_seed_range(name)
    return {
        "name": rng.name,
        "start": rng.start,
        "count": rng.count,
        "end": rng.end,
    }


def _assert_no_train_holdout_overlap() -> None:
    ranges = load_seed_registry()
    train_names = [n for n in ranges if n.startswith("train_") or n == "val_split_pool"]
    hold = ranges["eval_holdout"]
    for name in train_names:
        other = ranges[name]
        overlap = other.start < hold.end and hold.start < other.end
        if overlap:
            raise RuntimeError(f"Seed overlap between {name} and eval_holdout")


@click.group()
def main():
    """Run Phase-1 train / holdout cohort jobs."""


@main.command("train")
@click.option("--target-decisions", default=100_000, show_default=True, type=int)
@click.option("--max-games", default=2_000, show_default=True, type=int)
@click.option("--mini-games", default=300, show_default=True, type=int)
@click.option("--workers", default=4, show_default=True, type=int)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phase1"),
    show_default=True,
)
@click.option("--skip-mini", is_flag=True, help="Skip train_mini_curriculum slice")
@click.option("--skip-dataset", is_flag=True, help="Only generate trajectories")
def train_cmd(target_decisions, max_games, mini_games, workers, out_dir, skip_mini, skip_dataset):
    """Generate train_main (+ optional MINI) until filtered decision target."""
    out_dir = Path(out_dir)
    raw_main = out_dir / "raw" / "train_main.jsonl"
    report: dict = {"slices": [], "target_decisions": target_decisions}

    summary_main = generate_trajectories(
        bot_names=DEFAULT_LADDER,
        num_games=max_games,
        seed_range_name="train_main",
        out_path=raw_main,
        map_type="BASE",
        vps_to_win=10,
        workers=workers,
        resume=True,
        target_decisions=target_decisions,
        rotate_seats=True,
    )
    report["slices"].append({"slice": "A_train_main", **summary_main})
    filtered = int(summary_main["num_filtered_decisions"])

    if not skip_mini and filtered < target_decisions:
        remaining = target_decisions - filtered
        raw_mini = out_dir / "raw" / "train_mini_curriculum.jsonl"
        summary_mini = generate_trajectories(
            bot_names=DEFAULT_LADDER,
            num_games=mini_games,
            seed_range_name="train_mini_curriculum",
            out_path=raw_mini,
            map_type="MINI",
            vps_to_win=8,
            workers=workers,
            resume=True,
            target_decisions=remaining,
            rotate_seats=True,
        )
        report["slices"].append({"slice": "B_train_mini", **summary_mini})
        # Merge mini into a combined train trajectories file for dataset build.
        combined = out_dir / "raw" / "train_combined.jsonl"
        with combined.open("w", encoding="utf-8") as out_fh:
            for path in (raw_main, raw_mini):
                if path.exists():
                    out_fh.write(path.read_text(encoding="utf-8"))
        traj_for_ds = combined
        seed_range = {
            "names": ["train_main", "train_mini_curriculum"],
            "train_main": _range_meta("train_main"),
            "train_mini_curriculum": _range_meta("train_mini_curriculum"),
        }
    else:
        traj_for_ds = raw_main
        seed_range = _range_meta("train_main")

    report["num_filtered_decisions"] = count_filtered_decisions(traj_for_ds)
    report["target_met"] = report["num_filtered_decisions"] >= target_decisions

    if not skip_dataset:
        manifest = build_chat_dataset(
            traj_for_ds,
            out_dir / "processed" / "expert-v1",
            name="expert-v1",
            version="v1",
            seed_range=seed_range if isinstance(seed_range, dict) else seed_range,
            immutable=False,
            role="train",
            split=True,
        )
        # When both slices used, seed_range is a composite dict — OK for manifest.
        report["dataset_manifest"] = json.loads(manifest.model_dump_json())

    report_path = out_dir / "train_cohort_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(json.dumps(report, indent=2))


@main.command("holdout")
@click.option("--games", default=5_000, show_default=True, type=int)
@click.option("--workers", default=4, show_default=True, type=int)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/phase1"),
    show_default=True,
)
@click.option("--skip-dataset", is_flag=True)
def holdout_cmd(games, workers, out_dir, skip_dataset):
    """Generate immutable eval_holdout (ticket 13)."""
    _assert_no_train_holdout_overlap()
    out_dir = Path(out_dir)
    raw = out_dir / "raw" / "eval_holdout.jsonl"
    summary = generate_trajectories(
        bot_names=DEFAULT_LADDER,
        num_games=games,
        seed_range_name="eval_holdout",
        out_path=raw,
        map_type="BASE",
        vps_to_win=10,
        workers=workers,
        resume=True,
        rotate_seats=False,  # fixed ladder for holdout
    )
    report = {"slice": "eval_holdout", **summary}
    seeds_used = set()
    if raw.exists():
        for line in raw.open(encoding="utf-8"):
            if line.strip():
                seeds_used.add(int(json.loads(line)["seed"]))
    hold = get_seed_range("eval_holdout")
    train = get_seed_range("train_main")
    bad = [s for s in seeds_used if train.start <= s < train.end]
    if bad:
        raise RuntimeError(f"Holdout leaked into train_main seeds: {bad[:5]}")
    outside = [s for s in seeds_used if not (hold.start <= s < hold.end)]
    if outside:
        raise RuntimeError(f"Holdout seeds outside eval_holdout range: {outside[:5]}")
    report["seed_overlap_with_train_main"] = 0
    report["seeds_in_eval_holdout_range"] = True

    if not skip_dataset:
        manifest = build_chat_dataset(
            raw,
            out_dir / "processed" / "eval-holdout-v1",
            name="eval-holdout-v1",
            version="v1",
            seed_range=_range_meta("eval_holdout"),
            immutable=True,
            role="eval_holdout",
            split=False,
        )
        report["dataset_manifest"] = json.loads(manifest.model_dump_json())
        if not manifest.immutable:
            raise RuntimeError("Holdout manifest missing immutable=true")

    report_path = out_dir / "holdout_cohort_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
