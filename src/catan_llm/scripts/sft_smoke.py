"""CLI: Phase-0 end-to-end smoke — generate → dataset → tiny SFT → legality check."""

from __future__ import annotations

import json
from pathlib import Path

import click
from catanatron.models.player import Color, RandomPlayer

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.eval.arena import SeatSpec, run_match
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.sim.adapter import generate_trajectories
from catan_llm.training.sft import DEFAULT_SMOKE_MODEL, run_sft_smoke


@click.command()
@click.option("--games", default=8, show_default=True, help="Expert games to generate")
@click.option("--seed", default=0, show_default=True)
@click.option("--max-steps", default=20, show_default=True)
@click.option("--max-samples", default=128, show_default=True)
@click.option("--model", default=DEFAULT_SMOKE_MODEL, show_default=True)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/sft_smoke"),
    show_default=True,
)
@click.option("--skip-train", is_flag=True, help="Only generate data (no torch)")
@click.option("--eval-games", default=2, show_default=True)
def main(games, seed, max_steps, max_samples, model, work_dir, skip_train, eval_games):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    traj_path = work_dir / "trajectories.jsonl"
    click.echo(f"Generating {games} expert games → {traj_path}")
    summary = generate_trajectories(
        bot_names=["random", "weightedrandom", "valuefunction", "random"],
        num_games=games,
        seed=seed,
        out_path=traj_path,
        vps_to_win=8,
        workers=1,
    )
    click.echo(json.dumps(summary, indent=2))

    ds_dir = work_dir / "dataset"
    manifest = build_chat_dataset(traj_path, ds_dir, name="sft-smoke", version="v0")
    click.echo(f"Dataset decisions: {manifest.num_decisions}")

    report = {
        "generate": summary,
        "dataset": json.loads(manifest.model_dump_json()),
        "train": None,
        "eval": None,
    }

    if skip_train:
        (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo("Skipping train (--skip-train)")
        return

    ckpt = run_sft_smoke(
        ds_dir / "train.jsonl",
        work_dir / "train",
        model_name=model,
        max_steps=max_steps,
        max_samples=max_samples,
    )
    report["train"] = {"checkpoint": str(ckpt), "model": model, "max_steps": max_steps}

    from catan_llm.training.sft import local_complete_fn_from_checkpoint

    complete = local_complete_fn_from_checkpoint(ckpt)
    llm = LLMPlayer(Color.RED, complete_fn=complete, model=model)
    seats = [
        SeatSpec(name="llm", kind="llm", player=llm),
        SeatSpec(name="random", kind="random", player=RandomPlayer(Color.BLUE)),
        SeatSpec(name="random2", kind="random", player=RandomPlayer(Color.ORANGE)),
        SeatSpec(name="random3", kind="random", player=RandomPlayer(Color.WHITE)),
    ]
    stats = run_match(seats, num_games=eval_games, seed=seed + 1000, vps_to_win=8)
    report["eval"] = stats.summary()
    (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(json.dumps(report["eval"], indent=2))
    click.echo(f"Wrote {work_dir / 'report.json'}")


if __name__ == "__main__":
    main()
