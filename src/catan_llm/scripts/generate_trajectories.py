"""CLI: generate expert trajectories from Catanatron bots."""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.sim.adapter import generate_trajectories


@click.command()
@click.option("--games", default=10, show_default=True, help="Number of games")
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--bots",
    default="random,weightedrandom,valuefunction,alphabeta",
    show_default=True,
    help="Comma-separated bot seats",
)
@click.option("--map-type", default="BASE", show_default=True)
@click.option("--vps", default=10, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=int)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("data/raw/trajectories.jsonl"),
    show_default=True,
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip game_keys already recorded in the sidecar journal (default: resume).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Wipe existing jsonl + journal before writing (explicit; never done implicitly).",
)
def main(games, seed, bots, map_type, vps, workers, out_path, resume, overwrite):
    bot_names = [b.strip() for b in bots.split(",") if b.strip()]
    summary = generate_trajectories(
        bot_names=bot_names,
        num_games=games,
        seed=seed,
        out_path=out_path,
        map_type=map_type,
        vps_to_win=vps,
        workers=workers,
        resume=resume,
        overwrite=overwrite,
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
