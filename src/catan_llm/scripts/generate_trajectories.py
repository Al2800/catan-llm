"""CLI: generate expert trajectories from Catanatron bots."""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.sim.adapter import generate_trajectories


@click.command()
@click.option("--games", default=10, show_default=True, help="Number of games (upper bound)")
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Base seed (default 0, or range start when --seed-range-name is set)",
)
@click.option(
    "--seed-range-name",
    default=None,
    help=(
        "Named range from docs/SEED_REGISTRY.md (e.g. train_main, hw_smoke). "
        "SCOPE §5.2: stop at filtered decision targets — do not blindly burn "
        "the full reserved count (e.g. 50k train_main)."
    ),
)
@click.option(
    "--bots",
    default="alphabeta,valuefunction,weightedrandom,random",
    show_default=True,
    help="Comma-separated bot seats (SCOPE §5.2 ladder order)",
)
@click.option("--map-type", default="BASE", show_default=True)
@click.option("--vps", default=10, show_default=True, type=int)
@click.option("--workers", default=1, show_default=True, type=int)
@click.option(
    "--target-decisions",
    default=None,
    type=int,
    help="Stop once filtered (action_index>=0) decisions reach this count.",
)
@click.option(
    "--rotate-seats/--no-rotate-seats",
    default=False,
    show_default=True,
    help="Rotate bot seats by seed (SCOPE §5.2 train_main).",
)
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
def main(
    games,
    seed,
    seed_range_name,
    bots,
    map_type,
    vps,
    workers,
    target_decisions,
    rotate_seats,
    out_path,
    resume,
    overwrite,
):
    bot_names = [b.strip() for b in bots.split(",") if b.strip()]
    summary = generate_trajectories(
        bot_names=bot_names,
        num_games=games,
        seed=seed,
        seed_range_name=seed_range_name,
        out_path=out_path,
        map_type=map_type,
        vps_to_win=vps,
        workers=workers,
        resume=resume,
        overwrite=overwrite,
        target_decisions=target_decisions,
        rotate_seats=rotate_seats,
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
