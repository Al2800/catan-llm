"""CLI: bot-ladder / protocol fixture evaluation arena."""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.eval.arena import bot_ladder_arena, list_fixtures, run_fixture, write_report


@click.command()
@click.option(
    "--fixture",
    default="bot-ladder",
    show_default=True,
    help=f"Fixture name: bot-ladder | {' | '.join(list_fixtures())}",
)
@click.option("--games", default=12, show_default=True)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Base seed (default: fixture seed-range start, or 0 for bot-ladder)",
)
@click.option(
    "--seed-range-name",
    default=None,
    help="Override fixture seed range from docs/SEED_REGISTRY.md",
)
@click.option("--vps", default=10, show_default=True)
@click.option(
    "--candidate-bot",
    default="random",
    show_default=True,
    help="Bot stand-in for the candidate seat when no LLM checkpoint is loaded",
)
@click.option("--no-alphabeta", is_flag=True, help="Skip AlphaBeta for bot-ladder CI")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("outputs/arena/report.json"),
    show_default=True,
)
def main(fixture, games, seed, seed_range_name, vps, candidate_bot, no_alphabeta, out):
    if fixture == "bot-ladder":
        report = bot_ladder_arena(
            num_games=games,
            seed=0 if seed is None else seed,
            vps_to_win=vps,
            include_alphabeta=not no_alphabeta,
        )
    else:
        report = run_fixture(
            fixture,
            num_games=games,
            seed=seed,
            seed_range_name=seed_range_name,
            vps_to_win=vps,
            candidate_kind=candidate_bot,
        )
    write_report(report, out)
    click.echo(json.dumps(report, indent=2))
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
