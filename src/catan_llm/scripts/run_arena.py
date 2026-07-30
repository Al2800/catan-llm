"""CLI: bot-ladder / checkpoint evaluation arena."""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.eval.arena import bot_ladder_arena, write_report


@click.command()
@click.option("--games", default=12, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--vps", default=10, show_default=True)
@click.option("--no-alphabeta", is_flag=True, help="Skip AlphaBeta for faster CI")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("outputs/arena/bot_ladder.json"),
    show_default=True,
)
def main(games, seed, vps, no_alphabeta, out):
    report = bot_ladder_arena(
        num_games=games,
        seed=seed,
        vps_to_win=vps,
        include_alphabeta=not no_alphabeta,
    )
    write_report(report, out)
    click.echo(json.dumps(report, indent=2))
    click.echo(f"Wrote {out}")


if __name__ == "__main__":
    main()
