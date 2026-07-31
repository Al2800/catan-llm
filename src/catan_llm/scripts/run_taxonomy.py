"""CLI: build failure taxonomy from a Gate B / arena JSON report (ticket 18)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.eval.taxonomy import write_taxonomy


@click.command()
@click.option(
    "--report",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Arena / Gate B JSON (e.g. outputs/arena/gate_b_ladder4p.json)",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("docs/reports/failure_taxonomy_v1.json"),
    show_default=True,
)
@click.option(
    "--md",
    type=click.Path(path_type=Path),
    default=Path("docs/reports/failure_taxonomy_v1.md"),
    show_default=True,
)
def main(report, out, md):
    tax = write_taxonomy(report, out_json=out, out_md=md)
    click.echo(json.dumps(tax, indent=2))
    click.echo(f"Wrote {out} and {md}")


if __name__ == "__main__":
    main()
