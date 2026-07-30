"""CLI: trajectories JSONL → chat-format dataset + manifest."""

from __future__ import annotations

from pathlib import Path

import click

from catan_llm.data.dataset import build_chat_dataset


@click.command()
@click.option(
    "--trajectories",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed/expert-smoke"),
    show_default=True,
)
@click.option("--name", default="expert-smoke", show_default=True)
@click.option("--version", default="v0", show_default=True)
def main(trajectories, out_dir, name, version):
    manifest = build_chat_dataset(
        trajectories, out_dir, name=name, version=version, include_rationale=True
    )
    click.echo(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
