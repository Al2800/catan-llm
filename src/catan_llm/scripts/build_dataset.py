"""CLI: trajectories JSONL → chat-format dataset + manifest."""

from __future__ import annotations

from pathlib import Path

import click

from catan_llm.data.dataset import build_chat_dataset
from catan_llm.data.seed_registry import get_seed_range


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
@click.option(
    "--seed-range-name",
    default=None,
    help="Record this SEED_REGISTRY name on the manifest.",
)
@click.option(
    "--immutable/--mutable",
    default=False,
    show_default=True,
    help="Mark dataset immutable (eval holdout).",
)
@click.option(
    "--role",
    default=None,
    help="Manifest role tag (train / eval_holdout).",
)
@click.option(
    "--no-split",
    is_flag=True,
    default=False,
    help="Write a single holdout.jsonl instead of train/val/test.",
)
def main(trajectories, out_dir, name, version, seed_range_name, immutable, role, no_split):
    seed_range = None
    if seed_range_name:
        rng = get_seed_range(seed_range_name)
        seed_range = {
            "name": rng.name,
            "start": rng.start,
            "count": rng.count,
            "end": rng.end,
        }
    manifest = build_chat_dataset(
        trajectories,
        out_dir,
        name=name,
        version=version,
        include_rationale=True,
        seed_range=seed_range,
        immutable=immutable,
        role=role,
        split=not no_split,
    )
    click.echo(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
