"""CLI: production Qwen3.5-9B QLoRA SFT (ticket 15).

Examples:
  catan-qlora-train --dry-run
  catan-qlora-train --max-steps 50 --max-samples 512   # rental micro
  catan-qlora-train --resume-from outputs/sft/qwen3.5-9b-qlora/checkpoint-200
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from catan_llm.training.qlora import DEFAULT_CONFIG, run_qlora_sft


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help=f"YAML config (default: {DEFAULT_CONFIG})",
)
@click.option(
    "--train-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Override config data.train_file (schema-v2 chat JSONL)",
)
@click.option(
    "--val-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Override config data.val_file",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override config train.output_dir",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Override epochs with a fixed step budget (rental smokes)",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Cap train rows (debug / micro-runs)",
)
@click.option(
    "--resume-from",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="HF Trainer checkpoint directory to resume",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config + data + mask check without loading Qwen/CUDA",
)
def main(
    config_path,
    train_file,
    val_file,
    output_dir,
    max_steps,
    max_samples,
    resume_from,
    dry_run,
):
    report = run_qlora_sft(
        config_path,
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        max_steps=max_steps,
        max_samples=max_samples,
        resume_from=resume_from,
        dry_run=dry_run,
    )
    click.echo(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
