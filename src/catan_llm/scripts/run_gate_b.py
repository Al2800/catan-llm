"""CLI: Gate B ladder-4p eval against a QLoRA adapter (ticket 17)."""

from __future__ import annotations

import json
from pathlib import Path

import click
from catanatron.models.player import Color

from catan_llm.eval.arena import run_fixture, write_report
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.training.masking import qwen_model_name


@click.command()
@click.option(
    "--adapter",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="PEFT adapter directory from catan-qlora-train",
)
@click.option("--fixture", default="ladder-4p", show_default=True)
@click.option("--games", default=200, show_default=True)
@click.option("--seed", default=None, type=int)
@click.option("--vps", default=10, show_default=True)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("outputs/arena/gate_b_ladder4p.json"),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True),
    default=Path("configs/qwen3.5-9b-qlora.yaml"),
    show_default=True,
)
def main(adapter, fixture, games, seed, vps, out, config_path):
    from catan_llm.training.peft_infer import load_peft_generator

    _model, _tok, complete = load_peft_generator(adapter, config_path=config_path)
    llm = LLMPlayer(
        Color.RED,
        complete_fn=complete,
        model=qwen_model_name(config_path),
        temperature=0.0,
        max_tokens=128,
    )
    report = run_fixture(
        fixture,
        num_games=games,
        seed=seed,
        vps_to_win=vps,
        candidate_player=llm,
    )
    results = report.get("results") or {}
    parse_rate = _as_float(results.get("parse_rate_model"), default=0.0)
    legality = _as_float(results.get("legality_rate_model"), default=0.0)
    win_rates = results.get("win_rates") or {}
    cand = _win_rate(win_rates, "candidate")
    wr = _win_rate(win_rates, "weightedrandom")
    finished = int(results.get("finished") or results.get("games") or 0)
    gaps = results.get("win_share_gap") or {}
    gap = _as_float(
        gaps.get("candidate,weightedrandom")
        or gaps.get("candidate_vs_weightedrandom"),
        default=cand - wr,
    )

    # Always persist the arena report before pass/fail exit codes.
    gate = {
        "ticket": "17",
        "parse_rate_model": parse_rate,
        "legality_rate_model": legality,
        "finished": finished,
        "win_rate_candidate": cand,
        "win_rate_weightedrandom": wr,
        "win_share_gap_candidate_vs_wr": gap,
        "thresholds": {
            "parse_rate_model_min": 0.995,
            "legality_rate_model_min": 0.995,
            "finished_min": 200,
            "candidate_beats_wr": True,
        },
        "pass": (
            parse_rate >= 0.995
            and legality >= 0.995
            and finished >= 200
            and cand > wr
        ),
    }
    report["gate_b"] = gate
    write_report(report, out)
    click.echo(json.dumps(report, indent=2))
    click.echo(f"Wrote {out}")
    if not gate["pass"]:
        raise SystemExit(2)


def _as_float(value, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, dict):
        if "rate" in value:
            return float(value["rate"])
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _win_rate(win_rates: dict, name: str) -> float:
    return _as_float(win_rates.get(name), default=0.0)


if __name__ == "__main__":
    main()
