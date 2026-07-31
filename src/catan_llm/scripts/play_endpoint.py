"""CLI: play LLMPlayer against Random via an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import click
from catanatron.models.player import Color, RandomPlayer

from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.eval.arena import SeatSpec, run_match
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.serve.openai_client import chat_complete


@click.command()
@click.option("--base-url", default="http://127.0.0.1:8000/v1", show_default=True)
@click.option("--model", default="local-model", show_default=True)
@click.option("--api-key", default="EMPTY", show_default=True)
@click.option("--games", default=1, show_default=True, type=int)
@click.option("--seed", default=1007, show_default=True, type=int)
@click.option("--vps", default=8, show_default=True, type=int)
@click.option("--structured/--freeform", default=True, show_default=True)
@click.option(
    "--backend",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "openai", "vllm", "none"]),
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("outputs/play_endpoint/report.json"),
    show_default=True,
)
def main(base_url, model, api_key, games, seed, vps, structured, backend, out):
    assert FALLBACK_POLICY == "first_legal"
    meta_box: dict = {"calls": 0, "degraded": 0, "structured_applied": 0}

    def complete(system: str, user: str) -> str:
        text, meta = chat_complete(
            base_url=base_url,
            model=model,
            system=system,
            user=user,
            api_key=api_key,
            structured=structured,
            backend=backend,
        )
        meta_box["calls"] += 1
        if meta.get("degraded"):
            meta_box["degraded"] += 1
        if meta.get("structured_applied"):
            meta_box["structured_applied"] += 1
        return text

    llm = LLMPlayer(Color.RED, complete_fn=complete, model=model)
    seats = [
        SeatSpec(name="llm", kind="llm", player=llm),
        SeatSpec(name="random", kind="random", player=RandomPlayer(Color.BLUE)),
        SeatSpec(name="random2", kind="random", player=RandomPlayer(Color.ORANGE)),
        SeatSpec(name="random3", kind="random", player=RandomPlayer(Color.WHITE)),
    ]
    stats = run_match(
        seats,
        num_games=games,
        seed=seed,
        vps_to_win=vps,
        candidate_name="llm",
        versus_name="random",
    )
    report = {
        "fallback_policy": FALLBACK_POLICY,
        "base_url": base_url,
        "model": model,
        "structured": structured,
        "backend": backend,
        "client_meta": meta_box,
        "eval": stats.summary(candidate="llm", versus="random"),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
