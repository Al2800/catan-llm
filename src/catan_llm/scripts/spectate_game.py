"""CLI: watch an LLM (or bot) play Catan live in the terminal (ticket 24).

Examples:
  # Mock endpoint (CPU) — prints every action
  catan-serve --mock --port 8000 &
  catan-spectate --base-url http://127.0.0.1:8000/v1 --watch --vps 6

  # Against a real OpenAI-compatible server (vLLM / rental)
  catan-spectate --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-9B --watch

  # Bot-only dry watch (no LLM)
  catan-spectate --bots-only --watch --seed 7
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from catanatron.models.player import Color

from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.play.spectate import bot_seat, play_spectate_game, write_replay
from catan_llm.serve.openai_client import chat_complete


@click.command("spectate")
@click.option("--base-url", default="http://127.0.0.1:8000/v1", show_default=True)
@click.option("--model", default="local-model", show_default=True)
@click.option("--api-key", default="EMPTY", show_default=True)
@click.option("--seed", default=1007, show_default=True, type=int)
@click.option("--vps", default=10, show_default=True, type=int)
@click.option("--watch/--no-watch", default=True, show_default=True)
@click.option(
    "--delay",
    default=0.0,
    show_default=True,
    type=float,
    help="Optional per-action delay (seconds) for human-paced watching",
)
@click.option(
    "--bots-only",
    is_flag=True,
    default=False,
    help="Watch bot ladder only (no LLM endpoint)",
)
@click.option(
    "--adapter",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="PEFT adapter dir (in-process generate; needs CUDA + train extras)",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True),
    default=Path("configs/qwen3.5-9b-qlora.yaml"),
    show_default=True,
)
@click.option(
    "--opponents",
    default="random,weightedrandom,valuefunction",
    show_default=True,
    help="Comma-separated bot kinds for the three opponent seats",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("outputs/spectate/replay.json"),
    show_default=True,
)
@click.option("--structured/--freeform", default=True, show_default=True)
def main(
    base_url,
    model,
    api_key,
    seed,
    vps,
    watch,
    delay,
    bots_only,
    adapter,
    config_path,
    opponents,
    out,
    structured,
):
    """Ticket 24: terminal spectate + JSON replay for one game."""
    assert FALLBACK_POLICY == "first_legal"
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    opp_kinds = [x.strip() for x in opponents.split(",") if x.strip()]
    if len(opp_kinds) != 3:
        raise click.ClickException("--opponents must list exactly 3 bot kinds")

    seat_labels: dict[str, str] = {}
    if bots_only:
        players = [
            bot_seat("valuefunction", colors[0]),
            bot_seat(opp_kinds[0], colors[1]),
            bot_seat(opp_kinds[1], colors[2]),
            bot_seat(opp_kinds[2], colors[3]),
        ]
        seat_labels = {
            colors[0].value: "candidate-bot",
            colors[1].value: opp_kinds[0],
            colors[2].value: opp_kinds[1],
            colors[3].value: opp_kinds[2],
        }
    elif adapter is not None:
        from catan_llm.training.masking import qwen_model_name
        from catan_llm.training.peft_infer import load_peft_generator

        _m, _t, complete = load_peft_generator(adapter, config_path=config_path)
        label = qwen_model_name(config_path)
        llm = LLMPlayer(colors[0], complete_fn=complete, model=label)
        players = [
            llm,
            bot_seat(opp_kinds[0], colors[1]),
            bot_seat(opp_kinds[1], colors[2]),
            bot_seat(opp_kinds[2], colors[3]),
        ]
        seat_labels = {
            colors[0].value: f"llm-adapter:{adapter}",
            colors[1].value: opp_kinds[0],
            colors[2].value: opp_kinds[1],
            colors[3].value: opp_kinds[2],
        }
    else:

        def complete(system: str, user: str) -> str:
            text, _meta = chat_complete(
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                api_key=api_key,
                structured=structured,
                backend="auto",
            )
            return text

        llm = LLMPlayer(colors[0], complete_fn=complete, model=model)
        players = [
            llm,
            bot_seat(opp_kinds[0], colors[1]),
            bot_seat(opp_kinds[1], colors[2]),
            bot_seat(opp_kinds[2], colors[3]),
        ]
        seat_labels = {
            colors[0].value: f"llm:{model}",
            colors[1].value: opp_kinds[0],
            colors[2].value: opp_kinds[1],
            colors[3].value: opp_kinds[2],
        }

    result = play_spectate_game(
        players,
        seed=seed,
        vps_to_win=vps,
        watch=watch,
        delay_s=delay,
        seat_labels=seat_labels,
    )
    write_replay(result, out)
    summary = {
        "fallback_policy": FALLBACK_POLICY,
        "winner": result.winner,
        "turns": result.turns,
        "events": len(result.events),
        "replay": str(out),
        "bots_only": bots_only,
        "adapter": str(adapter) if adapter else None,
        "base_url": None if (bots_only or adapter) else base_url,
        "model": None if bots_only else model,
    }
    click.echo(json.dumps(summary, indent=2))
    click.echo(f"Wrote replay {out}")


if __name__ == "__main__":
    main()
