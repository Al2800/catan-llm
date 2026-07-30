"""Evaluation arena v0 — seeded matches vs the bot ladder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from catanatron import Game
from catanatron.models.player import Color, Player
from tqdm import tqdm

from catan_llm.eval.metrics import MatchStats
from catan_llm.sim.players import make_player


@dataclass
class SeatSpec:
    name: str
    kind: str  # bot alias or "llm"
    player: Player


def _seat(name: str, kind: str, color: Color, player: Player | None = None) -> SeatSpec:
    if player is not None:
        return SeatSpec(name=name, kind=kind, player=player)
    bot, _policy = make_player(kind, color)
    return SeatSpec(name=name, kind=kind, player=bot)


def run_match(
    seats: list[SeatSpec],
    *,
    num_games: int,
    seed: int,
    vps_to_win: int = 10,
) -> MatchStats:
    stats = MatchStats()
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

    for i in tqdm(range(num_games), desc="arena"):
        # Rotate seating to reduce seat bias.
        rotation = i % len(seats)
        ordered = seats[rotation:] + seats[:rotation]
        players = []
        name_by_color: dict[str, str] = {}
        for seat, color in zip(ordered, colors, strict=False):
            # Rebuild bots each game so internal state is fresh.
            if seat.kind == "llm":
                player = seat.player
                player.color = color
                if hasattr(player, "reset_state"):
                    player.reset_state()
            else:
                player, _ = make_player(seat.kind, color)
            players.append(player)
            name_by_color[color.value] = seat.name

        game = Game(players, seed=seed + i, vps_to_win=vps_to_win)
        winner_color = game.play()
        winner_name = name_by_color.get(winner_color.value) if winner_color else None
        stats.add_game(winner_name, game.state.num_turns)

        # Optional legality/parse hooks for LLM seats.
        for player in players:
            if hasattr(player, "consume_eval_counters"):
                parsed_ok, parsed_total, legal_ok, legal_total = player.consume_eval_counters()
                for _ in range(parsed_ok):
                    stats.add_decision(parsed=True)
                for _ in range(parsed_total - parsed_ok):
                    stats.add_decision(parsed=False)
                for _ in range(legal_ok):
                    stats.add_decision(legal=True)
                for _ in range(legal_total - legal_ok):
                    stats.add_decision(legal=False)

    return stats


def bot_ladder_arena(
    *,
    num_games: int = 20,
    seed: int = 0,
    vps_to_win: int = 10,
    include_alphabeta: bool = True,
) -> dict:
    """Round-ish bot-vs-bot ladder for Phase-0 CI smoke."""
    kinds = ["random", "weightedrandom", "valuefunction"]
    if include_alphabeta:
        kinds.append("alphabeta")
    seats = [
        _seat(kind, kind, color)
        for kind, color in zip(
            kinds, [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE], strict=False
        )
    ]
    stats = run_match(seats, num_games=num_games, seed=seed, vps_to_win=vps_to_win)
    return {
        "config": {
            "num_games": num_games,
            "seed": seed,
            "vps_to_win": vps_to_win,
            "seats": [s.kind for s in seats],
        },
        "results": stats.summary(),
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
