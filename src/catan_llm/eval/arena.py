"""Evaluation arena — seeded matches vs the bot ladder (EVAL_PROTOCOL v1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from catanatron import Game
from catanatron.models.player import Color, Player
from catanatron.state_functions import get_actual_victory_points
from tqdm import tqdm

from catan_llm.data.identity import CATANATRON_COMMIT, PROMPT_VERSION, resolve_source_commit
from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.data.seed_registry import resolve_generation_seeds
from catan_llm.eval.metrics import MatchStats
from catan_llm.sim.players import make_player

PROTOCOL_VERSION = "v1"

# Headline fixtures (EVAL_PROTOCOL §2). Seat kinds after "candidate" are bots.
FIXTURE_SEATS: dict[str, list[str]] = {
    "ladder-4p": ["candidate", "random", "weightedrandom", "valuefunction"],
    "ab-4p": ["candidate", "alphabeta", "valuefunction", "random"],
}

FIXTURE_SEED_RANGES: dict[str, str] = {
    "ladder-4p": "ladder_sft_gate",
    "ab-4p": "champion_ab",
}


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


def _candidate_vp_margin(game: Game, candidate_color: Color) -> float:
    vps = {
        color: get_actual_victory_points(game.state, color) for color in game.state.colors
    }
    cand = vps[candidate_color]
    best_opp = max(v for color, v in vps.items() if color != candidate_color)
    return float(cand - best_opp)


def run_match(
    seats: list[SeatSpec],
    *,
    num_games: int,
    seed: int,
    vps_to_win: int = 10,
    candidate_name: str = "candidate",
    versus_name: str = "weightedrandom",
) -> MatchStats:
    stats = MatchStats()
    stats.register_seats([s.name for s in seats])
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]

    for i in tqdm(range(num_games), desc="arena"):
        # Rotate seating to reduce seat bias.
        rotation = i % len(seats)
        ordered = seats[rotation:] + seats[:rotation]
        players = []
        name_by_color: dict[str, str] = {}
        color_by_name: dict[str, Color] = {}
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
            color_by_name[seat.name] = color

        game = Game(players, seed=seed + i, vps_to_win=vps_to_win)
        winner_color = game.play()
        winner_name = name_by_color.get(winner_color.value) if winner_color else None
        margin = None
        if winner_color is not None and candidate_name in color_by_name:
            margin = _candidate_vp_margin(game, color_by_name[candidate_name])
        stats.add_game(winner_name, game.state.num_turns, vp_margin=margin)

        for player in players:
            if hasattr(player, "consume_eval_counters"):
                counters = player.consume_eval_counters()
                if isinstance(counters, dict):
                    stats.absorb_llm_counters(
                        counters["parse_ok"],
                        counters["parse_total"],
                        counters["legal_ok"],
                        counters["legal_total"],
                        counters.get("fallback_count", 0),
                    )
                elif len(counters) >= 5:
                    stats.absorb_llm_counters(*counters[:5])
                else:
                    parsed_ok, parsed_total, legal_ok, legal_total = counters
                    stats.absorb_llm_counters(
                        parsed_ok,
                        parsed_total,
                        legal_ok,
                        legal_total,
                        parsed_total - parsed_ok,
                    )

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
    stats = run_match(
        seats,
        num_games=num_games,
        seed=seed,
        vps_to_win=vps_to_win,
        candidate_name=kinds[0],
        versus_name="weightedrandom" if "weightedrandom" in kinds else kinds[1],
    )
    summary = stats.summary(
        candidate=kinds[0],
        versus="weightedrandom" if "weightedrandom" in kinds else kinds[1],
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config": {
            "num_games": num_games,
            "seed": seed,
            "vps_to_win": vps_to_win,
            "seats": [s.kind for s in seats],
            "fallback_policy": FALLBACK_POLICY,
            "prompt_version": PROMPT_VERSION,
        },
        "results": summary,
    }


def run_fixture(
    format_name: str,
    *,
    num_games: int,
    seed: int | None = None,
    seed_range_name: str | None = None,
    vps_to_win: int = 10,
    candidate_kind: str = "random",
    candidate_player: Player | None = None,
    map_type: str = "BASE",
) -> dict:
    """Run a named EVAL_PROTOCOL fixture (``ladder-4p`` / ``ab-4p``).

    Without an LLM checkpoint, ``candidate_kind`` defaults to a bot stand-in so
    metric plumbing and CLI wiring can be exercised on CPU.
    """
    if format_name not in FIXTURE_SEATS:
        known = ", ".join(sorted(FIXTURE_SEATS))
        raise ValueError(f"Unknown fixture {format_name!r}. Known: {known}")

    default_range = FIXTURE_SEED_RANGES[format_name]
    range_name = seed_range_name or default_range
    base_seed, num_games, rng = resolve_generation_seeds(
        num_games=num_games, seed=seed, seed_range_name=range_name
    )

    seat_names = FIXTURE_SEATS[format_name]
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    seats: list[SeatSpec] = []
    for name, color in zip(seat_names, colors, strict=True):
        if name == "candidate":
            if candidate_player is not None:
                seats.append(SeatSpec(name="candidate", kind="llm", player=candidate_player))
            else:
                seats.append(_seat("candidate", candidate_kind, color))
        else:
            seats.append(_seat(name, name, color))

    versus = "weightedrandom" if "weightedrandom" in seat_names else "alphabeta"
    stats = run_match(
        seats,
        num_games=num_games,
        seed=base_seed,
        vps_to_win=vps_to_win,
        candidate_name="candidate",
        versus_name=versus,
    )
    results = stats.summary(candidate="candidate", versus=versus)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "fixture": {
            "format": format_name,
            "seed_range_name": range_name,
            "seed_start": base_seed,
            "num_games": num_games,
            "map_type": map_type,
            "vps_to_win": vps_to_win,
            "fallback_policy": FALLBACK_POLICY,
            "prompt_version": PROMPT_VERSION,
            "seats": seat_names,
            "candidate_kind": "llm" if candidate_player is not None else candidate_kind,
            "catanatron_commit": CATANATRON_COMMIT,
            "source_commit": resolve_source_commit(),
            "seed_range": {
                "name": rng.name,
                "start": rng.start,
                "count": rng.count,
                "end": rng.end,
            }
            if rng is not None
            else None,
        },
        "results": results,
    }


def list_fixtures() -> list[str]:
    return sorted(FIXTURE_SEATS)


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
