"""Simulator adapter: bulk games + trajectory recording."""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from catanatron import Game
from catanatron.models.map import CatanMap
from catanatron.models.player import Color
from tqdm import tqdm

from catan_llm.data.schema import DecisionRecord, ExpertPolicy, GameOutcome
from catan_llm.sim.players import make_player
from catan_llm.sim.trajectories import TrajectoryAccumulator, append_jsonl, write_jsonl


@dataclass
class GameResult:
    game_id: str
    seed: int
    winner: str | None
    turns: int
    num_decisions: int
    records: list[DecisionRecord]
    outcome: GameOutcome


def _build_map(map_type: str):
    key = map_type.upper()
    if key == "MINI":
        # Catanatron exposes MINI via from_template if available; fall back to standard.
        if hasattr(CatanMap, "from_template"):
            try:
                return CatanMap.from_template("MINI")  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(CatanMap, "from_random_template"):
            return CatanMap.from_random_template()
        return None
    return None  # default BASE map


def play_one(
    bot_names: list[str],
    seed: int,
    *,
    map_type: str = "BASE",
    vps_to_win: int = 10,
) -> GameResult:
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE][: len(bot_names)]
    players = []
    policy_by_color: dict[str, ExpertPolicy] = {}
    for name, color in zip(bot_names, colors, strict=True):
        player, policy = make_player(name, color)
        players.append(player)
        policy_by_color[color.value] = policy

    catan_map = _build_map(map_type)
    game_kwargs = {"players": players, "seed": seed, "vps_to_win": vps_to_win}
    if catan_map is not None:
        game_kwargs["catan_map"] = catan_map

    game = Game(**game_kwargs)
    acc = TrajectoryAccumulator(
        seed=seed, map_type=map_type, policy_by_color=policy_by_color
    )
    game.play(accumulators=[acc])
    assert acc.outcome is not None
    return GameResult(
        game_id=game.id,
        seed=seed,
        winner=acc.outcome.winner,
        turns=acc.outcome.turns,
        num_decisions=len(acc.records),
        records=acc.records,
        outcome=acc.outcome,
    )


def _play_one_job(args: tuple) -> GameResult:
    bot_names, seed, map_type, vps_to_win = args
    return play_one(bot_names, seed, map_type=map_type, vps_to_win=vps_to_win)


def generate_trajectories(
    *,
    bot_names: list[str],
    num_games: int,
    seed: int,
    out_path: Path,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    workers: int = 1,
    chunk_flush: int = 25,
) -> dict:
    """Generate decision trajectories and write crash-safe JSONL shards."""
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()

    seeds = [seed + i for i in range(num_games)]
    jobs = [(bot_names, s, map_type, vps_to_win) for s in seeds]

    total_decisions = 0
    finished_games = 0
    winners: dict[str, int] = {}
    buffer: list[DecisionRecord] = []

    def flush():
        nonlocal buffer
        if buffer:
            append_jsonl(out_path, buffer)
            buffer = []

    if workers <= 1:
        for s in tqdm(seeds, total=num_games, desc="games"):
            result = play_one(bot_names, s, map_type=map_type, vps_to_win=vps_to_win)
            buffer.extend(result.records)
            total_decisions += result.num_decisions
            finished_games += 1
            if result.winner:
                winners[result.winner] = winners.get(result.winner, 0) + 1
            if finished_games % chunk_flush == 0:
                flush()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_play_one_job, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=num_games, desc="games"):
                result = fut.result()
                buffer.extend(result.records)
                total_decisions += result.num_decisions
                finished_games += 1
                if result.winner:
                    winners[result.winner] = winners.get(result.winner, 0) + 1
                if finished_games % chunk_flush == 0:
                    flush()

    flush()
    if not out_path.exists():
        write_jsonl(out_path, [])

    return {
        "num_games": finished_games,
        "num_decisions": total_decisions,
        "winners": winners,
        "out_path": str(out_path),
        "seeds": seeds,
        "bot_names": bot_names,
        "map_type": map_type,
        "base_seed": seed,
    }


def sample_bot_mix(rng: random.Random | None = None) -> list[str]:
    """Default Phase-0 bot ladder mix (4 seats)."""
    rng = rng or random.Random()
    ladder = ["random", "weightedrandom", "valuefunction", "alphabeta"]
    # Keep diversity; shuffle seating for seat-bias control.
    seats = ladder.copy()
    rng.shuffle(seats)
    return seats
