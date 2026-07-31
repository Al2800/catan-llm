"""Simulator adapter: bulk games + trajectory recording."""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from catanatron import Game
from catanatron.models.map import BASE_MAP_TEMPLATE, MINI_MAP_TEMPLATE, CatanMap
from catanatron.models.player import Color
from tqdm import tqdm

from catan_llm.data.identity import (
    CATANATRON_COMMIT,
    bot_config_hash,
    hash_catan_map,
    make_game_key,
    resolve_source_commit,
)
from catan_llm.data.schema import DecisionRecord, ExpertPolicy, GameOutcome
from catan_llm.data.seed_registry import resolve_generation_seeds
from catan_llm.sim.players import bot_config_for_names, make_player
from catan_llm.sim.trajectories import (
    TrajectoryAccumulator,
    append_journal,
    append_jsonl,
    journal_path_for,
    load_completed_game_keys,
    write_jsonl,
)


@dataclass
class GameResult:
    game_id: str
    game_key: str
    seed: int
    map_hash: str
    winner: str | None
    turns: int
    num_decisions: int
    records: list[DecisionRecord]
    outcome: GameOutcome


def build_catan_map(map_type: str) -> CatanMap:
    """Build a CatanMap for the requested type. Failures raise (no silent BASE)."""
    key = map_type.upper()
    if key == "BASE":
        return CatanMap.from_template(BASE_MAP_TEMPLATE)
    if key == "MINI":
        return CatanMap.from_template(MINI_MAP_TEMPLATE)
    raise ValueError(f"Unsupported map_type={map_type!r}; expected BASE or MINI")


def _burn_player_shuffle(n_players: int) -> None:
    """Consume the same RNG draws State uses before map construction."""
    placeholders = list(range(n_players))
    random.sample(placeholders, len(placeholders))


def map_for_seed(map_type: str, seed: int, *, n_players: int = 4) -> CatanMap:
    """Build map under the RNG stream Game/State use for the default BASE path."""
    random.seed(seed)
    _burn_player_shuffle(n_players)
    return build_catan_map(map_type)


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

    bot_config = bot_config_for_names(bot_names)
    cfg_hash = bot_config_hash(bot_config)

    # Match Catanatron's seed → player-shuffle → map construction order so MINI
    # sits on the same RNG path BASE would use, then hand the map to Game.
    catan_map = map_for_seed(map_type, seed, n_players=len(players))
    map_hash = hash_catan_map(catan_map)
    game_key = make_game_key(seed, map_hash, cfg_hash)

    game = Game(
        players=players,
        seed=seed,
        vps_to_win=vps_to_win,
        catan_map=catan_map,
    )
    acc = TrajectoryAccumulator(
        seed=seed,
        map_type=map_type.upper(),
        map_hash=map_hash,
        bot_config=bot_config,
        bot_config_hash=cfg_hash,
        game_key=game_key,
        policy_by_color=policy_by_color,
        catanatron_commit=CATANATRON_COMMIT,
        source_commit=resolve_source_commit(),
    )
    game.play(accumulators=[acc])
    assert acc.outcome is not None
    return GameResult(
        game_id=game.id,
        game_key=game_key,
        seed=seed,
        map_hash=map_hash,
        winner=acc.outcome.winner,
        turns=acc.outcome.turns,
        num_decisions=len(acc.records),
        records=acc.records,
        outcome=acc.outcome,
    )


def _play_one_job(args: tuple) -> GameResult:
    bot_names, seed, map_type, vps_to_win = args
    return play_one(bot_names, seed, map_type=map_type, vps_to_win=vps_to_win)


def _expected_game_key(bot_names: list[str], seed: int, map_type: str) -> str:
    bot_config = bot_config_for_names(bot_names)
    cfg_hash = bot_config_hash(bot_config)
    catan_map = map_for_seed(map_type, seed, n_players=len(bot_names))
    return make_game_key(seed, hash_catan_map(catan_map), cfg_hash)


def rotate_bot_names(bot_names: list[str], seed: int) -> list[str]:
    """Deterministic seat rotation for SCOPE §5.2 cohort mixes."""
    if not bot_names:
        return bot_names
    k = seed % len(bot_names)
    if k == 0:
        return list(bot_names)
    return list(bot_names[k:] + bot_names[:k])


def count_filtered_decisions(path: Path) -> int:
    """Count schema rows with ``action_index >= 0`` in a trajectory JSONL."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("action_index", -1)) >= 0:
                n += 1
    return n


def _filtered_in_records(records: list[DecisionRecord]) -> int:
    return sum(1 for r in records if r.action_index >= 0)


def generate_trajectories(
    *,
    bot_names: list[str],
    num_games: int,
    seed: int | None = None,
    out_path: Path,
    map_type: str = "BASE",
    vps_to_win: int = 10,
    workers: int = 1,
    chunk_flush: int = 25,
    resume: bool = True,
    overwrite: bool = False,
    seed_range_name: str | None = None,
    target_decisions: int | None = None,
    rotate_seats: bool = False,
) -> dict:
    """Generate decision trajectories with append-only JSONL + game_key journal.

    Resume behavior (DATA_CONTRACT §10):
    - Never `unlink()` completed outputs at job start.
    - Default ``resume=True``: skip ``game_key``s already in the sidecar journal.
    - ``resume=False`` refuses to write over an existing non-empty output unless
      ``overwrite=True`` (explicit wipe of jsonl + journal).

    Seed ranges: pass ``seed_range_name`` from ``docs/SEED_REGISTRY.md``.
    Cohort stop (SCOPE §5.2): when ``target_decisions`` is set, stop once filtered
    (``action_index >= 0``) decisions reach that count — including prior resume rows.
    """
    seed, num_games, seed_range = resolve_generation_seeds(
        num_games=num_games, seed=seed, seed_range_name=seed_range_name
    )
    seed_range_meta = (
        {
            "name": seed_range.name,
            "start": seed_range.start,
            "count": seed_range.count,
            "end": seed_range.end,
        }
        if seed_range is not None
        else None
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = journal_path_for(out_path)

    if overwrite:
        if out_path.exists():
            out_path.write_text("", encoding="utf-8")
        if journal_path.exists():
            journal_path.write_text("", encoding="utf-8")
    elif not resume and (
        (out_path.exists() and out_path.stat().st_size > 0)
        or (journal_path.exists() and journal_path.stat().st_size > 0)
    ):
        raise FileExistsError(
            f"Refusing to clobber existing trajectories at {out_path} "
            "(pass resume=True to continue, or overwrite=True to wipe)."
        )

    completed = load_completed_game_keys(journal_path) if resume else set()
    prior_filtered = count_filtered_decisions(out_path) if resume else 0

    seeds = [seed + i for i in range(num_games)]
    planned: list[tuple[int, str, list[str]]] = []
    skipped = 0
    for s in seeds:
        seats = rotate_bot_names(bot_names, s) if rotate_seats else list(bot_names)
        gkey = _expected_game_key(seats, s, map_type)
        if gkey in completed:
            skipped += 1
        else:
            planned.append((s, gkey, seats))

    total_decisions = 0
    filtered_decisions = prior_filtered
    finished_games = 0
    winners: dict[str, int] = {}
    buffer: list[DecisionRecord] = []
    pending_journal: list[tuple[str, int]] = []
    stopped_early = False

    def flush():
        nonlocal buffer, pending_journal
        if buffer:
            append_jsonl(out_path, buffer)
            buffer = []
        for gkey, s in pending_journal:
            append_journal(journal_path, game_key=gkey, seed=s)
        pending_journal = []

    def handle_result(result: GameResult) -> None:
        nonlocal total_decisions, filtered_decisions, finished_games, buffer, pending_journal
        buffer.extend(result.records)
        pending_journal.append((result.game_key, result.seed))
        total_decisions += result.num_decisions
        filtered_decisions += _filtered_in_records(result.records)
        finished_games += 1
        if result.winner:
            winners[result.winner] = winners.get(result.winner, 0) + 1
        if finished_games % chunk_flush == 0:
            flush()

    def hit_target() -> bool:
        return target_decisions is not None and filtered_decisions >= target_decisions

    def summary(*, early: bool) -> dict:
        return {
            "num_games": finished_games,
            "num_decisions": total_decisions,
            "num_filtered_decisions": filtered_decisions,
            "prior_filtered_decisions": prior_filtered,
            "target_decisions": target_decisions,
            "stopped_early": early,
            "skipped_games": skipped,
            "winners": winners,
            "out_path": str(out_path),
            "journal_path": str(journal_path),
            "seeds": seeds,
            "bot_names": bot_names,
            "rotate_seats": rotate_seats,
            "map_type": map_type.upper(),
            "base_seed": seed,
            "seed_range": seed_range_meta,
            "resume": resume,
            "catanatron_commit": CATANATRON_COMMIT,
            "source_commit": resolve_source_commit(),
            "cohort_note": (
                "SCOPE §5.2: stop at filtered decision targets; "
                "do not blindly burn full reserved seed counts."
            ),
        }

    if hit_target():
        if not out_path.exists():
            write_jsonl(out_path, [])
        return summary(early=True)

    if not planned:
        if not out_path.exists():
            write_jsonl(out_path, [])
        return summary(early=False)

    if workers <= 1:
        for s, _gkey, seats in tqdm(planned, total=len(planned), desc="games"):
            result = play_one(seats, s, map_type=map_type, vps_to_win=vps_to_win)
            handle_result(result)
            if hit_target():
                stopped_early = True
                break
    else:
        # Batch so target_decisions can stop without scheduling the whole range.
        batch_size = max(workers * 2, 8)
        idx = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pbar = tqdm(total=len(planned), desc="games")
            while idx < len(planned) and not hit_target():
                batch = planned[idx : idx + batch_size]
                idx += len(batch)
                futures = [
                    pool.submit(_play_one_job, (seats, s, map_type, vps_to_win))
                    for s, _gkey, seats in batch
                ]
                # Always drain the batch so completed games are recorded.
                for fut in as_completed(futures):
                    handle_result(fut.result())
                    pbar.update(1)
                if hit_target():
                    stopped_early = True
                    break
            pbar.close()

    flush()
    if not out_path.exists():
        write_jsonl(out_path, [])

    return summary(early=stopped_early)


def sample_bot_mix(rng: random.Random | None = None) -> list[str]:
    """Default Phase-0 bot ladder mix (4 seats)."""
    rng = rng or random.Random()
    ladder = ["random", "weightedrandom", "valuefunction", "alphabeta"]
    seats = ladder.copy()
    rng.shuffle(seats)
    return seats
