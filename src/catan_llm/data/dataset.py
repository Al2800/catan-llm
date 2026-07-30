"""Build chat-format JSONL datasets from trajectory shards."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from catan_llm import __version__
from catan_llm.data.parser import format_assistant_target
from catan_llm.data.renderer import render_system_prompt, render_user_prompt
from catan_llm.data.schema import DatasetManifest, DecisionRecord
from catan_llm.sim.trajectories import read_jsonl


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tier_a_rationale(record: DecisionRecord) -> str:
    """Free template rationale derived from action type / phase."""
    at = record.action_taken.action_type
    phase = record.phase
    policy = record.expert_policy.value
    if phase.startswith("BUILD_INITIAL"):
        return f"{policy} initial placement via {at}"
    if at in {"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD"}:
        return f"{policy} expands board position with {at}"
    if at in {"PLAY_KNIGHT_CARD", "MOVE_ROBBER"}:
        return f"{policy} applies robber pressure via {at}"
    if at.startswith("PLAY_") or at == "BUY_DEVELOPMENT_CARD":
        return f"{policy} uses development-card line ({at})"
    if at == "MARITIME_TRADE":
        return f"{policy} balances hand via maritime trade"
    if at == "END_TURN":
        return f"{policy} ends turn; no higher-value legal build"
    if at == "ROLL":
        return "Must roll to start the turn"
    return f"{policy} selects {at}"


def decision_to_chat(
    record: DecisionRecord,
    *,
    include_rationale: bool = True,
) -> dict:
    """Convert a DecisionRecord into chat messages.

    Note: for SFT we reconstruct prompts from the stored structured state +
    valid action list rather than requiring a live Game object. The live
    renderer is used at play time; for smoke/SFT we emit a compact prompt that
    mirrors the same action-index contract.
    """
    action_lines = []
    for i, action in enumerate(record.valid_actions):
        action_lines.append(
            f"  [{i}] {action.action_type}"
            + (f" {action.value}" if action.value is not None else "")
        )
    you = next(
        (p for p in record.state.get("players", []) if p.get("color") == record.player_color),
        {},
    )
    system = (
        "You are an expert Settlers of Catan player. "
        f"You are playing as {record.player_color}. "
        'Respond with ONLY JSON: {"action": <index>, "reasoning": "<brief>"}.'
    )
    user = (
        f"Turn {record.turn} phase={record.phase}\n"
        f"Your status: {json.dumps(you, ensure_ascii=True)}\n"
        f"Board buildings: {json.dumps(record.state.get('buildings', {}), ensure_ascii=True)}\n"
        f"Robber: {record.state.get('robber')}\n"
        "AVAILABLE ACTIONS:\n" + "\n".join(action_lines)
    )
    reasoning = tier_a_rationale(record) if include_rationale else ""
    assistant = format_assistant_target(record.action_index, reasoning)
    return {
        "game_id": record.game_id,
        "decision_idx": record.decision_idx,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "expert_policy": record.expert_policy.value,
            "action_type": record.action_taken.action_type,
            "phase": record.phase,
        },
    }


def split_by_game(
    records: list[DecisionRecord],
    *,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
) -> dict[str, list[DecisionRecord]]:
    games: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in records:
        games[r.game_id].append(r)
    game_ids = sorted(games)
    n = len(game_ids)
    n_train = max(1, int(n * train_frac)) if n else 0
    n_val = max(0, int(n * val_frac)) if n > 1 else 0
    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train : n_train + n_val])
    test_ids = set(game_ids[n_train + n_val :])

    splits: dict[str, list[DecisionRecord]] = {"train": [], "val": [], "test": []}
    for gid, rows in games.items():
        if gid in train_ids:
            splits["train"].extend(rows)
        elif gid in val_ids:
            splits["val"].extend(rows)
        else:
            splits["test"].extend(rows)
    return splits


def build_chat_dataset(
    trajectory_path: Path,
    out_dir: Path,
    *,
    name: str = "expert-smoke",
    version: str = "v0",
    include_rationale: bool = True,
) -> DatasetManifest:
    records = read_jsonl(Path(trajectory_path))
    # Drop decisions where action wasn't in the listed legal set.
    records = [r for r in records if r.action_index >= 0]
    splits = split_by_game(records)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split_name, rows in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in rows:
                chat = decision_to_chat(record, include_rationale=include_rationale)
                fh.write(json.dumps(chat, ensure_ascii=True) + "\n")
        split_counts[split_name] = len(rows)
        checksums[path.name] = _sha256_file(path)

    bot_mix = sorted({r.expert_policy.value for r in records})
    seeds = sorted({r.seed for r in records})
    manifest = DatasetManifest(
        name=name,
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        generator_versions={"catan_llm": __version__},
        bot_mix=bot_mix,
        seeds=seeds,
        num_games=len({r.game_id for r in records}),
        num_decisions=len(records),
        split_counts=split_counts,
        checksums=checksums,
        notes="Phase-0 smoke dataset; splits by game_id",
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def render_live_chat_example(game, player_color, playable_actions, action_index: int) -> dict:
    """Helper used by tests to assert train/play renderer parity contract."""
    system = render_system_prompt(game, player_color)
    user = render_user_prompt(game, player_color, playable_actions)
    assistant = format_assistant_target(action_index, "live renderer example")
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
