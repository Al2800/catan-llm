"""Build chat-format JSONL datasets from trajectory shards."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from catan_llm import __version__
from catan_llm.data.identity import CATANATRON_COMMIT, PROMPT_VERSION, resolve_source_commit
from catan_llm.data.parser import format_assistant_target
from catan_llm.data.pov import assert_tier_a_pov_safe
from catan_llm.data.quality import (
    filter_decision_records,
    try_load_truncation_tokenizer,
)
from catan_llm.data.renderer import render_system_prompt, render_user_prompt
from catan_llm.data.schema import DatasetManifest, DecisionRecord, require_schema_v2
from catan_llm.data.tier_a import render_tier_a_rationale
from catan_llm.sim.trajectories import read_jsonl
from catan_llm.training.masking import qwen_max_seq_length


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tier_a_rationale(record: DecisionRecord) -> str:
    """Feature-aware POV-safe Tier A rationale (SCOPE §7.4)."""
    return render_tier_a_rationale(record)


def decision_to_chat(
    record: DecisionRecord,
    *,
    include_rationale: bool = True,
) -> dict:
    """Convert a DecisionRecord into chat messages using stored live prompts.

    Compact alternate prompts are forbidden for labeled SFT (DATA_CONTRACT §4).
    ``system_prompt`` / ``user_prompt`` must have been captured via
    ``render_system_prompt`` / ``render_user_prompt`` at decision time.
    """
    if not record.system_prompt or not record.user_prompt:
        raise ValueError(
            "decision_to_chat requires system_prompt and user_prompt from the live renderer"
        )
    reasoning = tier_a_rationale(record) if include_rationale else ""
    if reasoning:
        assert_tier_a_pov_safe(reasoning, context="decision_to_chat")
    assistant = format_assistant_target(record.action_index, reasoning)
    return {
        "game_id": record.game_id,
        "game_key": record.game_key,
        "decision_idx": record.decision_idx,
        "prompt_version": record.prompt_version,
        "messages": [
            {"role": "system", "content": record.system_prompt},
            {"role": "user", "content": record.user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "expert_policy": record.expert_policy.value,
            "action_type": record.action_taken.action_type,
            "phase": record.phase,
        },
    }


def split_by_game_key(
    records: list[DecisionRecord],
    *,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
) -> dict[str, list[DecisionRecord]]:
    """Deterministic splits keyed by ``game_key`` (never UUID ``game_id``)."""
    games: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in records:
        games[r.game_key].append(r)
    game_keys = sorted(games)
    n = len(game_keys)
    n_train = max(1, int(n * train_frac)) if n else 0
    n_val = max(0, int(n * val_frac)) if n > 1 else 0
    train_ids = set(game_keys[:n_train])
    val_ids = set(game_keys[n_train : n_train + n_val])

    splits: dict[str, list[DecisionRecord]] = {"train": [], "val": [], "test": []}
    for gkey, rows in games.items():
        if gkey in train_ids:
            splits["train"].extend(rows)
        elif gkey in val_ids:
            splits["val"].extend(rows)
        else:
            splits["test"].extend(rows)
    return splits


# Back-compat alias — Phase-1 paths must use game_key (this function).
split_by_game = split_by_game_key


def build_chat_dataset(
    trajectory_path: Path,
    out_dir: Path,
    *,
    name: str = "expert-smoke",
    version: str = "v0",
    include_rationale: bool = True,
    require_v2: bool = True,
    seed_range: dict | None = None,
    immutable: bool = False,
    role: str | None = None,
    notes: str | None = None,
    split: bool = True,
    max_seq_length: int | None = None,
    check_truncation: bool = True,
    tokenizer=None,
) -> DatasetManifest:
    """Build chat JSONL + DATA_CONTRACT §9 manifest (ticket 12 quality filters)."""
    records = read_jsonl(Path(trajectory_path))
    if require_v2:
        require_schema_v2(records, context="build_chat_dataset")

    max_len = int(max_seq_length if max_seq_length is not None else qwen_max_seq_length())
    tok = tokenizer
    if check_truncation and tok is None:
        tok = try_load_truncation_tokenizer()

    records, filter_stats = filter_decision_records(
        records,
        max_seq_length=max_len,
        include_rationale=include_rationale,
        tokenizer=tok,
        check_truncation=check_truncation,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split:
        split_map = split_by_game_key(records)
    else:
        # Holdout / single-artifact mode (ticket 13).
        split_map = {"holdout": records}

    checksums: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split_name, rows in split_map.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in rows:
                chat = decision_to_chat(record, include_rationale=include_rationale)
                fh.write(json.dumps(chat, ensure_ascii=True) + "\n")
        split_counts[split_name] = len(rows)
        checksums[path.name] = _sha256_file(path)

    bot_mix = sorted({r.expert_policy.value for r in records})
    seeds = sorted({r.seed for r in records})
    bot_config = records[0].bot_config if records else []
    map_type = records[0].map_type if records else "BASE"
    default_notes = (
        "schema v2; splits by game_key; prompts from live renderer; "
        "DATA_CONTRACT §7 filters + §9 quality"
    )
    if immutable:
        default_notes = (
            "IMMUTABLE eval holdout — never train on this artifact; "
            "schema v2; prompts from live renderer; DATA_CONTRACT §7 filters"
        )
    manifest = DatasetManifest(
        name=name,
        version=version,
        schema_version="v2",
        prompt_version=PROMPT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=resolve_source_commit(),
        catanatron_commit=CATANATRON_COMMIT,
        generator_versions={"catan_llm": __version__},
        bot_mix=bot_mix,
        bot_config=bot_config,
        seeds=seeds,
        seed_range=seed_range,
        map_type=map_type,
        num_games=len({r.game_key for r in records}),
        num_decisions=len(records),
        split_counts=split_counts,
        checksums=checksums,
        notes=notes or default_notes,
        immutable=immutable,
        role=role or ("eval_holdout" if immutable else "train"),
        max_seq_length=max_len,
        quality=filter_stats.as_quality_dict(),
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    # Also write a small quality sidecar for humans / ticket 14.
    (out_dir / "quality.json").write_text(
        json.dumps(filter_stats.as_quality_dict(), indent=2), encoding="utf-8"
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
