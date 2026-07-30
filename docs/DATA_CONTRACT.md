# Data Contract v1

Normative contract for trajectories, prompts, splits, and manifests.
Implements SCOPE §4.1–4.2 and §5. If code disagrees with this doc, **this doc wins** until both are updated together.

## 1. Goals

1. Train and live play see **byte-identical prompts** for the same decision.
2. Rerunning generation with the same config produces **identical normalized shards and splits**.
3. Downstream training never trains on the eval holdout.

## 2. Stable identity

| Field | Meaning |
|---|---|
| `seed` | Engine RNG seed for the game |
| `map_type` | `BASE` or `MINI` |
| `map_hash` | Deterministic hash of land tiles / numbers / ports after map construction |
| `bot_config` | Ordered seat list with bot kind + params (e.g. `alphabeta:depth=2`) |
| `bot_config_hash` | Hash of canonical JSON `bot_config` |
| `game_key` | `sha256(f"{seed}:{map_hash}:{bot_config_hash}")` — **split key** |
| `game_id` | Opaque UUID for logging only — **not** used for splits |

## 3. Trajectory record (schema v1)

One JSONL row per decision, written **before** the action is applied.

Required fields:

```json
{
  "schema_version": "v1",
  "game_key": "...",
  "game_id": "...",
  "decision_idx": 0,
  "seed": 0,
  "map_type": "BASE",
  "map_hash": "...",
  "bot_config": [{"name": "alphabeta", "params": {"depth": 2}}, "..."],
  "bot_config_hash": "...",
  "catanatron_commit": "82aae93...",
  "source_commit": "<git sha of catan-llm>",
  "player_color": "RED",
  "turn": 0,
  "phase": "BUILD_INITIAL_SETTLEMENT",
  "board": { "...": "static layout needed by canonical renderer" },
  "state": { "...": "POV-aware dynamic state" },
  "valid_actions": [{"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3}],
  "action_taken": {"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3},
  "action_index": 3,
  "expert_policy": "alphabeta",
  "outcome": {"winner": "RED", "vps": {"RED": 10}, "turns": 72, "finished": true}
}
```

Rules:

- `action_index` must be `>= 0` and refer to `valid_actions[action_index] == action_taken` after normalization.
- Unfinished games (`outcome.finished == false`) are dropped from training sets; may be kept in a `debug/` shard.
- Opponent private hands must not appear in `state` for other colors (POV-aware).
- `board` must contain enough static layout for the canonical renderer without a live `Game` object **or** the builder must reconstruct a `Game` and call the live renderer. Either approach is fine; **output prompts must match live play**.

## 4. Canonical prompts

Training chat rows MUST be produced by the same functions used at play time:

- system: `render_system_prompt(game, player_color)`
- user: `render_user_prompt(game, player_color, playable_actions)`
- assistant: `{"action": <index>, "reasoning": "<tier-A text>"}`

Forbidden for labeled SFT:

- Compact alternate prompts that omit board layout / action formatting
- Different action pretty-printing between train and play

### Parity gate (CI + pre-scale)

Sample ≥1k decisions (or all decisions from ≥20 seeded games). For each:

1. Replay or reconstruct the pre-action game state.
2. Render live system/user prompts.
3. Assert equality with stored/dataset prompts.
4. Assert `record_to_action(valid_actions[action_index])` is legal in that state.

## 5. Maps

| `map_type` | Required construction |
|---|---|
| `BASE` | Default `CatanMap` / `BASE_MAP_TEMPLATE` |
| `MINI` | `CatanMap.from_template(MINI_MAP_TEMPLATE)` |

If construction fails → raise. Never silently substitute BASE while writing `map_type="MINI"`.

## 6. Splits

1. Group records by `game_key`.
2. Sort keys lexicographically.
3. Assign fractions: train 90% / val 5% / test 5% (configurable, recorded in manifest).
4. `eval-holdout` is a **separately generated** seed range, never overlapping training seeds.

Do **not** split on UUID `game_id`.

## 7. Filtering & balance (Phase 1)

Drop:

- unfinished games
- decisions with `action_index < 0`
- records that fail action round-trip / legality checks

Balance / oversample (targets for v1 report, not hard blockers on smoke sets):

| Bucket | Intent |
|---|---|
| Initial placement | Keep full coverage |
| Early / mid / late (`turn` terciles of finished games) | Roughly even exposure |
| Rare actions | Oversample `PLAY_MONOPOLY`, `PLAY_YEAR_OF_PLENTY`, `PLAY_ROAD_BUILDING`, `PLAY_KNIGHT_CARD`, `BUY_DEVELOPMENT_CARD`, maritime trades, `MOVE_ROBBER` |

Publish counts in the dataset quality report.

## 8. Tier A rationales

Acceptable: feature-aware templates (pip sums, port access, blocker value, value-fn delta if available).

Not sufficient for Phase-1 sign-off: `"policy selects BUILD_SETTLEMENT"` style restatements alone.

## 9. Manifest (required keys)

```json
{
  "name": "expert-v1",
  "version": "v1",
  "schema_version": "v1",
  "created_at": "ISO-8601",
  "source_commit": "...",
  "catanatron_commit": "...",
  "generator_versions": {"catan_llm": "0.x"},
  "bot_config": [],
  "map_type": "BASE",
  "seed_range": {"start": 0, "count": 1000},
  "num_games": 0,
  "num_decisions": 0,
  "split_counts": {"train": 0, "val": 0, "test": 0},
  "checksums": {"train.jsonl": "sha256:...", "val.jsonl": "...", "test.jsonl": "..."},
  "quality": {"unfinished_dropped": 0, "illegal_dropped": 0, "action_type_hist": {}}
}
```

## 10. Write durability

- Write to `*.jsonl.partial` then atomic rename, **or** append-only shards with a manifest of completed shard ids.
- Never `unlink()` a completed output at the start of a resume-capable job.
- Crash recovery: skip completed `game_key`s recorded in a sidecar journal.

## 11. Phase-1 exit checklist

- [ ] ≥100k train decisions after filtering
- [ ] Separate ≥5k-game eval holdout seed range
- [ ] Parity gate green
- [ ] Manifest complete + checksums
- [ ] Quality report checked in (`docs/` or `outputs/reports/`)
