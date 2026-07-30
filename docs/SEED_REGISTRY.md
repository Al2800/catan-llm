# Seed registry

Single source of truth for disjoint seed ranges.  
Scripts and agents **must not** invent ad-hoc ranges.

Update this table in the same PR that allocates a new range.

## Active allocations

| Name | Start | Count | End (exclusive) | Purpose | Status |
|---|---:|---:|---:|---|---|
| `train_main` | 0 | 50_000 | 50_000 | Phase-1 BASE expert games (use until ≥100k filtered decisions; see SCOPE §5.2 — do not blindly burn all) | reserved |
| `train_mini_curriculum` | 50_000 | 10_000 | 60_000 | Optional MINI curriculum slice | reserved |
| `val_split_pool` | 60_000 | 5_000 | 65_000 | Extra games if split-by-game_key needs topping up | reserved |
| `eval_holdout` | 100_000 | 5_000 | 105_000 | Immutable offline holdout (≥5k games) | reserved |
| `ladder_sft_gate` | 200_000 | 1_000 | 201_000 | Stage-1 ladder (≥200 finished used from here) | reserved |
| `champion_ab` | 300_000 | 2_000 | 302_000 | Phase-3 AlphaBeta fixture (≥1000 finished) | reserved |
| `champion_ab_rerun` | 302_000 | 2_000 | 304_000 | Independent reproducibility rerun | reserved |
| `hw_smoke` | 900_000 | 100 | 900_100 | Local 8B / tiny smokes only | reserved |
| `dev_adhoc` | 9_000_000 | 1_000 | 9_001_000 | Local debugging; never publish numbers from here | open |

## Rules

1. Training data may only use `train_*` / `val_split_pool` ranges.
2. Published eval numbers may only use `eval_holdout`, `ladder_*`, or `champion_*`.
3. Once a `champion_*` range is used in a promotion report, it is **immutable** for that protocol version.
4. New ranges are appended below existing ones; never reuse a retired range for a different purpose.
5. Configs (e.g. `configs/qwen3.5-9b-qlora.yaml`) must reference a **name** from this table, not a magic number alone.

## How to allocate

Add a row, open a PR, mention the consuming script/config. Do not leave gaps undocumented.
