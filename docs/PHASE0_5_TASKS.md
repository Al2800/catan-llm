# Phase 0.5 task cards

Assignable units before Phase-1 scale.

Normative refs: [`DATA_CONTRACT.md`](DATA_CONTRACT.md), [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md),
[`SEED_REGISTRY.md`](SEED_REGISTRY.md), [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md),
[`../AGENTS.md`](../AGENTS.md), [`SCOPE.md`](SCOPE.md) §14.

**Base branch:** `main` (do not use long-lived feature branches).

## Dependency graph

```
T1 (schema v2)
├── T2 (prompt parity) ──────────────┐
├── T4 (splits + seed registry) ─────┤
├── T6 (eval metrics) ───────────────┼──► T7 (CI gates)
└── T9 (POV audit + assistant mask) ─┘
T3 (MINI)  ── parallel, then into T7
T5 (resume writes) ── parallel
T8 (5060 Ti 8B smoke) ── parallel on owner GPU; blocks Phase 1
```

| Wave | Cards | Notes |
|---|---|---|
| Wave A (parallel) | T1, T3, T5 | no mutual deps |
| Wave B | T2, T4, T6, T9 | after T1 |
| Wave C | T7 | after T2/T3/T4/T6/T9 |
| Hardware | T8 | owner GPU anytime; required for Phase-1 exit |

---

## T1 — Schema v2 + identity fields ✅

**Goal:** Trajectory records carry contract v2 fields.  
**Status:** done (ticket 01)  
**Deps:** none  
**Touch:** `src/catan_llm/data/schema.py`, `data/identity.py`, `sim/trajectories.py`, tests.  
**Done when:**
- Records include `schema_version="v2"`, `game_key`, `map_hash`, `bot_config`, `bot_config_hash`, `prompt_version`, `catanatron_commit`, `source_commit`
- Unit tests cover round-trip
- Builder rejects v1 rows for Phase-1 paths

## T2 — Canonical prompt parity ✅

**Goal:** Train chat JSONL uses live renderer only.  
**Status:** done (ticket 04)  
**Deps:** T1  
**Touch:** `data/dataset.py`, `sim/trajectories.py`, tests.  
**Done when:**
- Compact alternate prompt path removed for labeled SFT
- `tests/test_prompt_parity.py` asserts system/user equality on ≥20 seeded games
- `PROMPT_VERSION` constant exists and is stored on records

## T3 — MINI maps fail-loud ✅

**Goal:** `map_type=MINI` uses `MINI_MAP_TEMPLATE`.  
**Status:** done (ticket 02)  
**Deps:** none  
**Touch:** `sim/adapter.py`, tests.  
**Done when:**
- `CatanMap.from_template(MINI_MAP_TEMPLATE)` used
- No silent BASE fallback
- Test asserts MINI topology/hash ≠ BASE for same seed construction path

## T4 — Stable splits + seed registry wiring ✅

**Goal:** Splits by `game_key`; scripts read [`SEED_REGISTRY.md`](SEED_REGISTRY.md) names.  
**Status:** done (ticket 05)  
**Deps:** T1  
**Touch:** `data/dataset.py`, `data/seed_registry.py`, generate CLI, tests.  
**Done when:**
- No UUID-based split
- Same inputs → same split assignment
- CLI accepts `--seed-range-name train_main` (or equivalent)
- Cohort stop condition documented (SCOPE §5.2) — do not blindly burn 50k games

## T5 — Resume-safe shard writes ✅

**Goal:** Crash-safe generation.  
**Status:** done (ticket 03)  
**Deps:** none  
**Touch:** `sim/adapter.py`, `scripts/generate_trajectories.py`, `docs/PHASE0.md`.  
**Done when:**
- No `unlink()` of completed outputs at job start
- Atomic rename or append-only + journal of completed `game_key`s
- Documented resume flag

## T6 — Eval protocol metrics + fixture hooks ✅

**Goal:** Arena reports Gate-B fields with correct WR comparison.  
**Status:** done (ticket 06)  
**Deps:** T1 (for prompt_version plumbing if exposed)  
**Touch:** `eval/metrics.py`, `eval/arena.py`, `play/llm_player.py`, `scripts/run_arena.py`.  
**Done when:**
- Reports `parse_rate_model`, `legality_rate_model`, `fallback_rate`, `vp_margin`
- Reports `win_share_gap[candidate,weightedrandom]`
- Fallback policy constant = `first_legal`
- Zero-win seats still appear in `win_rates`
- Fixture formats `ladder-4p` and `ab-4p` selectable

## T7 — Gate tests in CI

**Goal:** Agents see red, not prose.  
**Deps:** T2, T3, T4, T6, T9  
**Touch:** `tests/`, `.github/workflows/ci.yml`.  
**Done when:**
- Parity / MINI / schema-v2 / no-truncation / POV-rationale / assistant-mask tests run in CI
- Unfinished items may use `xfail(strict=True)` only with a task-id reason

## T8 — Local 8B QLoRA smoke (owner GPU)

**Goal:** Prove 5060 Ti path at label-safe context.  
**Deps:** none (ideally after T2 for real prompts)  
**Touch:** env pins, `configs/qwen3-8b-qlora.yaml`, `outputs/hw_smoke/report.json`.  
**Done when:**
- Checklist in [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md) completed
- Pinned Qwen **revision** recorded (no `revision: null` left in the successful report)
- `max_seq_length >= 4096`; peak VRAM logged
- Micro-train + one game vs Random
- If OOM: document prompt-compression or rental fallback — **do not** lower below label-safe length
- Report checked in or attached to PR

## T9 — Teacher POV audit + assistant-mask test

**Goal:** Lock distillation fairness and prove SFT loss hits labels.  
**Deps:** T1 (and T2 for real chat formatting)  
**Touch:** docs note in DATA_CONTRACT if needed; `tests/test_teacher_pov.py`; `tests/test_assistant_mask.py`; training helpers.  
**Done when:**
- Short audit doc or test comments state: experts may use full `Game`; learner prompts POV-limited; Tier A POV-safe
- Test fails if generated Tier A text includes opponent private-hand literals
- One-batch test on the **pinned** Qwen tokenizer/chat template proves:
  - system/user tokens masked from loss
  - assistant JSON tokens have nonzero loss
  - full `{"action": ...}` span present (no truncation)
- If the pin is not yet chosen, test may skip with clear message — but T8 must fix the pin and T9 must pass before Phase 1

---

## Exit

Phase 0.5 is done when **T1–T7 and T9 are merged** and **T8 report exists**
(local success **or** explicit approved rental fallback).  
Only then start Phase-1 generation toward ≥100k filtered decisions (SCOPE §5.2).
