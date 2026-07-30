# Phase 0.5 task cards

Assignable units before Phase-1 scale. Each card is independently shippable with tests.

Normative refs: [`DATA_CONTRACT.md`](DATA_CONTRACT.md), [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md), [`SEED_REGISTRY.md`](SEED_REGISTRY.md), [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md), [`../AGENTS.md`](../AGENTS.md).

---

## T1 — Schema v2 + identity fields

**Goal:** Trajectory records carry contract v2 fields.  
**Touch:** `src/catan_llm/data/schema.py`, `sim/trajectories.py`, tests.  
**Done when:**
- Records include `schema_version="v2"`, `game_key`, `map_hash`, `bot_config`, `bot_config_hash`, `prompt_version`, `catanatron_commit`, `source_commit`
- Unit tests cover round-trip
- Builder rejects v1 rows for Phase-1 paths

## T2 — Canonical prompt parity

**Goal:** Train chat JSONL uses live renderer only.  
**Touch:** `data/dataset.py`, `data/renderer.py`, tests.  
**Done when:**
- Compact alternate prompt path removed for labeled SFT
- `tests/test_prompt_parity.py` asserts system/user equality on ≥20 seeded games
- `PROMPT_VERSION` constant exists and is stored on records

## T3 — MINI maps fail-loud

**Goal:** `map_type=MINI` uses `MINI_MAP_TEMPLATE`.  
**Touch:** `sim/adapter.py`, tests.  
**Done when:**
- `CatanMap.from_template(MINI_MAP_TEMPLATE)` used
- No silent BASE fallback
- Test asserts MINI topology/hash ≠ BASE for same seed construction path

## T4 — Stable splits + seed registry wiring

**Goal:** Splits by `game_key`; scripts read [`SEED_REGISTRY.md`](SEED_REGISTRY.md) names.  
**Touch:** `data/dataset.py`, generate/build CLIs, tests.  
**Done when:**
- No UUID-based split
- Same inputs → same split assignment
- CLI accepts `--seed-range-name train_main` (or equivalent)

## T5 — Resume-safe shard writes

**Goal:** Crash-safe generation.  
**Touch:** `sim/adapter.py`.  
**Done when:**
- No `unlink()` of completed outputs at job start
- Atomic rename or append-only + journal of completed `game_key`s
- Documented resume flag

## T6 — Eval protocol metrics + fixture hooks

**Goal:** Arena reports Gate-B fields.  
**Touch:** `eval/metrics.py`, `eval/arena.py`, `play/llm_player.py`.  
**Done when:**
- Reports `parse_rate_model`, `legality_rate_model`, `fallback_rate`, `vp_margin`
- Fallback policy constant = `first_legal`
- Zero-win seats still appear in `win_rates`

## T7 — Gate tests in CI

**Goal:** Agents see red, not prose.  
**Touch:** `tests/`, `.github/workflows/ci.yml`.  
**Done when:**
- Parity / MINI / schema-v2 tests run in CI
- Optional: mark unfinished cards `xfail(strict=True)` only with a task-id reason, then remove xfail as each card lands

## T8 — Local 8B QLoRA smoke (owner GPU)

**Goal:** Prove 5060 Ti path.  
**Touch:** env pins, `configs/qwen3-8b-qlora.yaml`, `outputs/hw_smoke/report.json`.  
**Done when:**
- Checklist in [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md) completed
- `max_seq_length >= 4096`
- Peak VRAM logged; micro-train + one game vs Random
- Report checked in or attached to PR

---

## Exit

Phase 0.5 is done when **T1–T7 are merged** and **T8 report exists**. Only then start Phase-1 ≥100k generation.
