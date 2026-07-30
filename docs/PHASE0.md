# Phase 0 notes

## Exit criteria (from SCOPE §11)

Repo scaffold, pinned deps, simulator adapter, trajectory schema, canonical
renderer + action parser, tiny SFT smoke, eval arena v0 (bot-vs-bot), CI.

**Exit:** end-to-end loop proven with a small model on CPU/single GPU.

## Status

**Phase 0 plumbing: complete** (spike proves wiring).  
**Phase 0.5 (contract repair + local 8B proof): next** — required before Phase 1 scale.  
**Handoff base:** `main` (merged).

Handoff pack:

| Doc | Role |
|---|---|
| [`SCOPE.md`](SCOPE.md) | Goals, phases, locked decisions, handoff readiness |
| [`DATA_CONTRACT.md`](DATA_CONTRACT.md) | Schema **v2** + parity + truncation rules |
| [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) | Fixtures / gates; fallback=`first_legal` |
| [`SEED_REGISTRY.md`](SEED_REGISTRY.md) | Disjoint seed ranges |
| [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md) | 5060 Ti validation checklist |
| [`PHASE0_5_TASKS.md`](PHASE0_5_TASKS.md) | Assignable T1–T8 cards |
| [`RL_SPEC.md`](RL_SPEC.md) | Phase-3 entry template |
| [`../AGENTS.md`](../AGENTS.md) | Do/don't for coding agents |
| [`../configs/qwen3.5-9b-qlora.yaml`](../configs/qwen3.5-9b-qlora.yaml) | `max_seq_length: 4096` |

## Phase-0 decisions

| Topic | Decision |
|---|---|
| Package layout | `src/catan_llm/{sim,data,eval,play,training,serving}` |
| Catanatron pin | git `82aae93` (declared 3.3.0; PyPI still at 3.2.1) |
| Smoke base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (prove loop; Qwen3.5-9B is Phase 2) |
| Experiment tracking | Local JSON reports (`outputs/**/report.json`); W&B optional later |
| Domestic trading | Out of scope (SCOPE §12.5) — renderer/parser omit it |
| Fallback | `first_legal` |
| Train context | `max_seq_length >= 4096` for canonical prompts |

## Smoke result (CPU plumbing)

`catan-sft-smoke --games 2 --max-steps 5 --max-samples 32 --eval-games 1` on CPU with
`HuggingFaceTB/SmolLM2-135M-Instruct` completed successfully:

- train loss ~1.33 over 5 steps
- eval parse_rate **1.0**, legality_rate **1.0**

Interpretation: proves the loop wires together. It is **not** evidence of Catan skill
(a constant legal action-0 policy can also look “legal”).

## Known gaps → Phase 0.5 task cards

| Gap | Task |
|---|---|
| Schema v2 identity fields landed | T1 done |
| Train/play prompt parity via stored live renders | T2 done |
| MINI maps fail-loud via `MINI_MAP_TEMPLATE` | T3 done |
| Splits by `game_key`; `--seed-range-name` wired | T4 done |
| Resume-safe append + journal (no silent unlink) | T5 done |
| Gate-B metrics + `ladder-4p` / `ab-4p` fixtures | T6 done |
| Contract gate tests wired in CI | T7 done |
| No local 8B QLoRA proof at 4096; revision unpinned | T8 (owner GPU) |
| Privileged-teacher POV audit + assistant-mask proof | T9 done (Qwen one-batch skips until T8 pin) |
| Tier A feature-aware templates landed (ticket 10); bulk scale still waits on T8/09 | Phase 1 partial |
| `/setup-matt-pocock-skills` not run | optional (needed for ticket/triage skills) |
| GPL distribution posture | owner decision before combined packaging |

## Resume-safe generation (T5)

`catan-generate-data` / `generate_trajectories`:

| Flag | Behavior |
|---|---|
| `--resume` (default) | Append-only JSONL; skip `game_key`s listed in `<out>.journal` |
| `--no-resume` | Refuse if output/journal already non-empty |
| `--overwrite` | Explicitly wipe JSONL + journal, then write fresh |

Completed outputs are never `unlink()`'d at job start. Journal lines are
`{"game_key": "...", "seed": N}`.

## Commands

See root `README.md` and `AGENTS.md`.
