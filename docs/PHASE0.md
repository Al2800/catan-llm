# Phase 0 notes

## Exit criteria (from SCOPE §11)

Repo scaffold, pinned deps, simulator adapter, trajectory schema, canonical
renderer + action parser, tiny SFT smoke, eval arena v0 (bot-vs-bot), CI.

**Exit:** end-to-end loop proven with a small model on CPU/single GPU.

## Phase-0 decisions

| Topic | Decision |
|---|---|
| Package layout | `src/catan_llm/{sim,data,eval,play,training,serving}` |
| Catanatron pin | git `82aae93` (declared 3.3.0; PyPI still at 3.2.1) |
| Smoke base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (prove loop; Qwen3-8B is Phase 2) |
| Experiment tracking | Local JSON reports (`outputs/**/report.json`); W&B optional later |
| Domestic trading | Out of scope (SCOPE §12.5) — renderer/parser omit it |

## Smoke result (this environment)

`catan-sft-smoke --games 2 --max-steps 5 --max-samples 32 --eval-games 1` on CPU with
`HuggingFaceTB/SmolLM2-135M-Instruct` completed successfully:

- train loss ~1.33 over 5 steps
- eval parse_rate **1.0**, legality_rate **1.0** (fallback covers failures; none observed)

## Commands

See root `README.md`.
