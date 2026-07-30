# Phase 0 notes

## Exit criteria (from SCOPE §11)

Repo scaffold, pinned deps, simulator adapter, trajectory schema, canonical
renderer + action parser, tiny SFT smoke, eval arena v0 (bot-vs-bot), CI.

**Exit:** end-to-end loop proven with a small model on CPU/single GPU.

## Status

**Phase 0 plumbing: complete** (spike proves wiring).  
**Phase 0.5 (contract repair + local 8B proof): not done** — required before Phase 1 scale.

See plan updates in [`SCOPE.md`](SCOPE.md) §11–§12 and companion specs:

- [`DATA_CONTRACT.md`](DATA_CONTRACT.md)
- [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)
- [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md)
- [`../configs/qwen3-8b-qlora.yaml`](../configs/qwen3-8b-qlora.yaml)

## Phase-0 decisions

| Topic | Decision |
|---|---|
| Package layout | `src/catan_llm/{sim,data,eval,play,training,serving}` |
| Catanatron pin | git `82aae93` (declared 3.3.0; PyPI still at 3.2.1) |
| Smoke base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (prove loop; Qwen3-8B is Phase 2) |
| Experiment tracking | Local JSON reports (`outputs/**/report.json`); W&B optional later |
| Domestic trading | Out of scope (SCOPE §12.5) — renderer/parser omit it |

## Smoke result (CPU plumbing)

`catan-sft-smoke --games 2 --max-steps 5 --max-samples 32 --eval-games 1` on CPU with
`HuggingFaceTB/SmolLM2-135M-Instruct` completed successfully:

- train loss ~1.33 over 5 steps
- eval parse_rate **1.0**, legality_rate **1.0**

Interpretation: proves the loop wires together. It is **not** evidence of Catan skill
(a constant legal action-0 policy can also look “legal”).

## Known gaps carried into Phase 0.5

From the post-spike review (must fix before ≥100k data / 8B claims):

1. **Train/play prompt mismatch** — dataset builder uses a compact prompt; live play uses the full renderer.
2. **MINI maps** — `_build_map("MINI")` does not pass `MINI_MAP_TEMPLATE` and can silently use BASE.
3. **Splits on UUID `game_id`** — not stable across regeneration; contract requires `game_key`.
4. **Generation deletes existing output** — not resume-safe.
5. **No local 8B QLoRA / bitsandbytes / vLLM pin validated** on the 5060 Ti yet.
6. **Eval lacks** VP margin, fallback-separated legality, fixed AlphaBeta fixture, promotion JSON.
7. **Tier A rationales** are action-type restatements; Phase 1 needs feature-aware templates.
8. **License posture** — Catanatron is GPL-3; confirm distribution model before combined packaging.

## Commands

See root `README.md`.
