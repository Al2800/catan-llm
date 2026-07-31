# Project status — where we are

**Updated:** 2026-07-31  
**Current phase:** **Phase 2 — SFT model v1** (ticket **17** in progress)  
**Ticket index:** [`tickets/BACKLOG.md`](tickets/BACKLOG.md)

## What we are trying to achieve

Train an open-weights **~9B** LLM (**Qwen3.5-9B**, QLoRA) to play Settlers of Catan
inside [Catanatron](https://github.com/bcollazo/catanatron) well enough to:

1. **Near-perfect parse/legality** on structured JSON actions (`first_legal` fallback locked).
2. **Beat WeightedRandom** on the pre-registered 4p `ladder-4p` fixture (Gate B, ≥200 games).
3. Later **match/beat pinned AlphaBeta** on `ab-4p` via GRPO (Gate C — Phase 3).

Full vision, risks, and locked decisions: [`SCOPE.md`](SCOPE.md).

## Phase scoreboard

| Phase | Goal | Status |
|---|---|---|
| 0 | Plumbing spike (sim → data → tiny SFT → arena) | **done** |
| 0.5 | Schema v2, parity, seeds, CI gates, 9B QLoRA HW proof | **done** (local 16GB train **no-go**; L40S rental **go**) |
| 1 | ≥100k train decisions + 5k-game holdout + quality sign-off | **done** — [GO for Phase 2](reports/phase1_quality_signoff.md) |
| 2 | Production QLoRA SFT + Gate B + failure taxonomy | **in progress** (15/16/24 done; **17** claimed) |
| 3 | RL_SPEC → GRPO → Gate C | blocked on 17→19 |
| 4 | 12B / self-play / polish / blog | later |

## Done (high-signal artifacts)

| Area | Artifact |
|---|---|
| Model pin | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| HW smoke | [`reports/hw_smoke_rental_l40s.md`](reports/hw_smoke_rental_l40s.md) — peak train VRAM **14.9 GB** @ 4096 |
| Train cohort | 440 games → **101,303** train-split decisions — HF `AlCampbell/catan-llm-phase1` |
| Holdout | 5,000 games, immutable (Mac); checksum in cohort progress report |
| Quality | [`reports/phase1_quality_signoff.md`](reports/phase1_quality_signoff.md) |
| Train entrypoint | `catan-qlora-train` ([`TRAINING.md`](TRAINING.md)) |
| Serve / play | [`SERVING.md`](SERVING.md), `catan-serve`, `catan-play-endpoint` |
| Spectate | [`SPECTATE.md`](SPECTATE.md), `catan-spectate --watch` |
| Pipeline smoke | 40-step L40S train + 2 arena games; adapter on `AlCampbell/catan-llm-sft-v1` |
| Production SFT | 2000-step L40S job launched (cost-gated vs full 2-epoch) |

## In flight

- **Ticket 17:** 2000-step L40S SFT → then Gate B (200 games) → publish arena JSON.
- **Ticket 18 tooling:** `catan-taxonomy` ready; fill report when Gate B JSON exists.
- **Ops:** upload Mac `eval-holdout-v1` via `scripts/upload_holdout_hf.py`.

## Next (after 17)

1. **18** — failure taxonomy v1 (why the model loses).
2. **19** — fill [`RL_SPEC.md`](RL_SPEC.md).
3. **20 → 21** — GRPO loop → Gate C vs AlphaBeta.
4. Optional longer SFT (toward 1 epoch) if Gate B format is good but skill is short.

## Expectation for the 2000-step SFT

Sees ~16k examples (~16% of one epoch). Likely improves **JSON format / legality**;
unlikely alone to **clear Gate B vs WeightedRandom**. Treat as diagnostic SFT v0.5
checkpoint for spectate + taxonomy — extend train if metrics justify the GPU cost
(~$30 L40S for 2k steps; ~$180–200 for ~1 epoch).

## Where to read next

| If you want… | Open |
|---|---|
| Architecture diagrams + module map | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| How to train / rent GPU | [`TRAINING.md`](TRAINING.md) |
| Eval gates | [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) |
| Data rules | [`DATA_CONTRACT.md`](DATA_CONTRACT.md) |
| Ticket statuses | [`tickets/BACKLOG.md`](tickets/BACKLOG.md) |
| Agent rules | [`../AGENTS.md`](../AGENTS.md) |
