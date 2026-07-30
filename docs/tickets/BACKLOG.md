# Catan LLM — ticket backlog

Source of truth for outstanding work. Derived from [`SCOPE.md`](../SCOPE.md) and
[`PHASE0_5_TASKS.md`](../PHASE0_5_TASKS.md).

**Base branch:** `main` (no long-lived feature branches)

## How to use

1. Open [`BACKLOG.md`](BACKLOG.md) (this file) for status.
2. Pick a **frontier** ticket (`ready-for-agent`, blockers all `done`).
3. Set that ticket’s `Status: claimed`, implement on `main`, mark `done`.
4. Keep docs and code in the same commit when contracts change.

## Status summary

| ID | Title | Phase | Status | Blocked by |
|---|---|---|---|---|
| [00](issues/00-merge-phase0-branch.md) | Merge Phase-0 branch to main (or pin agents to it) | Meta | done | — |
| [01](issues/01-schema-v2.md) | Schema v2 + identity fields | 0.5 | done | — |
| [02](issues/02-mini-maps.md) | MINI maps fail-loud | 0.5 | done | — |
| [03](issues/03-resume-safe-writes.md) | Resume-safe shard writes | 0.5 | done | — |
| [04](issues/04-prompt-parity.md) | Canonical train/play prompt parity | 0.5 | done | 01 |
| [05](issues/05-splits-seed-registry.md) | Stable splits + seed registry wiring | 0.5 | done | 01 |
| [06](issues/06-eval-gate-b-metrics.md) | Eval Gate-B metrics + 4p fixtures | 0.5 | done | 01 |
| [07](issues/07-pov-audit-assistant-mask.md) | Teacher POV audit + assistant-mask test | 0.5 | done | 01, 04 |
| [08](issues/08-ci-gate-tests.md) | CI gate tests | 0.5 | done | 02, 04–07 |
| [09](issues/09-local-8b-qlora-smoke.md) | Local 8B QLoRA smoke on 5060 Ti | 0.5 | ready-for-agent | — (owner GPU) |
| [10](issues/10-tier-a-rationales.md) | Feature-aware POV-safe Tier A rationales | 1 | ready-for-agent | 04✓, 07✓ |
| [11](issues/11-bulk-trajectory-generation.md) | Bulk trajectory generation (≥100k decisions) | 1 | blocked | 01–09 |
| [12](issues/12-dataset-builder-quality.md) | Dataset builder quality + manifests | 1 | blocked | 05, 10, 11 |
| [13](issues/13-eval-holdout.md) | Immutable eval holdout set | 1 | blocked | 01–09 |
| [14](issues/14-dataset-quality-signoff.md) | Dataset quality report + Phase-1 sign-off | 1 | blocked | 12, 13 |
| [15](issues/15-qlora-training-pipeline.md) | Production QLoRA training pipeline | 2 | blocked | 09, 14 |
| [16](issues/16-serving-constrained-decoding.md) | Serving + constrained decoding path | 2 | blocked | 09 |
| [17](issues/17-sft-run-gate-b.md) | SFT run + Gate B ladder eval | 2 | blocked | 15, 16 |
| [18](issues/18-failure-taxonomy-v1.md) | Failure taxonomy v1 | 2 | blocked | 17 |
| [19](issues/19-fill-rl-spec.md) | Fill RL_SPEC (reward / anti-hack / cost) | 3 | blocked | 17 |
| [20](issues/20-grpo-loop.md) | GRPO training loop implementation | 3 | blocked | 19 |
| [21](issues/21-champion-gate-c.md) | RL run + Gate C AlphaBeta champion | 3 | blocked | 20 |
| [22](issues/22-12b-rental.md) | 12B-class rental run | 4 | blocked | 21 |
| [23](issues/23-self-play-iteration.md) | Self-play iteration loop | 4 | blocked | 21 |
| [24](issues/24-live-spectate.md) | Live spectate UX | 4 | blocked | 16 |
| [25](issues/25-blog-writeup.md) | Blog write-up + publishable artifacts | 4 | blocked | 21 |

## Frontier right now (can start)

- **09** — owner GPU 8B QLoRA smoke (on hold / local machine; last Phase 0.5 blocker)
- **10** — Tier A feature-aware rationales (Phase 1; unblocked for coding, scale still waits on 09)

## Already done (not ticketed as open work)

Phase 0 plumbing spike: package scaffold, Catanatron pin, basic trajectories, renderer/parser, bot arena v0, tiny SmolLM SFT smoke, CI ruff/pytest, plan docs + engineering skills vendored. See PR #1 history.
