# 11 — Bulk trajectory generation (≥100k decisions)

**What to build:** Generate expert trajectories per SCOPE §5.2 cohort plan until
≥100k **filtered** train decisions exist. Use registry seed ranges; stop at
target, don’t blindly burn the full reservation.

**Blocked by:** 01–09 (done)

**Status:** done (2026-07-31)

**Phase:** 1

## Entrypoints

- `catan-generate --target-decisions … --seed-range-name train_main --rotate-seats`
- `catan-phase1-cohort train --target-decisions 112000 --workers 4`

## Result

| Item | Value |
|---|---|
| Seed range | `train_main` |
| Filtered decisions | 112,754 |
| Train / val / test | **101,303** / 5,525 / 5,926 |
| Games | (see `data/phase1/train_cohort_report.json`) |
| Dataset | `data/phase1/processed/expert-v1/` |
| Report | [`docs/reports/phase1_cohort_progress.md`](../../reports/phase1_cohort_progress.md) |

## Acceptance criteria

- [x] Generation uses schema v2 + resume-safe writes + named seed ranges
- [x] Bot mix / map slices match cohort table (train_main rotated ladder)
- [x] ≥100k filtered **train-split** decisions produced (101,303)
- [x] Manifest records counts, checksums, commits, prompt_version
