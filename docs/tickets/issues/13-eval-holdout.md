# 13 — Immutable eval holdout set

**What to build:** Generate the separate `eval_holdout` seed-range games/decisions
that never enter training. Record immutability in the manifest.

**Blocked by:** 01–09 (done)

**Status:** done

**Phase:** 1

## Entrypoints

- `catan-phase1-cohort holdout --games 5000 --workers 8`
- Manifest: `immutable=true`, `role=eval_holdout`, single `holdout.jsonl`
- Report: [`docs/reports/phase1_cohort_progress.md`](../../reports/phase1_cohort_progress.md)

## Acceptance criteria

- [x] Uses `eval_holdout` from SEED_REGISTRY only (wired + tested)
- [x] ≥5k games generated (fresh Mac burn; cloud partial abandoned)
- [x] No overlap with train seed ranges (registry + runtime assert)
- [x] Holdout artifact + checksum published (`immutable` on manifest)
