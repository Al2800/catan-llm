# 13 — Immutable eval holdout set

**What to build:** Generate the separate `eval_holdout` seed-range games/decisions
that never enter training. Record immutability in the manifest.

**Blocked by:** 01–09 (done)

**Status:** claimed — **handed off to Mac** (cloud burn stopped ~1725/5000)

**Phase:** 1

## Entrypoints

- `catan-phase1-cohort holdout --games 5000 --workers 8`
- Manifest: `immutable=true`, `role=eval_holdout`, single `holdout.jsonl`
- Handoff: [`docs/reports/MAC_HANDOFF_PHASE1.md`](../../reports/MAC_HANDOFF_PHASE1.md)

## Acceptance criteria

- [x] Uses `eval_holdout` from SEED_REGISTRY only (wired + tested)
- [ ] ≥5k games generated (regenerate fresh on Mac — do not resume cloud partial)
- [x] No overlap with train seed ranges (registry + runtime assert)
- [ ] Holdout artifact + checksum published (`immutable` on manifest)
