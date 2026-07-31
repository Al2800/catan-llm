# 13 — Immutable eval holdout set

**What to build:** Generate the separate `eval_holdout` seed-range games/decisions
that never enter training. Record immutability in the manifest.

**Blocked by:** 01–09 (done)

**Status:** claimed

**Phase:** 1

## Entrypoints

- `catan-phase1-cohort holdout --games 5000 --workers 4`
- Manifest: `immutable=true`, `role=eval_holdout`, single `holdout.jsonl`

## Acceptance criteria

- [x] Uses `eval_holdout` from SEED_REGISTRY only (wired + tested)
- [ ] ≥5k games generated
- [x] No overlap with train seed ranges (registry + runtime assert)
- [ ] Holdout artifact + checksum published (`immutable` on manifest)
