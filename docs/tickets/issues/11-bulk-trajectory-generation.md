# 11 — Bulk trajectory generation (≥100k decisions)

**What to build:** Generate expert trajectories per SCOPE §5.2 cohort plan until
≥100k **filtered** train decisions exist. Use registry seed ranges; stop at
target, don’t blindly burn the full reservation.

**Blocked by:** 01–09 (done)

**Status:** claimed

**Phase:** 1

## Entrypoints

- `catan-generate --target-decisions … --seed-range-name train_main --rotate-seats`
- `catan-phase1-cohort train --target-decisions 100000 --workers 4`

## Acceptance criteria

- [x] Generation uses schema v2 + resume-safe writes + named seed ranges
- [x] Bot mix / map slices match cohort table (train_main rotated ladder; optional MINI)
- [ ] ≥100k filtered train decisions produced
- [ ] Manifest records counts, checksums, commits, prompt_version
