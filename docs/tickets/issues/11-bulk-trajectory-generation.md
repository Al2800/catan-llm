# 11 — Bulk trajectory generation (≥100k decisions)

**What to build:** Generate expert trajectories per SCOPE §5.2 cohort plan until
≥100k **filtered** train decisions exist. Use registry seed ranges; stop at
target, don’t blindly burn the full reservation.

**Blocked by:** 01, 02, 03, 04, 05, 06, 07, 08, 09

**Status:** blocked

**Phase:** 1

- [ ] Generation uses schema v2 + resume-safe writes + named seed ranges
- [ ] Bot mix / map slices match cohort table (or an approved PR update to it)
- [ ] ≥100k filtered train decisions produced
- [ ] Manifest records counts, checksums, commits, prompt_version
