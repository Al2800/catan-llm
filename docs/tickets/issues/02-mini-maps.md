# 02 — MINI maps fail-loud

**What to build:** Requesting `map_type=MINI` actually builds Catanatron’s
`MINI_MAP_TEMPLATE`. Failures raise; never silently label BASE as MINI.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

**Phase:** 0.5 (T3)

- [ ] MINI construction uses `CatanMap.from_template(MINI_MAP_TEMPLATE)`
- [ ] Unsupported/failed map construction raises
- [ ] Test proves MINI map hash/topology ≠ BASE for the same seed path
