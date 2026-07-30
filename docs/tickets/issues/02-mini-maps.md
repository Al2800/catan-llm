# 02 — MINI maps fail-loud

**What to build:** Requesting `map_type=MINI` actually builds Catanatron’s
`MINI_MAP_TEMPLATE`. Failures raise; never silently label BASE as MINI.

**Blocked by:** None — can start immediately.

**Status:** done

**Phase:** 0.5 (T3)

- [x] MINI construction uses `CatanMap.from_template(MINI_MAP_TEMPLATE)`
- [x] Unsupported/failed map construction raises
- [x] Test proves MINI map hash/topology ≠ BASE for the same seed path
