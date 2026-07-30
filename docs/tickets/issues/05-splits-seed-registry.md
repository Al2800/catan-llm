# 05 — Stable splits + seed registry wiring

**What to build:** Dataset splits key on `game_key` (not UUID). CLIs accept
named seed ranges from `SEED_REGISTRY.md`. Same inputs → same split assignment.

**Blocked by:** 01

**Status:** done

**Phase:** 0.5 (T4)

- [x] No UUID-based split logic remains on Phase-1 paths
- [x] Deterministic split assignment covered by a test
- [x] CLI supports `--seed-range-name` (or equivalent) from the registry
- [x] Cohort stop guidance referenced (SCOPE §5.2) — do not blindly burn 50k games
