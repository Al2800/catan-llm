# 10 — Feature-aware POV-safe Tier A rationales

**What to build:** Replace action-type restatement rationales with feature-aware
one-liners (pips, ports, blockers, races, optional valueΔ) that never leak
opponent private hands.

**Blocked by:** 04, 07

**Status:** done

**Phase:** 1

- [x] Tier A templates cover SCOPE §7.4 minimum features
- [x] POV leakage tests still pass
- [x] Sample rationales appear in a small generated shard for review

**Notes:** Templates in `src/catan_llm/data/tier_a.py`. Samples in
`docs/samples/tier_a_rationales.{md,jsonl}`.
