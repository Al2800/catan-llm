# 17 — SFT run + Gate B ladder eval

**What to build:** Train SFT v1 and evaluate on pre-registered `ladder-4p`.
Promote only if Gate B passes (legality floors + candidate win share >
WeightedRandom in the same fixture).

**Blocked by:** 15, 16 (done)

**Status:** ready-for-agent

**Phase:** 2

**Train entrypoint:** `catan-qlora-train` ([ticket 15](15-qlora-training-pipeline.md)) on rental ≥24GB.

- [ ] Fixture identity recorded before eval
- [ ] `parse_rate_model` / `legality_rate_model` ≥ 0.995
- [ ] `win_rate[candidate] > win_rate[weightedrandom]` at ≥200 finished games
- [ ] Eval JSON report published under `outputs/arena/`
