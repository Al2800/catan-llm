# 17 — SFT run + Gate B ladder eval

**What to build:** Train SFT v1 and evaluate on pre-registered `ladder-4p`.
Promote only if Gate B passes (legality floors + candidate win share >
WeightedRandom in the same fixture).

**Blocked by:** 15, 16 (done)

**Status:** claimed

**Phase:** 2

**Train entrypoint:** `catan-qlora-train` / `scripts/rental_sft_gate_b.py` ([ticket 15](15-qlora-training-pipeline.md)) on rental ≥24GB.  
**Gate B CLI:** `catan-gate-b --adapter … --fixture ladder-4p --games 200`

## Progress

- Pipeline smoke (40 steps + 2 games) proved L40S train+arena path; Gate B report
  parser bug fixed (`win_rates` dict shape).
- Full 2-epoch @ ~30s/step ≈ 200h — cost-gated. Production v1 train targets a
  **2000-step** rental run first (`--max-steps 2000 --skip-gate`), then Gate B 200.

- [ ] Fixture identity recorded before eval
- [ ] `parse_rate_model` / `legality_rate_model` ≥ 0.995
- [ ] `win_rate[candidate] > win_rate[weightedrandom]` at ≥200 finished games
- [ ] Eval JSON report published under `outputs/arena/`
