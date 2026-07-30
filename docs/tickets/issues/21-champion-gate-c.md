# 21 — RL run + Gate C AlphaBeta champion

**What to build:** Run RL iteration(s) and evaluate on pre-registered `ab-4p`
(≥1000 finished games). Promote champion only if Gate C passes, including the
independent rerun on `champion_ab_rerun`.

**Blocked by:** 20

**Status:** blocked

**Phase:** 3

- [ ] Fixture pre-registered (map hash, seeds, AlphaBeta depth=2)
- [ ] `win_rate[candidate] ≥ 0.55` with Wilson LB > 0.50
- [ ] Rerun agrees on `champion_ab_rerun`
- [ ] No severe reward-hacking flags in the RL report
