# 20 — GRPO training loop implementation

**What to build:** Implement the GRPO loop (rollouts via vLLM, engine-verifiable
rewards, KL to SFT reference) according to the signed RL_SPEC.

**Blocked by:** 19

**Status:** blocked

**Phase:** 3

- [ ] Decision-level rollouts with group size from RL_SPEC
- [ ] Reward stack matches signed weights; illegals zero downstream reward
- [ ] Legality floor monitored with abort
- [ ] Runnable on rental GPU path (local pilot optional)
