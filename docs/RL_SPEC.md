# RL specification (Phase-3 entry gate)

Skeleton to be **filled before any GRPO run**.  
SCOPE Stage-2 entry requires this document completed and reviewed.

Status: **TEMPLATE — not filled**.

## 1. Objective

Improve a Stage-1 SFT checkpoint on the pre-registered AlphaBeta fixture
([`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) §2.2) without collapsing legality or inducing degenerate stalling.

## 2. Policy / reference

| Item | Value |
|---|---|
| Init policy | path/revision of Stage-1 SFT checkpoint |
| Reference policy | frozen SFT copy for KL |
| Base model | Qwen3.5-9B (`Qwen/Qwen3.5-9B`) + QLoRA adapters |
| Prompt renderer | canonical only (`prompt_version=…`) |

## 3. Rollout settings

| Item | Proposed default | Final |
|---|---|---|
| Group size G | 8 | |
| Temperature | 0.9 | |
| Max new tokens | 128 | |
| Decisions per update | TBD | |
| On-policy fraction | TBD (rest from bot-state buffer) | |
| vLLM colocate vs separate | TBD | |
| Hardware | local 5060 Ti pilot / A100 burst | |

## 4. Reward terms (engine-verifiable)

Fill weights; sum/normalize scheme must be explicit.

| Term | Weight | Definition | Notes |
|---|---:|---|---|
| `r_format` | | 1 if JSON parses else 0 | gate |
| `r_legal` | | 1 if action legal else 0 | gate |
| `r_win` | | +1 win / 0 draw-unfinished / −1 loss | group-shared |
| `r_vp_margin` | | clipped VP margin | |
| `r_shape_vp` | | immediate VP gain | small |
| `r_discard_risk` | | penalty if hand >7 pre-roll | small |
| `r_stall` | | penalty for turn-length outliers / pass-heavy | anti-hack |
| `r_kl` | | KL to reference (via trainer) | stability |

Illegal / unparseable actions: all downstream rewards zeroed.

## 5. Anti-hacking tests

Before scale:

- [ ] Synthetic stall policy scores worse than SFT baseline under the reward
- [ ] Always-`END_TURN`-when-legal policy does not win on reward
- [ ] Legality floor monitored every N steps; abort if `< 0.99`
- [ ] Turn-length distribution vs SFT baseline within agreed band

## 6. Cost & abort

| Item | Limit |
|---|---|
| Max rented GPU-hours this iteration | |
| Max wall-clock | |
| Abort if legality < | 0.99 |
| Abort if no holdout improvement after | N evals |

## 7. Promotion

Must satisfy [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) Gate C on `champion_ab` + `champion_ab_rerun` seeds.

## 8. Sign-off

| Role | Name | Date |
|---|---|---|
| Author | | |
| Reviewer | | |
