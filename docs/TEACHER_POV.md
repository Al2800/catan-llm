# Teacher observability audit (SCOPE §5.1)

Phase 0.5 lock for privileged distillation. CI enforces the learner/Tier-A side
via `tests/test_teacher_pov.py`.

## Policy

| Role | Observability |
|---|---|
| **Teacher** (AlphaBeta, ValueFunction, …) | May use the full engine `Game` (hidden hands, search features) to **choose actions**. This is intentional privileged distillation. |
| **Learner prompts** (`LLMPlayer` / SFT system+user) | POV-limited via the canonical renderer: own hand + own devs; opponents expose **card counts only**. |
| **Tier A rationale text** | Learner-observable features only. Never opponent private resource/dev compositions. |

## What enters labels vs rationales

| Signal | In action label? | In learner prompt? | In Tier A text? |
|---|---|---|---|
| Expert's chosen legal action index | yes | yes (as available actions) | yes (action type / public features) |
| Opponent exact hand composition | no | **no** | **no** |
| Opponent card *count* | no | yes | yes (e.g. “ORANGE (7 cards)”) |
| Board layout, buildings, roads, robber | n/a | yes | yes |
| Teacher internal valueΔ (if POV-safe) | no | no | optional later (Phase 1) |

## Enforcement

- Trajectory `state` for non-ego colors must not include a resource `hand` map.
- User prompts must not contain opponent `hand=[…]` literals.
- `assert_tier_a_pov_safe(text)` rejects opponent private-hand leakage patterns.
- Assistant-mask / no-truncation: `tests/test_assistant_mask.py`, `tests/test_no_truncation.py`.
