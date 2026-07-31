# Agent handbook (catan-llm)

Read this before changing code. **Normative docs win over outdated implementations.**

## Base branch

**Prefer `main`.** Do not open long-lived feature branches unless the owner asks.
Cloud Agents may use short-lived `cursor/*` PR branches; merge back promptly.

## Where we are

**Phase 2 (SFT / Gate B)** — see [`docs/STATUS.md`](docs/STATUS.md).  
Phase 0 / 0.5 / 1 are **done**. Do not re-litigate “block ≥100k until 0.5”.

## Normative docs (in order)

1. [`docs/STATUS.md`](docs/STATUS.md) — current phase / done / next  
2. [`docs/SCOPE.md`](docs/SCOPE.md) — goals, phases, locked decisions  
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system + module map  
4. [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — schema **v2**, prompts, splits  
5. [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) — fixtures & promotion gates  
6. [`docs/SEED_REGISTRY.md`](docs/SEED_REGISTRY.md) — only allowed seed ranges  
7. [`docs/TRAINING.md`](docs/TRAINING.md) — QLoRA / rental ops  
8. [`docs/SERVING.md`](docs/SERVING.md) / [`docs/SPECTATE.md`](docs/SPECTATE.md) — deploy & watch  
9. [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) — local GPU stack  
10. [`docs/tickets/BACKLOG.md`](docs/tickets/BACKLOG.md) — **ticket index**  
11. [`docs/RL_SPEC.md`](docs/RL_SPEC.md) — **must fill before GRPO**  
12. [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) — local ticket conventions  

Historical Phase 0.5 cards: [`docs/PHASE0_5_TASKS.md`](docs/PHASE0_5_TASKS.md) (complete).

## Agent skills (project-level)

Installed under [`.cursor/skills/engineering/`](.cursor/skills/engineering/) from
[mattpocock/skills](https://github.com/mattpocock/skills) (MIT).

Useful entry points (type `/` in Agent chat): `/ask-matt`, `/setup-matt-pocock-skills`,
`/implement`, `/tdd`, `/code-review`, `/to-tickets`, `/to-spec`, `/wayfinder`.

## Do / don't

**Do**

- Pick work from [`docs/tickets/BACKLOG.md`](docs/tickets/BACKLOG.md) frontier
- Keep train prompts byte-identical to live `render_*` functions
- Bump `PROMPT_VERSION` whenever renderer text changes
- Use `schema_version: "v2"` for new trajectories
- Allocate seeds only from `SEED_REGISTRY.md`; stop at cohort targets (SCOPE §5.2)
- Keep fallback policy = `first_legal`
- Use `max_seq_length >= 4096` for canonical-prompt SFT
- Keep Tier A rationales POV-safe (experts may use full `Game`; text must not leak hands)
- Add/adjust tests in the same PR as behavior changes
- Update docs in the same PR when changing contracts or phase status

**Don't**

- Invent compact alternate training prompts
- Split datasets by UUID `game_id`
- Silently fall back MINI → BASE
- Treat smoke parse/legality as skill evidence
- Treat candidate WR > 50% in a 4p ladder as “beats WeightedRandom”
- Start Phase-3 GRPO without a filled `RL_SPEC.md`
- Download 9B models in default CI
- Set both `assistant_only_loss` and `completion_only_loss` on chat SFT
- Lower `max_seq_length` below 4096 to fix OOM
- Claim “beat AlphaBeta” without pinned `ab-4p`
- Train on `eval_holdout` / immutable holdout artifacts

## Commands

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
catan-arena --fixture bot-ladder --games 4 --vps 6 --no-alphabeta --out outputs/arena/ci.json
catan-arena --fixture ladder-4p --games 4 --vps 6 --out outputs/arena/ladder.json
catan-qlora-train --dry-run
catan-spectate --bots-only --watch --vps 6 --seed 7
```

Training extras: `pip install -e ".[train]"`. Real 9B QLoRA needs rental CUDA — [`docs/TRAINING.md`](docs/TRAINING.md).

## Model / hardware pin

- Model: `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Config: `configs/qwen3.5-9b-qlora.yaml`
- Local 16GB train @ 4096: **no-go** (`docs/reports/hw_smoke_5060ti.md`)
- Rental L40S: **go** (`docs/reports/hw_smoke_rental_l40s.md`)

## License

Catanatron is GPL-3.0. Keep it as an external dependency; do not relicense or vend
blindly into a combined MIT distribution without an explicit owner decision.
