# Agent handbook (catan-llm)

Read this before changing code. Normative docs win over outdated implementations.

## Base branch (read first)

**Work directly on `main`.** Do not open long-lived feature branches for this project
unless the owner explicitly asks. Phase 0 foundations are already merged.

## Normative docs (in order)

1. [`docs/SCOPE.md`](docs/SCOPE.md) — goals, phases, locked decisions
2. [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — schema **v2**, prompts, splits
3. [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) — fixtures & promotion gates
4. [`docs/SEED_REGISTRY.md`](docs/SEED_REGISTRY.md) — only allowed seed ranges
5. [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) — local GPU stack
6. [`docs/PHASE0_5_TASKS.md`](docs/PHASE0_5_TASKS.md) — current assignable work
7. [`docs/tickets/BACKLOG.md`](docs/tickets/BACKLOG.md) — **ticket index** (outstanding work)
8. [`docs/RL_SPEC.md`](docs/RL_SPEC.md) — fill before GRPO
9. [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) — local ticket conventions

## Agent skills (project-level)

Installed under [`.cursor/skills/engineering/`](.cursor/skills/engineering/) from
[mattpocock/skills](https://github.com/mattpocock/skills) (MIT). Available in this
repo and to Cloud Agents that clone it.

Useful entry points (type `/` in Agent chat):

- `/ask-matt` — router over the engineering skills
- `/setup-matt-pocock-skills` — run once to configure issue tracker / triage / domain docs
- `/implement`, `/tdd`, `/code-review`, `/to-tickets`, `/to-spec`, `/wayfinder`

See [`.cursor/skills/engineering/README.md`](.cursor/skills/engineering/README.md).

## Do / don't

**Do**

- Implement Phase 0.5 task cards (**T1–T9**) before any ≥100k dataset job; respect the dependency graph
- Keep train prompts byte-identical to live `render_*` functions
- Bump `PROMPT_VERSION` whenever renderer text changes
- Use `schema_version: "v2"` for new trajectories
- Allocate seeds only from `SEED_REGISTRY.md`; stop at cohort targets (SCOPE §5.2)
- Keep fallback policy = `first_legal`
- Use `max_seq_length >= 4096` for canonical-prompt SFT
- Keep Tier A rationales POV-safe (experts may use full `Game`; text must not leak opponent hands)
- Add/adjust tests in the same PR as behavior changes
- Update docs in the same PR when changing contracts

**Don't**

- Invent compact alternate training prompts
- Split datasets by UUID `game_id`
- Silently fall back MINI → BASE
- Treat smoke parse/legality as skill evidence
- Treat candidate WR > 50% in a 4p ladder as “beats WeightedRandom”
- Start Phase-3 GRPO without a filled `RL_SPEC.md`
- Download 8B models in default CI
- Set both `assistant_only_loss` and `completion_only_loss` on chat SFT — use **assistant_only_loss only**
- Lower `max_seq_length` below the no-truncation budget to fix OOM
- Claim “beat AlphaBeta” without the pinned `ab-4p` fixture in `EVAL_PROTOCOL.md`
- Assume `/setup-matt-pocock-skills` has been run (skills are optional until configured)

## Commands

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
catan-arena --fixture bot-ladder --games 4 --vps 6 --no-alphabeta --out outputs/arena/ci.json
catan-arena --fixture ladder-4p --games 4 --vps 6 --out outputs/arena/ladder.json
catan-generate --seed-range-name hw_smoke --games 2 --vps 6 --overwrite --out /tmp/t.jsonl
```

Training extras: `pip install -e ".[train]"` (+ bitsandbytes / Blackwell torch on the owner GPU).

## Branch / commits

- **Work on `main` only** (owner policy). Do not open long-lived feature branches.
- Prefer small commits that cite the ticket id (`tickets/01`, `T1`, …) or SCOPE section.
- Keep docs and code in the same commit when contracts change.

## Current code vs docs

Phase 0.5 software cards **01–08** landed on `main`; ticket **10** (Tier A feature
rationales) also landed. Remaining Phase 0.5: **09** (owner-GPU Qwen3.5-9B QLoRA smoke
+ pin `model.revision` in `configs/qwen3.5-9b-qlora.yaml`). Assistant-mask one-batch
skips until that pin exists. Do not start the ≥100k data burn until **09** exits.

## License

Catanatron is GPL-3.0. Keep it as an external dependency; do not relicense or vend blindly into a combined MIT distribution without an explicit owner decision.
