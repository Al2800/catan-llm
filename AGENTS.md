# Agent handbook (catan-llm)

Read this before changing code. Normative docs win over outdated implementations.

## Normative docs (in order)

1. [`docs/SCOPE.md`](docs/SCOPE.md) — goals, phases, locked decisions
2. [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) — schema **v2**, prompts, splits
3. [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) — fixtures & promotion gates
4. [`docs/SEED_REGISTRY.md`](docs/SEED_REGISTRY.md) — only allowed seed ranges
5. [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) — local GPU stack
6. [`docs/PHASE0_5_TASKS.md`](docs/PHASE0_5_TASKS.md) — current assignable work
7. [`docs/RL_SPEC.md`](docs/RL_SPEC.md) — fill before GRPO

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

- Implement Phase 0.5 task cards (T1–T8) before any ≥100k dataset job
- Keep train prompts byte-identical to live `render_*` functions
- Bump `PROMPT_VERSION` whenever renderer text changes
- Use `schema_version: "v2"` for new trajectories
- Allocate seeds only from `SEED_REGISTRY.md`
- Keep fallback policy = `first_legal`
- Use `max_seq_length >= 4096` for canonical-prompt SFT
- Add/adjust tests in the same PR as behavior changes
- Update docs in the same PR when changing contracts

**Don't**

- Invent compact alternate training prompts
- Split datasets by UUID `game_id`
- Silently fall back MINI → BASE
- Treat smoke parse/legality as skill evidence
- Start Phase-3 GRPO without a filled `RL_SPEC.md`
- Download 8B models in default CI
- Set both `assistant_only_loss` and `completion_only_loss` on chat SFT without verifying TRL behavior — use **assistant_only_loss only**
- Claim “beat AlphaBeta” without the pinned fixture in `EVAL_PROTOCOL.md`

## Commands

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
catan-arena --games 4 --vps 6 --no-alphabeta --out outputs/arena/ci.json
```

Training extras: `pip install -e ".[train]"` (+ bitsandbytes / Blackwell torch on the owner GPU).

## Branch / PR

- Feature branches: `cursor/<descriptive-name>-9ca9` (repo convention for cloud agents)
- Base branch: `main`
- One task card ≈ one PR when possible
- PR description must cite the task id (`T1`…`T8`) or SCOPE section

## Current code vs docs

As of the handoff-review update, **docs describe v2 / Phase 0.5**; much of `src/` is still Phase-0 plumbing (v1-ish records, compact dataset prompts, broken MINI helper). Prefer implementing task cards over “fixing forward” on scale features.

## License

Catanatron is GPL-3.0. Keep it as an external dependency; do not relicense or vend blindly into a combined MIT distribution without an explicit owner decision.
