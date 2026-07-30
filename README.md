# catan-llm

Training an 8–12B parameter open-weights LLM to play Settlers of Catan inside the
[Catanatron](https://github.com/bcollazo/catanatron) simulator.

Pipeline: bulk self-play in Catanatron → structured + natural-language decision
datasets → SFT (QLoRA) on **Qwen3-8B** → GRPO/RLVR with engine-verifiable rewards →
live play back inside the simulator against Catanatron's bot ladder.

## Agent skills

Project-level engineering skills live in [`.cursor/skills/engineering/`](.cursor/skills/engineering/)
(from [mattpocock/skills](https://github.com/mattpocock/skills), MIT). Invoke with `/` in Agent chat
(e.g. `/ask-matt`, `/setup-matt-pocock-skills`, `/implement`). Run `/setup-matt-pocock-skills` once
to wire issue tracker / triage defaults for this repo.

## Docs (read before building)

| Doc | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Handoff rules for coding agents |
| [`docs/SCOPE.md`](docs/SCOPE.md) | Full scope, risks, phased roadmap |
| [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | Trajectory **schema v2**, prompt parity, truncation |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Fixtures, metrics, promotion gates |
| [`docs/SEED_REGISTRY.md`](docs/SEED_REGISTRY.md) | Disjoint seed ranges |
| [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) | 5060 Ti / PyTorch / vLLM validation plan |
| [`docs/PHASE0_5_TASKS.md`](docs/PHASE0_5_TASKS.md) | Assignable Phase 0.5 work (T1–T8) |
| [`docs/RL_SPEC.md`](docs/RL_SPEC.md) | Phase-3 RL entry template |
| [`docs/PHASE0.md`](docs/PHASE0.md) | Phase 0 status + known gaps |
| [`configs/qwen3-8b-qlora.yaml`](configs/qwen3-8b-qlora.yaml) | Phase-2 QLoRA sketch (`max_seq_length: 4096`) |

## Status

**Phase 0 plumbing complete. Phase 0.5 is the handoff surface.**

Locked decisions (see SCOPE §12):

- Base model: **Qwen3-8B** (Apache-2.0), QLoRA-first
- Primary hardware: RTX 5060 Ti 16GB (local), burst rentals for RL scale
- Train/play prompts must be identical; schema **v2** + `prompt_version`
- Fallback: **`first_legal`**
- SFT context: **`max_seq_length ≥ 4096`** (canonical prompts are ~2.3–2.5k before the label)
- Seeds only from `docs/SEED_REGISTRY.md`
- No ≥100k dataset until local 8B QLoRA smoke succeeds
- Docs win over code until both are updated together

## Layout

```
AGENTS.md
configs/qwen3-8b-qlora.yaml
docs/         # scope + contracts + task cards
src/catan_llm/
  sim/ data/ eval/ play/ training/ serving/ scripts/
tests/
```

## Setup

```bash
# Python 3.11+
pip install -e ".[dev]"          # core + tests
pip install -e ".[train]"        # torch / transformers / TRL (smoke)
# Local 8B QLoRA also needs bitsandbytes + Blackwell-capable torch/vLLM
# — see docs/ENV_BLACKWELL.md
```

Catanatron is pinned to git commit `82aae93` (v3.3.0; not yet on PyPI). It is GPL-3.0;
treat it as an external engine dependency when distributing.

## Quickstart (Phase 0 plumbing)

```bash
catan-arena --games 12 --seed 0 --out outputs/arena/bot_ladder.json
catan-generate --games 20 --seed 0 --out data/raw/trajectories.jsonl
catan-build-dataset --trajectories data/raw/trajectories.jsonl --out data/processed/expert-smoke
catan-sft-smoke --games 8 --max-steps 20 --work-dir outputs/sft_smoke
```

## Roadmap snapshot

1. **Phase 0** — plumbing spike *(done)*
2. **Phase 0.5** — schema v2, parity, MINI, seeds, CI gates, 5060 Ti 8B smoke *(next — see task cards)*
3. **Phase 1** — ≥100k dataset + holdout + quality report
4. **Phase 2** — Qwen3-8B QLoRA SFT vs bot ladder
5. **Phase 3** — GRPO toward pinned AlphaBeta fixture (after `RL_SPEC.md` filled)
6. **Phase 4** — scale / polish / write-up
