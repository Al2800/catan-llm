# catan-llm

Training an 8–12B parameter open-weights LLM to play Settlers of Catan inside the
[Catanatron](https://github.com/bcollazo/catanatron) simulator.

Pipeline: bulk self-play in Catanatron → structured + natural-language decision
datasets → SFT (QLoRA) on **Qwen3-8B** → GRPO/RLVR with engine-verifiable rewards →
live play back inside the simulator against Catanatron's bot ladder.

## Docs

| Doc | Purpose |
|---|---|
| [`docs/SCOPE.md`](docs/SCOPE.md) | Full scope, risks, phased roadmap |
| [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | Trajectory / prompt / split contract |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Fixtures, metrics, promotion gates |
| [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) | 5060 Ti / PyTorch / vLLM validation plan |
| [`docs/PHASE0.md`](docs/PHASE0.md) | Phase 0 status + known gaps |
| [`configs/qwen3-8b-qlora.yaml`](configs/qwen3-8b-qlora.yaml) | Phase-2 QLoRA sketch |

## Status

**Phase 0 plumbing complete. Phase 0.5 (contract repair + local 8B proof) is next.**

Locked decisions:

- Base model: **Qwen3-8B** (Apache-2.0), QLoRA-first
- Primary hardware: RTX 5060 Ti 16GB (local), burst rentals for RL scale
- Action space: Catanatron-native only (negotiation trading later)
- Train/play prompts must be identical (hard gate)
- No ≥100k dataset until local 8B QLoRA smoke succeeds
- Experiment tracking: local JSON reports first; W&B optional later

## Layout

```
src/catan_llm/
  sim/        # Catanatron adapter + trajectory recording
  data/       # schema, renderer, parser, dataset builder
  eval/       # seeded arena + metrics
  play/       # LLMPlayer
  training/   # SFT smoke (TRL); QLoRA lands in Phase 2
  serving/    # vLLM client path (stub)
  scripts/    # CLIs
configs/      # training sketches
docs/         # scope + contracts
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
# Bot-ladder arena (CPU)
catan-arena --games 12 --seed 0 --out outputs/arena/bot_ladder.json

# Generate expert trajectories
catan-generate --games 20 --seed 0 --out data/raw/trajectories.jsonl

# Build chat-format dataset + manifest
catan-build-dataset --trajectories data/raw/trajectories.jsonl --out data/processed/expert-smoke

# End-to-end smoke: generate → tiny SFT → play vs Random
# Uses HuggingFaceTB/SmolLM2-135M-Instruct by default (not Qwen3-8B).
catan-sft-smoke --games 8 --max-steps 20 --work-dir outputs/sft_smoke
```

## Roadmap snapshot

1. **Phase 0** — plumbing spike *(done)*
2. **Phase 0.5** — renderer parity, MINI maps, stable splits, 5060 Ti 8B smoke *(next)*
3. **Phase 1** — ≥100k dataset + holdout + quality report
4. **Phase 2** — Qwen3-8B QLoRA SFT vs bot ladder
5. **Phase 3** — GRPO toward pinned AlphaBeta fixture
6. **Phase 4** — scale / polish / write-up
