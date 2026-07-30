# catan-llm

Training an 8–12B parameter open-weights LLM to play Settlers of Catan inside the
[Catanatron](https://github.com/bcollazo/catanatron) simulator.

Pipeline: bulk self-play in Catanatron → structured + natural-language decision
datasets → SFT (QLoRA) on **Qwen3-8B** → GRPO/RLVR with engine-verifiable rewards →
live play back inside the simulator against Catanatron's bot ladder.

Full scope: [`docs/SCOPE.md`](docs/SCOPE.md).

## Status

**Phase 0 in progress** — foundations & spike.

Locked decisions (2026-07-30):

- Base model: **Qwen3-8B** (Apache-2.0), QLoRA-first
- Primary hardware: RTX 5060 Ti 16GB (local), burst rentals for RL scale
- Action space: Catanatron-native only (negotiation trading is a later extension)
- Endgame: semi-public — write-up on the owner's blog
- Experiment tracking: **local JSON reports by default**; optional W&B later (`report_to`)

## Layout

```
src/catan_llm/
  sim/        # Catanatron adapter + trajectory recording
  data/       # schema, renderer, parser, dataset builder
  eval/       # seeded arena + metrics
  play/       # LLMPlayer
  training/   # SFT smoke (TRL)
  serving/    # vLLM client path (stub)
  scripts/    # CLIs
tests/
docs/SCOPE.md
```

## Setup

```bash
# Python 3.11+
pip install -e ".[dev]"          # core + tests
pip install -e ".[train]"        # torch / transformers / TRL (for SFT smoke)
```

Catanatron is pinned to git commit `82aae93` (v3.3.0; not yet on PyPI).

## Quickstart (Phase 0)

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

## Phase 0 exit criteria

- [x] Repo scaffold + pinned deps
- [x] Simulator adapter + trajectory schema v1
- [x] Canonical renderer + action parser
- [x] Eval arena v0 (bot-vs-bot)
- [x] CI (ruff + pytest + arena smoke)
- [ ] Tiny SFT smoke on CPU/GPU proving legality-capable play loop
