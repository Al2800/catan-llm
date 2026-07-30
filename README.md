# catan-llm

Training an 8–12B parameter open-weights LLM to play Settlers of Catan inside the
[Catanatron](https://github.com/bcollazo/catanatron) simulator.

The pipeline: bulk self-play in Catanatron → structured + natural-language decision
datasets → SFT (QLoRA) on **Qwen3-8B** → GRPO/RLVR with engine-verifiable rewards →
live play back inside the simulator against Catanatron's bot ladder.

## Status

**Pre-Phase 0 — scope locked, build not started.** The full scope (architecture, data
strategy, training plan, evaluation, infra, risks, roadmap, locked decisions) lives in
[`docs/SCOPE.md`](docs/SCOPE.md).

Locked decisions (2026-07-30):

- Base model: **Qwen3-8B** (Apache-2.0), QLoRA-first
- Primary hardware: RTX 5060 Ti 16GB (local), burst rentals for RL scale
- Action space: Catanatron-native only (negotiation trading is a later extension)
- Endgame: semi-public — write-up on the owner's blog

## Planned layout (Phase 0)

- `sim/` — Catanatron adapter: bulk games, trajectory recording
- `data/` — dataset builders, state renderers, manifests
- `training/` — SFT and GRPO configs/scripts
- `eval/` — seeded arena vs the bot ladder, Elo tracking
- `serving/`, `play/` — vLLM serving + `LLMPlayer` for live games
