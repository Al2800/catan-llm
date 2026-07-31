# 12 — Dataset builder quality + manifests

**What to build:** Build chat-format datasets from trajectories with dedup,
unfinished/illegal drops, no-truncation checks, balancing/rare-action
oversample, game_key splits, and complete manifests.

**Blocked by:** 05, 10, 11 (done)

<<<<<<< HEAD
**Status:** done (2026-07-31)
=======
**Status:** ready-for-agent
>>>>>>> f5e0a10 (Close ticket 13: fresh 5k-game immutable eval holdout on Mac.)

**Phase:** 1

## Entrypoints

- `catan_llm.data.quality.filter_decision_records`
- `build_chat_dataset` → `manifest.json` + `quality.json` (DATA_CONTRACT §9)

## Acceptance criteria

- [x] Filters and no-truncation assert at max_seq_length≥4096 (enforced when tokenizer available; else recorded as skipped)
- [x] Rare-action / phase / turn-tercile balance reported in `quality`
- [x] train/val/test emitted with checksums
- [x] Manifest matches DATA_CONTRACT §9 (`max_seq_length`, `quality`, checksums, commits, prompt_version)
