# 12 — Dataset builder quality + manifests

**What to build:** Build chat-format datasets from trajectories with dedup,
unfinished/illegal drops, no-truncation checks, balancing/rare-action
oversample, game_key splits, and complete manifests.

**Blocked by:** 05, 10, 11

**Status:** blocked

**Phase:** 1

- [ ] Filters and no-truncation assert at max_seq_length≥4096
- [ ] Rare-action / phase balance reported
- [ ] train/val/test emitted with checksums
- [ ] Manifest matches DATA_CONTRACT §9
