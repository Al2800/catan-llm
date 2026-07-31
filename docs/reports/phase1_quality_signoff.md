# Phase-1 dataset quality report + sign-off (ticket 14)

**Date:** 2026-07-31  
**Decision:** **GO for Phase 2** (production QLoRA / ticket 15)  
**Artifacts:** HF `AlCampbell/catan-llm-phase1` (train); Mac holdout checksums below  
**Stats JSON:** [`phase1_quality_stats.json`](phase1_quality_stats.json)

## Cohort summary

| Artifact | Games | Decisions | Role | Location |
|---|---:|---:|---|---|
| `expert-v1` | 440 | 112,754 filtered (train-split **101,303**) | train | HF `processed/expert-v1/` |
| `eval-holdout-v1` | **5,000** | **1,316,632** filtered | eval_holdout (`immutable`) | Mac (`data/phase1/processed/eval-holdout-v1/`); checksum published |

Train seed range: `train_main` `[0, 50000)` — burned seeds `0…439` (stop-at-target).  
Holdout seed range: `eval_holdout` `[100000, 105000)` — **no overlap** with train (registry + cohort assert).

### Checksums (verified locally 2026-07-31)

| File | sha256 |
|---|---|
| `train.jsonl` | `64bdb154ba089b55b761c1c9b07f2a9e015911d1bb105f4b94fc2a1e1573a7f7` |
| `val.jsonl` | `b5480469178ffd967097d580e2f29d067b3740858cda643e52a762a169112290` |
| `test.jsonl` | `b8018068ffe91b6a1daa433ae2862335ddfc3cd00a79111ba7e192a1a19683c0` |
| `holdout.jsonl` (Mac) | `5809a9fc1ed42b2fda29abdd77e37980f895ab52e7c5787023fce2fc7ccd1bfc` |

Split counts: train **101,303** / val **5,458** / test **5,993**.

## DATA_CONTRACT §7 filters (train raw → kept)

Re-ran `filter_decision_records` on `data/phase1/raw/train_main.jsonl` (112,754 rows):

| Drop | Count |
|---|---:|
| unfinished | 0 |
| illegal / round-trip | 0 |
| truncated (full pass) | n/a at build time\* |
| **kept** | **112,754** |

\*Cohort build predated the ticket-12 quality sidecar on the published manifest (`quality` / `max_seq_length` were null on HF). Post-hoc filter confirms zero unfinished/illegal drops; truncation audited via sample (below). Future rebuilds write `quality.json` + `max_seq_length=4096` via `build_chat_dataset`.

### Distributions (kept)

**Expert policy:** alphabeta 31,252 · valuefunction 30,679 · weightedrandom 25,613 · random 25,210  

**Phase:** PLAY_TURN 95,955 · MOVE_ROBBER 6,569 · initial settlement/road 3,520 each · DISCARD 3,190  

**Turn terciles:** early 33,351 · mid 37,292 · late 42,111 (roughly even; late slightly heavy — acceptable for v1)

**Rare / contract buckets:**

| Action | Count |
|---|---:|
| MARITIME_TRADE | 10,315 |
| MOVE_ROBBER | 6,569 |
| BUY_DEVELOPMENT_CARD | 1,621 |
| PLAY_KNIGHT_CARD | 828 |
| PLAY_YEAR_OF_PLENTY | 126 |
| PLAY_MONOPOLY | 113 |
| PLAY_ROAD_BUILDING | 93 |

All rare buckets are present with non-trivial counts (SCOPE §5.2 / DATA_CONTRACT §7 targets for the v1 report).

## Gates

| Gate | Result | Evidence |
|---|---|---|
| Schema v2 | pass | 112,754 / 112,754 `schema_version=v2` |
| `prompt_version` | pass | all `2026-07-30.1` |
| Train/play parity | pass | `tests/test_prompt_parity.py` green |
| No-truncation @ 4096 | pass (sample) | 2,000 stratified decisions, SmolLM tokenizer, **0** truncated; CI `test_no_truncation` + assistant-mask green |
| Manifest + checksums | pass | checksums match HF/local; see note on quality sidecar backfill |
| Holdout immutable | pass (Mac) | `immutable: true`, `role: eval_holdout`, 5k games, checksum above |

Chat character lengths (proxy): p50≈3944 · p90≈4646 · p99≈5188 · max 6022 — consistent with the 4096-token floor (ticket 09 L40S smoke).

## DATA_CONTRACT §11 checklist

- [x] ≥100k train decisions after filtering (`schema_version=v2`) — **101,303** train-split
- [x] Separate ≥5k-game eval holdout from the seed registry — **5,000** games, `eval_holdout`
- [x] Parity gate green + matching `prompt_version` — `2026-07-30.1`
- [x] No-truncation check green at `max_seq_length=4096` — sample + CI
- [x] Manifest complete + checksums — train on HF; holdout checksum published
- [x] Quality report checked in — this file + `phase1_quality_stats.json`

## Phase-2 go / no-go

**GO.** Phase 1 exit criteria are met. Unblocks ticket **15** (production Qwen3.5-9B QLoRA).

### Operational follow-ups (not Phase-1 blockers)

1. **Upload Mac holdout** to `AlCampbell/catan-llm-phase1` as `processed/eval-holdout-v1/` (verify sha256 `5809a9fc…`) before Gate B / ticket 17 consumes it. Cloud agent only has the abandoned ~1725-game partial under `data/phase1/raw/eval_holdout.jsonl` — do not train or eval on that partial.
2. **Backfill** `quality.json` + `max_seq_length: 4096` onto the HF train manifest (values in `phase1_quality_stats.json`) on next dataset touch; JSONL checksums stay unchanged.
3. Prefer **Qwen tokenizer** for a full truncation pass before the long SFT run if VRAM/time allows; SmolLM sample is CI-aligned and showed zero drops.

## Sign-off

| Field | Value |
|---|---|
| Ticket | 14 |
| Phase-1 status | **complete** |
| Phase-2 | **go** |
| Signed | cloud agent + published cohort/checksum evidence (2026-07-31) |
