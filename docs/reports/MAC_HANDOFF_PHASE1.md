# Mac handoff — Phase 1 tickets 11 / 13

**Date:** 2026-07-31  
**Cloud agent stopped holdout at a clean cut.**  
**Branch:** `cursor/phase1-cohort-11-13-9ca9` (PR #3)  
**Also needed:** ticket 09 pin on `cursor/rental-9b-smoke-9ca9` (PR #2) — merge #2 first or keep this branch (already based on it).

## What is done (ticket 11 ✅)

| Item | Value |
|---|---|
| Seed range | `train_main` |
| Games | 440 |
| Filtered decisions | 112,754 |
| Train / val / test | **101,303** / 5,525 / 5,926 |
| Hub mirror | [`AlCampbell/catan-llm-phase1`](https://huggingface.co/datasets/AlCampbell/catan-llm-phase1) |

Hub layout:

```
processed/expert-v1/{train,val,test}.jsonl + manifest.json
raw/train_main.jsonl(+.journal)
train_cohort_report.json
```

## What was stopped (ticket 13 ⏸)

| Item | Value |
|---|---|
| Progress on cloud | **~1,725 / 5,000** games (abandoned — do **not** resume this partial) |
| Why abandon | Partial JSONL lived only on the cloud VM; cleaner to regenerate holdout on Mac |
| Your job | Run a **fresh** 5k-game `eval_holdout` burn locally |

## Mac setup

```bash
git clone https://github.com/Al2800/catan-llm.git
cd catan-llm
git checkout cursor/phase1-cohort-11-13-9ca9
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
# optional: pull finished train set
pip install huggingface_hub
huggingface-cli download AlCampbell/catan-llm-phase1 \
  --repo-type dataset --local-dir data/hf-phase1
mkdir -p data/phase1
cp -R data/hf-phase1/processed data/phase1/
cp -R data/hf-phase1/raw data/phase1/ 2>/dev/null || true
```

## Finish ticket 13 on Mac (fresh holdout)

Use as many workers as you like (e.g. 8 on Apple Silicon). This is CPU-only.

```bash
source .venv/bin/activate
# Fresh 5k holdout — do not reuse cloud partial
catan-phase1-cohort holdout --games 5000 --workers 8 --out-dir data/phase1
```

That command:

1. Generates `data/phase1/raw/eval_holdout.jsonl` from seed range `eval_holdout` `[100000, 105000)`
2. Asserts no overlap with `train_main`
3. Builds `data/phase1/processed/eval-holdout-v1/` with `immutable: true`
4. Writes `data/phase1/holdout_cohort_report.json`

Expect ~1–2 hours at `--workers 8` (AlphaBeta in the ladder). Disk ~8–12 GB.

## After holdout completes

1. Mark ticket 13 done; update `docs/reports/phase1_cohort_progress.md`
2. Optionally upload holdout to the same HF dataset:
   ```bash
   hf upload AlCampbell/catan-llm-phase1 \
     data/phase1/processed/eval-holdout-v1 \
     processed/eval-holdout-v1 --repo-type dataset
   ```
3. Next ticket: **12** (dataset builder quality + manifests)

## Do not

- Resume the cloud partial holdout (journal/jsonl not transferred)
- Lower `max_seq_length` below 4096
- Train on holdout (`immutable: true` is the gate)
- Invent seed ranges outside `docs/SEED_REGISTRY.md`

## Prompt for Mac agent

> Continue catan-llm Phase 1 from `docs/reports/MAC_HANDOFF_PHASE1.md` on branch `cursor/phase1-cohort-11-13-9ca9`. Ticket 11 is done (train set on HF `AlCampbell/catan-llm-phase1`). Finish ticket 13: fresh `catan-phase1-cohort holdout --games 5000 --workers 8`, verify immutable manifest, update docs/BACKLOG, commit + push.
