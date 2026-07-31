# Phase-1 cohort progress (tickets 11 + 13)

**Date:** 2026-07-31  
**Branch:** `cursor/phase1-cohort-11-13-9ca9`  
**Entrypoint:** `catan-phase1-cohort`

## Ticket 11 — train bulk ✅

| Item | Value |
|---|---|
| Seed range | `train_main` `[0, 50000)` |
| Bot mix | alphabeta, valuefunction, weightedrandom, random (rotated) |
| Filtered decisions (raw) | **112,754** |
| Train / val / test | **101,303** / 5,525 / 5,926 |
| Target | ≥100k train-split decisions — **met** |
| Dataset | `data/phase1/processed/expert-v1/` |
| Manifest | `data/phase1/processed/expert-v1/manifest.json` |
| Cohort report | `data/phase1/train_cohort_report.json` |
| Games | 440 |
| train.jsonl sha256 | `64bdb154ba089b55b761c1c9b07f2a9e015911d1bb105f4b94fc2a1e1573a7f7` |
| val.jsonl sha256 | `b5480469178ffd967097d580e2f29d067b3740858cda643e52a762a169112290` |
| test.jsonl sha256 | `b8018068ffe91b6a1daa433ae2862335ddfc3cd00a79111ba7e192a1a19683c0` |

Default generator target is **112k** filtered so the 90% train split clears 100k.

## Ticket 13 — eval holdout (in progress)

| Item | Value |
|---|---|
| Seed range | `eval_holdout` `[100000, 105000)` |
| Target | 5,000 games |
| Status | generating on agent VM |
| Artifact (when done) | `data/phase1/processed/eval-holdout-v1/` (`immutable: true`) |

## Ticket 09 smoke recap

See [`hw_smoke_rental_l40s.md`](hw_smoke_rental_l40s.md): L40S **go**, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, peak train VRAM **14.9 GB** @ 4096.
