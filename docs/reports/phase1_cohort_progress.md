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
| Train / val / test | **101,303** / 5,458 / 5,993 |
| Target | ≥100k train-split decisions — **met** |
| Dataset | `data/phase1/processed/expert-v1/` |
| Manifest | `data/phase1/processed/expert-v1/manifest.json` |
| Cohort report | `data/phase1/train_cohort_report.json` |
| Games | 440 |
| train.jsonl sha256 | `64bdb154ba089b55b761c1c9b07f2a9e015911d1bb105f4b94fc2a1e1573a7f7` |
| val.jsonl sha256 | `b5480469178ffd967097d580e2f29d067b3740858cda643e52a762a169112290` |
| test.jsonl sha256 | `b8018068ffe91b6a1daa433ae2862335ddfc3cd00a79111ba7e192a1a19683c0` |

Default generator target is **112k** filtered so the 90% train split clears 100k.

## Ticket 13 — eval holdout ✅

| Item | Value |
|---|---|
| Seed range | `eval_holdout` `[100000, 105000)` |
| Games | **5,000** (fresh Mac burn; cloud partial abandoned) |
| Filtered decisions | **1,316,632** |
| Workers | 8 |
| Runtime | ~35 min (2026-07-31T10:21Z → 10:56Z) |
| Artifact | `data/phase1/processed/eval-holdout-v1/` |
| Manifest | `immutable: true`, `role: eval_holdout` |
| Cohort report | `data/phase1/holdout_cohort_report.json` |
| Seed overlap with `train_main` | **0** |
| holdout.jsonl sha256 | `5809a9fc1ed42b2fda29abdd77e37980f895ab52e7c5787023fce2fc7ccd1bfc` |

## Ticket 09 smoke recap

See [`hw_smoke_rental_l40s.md`](hw_smoke_rental_l40s.md): L40S **go**, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, peak train VRAM **14.9 GB** @ 4096.
