# Phase-1 cohort progress (tickets 11 + 13)

**Date:** 2026-07-31  
**Branch:** `cursor/phase1-cohort-11-13-9ca9`  
**Entrypoint:** `catan-phase1-cohort`

## Ticket 11 — train bulk

| Item | Value |
|---|---|
| Seed range | `train_main` |
| Filtered decisions (raw) | **100,458** (first pass) |
| Train split after 90/5/5 | 89,912 (short of ≥100k) |
| Follow-up | Top-up to **112k** filtered so train split ≥100k (running) |
| Dataset | `data/phase1/processed/expert-v1/` (+ `manifest.json`) |
| Games (first pass) | 392 |

## Ticket 13 — eval holdout

| Item | Value |
|---|---|
| Seed range | `eval_holdout` `[100000, 105000)` |
| Target | 5,000 games |
| Status | **in progress** on agent VM |
| Artifact (when done) | `data/phase1/processed/eval-holdout-v1/` with `immutable: true` |

## Smoke report recap (ticket 09)

See [`hw_smoke_rental_l40s.md`](hw_smoke_rental_l40s.md): L40S **go**, revision pinned, peak train VRAM 14.9GB @ 4096.
