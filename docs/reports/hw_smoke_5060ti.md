# Hardware smoke report — RTX 5060 Ti 16GB

**Status:** local **no-go** for Qwen3.5-9B QLoRA train at label-safe context  
**Date:** 2026-07-31  
**Ticket:** [09](../tickets/issues/09-local-8b-qlora-smoke.md) / Phase 0.5 T8  
**Device:** NVIDIA RTX 5060 Ti 16GB (Blackwell)

## Finding

Owner report: **16GB is not enough** to hold `Qwen/Qwen3.5-9B` plus the other
elements required for QLoRA training (4-bit base + LoRA adapters + optimizer
state + activations at `max_seq_length≥4096`).

This matches the risk called out when locking Qwen3.5-9B (multimodal/conversational
9B checkpoint is heavier than classic text-only 8B estimates of ~10–14GB).

## Allowed mitigations (do / don't)

| Option | Allowed? | Notes |
|---|---|---|
| Lower `max_seq_length` below 4096 | **No** | Truncates assistant JSON labels (DATA_CONTRACT) |
| Prompt compression (shorter rules / board cache) | Optional try | Unlikely to fix load+train OOM if base+train state already exceeds 16GB |
| Smaller local model (e.g. Qwen3.5-4B) | Only via explicit lock change | Would reopen SCOPE §12.2 |
| **Burst rental for QLoRA SFT** | **Yes — approved path** | 1× 24–80GB class GPU (A6000 / A100 / H100 / L40S, etc.) |

## Decision

1. **Local 5060 Ti:** keep for data generation, dataset builds, bot arena, CI-adjacent CPU work, and (if it fits) **inference-only** serving experiments.
2. **Training / ticket 09 exit:** use **rental GPU** for QLoRA micro-smoke + Phase-2 SFT.
3. **Closed:** rental smoke **go** on HF Jobs L40S — see
   [`hw_smoke_rental_l40s.md`](hw_smoke_rental_l40s.md). Phase-1 ≥100k generation unblocked.

## Rental smoke checklist (replaces local T8 train steps)

Run on rented GPU; save under `outputs/hw_smoke/`:

1. Load `Qwen/Qwen3.5-9B` 4-bit + QLoRA per `configs/qwen3.5-9b-qlora.yaml`
2. Pin `model.revision` SHA in that config
3. 10–20 optimizer steps at `max_seq_length: 4096`; log peak VRAM
4. Assistant-mask test (ticket 07) on that pin
5. One game vs Random; log parse/fallback
6. Write `outputs/hw_smoke/report.json` with go/no-go for Phase-2 SFT

## Local follow-ups (optional, non-blocking)

- Measure **inference-only** 4-bit load VRAM on 5060 Ti (useful for live play later).
- If inference also OOMs, serving is rental/API until a smaller deploy checkpoint exists.
