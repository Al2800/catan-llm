# Hardware smoke report — HF Jobs L40S (rental)

**Status:** **go** for Phase-2 QLoRA SFT entry (ticket 09 closed via rental)  
**Date:** 2026-07-31  
**Ticket:** [09](../tickets/issues/09-local-8b-qlora-smoke.md) / Phase 0.5 T8  
**Job:** [AlCampbell/6a6c48a8b36a6516e96a38c7](https://huggingface.co/jobs/AlCampbell/6a6c48a8b36a6516e96a38c7)  
**Artifact:** `hf://datasets/AlCampbell/catan-llm-hw-smoke/report.json`

## Hardware

| Item | Value |
|---|---|
| Flavor | `l40sx1` (NVIDIA L40S 48GB) |
| Peak VRAM (load 4-bit+LoRA) | **12.808 GB** |
| Peak VRAM (15-step micro-train @ 4096) | **14.906 GB** |
| Runtime | ~20m (incl. download + one game) |

## Model pin

| Item | Value |
|---|---|
| Hub id | `Qwen/Qwen3.5-9B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Config | `configs/qwen3.5-9b-qlora.yaml` (`model.revision` pinned) |
| `max_seq_length` | 4096 (label-safety floor; not lowered) |

## Checklist results

| Check | Result |
|---|---|
| 4-bit QLoRA load | pass |
| Micro-train 15 steps @ ≥4096 | pass (train_loss ≈ 0.72) |
| Assistant-mask on pin | pass (`assistant_span_intact`, no truncation) |
| One game vs Random | pass (finished; 585 turns) |
| Parse / fallback logged | yes — `parse_rate_model=0.0`, `fallback_rate=1.0`, policy `first_legal` |

Smoke-quality note: the micro-trained adapter does **not** produce legal JSON yet
(100% `first_legal` fallback). That is expected for 15 steps; ticket 09 only
requires the train+mask+one-game loop to complete with parse/fallback metrics.

## Decision

1. **Ticket 09 → done** via rental path (local 5060 Ti remains no-go for train).
2. Phase-1 ≥100k generation (**11+**) is unblocked on the hardware/revision gate.
3. Phase-2 SFT should target **≥24GB** rental GPUs (L40S / A10G / A100); 16GB local
   train stays out of scope unless a smaller deploy model is locked.
