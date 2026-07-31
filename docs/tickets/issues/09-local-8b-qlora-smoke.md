# 09 — Qwen3.5-9B QLoRA smoke (local 5060 Ti → rental fallback)

**What to build:** Prove `Qwen/Qwen3.5-9B` QLoRA at `max_seq_length≥4096` with a
pinned revision, peak VRAM log, assistant-mask pass, and one game vs Random —
**on local 5060 Ti if possible, otherwise on an approved rental GPU**.

**Blocked by:** None. **Unblocks Phase 1 scale.**

**Status:** done — rental go on HF Jobs L40S (2026-07-31)

**Phase:** 0.5 (T8) — owner GPU / rental

## Model lock

| Item | Value |
|---|---|
| Hub id | `Qwen/Qwen3.5-9B` |
| Config | [`configs/qwen3.5-9b-qlora.yaml`](../../../configs/qwen3.5-9b-qlora.yaml) |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Why not Qwen3-8B | `Qwen/Qwen3-8B-Instruct` does not resolve on HF Hub |
| Why 9B not 8B | Qwen3.5 has no dense 8B |

## Local result (2026-07-31)

Owner finding: **RTX 5060 Ti 16GB cannot hold the model + QLoRA training state**
at label-safe context. Documented in
[`docs/reports/hw_smoke_5060ti.md`](../../reports/hw_smoke_5060ti.md).

Do **not** “fix” this by lowering `max_seq_length` below 4096.

## Rental result (2026-07-31)

HF Jobs `l40sx1` (L40S 48GB): **go**. Report:
[`docs/reports/hw_smoke_rental_l40s.md`](../../reports/hw_smoke_rental_l40s.md).

Entrypoint: `scripts/rental_hw_smoke.py` (+ `scripts/rental_hw_smoke_job.sh`).

## Acceptance criteria

### Local path (closed)

- [x] Attempt / owner report: 16GB insufficient for train smoke
- [x] Local no-go + rental fallback documented (`docs/reports/hw_smoke_5060ti.md`)
- [ ] Optional: inference-only VRAM probe on 5060 Ti (non-blocking)

### Rental path (required to close ticket)

- [x] Rental GPU micro-train at ≥4096 with peak VRAM logged (14.906 GB)
- [x] Concrete `model.revision` SHA recorded in `configs/qwen3.5-9b-qlora.yaml`
- [x] Assistant-mask check (ticket 07) passes on that pin
- [x] One game vs Random completes; parse/fallback logged
- [x] Rental report under `docs/reports/hw_smoke_rental_l40s.md` (+ Hub dataset artifact)

## Notes for implementers

- Local box stays valuable for **CPU data gen / arena / dataset builds**.
- Phase-2 SFT entrypoint (ticket 15) should assume **rental** unless a future
  smaller deploy model is locked.
