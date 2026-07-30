# 09 — Local Qwen3.5-9B QLoRA smoke on 5060 Ti

**What to build:** Prove the owner’s RTX 5060 Ti can load/train/serve
**`Qwen/Qwen3.5-9B`** QLoRA at `max_seq_length≥4096`, or document an approved
rental fallback. Pin the model revision and record peak VRAM.

**Blocked by:** None for starting (ideally after 04 for real prompts). **Blocks Phase 1.**

**Status:** ready-for-agent

**Phase:** 0.5 (T8) — owner GPU

## Model lock (updated 2026-07-30)

| Item | Value |
|---|---|
| Hub id | `Qwen/Qwen3.5-9B` |
| Config | [`configs/qwen3.5-9b-qlora.yaml`](../../../configs/qwen3.5-9b-qlora.yaml) |
| Why not Qwen3-8B | `Qwen/Qwen3-8B-Instruct` **does not resolve** on HF Hub |
| Why 9B not 8B | Qwen3.5 dense lineup has **no 8B** — nearest 8B-class post-trained checkpoint is **9B** |
| Note | Qwen3.5-9B is a multimodal/conversational checkpoint; VRAM may be tighter than classic text-only 8B — smoke must prove it or choose rental fallback |

Do **not** fall back to inventing `Qwen3.5-8B` / `Qwen3-8B-Instruct` ids.

## Acceptance criteria

- [ ] `ENV_BLACKWELL.md` checklist completed against `Qwen/Qwen3.5-9B`
- [ ] Concrete `model.revision` SHA recorded in `configs/qwen3.5-9b-qlora.yaml` (not null)
- [ ] Micro-train at ≥4096 with peak VRAM logged
- [ ] Assistant-mask check (ticket 07) passes on that pin
- [ ] One game vs Random completes; parse/fallback logged
- [ ] Report at `outputs/hw_smoke/report.json` (or attached)
- [ ] If OOM: prompt-compression or rental fallback documented — do **not** lower below label-safe length
