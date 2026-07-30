# 09 — Local 8B QLoRA smoke on 5060 Ti

**What to build:** Prove the owner’s RTX 5060 Ti can load/train/serve Qwen3-8B
QLoRA at `max_seq_length≥4096`, or document an approved rental fallback. Pin the
model revision and record peak VRAM.

**Blocked by:** None for starting (ideally after 04 for real prompts). **Blocks Phase 1.**

**Status:** ready-for-agent

**Phase:** 0.5 (T8) — owner GPU

- [ ] `ENV_BLACKWELL.md` checklist completed
- [ ] Concrete `model.revision` SHA recorded (not null)
- [ ] Micro-train at ≥4096 with peak VRAM logged
- [ ] Assistant-mask check (ticket 07) passes on that pin
- [ ] One game vs Random completes; parse/fallback logged
- [ ] Report at `outputs/hw_smoke/report.json` (or attached to PR)
- [ ] If OOM: prompt-compression or rental fallback documented — do **not** lower below label-safe length
