# 07 — Teacher POV audit + assistant-mask test

**What to build:** Codify privileged-teacher distillation: experts may use full
`Game`, learner prompts stay POV-limited, Tier A text is POV-safe. Prove SFT loss
hits the assistant JSON on the pinned Qwen chat template.

**Blocked by:** 01, 04

**Status:** done

**Phase:** 0.5 (T9)

- [x] Audit/test documents teacher vs learner observability (SCOPE §5.1)
- [x] Test fails if Tier A text leaks opponent private-hand literals
- [x] One-batch mask test: system/user masked; assistant JSON has nonzero loss; full action span present
- [x] If Qwen revision not yet pinned, test skips clearly; must pass after ticket 09 pins it, before Phase 1

**Notes:** SmolLM one-batch runs in CI when transformers/torch present. Qwen
revision one-batch skips until `configs/qwen3.5-9b-qlora.yaml` pins `model.revision`
(ticket 09; model = `Qwen/Qwen3.5-9B`).
