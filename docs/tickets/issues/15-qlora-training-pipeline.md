# 15 — Production QLoRA training pipeline

**What to build:** Replace the tiny SmolLM smoke with a real **Qwen3.5-9B** QLoRA
training entrypoint driven by the pinned config (assistant-only loss, grad
checkpointing, revision pin, VRAM telemetry). Config:
[`configs/qwen3.5-9b-qlora.yaml`](../../../configs/qwen3.5-9b-qlora.yaml).

**Blocked by:** 09, 14 (done)

**Status:** done (2026-07-31) — pipeline landed; full rental train is ticket 17

**Phase:** 2

## Entrypoints

- `catan-qlora-train` → `catan_llm.scripts.run_qlora_sft`
- Library: `catan_llm.training.qlora.run_qlora_sft`
- Config paths point at Phase-1 `data/phase1/processed/expert-v1/`

```bash
# CPU-safe validation
catan-qlora-train --dry-run

# Rental micro / resume
catan-qlora-train --max-steps 50 --max-samples 512
catan-qlora-train --resume-from outputs/sft/qwen3.5-9b-qlora/checkpoint-200
```

Writes `train_report.json` (peak VRAM, step time, mask check) under the output dir.

## Acceptance criteria

- [x] Config + script train from schema-v2 chat JSONL
- [x] Assistant-only loss verified on the pinned template (`assistant_only_loss` + mask check)
- [x] Checkpoints resume/save cleanly (`--resume-from`, `save_steps` / `save_total_limit`)
- [x] Peak VRAM / step time logged (`train_report.json`)
