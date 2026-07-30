# 15 — Production QLoRA training pipeline

**What to build:** Replace the tiny SmolLM smoke with a real **Qwen3.5-9B** QLoRA
training entrypoint driven by the pinned config (assistant-only loss, grad
checkpointing, revision pin, VRAM telemetry). Config:
[`configs/qwen3.5-9b-qlora.yaml`](../../../configs/qwen3.5-9b-qlora.yaml).

**Blocked by:** 09, 14

**Status:** blocked

**Phase:** 2

- [ ] Config + script train from schema-v2 chat JSONL
- [ ] Assistant-only loss verified on the pinned template
- [ ] Checkpoints resume/save cleanly
- [ ] Peak VRAM / step time logged
