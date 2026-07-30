# Blackwell / RTX 5060 Ti environment plan

Local primary box: **NVIDIA RTX 5060 Ti 16GB** (sm_120).  
This doc is the Phase 0.5 hardware gate for SCOPE §9 / §11.

## 1. Why this exists

Phase 0 CPU smoke does **not** prove the local training path. Before building ≥100k decisions or claiming QLoRA readiness, run the checklist below on the owner's workstation and attach the report.

## 2. Target stack (to be pinned after first successful smoke)

| Component | Intent |
|---|---|
| OS / driver | Recent NVIDIA driver with sm_120 support |
| Python | 3.11 or 3.12 |
| PyTorch | CUDA 12.8+ wheel with Blackwell kernels (`cu128` or newer) |
| transformers / accelerate / peft / trl | Compatible with chosen torch |
| bitsandbytes | Required for 4-bit QLoRA |
| vLLM | Version that supports the installed torch + sm_120 |
| catanatron | git pin from `pyproject.toml` (`82aae93…`) |

Do not leave these as open-ended “latest” once a working set is found — freeze versions in a lockfile (`uv.lock` or `requirements-blackwell.txt`).

## 3. Local validation checklist (Phase 0.5 exit)

Run on the 5060 Ti; save outputs under `outputs/hw_smoke/`:

1. **Device probe**
   - `nvidia-smi` shows 16GB and a non-zero CUDA capability
   - `torch.cuda.is_available()` true; print `torch.__version__`, CUDA version, device name
2. **4-bit model load**
   - Load `Qwen/Qwen3.5-9B` (exact revision recorded) in 4-bit with bitsandbytes
   - Record peak VRAM (9B multimodal may be tighter than classic text-only 8B)
3. **QLoRA train micro-step**
   - 10–20 optimizer steps using [`configs/qwen3.5-9b-qlora.yaml`](../configs/qwen3.5-9b-qlora.yaml) on a tiny JSONL shard built from the **canonical** renderer
   - `gradient_checkpointing: true`, batch size 1, **`max_seq_length: 4096`** (label-safety floor; 2048 is unsafe)
   - **Pin** `model.revision` to a concrete commit SHA in the successful report
   - Peak VRAM logged. Target ≤14GB; if OOM at 4096, try prompt compression (shorter rules / board caching) or document rental fallback — **never** lower `max_seq_length` below the no-truncation budget
4. **Assistant-mask check**
   - Run the T9 one-batch mask test against this pinned revision/tokenizer
5. **Serve / generate**
   - Either vLLM OpenAI server **or** HF generate path used by `LLMPlayer`
   - One game vs Random completes; log `parse_rate_model` / `fallback_rate`
6. **Report**
   - Write `outputs/hw_smoke/report.json` with versions, **model revision**, peak VRAM, tokens/sec, step time, OOMs/workarounds, and go/no-go for local Phase-2 SFT

## 4. Known constraints

- Full fine-tune of ~9B: **not** local
- 12B training: **rental only**
- Serious GRPO (large G, long contexts): prefer **A100/H100 burst**
- Default CI runners have no GPU — keep HF Qwen3.5-9B downloads out of CI
- Do not use `Qwen/Qwen3-8B-Instruct` (unresolvable) or invent `Qwen3.5-8B`

## 5. Failure modes to expect

| Symptom | Likely cause | Mitigation |
|---|---|---|
| Torch has no kernel for sm_120 | Torch too old | Upgrade to cu128+ build |
| bitsandbytes import / 4bit fail | Incompatible bnb/torch | Pin matching pair; try `transformers` quantization config variants |
| OOM at seq 2k+ | Context too long / no checkpointing | Lower `max_seq_length`, enable grad checkpointing, smaller LoRA rank |
| vLLM won't start | Build lacks Blackwell | Pin newer vLLM or serve via HF for Phase 2 interim |

## 6. Gate language

SCOPE decision §12.10: **No ≥100k dataset build until local 8B QLoRA smoke succeeds** and this report exists.
