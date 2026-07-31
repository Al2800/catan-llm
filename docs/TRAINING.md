# Training guide (Phase 2 QLoRA)

How to run **Qwen3.5-9B** QLoRA SFT for catan-llm. Product gates live in
[`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md); architecture in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Goals

- Behavior-clone Phase-1 expert decisions (+ Tier A rationales) into a LoRA adapter.
- Preserve **assistant-only** loss and **`max_seq_length ≥ 4096`**.
- Prove skill on **`ladder-4p` Gate B** (≥200 games) — not on smoke parse rates alone.

## Locked train config

File: [`configs/qwen3.5-9b-qlora.yaml`](../configs/qwen3.5-9b-qlora.yaml)

| Key | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Quant | 4-bit NF4 (+ double quant), bf16 compute |
| LoRA | r=16, α=32, dropout 0.05 |
| Data | `data/phase1/processed/expert-v1/{train,val}.jsonl` |
| Seq len | **4096** (do not lower) |
| Loss | `assistant_only_loss: true` |
| Grad ckpt | on |
| Effective batch | 1 × accum 8 |

## Hardware

| Machine | Train @ 4096 | Notes |
|---|---|---|
| RTX 5060 Ti 16GB | **no-go** | See [`reports/hw_smoke_5060ti.md`](reports/hw_smoke_5060ti.md) |
| HF Jobs L40S 48GB | **go** | Peak ~15–16 GB; ~29–30 s/step in smoke | 
| Other ≥24GB rental | expected OK | Measure VRAM on first run |

L40S Jobs rate (2026-07-31): **~$1.80 / hour**.

### Cost cheatsheet

| Run | Steps (approx) | Wall time | $ on L40S |
|---|---:|---:|---:|
| Pipeline smoke | 40 | ~20–30 min | ~$1 |
| Diagnostic SFT (current) | 2000 | ~16–17 h | ~$30–32 |
| ~1 epoch | ~12.6k | ~100+ h | ~$180–200 |
| Config 2 epochs | ~25k | ~200 h | ~$350+ |

## Entrypoints

### Local / rental shell (code already on machine)

```bash
pip install -e '.[train]'
# download Phase-1 chat JSONL if needed (HF dataset AlCampbell/catan-llm-phase1)

# CPU-safe wiring check
catan-qlora-train --dry-run

# Real train (CUDA required)
catan-qlora-train --max-steps 2000
catan-qlora-train --resume-from outputs/sft/qwen3.5-9b-qlora/checkpoint-200
```

Outputs under `outputs/sft/qwen3.5-9b-qlora/`:

| File | Meaning |
|---|---|
| `adapter/` | PEFT weights to load / upload |
| `train_report.json` | VRAM, step time, mask check |
| `train_history.json` / `.md` | Loss telemetry / sparkline |
| `resolved_config.yaml` | Frozen train snapshot |

### HF Jobs bootstrap

```bash
# From a machine with `hf` CLI + HF_TOKEN
hf jobs run \
  --flavor l40sx1 \
  --timeout 24h \
  --detach \
  --secrets HF_TOKEN \
  --name catan-ticket17-sft-2k \
  -e BRANCH=main \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  bash -c 'set -euo pipefail; apt-get update -qq && apt-get install -y -qq git >/dev/null; \
    git clone --depth 1 --branch "$BRANCH" https://github.com/Al2800/catan-llm.git /tmp/catan-llm; \
    bash /tmp/catan-llm/scripts/rental_sft_gate_b_job.sh --max-steps 2000 --skip-gate'
```

Scripts: `scripts/rental_sft_gate_b_job.sh`, `scripts/rental_sft_gate_b.py`  
(Older HW-only smoke: `scripts/rental_hw_smoke*.sh`.)

Artifacts upload (when token present) to **`AlCampbell/catan-llm-sft-v1`**.

## Gate B after train

```bash
catan-gate-b \
  --adapter outputs/sft/qwen3.5-9b-qlora/adapter \
  --fixture ladder-4p \
  --games 200 \
  --out outputs/arena/gate_b_ladder4p.json
```

Pass criteria (EVAL_PROTOCOL): parse/legality ≥ 0.995, finished ≥ 200,
`win_rate[candidate] > win_rate[weightedrandom]`.

In-process PEFT generate is used (no HTTP). For watched games, serve the adapter
and use [`SPECTATE.md`](SPECTATE.md).

## Tiny smoke (not Phase-2 skill)

```bash
catan-sft-smoke --games 8 --max-steps 20 --work-dir outputs/sft_smoke
```

Uses SmolLM — proves package wiring only.

## What “good” looks like at each scale

| Scale | Expect |
|---|---|
| 15–40 steps | Format mostly broken; high fallback — OK for VRAM/mask proof |
| ~2k steps | Format/legality much better; skill still weak vs WR |
| ≥1 epoch + tune | Plausible Gate B contention if data quality holds |

## Related

- Ticket **15** (pipeline), **17** (SFT + Gate B), **09** (HW smoke)
- Dataset sign-off: [`reports/phase1_quality_signoff.md`](reports/phase1_quality_signoff.md)
