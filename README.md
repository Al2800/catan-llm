# catan-llm

Train an open-weights **~9B** language model to play **Settlers of Catan** inside
[Catanatron](https://github.com/bcollazo/catanatron) — from simulator self-play data
through QLoRA SFT and (later) GRPO, back to live eval and spectate.

| | |
|---|---|
| **Goal** | A self-contained Catan policy model that clears bot-ladder gates (WR → AlphaBeta) |
| **Model** | `Qwen/Qwen3.5-9B` (QLoRA), revision pinned in config |
| **Where we are** | **Phase 2** — production SFT / Gate B ([`docs/STATUS.md`](docs/STATUS.md)) |
| **Engine** | Catanatron `@82aae93` (GPL-3.0 external dependency) |

---

## What we’re trying to achieve

1. **Data engine** — millions of expert decisions from Catanatron bots (structured + Tier A text).
2. **Training** — QLoRA SFT on canonical train/play-identical prompts, then GRPO with engine-verifiable rewards.
3. **Live loop** — serve the model, play as a Catanatron `Player`, benchmark on fixed fixtures, watch games.

Success is **gate-based**, not vibes:

- **Gate B (SFT):** legality/parse floors + beat WeightedRandom on `ladder-4p` (≥200 games).
- **Gate C (RL):** pinned `ab-4p` vs AlphaBeta (see [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)).

Full vision, risks, locked decisions: [`docs/SCOPE.md`](docs/SCOPE.md).

---

## Where we are (2026-07-31)

```text
Phase 0 ──► Phase 0.5 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
 done        done          done        in progress  blocked     later
                                       (ticket 17)
```

| Done | In flight | Next |
|---|---|---|
| Schema v2, parity, seeds, CI | **17** — SFT + Gate B | **18** failure taxonomy |
| 9B QLoRA HW proof (L40S) | 2000-step rental train | **19** fill RL_SPEC |
| ≥100k train + 5k holdout + quality **GO** | Spectate CLI landed (**24**) | **20–21** GRPO → Gate C |
| `catan-qlora-train`, serve, spectate | Upload Mac holdout to HF | Longer SFT if metrics warrant |

Details: **[`docs/STATUS.md`](docs/STATUS.md)** · tickets: **[`docs/tickets/BACKLOG.md`](docs/tickets/BACKLOG.md)**.

---

## Architecture (outline)

```text
Simulator (Catanatron) ──trajectories──► Data pipeline ──chat JSONL──► QLoRA SFT
         ▲                                      │                        │
         │                                      ▼                        ▼
    LLMPlayer / spectate                  manifests/quality         adapter ckpt
         ▲                                                           │
         └──────────── OpenAI serve ◄────────────────────────────────┘
                              │
                         Eval arena (Gate B / C)
```

Deeper diagrams (data, train, serve sequence, eval gates, CLI map):
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

---

## Documentation map

### Start here

| Doc | Purpose |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | **Current phase, done/next, expectations** |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **System + module architecture** |
| [`docs/SCOPE.md`](docs/SCOPE.md) | Vision, goals/non-goals, roadmap, locked decisions |
| [`AGENTS.md`](AGENTS.md) | Rules for coding agents |

### Contracts (normative)

| Doc | Purpose |
|---|---|
| [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md) | Schema **v2**, prompts, splits, manifests, Phase-1 checklist |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Fixtures, metrics, Gate B / C |
| [`docs/SEED_REGISTRY.md`](docs/SEED_REGISTRY.md) | Disjoint seed ranges |
| [`docs/TEACHER_POV.md`](docs/TEACHER_POV.md) | Teacher vs learner observability |

### How-to / ops

| Doc | Purpose |
|---|---|
| [`docs/TRAINING.md`](docs/TRAINING.md) | QLoRA train, HF Jobs, cost cheatsheet |
| [`docs/SERVING.md`](docs/SERVING.md) | OpenAI-compatible serve + constrained JSON |
| [`docs/SPECTATE.md`](docs/SPECTATE.md) | Terminal watch + replay JSON |
| [`docs/ENV_BLACKWELL.md`](docs/ENV_BLACKWELL.md) | Local 5060 Ti / stack notes |
| [`configs/qwen3.5-9b-qlora.yaml`](configs/qwen3.5-9b-qlora.yaml) | Pinned train config |

### Evidence / history

| Doc | Purpose |
|---|---|
| [`docs/reports/phase1_quality_signoff.md`](docs/reports/phase1_quality_signoff.md) | Phase-1 **GO** |
| [`docs/reports/phase1_cohort_progress.md`](docs/reports/phase1_cohort_progress.md) | Train + holdout checksums |
| [`docs/reports/hw_smoke_rental_l40s.md`](docs/reports/hw_smoke_rental_l40s.md) | L40S QLoRA go |
| [`docs/reports/hw_smoke_5060ti.md`](docs/reports/hw_smoke_5060ti.md) | Local 16GB train no-go |
| [`docs/PHASE0.md`](docs/PHASE0.md) / [`PHASE0_5_TASKS.md`](docs/PHASE0_5_TASKS.md) | Historical Phase 0 / 0.5 cards |
| [`docs/RL_SPEC.md`](docs/RL_SPEC.md) | Phase-3 RL template (**fill before GRPO**) |
| [`docs/tickets/BACKLOG.md`](docs/tickets/BACKLOG.md) | All tickets + frontier |

---

## Locked decisions (short list)

- Base: **Qwen3.5-9B** QLoRA-first; revision pinned after L40S smoke.
- Local 16GB: data/eval/serve probes OK; **not** for train @ 4096 — use rental ≥24GB.
- Train prompts **byte-identical** to live renderer; `schema_version=v2` + `prompt_version`.
- Fallback: **`first_legal`**. Never drop `max_seq_length` below **4096**.
- Headline eval is **4-player**; “beats WR” = higher win share in the same ladder games.
- Seeds only from `SEED_REGISTRY.md`; stop generation at cohort targets.
- Docs win over code until both update together.

---

## Repository layout

```text
AGENTS.md                 Agent handbook
README.md                 This file
configs/                  QLoRA YAML (pinned revision)
docs/                     Scope, contracts, architecture, status, reports, tickets
scripts/                  HF Jobs rental bootstraps
src/catan_llm/
  sim/ data/ training/ eval/ play/ serve/ scripts/
tests/
```

---

## Setup

```bash
# Python 3.11+
pip install -e ".[dev]"     # core + tests
pip install -e ".[train]"   # torch / transformers / TRL / peft / bitsandbytes
```

Catanatron is pulled from git (`82aae93`). GPL-3.0 — keep as an external engine dependency.

---

## Quickstart

### CPU plumbing

```bash
catan-arena --games 12 --seed 0 --out outputs/arena/bot_ladder.json
catan-generate --games 20 --seed 0 --out data/raw/trajectories.jsonl
catan-build-dataset --trajectories data/raw/trajectories.jsonl --out data/processed/expert-smoke
catan-sft-smoke --games 8 --max-steps 20 --work-dir outputs/sft_smoke
```

### Spectate (mock server)

```bash
catan-serve --mock --port 8000 &
catan-spectate --base-url http://127.0.0.1:8000/v1 --watch --vps 6 \
  --out outputs/spectate/replay.json
```

### Production QLoRA (rental GPU)

```bash
catan-qlora-train --dry-run          # validate config + data paths
catan-qlora-train --max-steps 2000   # CUDA + bitsandbytes required
catan-gate-b --adapter outputs/sft/qwen3.5-9b-qlora/adapter --games 200
```

Full Jobs recipe, costs, and expectations: [`docs/TRAINING.md`](docs/TRAINING.md).

---

## CLI reference

| Command | Role |
|---|---|
| `catan-generate` / `catan-phase1-cohort` | Trajectory / cohort generation |
| `catan-build-dataset` | Trajectories → chat JSONL + manifest |
| `catan-sft-smoke` | Tiny SmolLM e2e smoke |
| `catan-qlora-train` | Qwen3.5-9B QLoRA production train |
| `catan-gate-b` | Gate B ladder vs adapter |
| `catan-arena` | Fixture / bot-ladder eval |
| `catan-serve` / `catan-play-endpoint` | Serve + HTTP play |
| `catan-spectate` | Terminal spectate + replay |

---

## Roadmap

| Phase | Exit | Status |
|---|---|---|
| 0 | E2E plumbing with tiny model | done |
| 0.5 | Contracts + 9B HW proof | done |
| 1 | ≥100k train + holdout + quality GO | done |
| 2 | SFT v1 clears Gate B; failure taxonomy | **active** |
| 3 | GRPO; Gate C vs AlphaBeta | next |
| 4 | Scale / self-play / polish / write-up | later |

---

## Hub artifacts

| Repo | Contents |
|---|---|
| `AlCampbell/catan-llm-phase1` | Phase-1 `expert-v1` chat JSONL (+ quality sidecar) |
| `AlCampbell/catan-llm-sft-v1` | QLoRA adapters + train reports (rental uploads) |
| `AlCampbell/catan-llm-hw-smoke` | Ticket 09 smoke report |

---

## Agent skills

Optional engineering skills under [`.cursor/skills/engineering/`](.cursor/skills/engineering/)
(from [mattpocock/skills](https://github.com/mattpocock/skills)). See [`AGENTS.md`](AGENTS.md).

---

## License notes

This project’s application code is MIT-intended unless stated otherwise.
**Catanatron is GPL-3.0** — do not vend or relicense it into a combined distribution
without an explicit owner decision.
