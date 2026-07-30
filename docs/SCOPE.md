# Catan LLM — Full Project Scope

**Working title:** Catan LLM (training an 8–12B parameter language model to play Settlers of Catan in the Catanatron simulator)

**Status:** Scope locked (see §12). Phase 0 plumbing spike landed in-repo; plan tightened after post-spike review (2026-07-30). Canonical docs now live in this `catan-llm` repository.

**Companion specs (normative for implementation — docs win over code until both update together):**

- [`DATA_CONTRACT.md`](DATA_CONTRACT.md) — trajectory **schema v2**, renderer parity, splits, manifests
- [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) — fixed fixtures, metrics, promotion gates
- [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md) — local 5060 Ti / PyTorch / vLLM pin plan
- [`SEED_REGISTRY.md`](SEED_REGISTRY.md) — disjoint seed ranges (train/val/test/holdout/champion)
- [`RL_SPEC.md`](RL_SPEC.md) — Phase-3 reward / anti-hacking skeleton (entry gate)
- [`PHASE0_5_TASKS.md`](PHASE0_5_TASKS.md) — assignable Phase 0.5 work units
- [`../AGENTS.md`](../AGENTS.md) — handoff rules for coding agents
- [`../configs/qwen3-8b-qlora.yaml`](../configs/qwen3-8b-qlora.yaml) — Phase-2 QLoRA training sketch

---

## 1. Vision

Build a system that trains a dedicated open-weights LLM (8–12B parameters) to play Settlers of Catan at a high level inside the **Catanatron** simulator, using large-scale simulator-generated data — both **structured** (game-state/action records) and **unstructured** (natural-language renderings and reasoning traces) — mirroring the data-pipeline pattern used in our other large-dataset project.

The end state:

1. A **data engine** that generates millions of Catan decisions from Catanatron self-play (expert bots, search-based bots, and later our own model).
2. A **training pipeline** that turns that data into a fine-tuned 8–12B model (SFT → RL).
3. A **live playing loop** where the trained model is plugged into Catanatron as a `Player`, plays full games against built-in bots and other models, and is benchmarked on win rate — watchable live via the Catanatron web UI.

## 2. Goals and non-goals

### Goals

- **G1.** Reproducible pipeline from "raw simulator" → "training-ready dataset" → "trained checkpoint" → "live evaluation in the simulator".
- **G2.** A fine-tuned 8–12B model whose Catan win rate beats Catanatron's strongest built-in bot (`AlphaBetaPlayer`) in head-to-head play, with statistically meaningful sample sizes.
- **G3.** The model outputs structured, machine-parseable actions with a near-100% legality/parse rate, plus short natural-language reasoning (useful for the unstructured-data side and for debugging).
- **G4.** Everything runnable as code: data generation, dataset builds, training configs, eval harness, and a live-spectate path.
- **G5.** Local-first QLoRA path on a single 16GB GPU, with portable configs. Multi-GPU / full-FT is a **deferred Phase-4 rental path**, not a Phase-0–3 delivery goal.

### Non-goals (for this project)

- Training a foundation model from scratch (we fine-tune an existing open-weights 8–12B model).
- Human-facing trade negotiation via natural-language chat between LLMs (Catanatron's engine has no domestic player-to-player trading; we stay within the engine's action space. See §10, R6).
- A production product/UI beyond what is needed to run and watch games.
- Beating frontier LLMs (Claude/GPT-class) prompted as agents — interesting comparison, not a success criterion.
- Shipping a multi-GPU training stack in the initial phases (rental scripts may appear in Phase 4).

## 3. Background and why this is differentiated

### 3.1 The simulator: Catanatron

[Catanatron](https://github.com/bcollazo/catanatron) (Bryan Collazo) is the de-facto open-source Settlers of Catan engine, and it is purpose-built for exactly this kind of project:

- **Fast:** thousands of games per minute in pure Python — data generation at scale is cheap (CPU only).
- **Programmatic:** `Game` / `Player` Python API plus a `catanatron-play` CLI for bulk simulation.
- **Built-in opponents of graduated strength:** `RandomPlayer` → `WeightedRandomPlayer` → `ValueFunctionPlayer` → `AlphaBetaPlayer` (the strongest, a hand-crafted value function + alpha-beta search).
- **Gymnasium environment** (`catanatron_gym`): `gymnasium.make("catanatron/Catanatron-v0")` with configurable map type (`MINI`, `BASE`), `vps_to_win`, enemy mix, custom reward functions, and vector-friendly state/action representations — designed for RL.
- **Dataset tooling:** `catanatron_experimental` already contains scripts for generating ML datasets of games.
- **Web UI** for watching games live.

### 3.2 What already exists (and the gap)

| Existing work | Approach | Limitation we exploit |
|---|---|---|
| [catan-bench](https://github.com/vmmadathil/catan-bench) | Prompts frontier LLMs (Claude, Gemini, GPT) turn-by-turn on Catanatron; JSON action choice + scratchpad | No training — models make narrow mechanical errors (rule misunderstandings, hallucinated scores, no plan B); games average 163 turns vs ~70 for humans |
| [HexMachina / "Agents of Change"](https://arxiv.org/abs/2506.04651) | Self-evolving multi-agent system that rewrites its own player *code*/prompts; beat `AlphaBetaPlayer` (54% win rate) | The intelligence lives in scaffolding and a frontier API model, not in a trained, self-contained model |
| CATArena | Tournament platform for agent learning | Evaluation framework, not a trained model |
| Catanatron's own RL attempts | Small value/policy networks on the Gym env | Not language models; no reasoning transfer |

**The gap:** nobody has trained the *weights* of a mid-size (8–12B) open model to be a strong Catan player. We get a self-contained, cheap-to-serve model with Catan-specific strategic skill — and a reusable simulator→data→training→eval loop that transfers to other games/decision domains.

### 3.3 Why Catan is a good training target

- **Long-horizon, partially observable, stochastic, 4-player adversarial** — a hard proxy for real multi-stakeholder decision-making.
- **Verifiable rewards for free:** win/loss, victory points, action legality, turn efficiency — all checkable by the engine, which is exactly what modern RL (RLVR/GRPO) needs.
- **Cheap ground truth at scale:** expert bots provide unlimited "teacher" decisions without human labeling.
- **Both data modalities:** every decision is simultaneously a structured record (state JSON, action enum) and renderable as natural language (board description, rationale) — matching our structured + unstructured dataset pattern.

## 4. System architecture

Five components, each independently runnable, connected by artifacts on disk:

```
┌──────────────┐   games/trajectories   ┌───────────────┐   chat-format JSONL   ┌───────────────┐
│ 1. SIMULATOR │ ─────────────────────▶ │ 2. DATA       │ ─────────────────────▶│ 3. TRAINING   │
│   ADAPTER    │                        │   PIPELINE    │                       │   (SFT → RL)  │
│ (Catanatron) │ ◀───────────────────── │               │                       │               │
└──────┬───────┘   LLMPlayer queries    └───────┬───────┘                       └───────┬───────┘
       │           model server                  │ curated datasets                    │ checkpoints
       │                                         ▼                                     ▼
┌──────┴───────┐                        ┌───────────────┐                       ┌───────────────┐
│ 5. LIVE PLAY │ ◀─── checkpoints ───── │ 4. EVALUATION │ ◀──────────────────── │    SERVING    │
│  & SPECTATE  │                        │    ARENA      │    win rates / Elo    │ (vLLM server) │
└──────────────┘                        └───────────────┘                       └───────────────┘
```

### 4.1 Simulator adapter (`sim/`)

- Wraps `catanatron` (and `catanatron_gym` where useful) as a pinned dependency.
- Runs bulk games between any mix of built-in bots and our `LLMPlayer`.
- Records **full trajectories**: per decision — serialized game state (before action), valid action list, action taken, acting player, dice/events, and final game outcome (winner, VP counts, turn count). Exact fields: [`DATA_CONTRACT.md`](DATA_CONTRACT.md).
- Deterministic seeding for reproducibility; multiprocessing for throughput; **append-only / atomic shard writes** (never truncate an existing shard mid-run; crash-safe resume required).
- Map types must use real Catanatron templates (`BASE_MAP_TEMPLATE`, `MINI_MAP_TEMPLATE`). Unsupported map requests fail loudly; never silently fall back while labeling another map type.

### 4.2 Data pipeline (`data/`)

- **Trajectory schema v2** (structured): one record per decision. Normative schema in [`DATA_CONTRACT.md`](DATA_CONTRACT.md). Legacy plumbing labelled v1 is not Phase-1-valid.
- **Renderers** (unstructured): **one canonical renderer** used for dataset chat JSONL *and* live `LLMPlayer` prompts. A compact alternate prompt is **not allowed** for training labels. Parity is CI-gated. Every record stores `prompt_version`.
- **Teacher observability (locked):** expert bots (AlphaBeta / ValueFunction / …) may use the **full engine `Game`** to choose actions — this is privileged distillation. The learner prompt remains **POV-limited** (own hand only). Tier A rationales may cite only learner-observable features (no opponent private hands). See §5.1 / §12.17.
- **Rationale generation** (unstructured, two tiers):
  - *Tier A (free):* feature-aware templates from learner-observable state (+ optional valueΔ if computed without leaking hidden info into the text).
  - *Tier B (sampled):* deferred until after Phase-2 SFT results (locked decision §12.4).
- **Dataset builder:** dedup, filter (drop corrupted/unfinished games), balance (by phase + rare action types), **split by stable `game_key`**, emit chat-format JSONL plus raw structured records for RL.
- **Versioning:** every dataset carries a manifest (source commit, Catanatron pin, `prompt_version`, generator versions, bot/depth/seat config, map hash, seeds, counts, checksums).

### 4.3 Training (`training/`)

Two stages (details in §7), driven by config files:

- **SFT:** behavior-clone expert decisions (+ Tier A rationales) into Qwen3-8B-Instruct via **QLoRA**. Config sketch: [`configs/qwen3-8b-qlora.yaml`](../configs/qwen3-8b-qlora.yaml). Required: assistant-token-only loss, gradient checkpointing, pinned model revision, VRAM telemetry.
- **RL (GRPO with verifiable rewards):** decision-level prompts with group rollouts scored by the engine (legality, immediate VP/shape heuristics) plus full-game outcome rewards (win, VP margin). Framework target: TRL or verl with vLLM rollouts. Full reward spec is a Phase-3 entry gate (see §7.2 / [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)).

### 4.4 Evaluation arena (`eval/`)

Normative protocol: [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).

- Plays N-game matches of a checkpoint vs fixed opponents on **pre-registered fixtures** ([`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)).
- **Headline format is 4-player** (`ladder-4p`, `ab-4p`). Two-player / 1v1 mirrors are secondary diagnostics only (mechanically supported, not standard Catan).
- Metrics: win rates for **all seats**, Wilson CIs, VP margin, turns-to-win, **model parse/legality before fallback**, fallback rate, per-action-type errors. “Beats WeightedRandom” means **candidate win share > WeightedRandom win share in the same ladder games** (not absolute candidate WR > 50% in a 4p table).
- Regression / promotion gates are pre-registered; smoke parse/legality alone is never treated as skill evidence.
- Baselines to contextualize: catan-bench's prompt-only results and HexMachina's ~54% vs AlphaBeta.

### 4.5 Serving + live play (`serving/`, `play/`)

- `LLMPlayer`: implements Catanatron's `Player.decide(game, playable_actions)` by rendering state (**exact same renderer as training**), querying an OpenAI-compatible endpoint (vLLM server hosting the checkpoint), parsing the chosen action, with constrained decoding when available (see §7.3), and a locked safe fallback of **`first_legal`** (engine-ordered first playable action) on parse failure.
- Live spectating: run games under the Catanatron web UI so matches are watchable in real time.
- A small CLI: `play --model <ckpt> --opponents AlphaBeta,ValueFunction,Random --games 100 --watch`.

## 5. Data strategy (the "large datasets, structured + unstructured" tie-in)

| Dataset | Type | Source | Target size (initial) | Purpose |
|---|---|---|---|---|
| `expert-trajectories` | Structured | AlphaBeta / ValueFunction / WeightedRandom self-play | 100k–1M decisions | SFT behavior cloning |
| `expert-commentary` | Unstructured | Template rationales (+5–10% teacher-model annotated) | paired with above | Reasoning-style SFT; dataset richness |
| `mixed-quality` | Structured | Games between bots of *different* strengths, labeled with outcome | 100k+ decisions | Preference/quality signal; teaches recovering from bad positions |
| `self-play-rollouts` | Structured | Our checkpoints playing each other (post-SFT) | grows each RL iteration | GRPO advantage estimation; on-policy improvement |
| `eval-holdout` | Structured | Fixed seeded game set, never trained on | 5–10k games | Honest benchmark |

Key properties:

- **Leakage control:** splits by stable game key `(seed, map_hash, bot_config_hash)`, never by decision and never by UUID alone.
- **Curriculum:** start on `MINI` maps / lower `vps_to_win` (shorter games, cleaner signal), graduate to full `BASE` maps. MINI must actually instantiate `MINI_MAP_TEMPLATE`.
- **Class balance:** Catan decisions are dominated by `ROLL`/build/end-turn; rare high-leverage actions (dev cards, monopoly, year-of-plenty, 4:1/3:1 maritime trades, robber placement) are oversampled. Target quotas live in [`DATA_CONTRACT.md`](DATA_CONTRACT.md).
- **Train/play consistency:** one canonical state renderer + action indexing used by data gen, training, and live play alike. Parity is a hard gate before any ≥100k dataset build.

### 5.1 Teacher observability & distillation (locked)

| Role | What they may see |
|---|---|
| Learner (`LLMPlayer` / SFT prompts) | POV-limited canonical renderer (own hand; opponent card *counts* only) |
| Expert teacher (AlphaBeta, ValueFunction, …) | Full engine `Game` when choosing `action_taken` |
| Tier A rationale text | Learner-observable features only — never opponent private resources/devs |

Rationale: Catanatron's strongest bots are privileged search/value functions; cloning their *actions* from POV prompts is intentional distillation and will be noisy. We accept that noise for Phases 1–2 rather than rewriting the bots. A **teacher-observability audit** (Phase 0.5 T9) documents which features enter labels vs rationales and adds a CI check that rationale text does not contain opponent hand literals.

### 5.2 Phase-1 generation cohort (capacity plan)

Target: **≥100k train decisions** after filtering, not “as many games as possible.”

Default cohort (adjust only via PR to this table + [`SEED_REGISTRY.md`](SEED_REGISTRY.md)):

| Slice | Seed range name | Games (order) | Bot mix (seats) | Map | Intent |
|---|---|---:|---|---|---|
| A | `train_main` | ~800–2_000 | alphabeta, valuefunction, weightedrandom, random (rotated) | BASE | bulk expert decisions |
| B | `train_mini_curriculum` | ~200–500 | same ladder, lower `vps_to_win` optional | MINI | shorter games / placement signal |
| Holdout | `eval_holdout` | 5_000 | fixed ladder, never trained on | BASE | offline eval |

Stop generation when filtered train decisions ≥100k **and** rare-action floors in the quality report are met. Do not blindly burn the entire 50k `train_main` reservation.

## 6. Model selection (8–12B)

Candidates (final pick is a Phase-0 spike; all are fine-tunable and vLLM-servable):

| Candidate | Size | Notes |
|---|---|---|
| **Qwen3-8B / Qwen3-14B** | 8–14B | Strong reasoning-per-param; popular RLVR base; Apache-2.0 |
| **Llama-3.1-8B-Instruct** | 8B | Best ecosystem support; community license |
| **Gemma-3-12B-it** | 12B | Top of the target band; strong instruction following |
| **Mistral-Nemo-12B** | 12B | 128k context — useful for long game histories |

**Decision (locked): Qwen3-8B** (Apache-2.0) as the base model — fastest iteration, cheapest RL rollouts — using its instruct checkpoint, since we need reliable structured output. Prove the loop end-to-end at 8B, then evaluate a 12B-class step-up with the same pipeline (rental-only on current hardware, see §9).

## 7. Training strategy

### Stage 0 — Spike (prove the loop) — done as plumbing

- Tiny run: SmolLM2-135M through short SFT → legal moves vs `RandomPlayer`.
- **What this proved:** schema/parser/arena wiring works.
- **What this did not prove:** skill, renderer parity, or 8B QLoRA on the 5060 Ti.
- Remaining Stage-0 hard gates before Phase-1 scale (see §11 Phase 0.5):
  1. Train prompts == live renderer prompts (parity test).
  2. MINI maps use `MINI_MAP_TEMPLATE` (no silent BASE fallback).
  3. Local 8B QLoRA load/train/serve smoke on the owner's 5060 Ti with VRAM telemetry.

### Stage 1 — SFT (behavior cloning)

- Data: `expert-trajectories` + Tier A commentary (assistant = short rationale + chosen action in strict JSON), built from the **canonical renderer** (schema v2 / [`DATA_CONTRACT.md`](DATA_CONTRACT.md)).
- Loss on **assistant tokens only** (`assistant_only_loss: true`; do **not** also set conflicting `completion_only_loss` on chat data). 2–3 epochs; cosine LR ~1e-4 (QLoRA). Config: [`configs/qwen3-8b-qlora.yaml`](../configs/qwen3-8b-qlora.yaml).
- **Context budget (measured 2026-07-30):** canonical system ≈1.25k tokens; user ≈1.0–1.1k; total ≈2.3–2.5k before the assistant label (SmolLM2 tokenizer; Qwen same order). Therefore SFT `max_seq_length` must be **≥4096**. Truncating at 2048 with `keep_start` would cut the assistant JSON and silently destroy the learning signal.
- **4096 is a label-safety floor, not a proven VRAM fit.** Prior “~10–14GB” estimates assumed shorter contexts. Phase 0.5 T8 must measure peak VRAM on the 5060 Ti with the **pinned Qwen revision** at 4096. If OOM: compress the canonical prompt (move static board to a cached system prefix / shorten rules) or use rental compute — **never** lower `max_seq_length` below the no-truncation budget.
- **Masking gate:** one-batch test on the pinned Qwen tokenizer/chat template must prove system/user tokens are masked, assistant JSON tokens have nonzero loss, and the full `{"action":…}` span is present (Phase 0.5 T9).
- Exit criteria (held-out, fallbacks counted separately) — see [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) Gate B:
  - model parse rate ≥ 99.5% and model legality rate ≥ 99.5% **before** fallback
  - on `ladder-4p`, candidate win share **strictly exceeds** WeightedRandom's win share (paired same-fixture comparison), ≥200 finished games
  - competitive with `ValueFunctionPlayer` (not required to exceed yet)

### Stage 2 — RL (GRPO with verifiable rewards)

Entry gate: Stage-1 exit criteria met **and** a filled [`RL_SPEC.md`](RL_SPEC.md) (reward weights, anti-hacking tests, cost caps) reviewed.

- **Prompts:** decision points sampled from games (mix of bot-generated states and on-policy states), rendered with the canonical renderer.
- **Rollouts:** G=8–16 completions per decision via vLLM; temperature ~0.8–1.0.
- **Rewards (all engine-verifiable):**
  - Format/legality: parseable + legal action (gates everything else).
  - Outcome: win/loss and VP margin for the full game containing the decision (group-shared credit assignment).
  - Shaping (small weights): immediate VP gain, avoiding hand >7 before a roll, longest-road/army threats.
  - Penalties: illegal action, parse failure, degenerate repetition / stalling.
- **Stability:** KL penalty to the SFT reference policy; standard GRPO clipping.
- **Self-play loop (optional phase 3b):** refresh opponent pool with recent checkpoints; regenerate `self-play-rollouts`; iterate.
- **Hardware path:** small-scale GRPO may run locally (slow); meaningful iterations assume burst-rented A100/H100 (see §9). Cost cap and abort criteria in the RL spec.
- Exit criteria: on the pre-registered AlphaBeta fixture ([`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md)), ≥55% win rate over ≥1k games with 95% Wilson lower bound > 50%, plus an independent reproducibility rerun.

### What we deliberately avoid (for now)

- PPO with a learned value model (GRPO removes the critic — roughly half the memory, cleaner with binary-ish game rewards).
- Training on full-game single-context rollouts initially (context bloat); decision-level training first, full-game trajectories as a later extension.

### 7.3 Inference decoding & fallback (locked for Phases 1–2)

| Setting | Locked choice |
|---|---|
| Fallback policy | **`first_legal`** — engine-ordered first playable action |
| Assistant JSON | `{"action": <int>, "reasoning": "<short>"}` — reasoning **is** generated at train and inference |
| Constrained decoding | Prefer vLLM guided JSON / structured outputs enforcing `action` ∈ `[0, n_actions)` when serving; if unavailable, temperature-0 + parser + fallback |
| Max new tokens (play) | 128 (enough for short reasoning + JSON) |
| Temperature (eval) | 0.0 for headline ladders unless a report says otherwise |

### 7.4 Tier A rationale templates (Phase 1 minimum)

Feature-aware templates only (not `"policy selects BUILD_SETTLEMENT"`). Minimum feature set:

- Settlement/city: adjacent pip sum, resource diversity, port access, distance-to-opponent blocking
- Road: extends toward settlement spot / longest-road threat
- Robber/knight: tiles blocked (pips), steal target hand size
- Maritime trade: hand imbalance corrected
- Dev-card buys/plays: bank/dev-deck remaining if known, army/road race state
- ValueFunction / AlphaBeta: include value delta when the expert exposes it

Exact format strings live in [`DATA_CONTRACT.md`](DATA_CONTRACT.md) §8.

## 8. Evaluation plan

| Metric | Definition | Target |
|---|---|---|
| Win rate vs bot ladder | Seeded N-game matches vs Random / WeightedRandom / ValueFunction / AlphaBeta | ≥ AlphaBeta parity, then exceed |
| Elo (checkpoint pool) | Round-robin between all checkpoints + bots | monotonic improvement per iteration |
| Legality & parse rate | % model calls with parseable, engine-legal action **before** fallback | ≥ 99.5% (fallback=`first_legal` covers the rest) |
| Turn efficiency | Avg turns to finish (LLM-prompted games averaged 163 vs ~70 human in catan-bench) | trending toward bot norms |
| VP margin | Avg final VP diff | secondary tiebreaker |
| Failure taxonomy | per-action-type error breakdown from eval logs | drives data fixes each iteration |

All evals run through the same arena code with pinned seeds and published configs, so numbers are reproducible.

## 9. Infrastructure and compute

### 9.1 Compute tiers

| Tier | Hardware | What it supports |
|---|---|---|
| **Local (locked): RTX 5060 Ti 16GB** | Owner's workstation | Data generation, dataset builds, eval arena, **QLoRA SFT of Qwen3-8B** (~10–14GB VRAM with 4-bit base + gradient checkpointing), 4-bit vLLM serving for live play, **small-scale GRPO** (feasible but slow) |
| **Dev / CPU** | Any modern machine | Simulator (thousands of games/min), dataset builds, bot-vs-bot eval, CI, Stage-0 spike with a small model |
| **Burst rental** | 1× A100/H100 80GB | Larger-batch QLoRA/LoRA SFT, meaningfully faster GRPO iterations, 12B QLoRA |
| **Scale rental** | 2–4× 80GB GPUs | Full-FT of 8B (≈60–88GB), serious GRPO at 8–12B, 12–14B full-FT (≈140–174GB) |

**Toolchain note:** the 5060 Ti is a Blackwell card (sm_120) — it needs recent PyTorch (cu128+ builds) and a current vLLM. Exact pins and a local validation checklist live in [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md). The 16GB VRAM ceiling is the binding local constraint: no full fine-tuning and no 12B training locally; both are rental-only paths.

Data generation itself is CPU-only and cheap; the GPU budget is dominated by RL rollouts, which is why Stage 2 is scoped at 8B first.

**Hard gates before ≥100k generation:**

1. Phase 0.5 task cards T1–T7 + T9 merged (contracts, parity, CI gates, masking/POV audit).
2. T8 local 8B QLoRA smoke on the 5060 Ti succeeds **or** an explicit rental fallback is approved in the T8 report.
3. Peak VRAM / tokens/sec / approved cohort size recorded.

### 9.2 Software stack

- Python 3.11+, `catanatron` (pinned git commit; currently `82aae93`).
- Decision-level training does **not** require `catanatron_gym` / Gymnasium; those remain optional later if vectorized RL needs them.
- Training: PyTorch (Blackwell-capable build), transformers, TRL (SFT + GRPO), PEFT (LoRA/QLoRA), bitsandbytes; optional verl/OpenRLHF for scaled RL; DeepSpeed/FSDP for full-FT (rental only).
- Serving: vLLM (OpenAI-compatible endpoint), pinned per [`ENV_BLACKWELL.md`](ENV_BLACKWELL.md).
- Storage: datasets as JSONL/Parquet shards + manifests (checksums required; DVC optional); checkpoints on disk/object storage.
- Experiment tracking: **local JSON reports by default**; optional W&B later.
- CI: unit tests for schema/renderer/parser/**prompt parity**/MINI maps/contract fields; fast smoke tests on CPU. No HF 8B downloads in default CI. Known Phase-0.5 gaps should appear as failing or `xfail` tests so agents see red, not prose.
- Seed ranges: single registry in [`SEED_REGISTRY.md`](SEED_REGISTRY.md) — never invent ad-hoc ranges in scripts.
- **Licensing note:** Catanatron is GPL-3.0. Before publishing a combined package or redistributing binaries that link the engine, confirm distribution posture (engine as separate dependency vs combined release). Training artifacts/datasets/write-ups are separable from engine redistribution.

### 9.3 Cost shape (order of magnitude)

- Data generation + dataset builds + eval arena + QLoRA SFT of 8B + 4-bit serving: **~zero marginal cost** on the owner's 5060 Ti (electricity aside).
- Burst-rented A100/H100 for larger SFT sweeps: tens of dollars per run.
- Full-FT 8B (rental only): low hundreds of dollars per run.
- GRPO RL: the real budget — small iterations can run locally for free; serious iterations are hundreds of rented GPU-hours at 8B. This is the line item to watch and why we gate it behind a working SFT loop.

## 10. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **Action legality / parse failures** | Games crash or degrade | Constrained decoding (JSON schema/grammar), numbered-action output format (proven by catan-bench), deterministic fallback action, legality reward in RL |
| R2 | **Long games blow context / slow RL** | Cost and instability | Compact state renderer; decision-level (not full-history) prompts; scratchpad summary instead of full log; MINI-map curriculum |
| R3 | **Credit assignment over ~100+ decisions/game** | Weak RL signal | Group-shared outcome reward first; small shaped rewards; later: per-decision value targets from search bots |
| R4 | **Reward hacking / degenerate play** (stalling, pass-heavy) | Model games the metric | Turn-efficiency metric in eval; penalties for game-length outliers; arena always includes real opponents |
| R5 | **Distribution shift from bot states → model states** | SFT model drifts off-distribution | DAgger-style iteration: generate states with current model, label with expert; on-policy GRPO in Stage 2 |
| R6 | **Engine scope:** Catanatron has no player-to-player negotiation trading | Model can't learn human-style deals | Accept engine's action space (maritime trades exist); trading-bolt-on (like catan-bench did) is an optional later extension |
| R7 | **Compute budget overrun** | Project stalls | Stage gates (Stage 0 spike, SFT exit criteria) before RL; QLoRA-first; 8B-first |
| R8 | **Model license** | Can't distribute/use | Prefer Apache-2.0 bases (Qwen); confirm license before committing |
| R9 | **Catanatron API drift** | Breakage | Pin git commit; thin adapter layer; map helpers fail loudly |
| R10 | **Partial observability / privileged teachers** | Confusing fairness claims; rationale leakage | Locked distillation policy (§5.1); learner POV prompts; Tier A POV-safe; teacher-observability audit (T9) |
| R11 | **16GB VRAM ceiling on the local card** | No local full-FT, 12B, or heavy RL | 8B-first + QLoRA; burst rentals gated on Stage-1; Blackwell pins in `ENV_BLACKWELL.md` |
| R12 | **Train/play prompt mismatch** | SFT looks fine offline, fails live | Canonical renderer only; `prompt_version` on every record; parity CI; block scale data until green |
| R13 | **Smoke metrics mistaken for skill** | False confidence | Separate parse/legality-before-fallback from win-rate; fixed holdouts |
| R14 | **GPL engine redistribution ambiguity** | License conflict if packaged carelessly | Keep engine as external dependency; review before combined releases |
| R15 | **Silent label truncation / bad masking** | SFT trains with no assistant signal | `max_seq_length ≥ 4096`; no-truncation assert; one-batch assistant-mask test on pinned Qwen template |
| R15b | **4096 OOM on 16GB** | Local QLoRA path fails | T8 measures VRAM; compress prompt or rent — never drop below label-safe length |
| R16 | **Schema / prompt version drift** | Agents write incompatible JSONL | Schema **v2**; bump `prompt_version` on any renderer text change; reject mismatched manifests |
| R17 | **Seed-range collisions across agents** | Train/eval leakage | Only allocate from [`SEED_REGISTRY.md`](SEED_REGISTRY.md) |

## 11. Phased roadmap (deliverable-gated, not time-gated)

- **Phase 0 — Foundations & spike.** *(plumbing largely complete)* Repo scaffold, pinned Catanatron, simulator adapter, trajectory schema, renderer + parser, tiny SFT smoke, eval arena v0, CI. **Exit:** end-to-end loop proven with a small model on CPU.
- **Phase 0.5 — Contract repair & local hardware proof.** *(next — task cards + dependency graph in [`PHASE0_5_TASKS.md`](PHASE0_5_TASKS.md))* Schema v2 + `prompt_version`; canonical parity; MINI fail-loud; `game_key` splits; resume-safe writes; seed registry; eval Gate-B metrics; CI gate tests; teacher-observability audit + assistant-mask test; 8B QLoRA smoke at ≥4096 with VRAM log. **Exit:** T1–T7+T9 merged and T8 report exists (local success or approved rental fallback).
- **Phase 1 — Data engine at scale.** Bulk generation (bot ladder), dataset builder + manifests, Tier A rationales (feature-aware), immutable holdout set, dataset quality report. **Exit:** v1 dataset (≥100k train decisions + separate ≥5k-game holdout) with quality sign-off per [`DATA_CONTRACT.md`](DATA_CONTRACT.md).
- **Phase 2 — SFT model v1.** 8B QLoRA SFT on canonical prompts, eval vs bot ladder, failure taxonomy v1. **Exit:** Stage-1 criteria in §7 / [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).
- **Phase 3 — RL v1.** Written reward/anti-hacking spec → GRPO loop on 8B → first champion checkpoint. Prefer rented GPU for meaningful iterations. **Exit:** ≥ AlphaBeta parity on the pre-registered fixture (≥55% / ≥1k games, Wilson LB > 50%).
- **Phase 4 — Scale & polish.** 12B-class run (rental), self-play iteration loop, optional Tier B commentary, live-spectate UX, docs, blog write-up. **Exit:** champion beats AlphaBeta decisively; reproducible from README; publishable artifacts.

### Feasibility posture (post-review)

| Outcome | Posture |
|---|---|
| Beat Random / WeightedRandom after SFT | Achievable if parity + data quality land |
| Competitive with ValueFunction | Plausible; depends on placement/rare-action coverage |
| Beat pinned AlphaBeta (depth/config locked) via GRPO | Ambitious research target — believable, not promised until Phase-2 gates pass |
| Unspecified “beat AlphaBeta” without a fixture | Not a meaningful target |

## 12. Decisions (locked with the owner)

### Locked 2026-07-30

1. **Repo strategy → split.** Canonical home is `catan-llm` (this repo), not `dataversen`.
2. **Base model → Qwen3-8B** (Apache-2.0, instruct checkpoint). 12B-class step-up deferred to Phase 4 and rental-only.
3. **Compute → owner's RTX 5060 Ti 16GB is the primary box.** QLoRA-first mandatory; full-FT / 12B / serious GRPO behind burst rentals.
4. **Teacher-model commentary → deferred.** Tier A only through Phase 2; revisit Tier B after SFT results.
5. **Trading → later.** Catanatron-native action space only for the initial project.
6. **Endgame → semi-public.** Blog write-up + publishable benchmark artifacts.

### Locked / adopted from post-spike review (2026-07-30)

7. **Experiment tracking → local JSON reports first**; W&B optional later.
8. **Train/play consistency is a hard gate.** One canonical renderer; compact alternate training prompts are forbidden for labeled SFT data.
9. **AlphaBeta claims require a pre-registered fixture** (commit, depth, seats, maps, seeds, metrics) in [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).
10. **No ≥100k dataset build until local 8B QLoRA smoke succeeds** on the 5060 Ti.
11. **Smoke parse/legality ≠ skill.** Promotion uses held-out win-rate / VP / failure taxonomy with fallbacks accounted separately.
12. **Fallback policy → `first_legal`** (engine-ordered). Do not use highest-pip fallback unless a future protocol version changes this.
13. **Trajectory schema → v2** for contract fields (`game_key`, `map_hash`, `prompt_version`, …). Legacy plumbing records labelled v1 are not Phase-1-valid.
14. **SFT context → `max_seq_length ≥ 4096`** for canonical prompts; verify assistant labels are not truncated.
15. **Seed allocation → [`SEED_REGISTRY.md`](SEED_REGISTRY.md) only.**
16. **Docs win over code** until an agent updates both in the same change.
17. **Teacher observability → privileged distillation accepted.** Experts may use full `Game`; learner prompts stay POV-limited; Tier A text is POV-safe only.
18. **Headline eval format → 4-player.** `ab-4p` / `ladder-4p` are primary; 1v1 is secondary diagnostic.
19. **“Beats WeightedRandom” → same-fixture win-share comparison**, not absolute candidate WR > 50% in a 4p table.
20. **Handoff base → merged `main` or explicit `cursor/phase-0-foundations-9ca9`.**

## 13. References

- Catanatron: [github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron), [docs.catanatron.com](https://docs.catanatron.com)
- catan-bench: [github.com/vmmadathil/catan-bench](https://github.com/vmmadathil/catan-bench), [write-up](https://visakhmadathil.com/blog/2026-02-09-can-llms-play-catan)
- HexMachina / Agents of Change: [arXiv:2506.04651](https://arxiv.org/abs/2506.04651)
- GRPO / RLVR background: [arXiv:2503.06639](https://arxiv.org/html/2503.06639v3), [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)
- Fine-tuning VRAM/cost planning: [computecomparison.com Llama fine-tuning guide](https://computecomparison.com/guides/fine-tuning-llama-cost-guide)

## 14. Handoff readiness (for other agents)

### Branch / base of work (critical)

Canonical planning + Phase-0 code currently live on branch
`cursor/phase-0-foundations-9ca9` (PR [#1](https://github.com/Al2800/catan-llm/pull/1)).
**`main` only has the initial scope seed unless/until that PR is merged.**

Before delegating build work:

1. Merge PR #1 into `main`, **or**
2. Explicitly instruct every agent: *base branch = `cursor/phase-0-foundations-9ca9`*.

Do not start Phase 0.5 agents against stale `main`.

### Ready now

- Locked product decisions (model, hardware, non-goals, phases, teacher POV policy)
- Measurable SFT / AlphaBeta gates (with correct 4p “beats WR” definition)
- Companion contracts + Phase 0.5 task cards + `AGENTS.md` + vendored engineering skills

### Conditional go / no-go

| Work | Verdict |
|---|---|
| Phase 0.5 T1, T3, T5 (schema, MINI, resume writes) | **Go** in parallel |
| Phase 0.5 T2/T4/T6/T7/T9 (parity, splits, eval, CI, POV/mask) | **Go** after T1 (see task deps) |
| Phase 0.5 T8 (5060 Ti 8B smoke) | **Go** on owner GPU; blocks scale |
| Phase 1 ≥100k generation / Phase 2–3 train·RL | **No-go** until Phase 0.5 exit |

### Skills setup

Engineering skills are vendored under `.cursor/skills/engineering/`. They are **optional guidance** for coding agents unless `/setup-matt-pocock-skills` has been run (writes `docs/agents/*`). Required for `/to-tickets` / `/triage` workflows; not required for T1–T9 code cards.

## 15. Plan changelog

| Date | Change |
|---|---|
| 2026-07-30 | Initial scope locked (model, hardware, trading, Tier B deferral). |
| 2026-07-30 | Post-spike review → Phase 0.5, DATA_CONTRACT, EVAL_PROTOCOL, ENV_BLACKWELL, QLoRA sketch. |
| 2026-07-30 | Handoff review → schema v2 + `prompt_version`, token budget ≥4096, seed registry, locked `first_legal` fallback, softened G5, Tier A feature list, decoding locks, `AGENTS.md` / Phase 0.5 task cards / `RL_SPEC.md` skeleton, handoff readiness section. |
| 2026-07-30 | Second review → privileged-teacher POV policy; Gate B win-share fix; 4p headline fixtures; 4096 VRAM caveat; assistant-mask + teacher-audit task (T9); Phase-1 cohort plan; branch/base handoff warning; Phase 0.5 dependency graph. |
