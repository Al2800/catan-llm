# Catan LLM — Full Project Scope

**Working title:** Catan LLM (training an 8–12B parameter language model to play Settlers of Catan in the Catanatron simulator)

**Status:** Scope reviewed by owner; key decisions locked (see §12). No build work started.

**Repo note:** this project is being split out of `dataversen` into a standalone `catan-llm` repository. This copy remains in the `dataversen` PR until the new repo exists, then the new repo becomes canonical.

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
- **G5.** Model- and infra-portable: configs for QLoRA on a single GPU up to full fine-tuning / RL on a multi-GPU cluster.

### Non-goals (for this project)

- Training a foundation model from scratch (we fine-tune an existing open-weights 8–12B model).
- Human-facing trade negotiation via natural-language chat between LLMs (Catanatron's engine has no domestic player-to-player trading; we stay within the engine's action space. See §10, R6).
- A production product/UI beyond what is needed to run and watch games.
- Beating frontier LLMs (Claude/GPT-class) prompted as agents — interesting comparison, not a success criterion.

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
- Records **full trajectories**: per decision — serialized game state (before action), valid action list, action taken, acting player, dice/events, and final game outcome (winner, VP counts, turn count).
- Deterministic seeding for reproducibility; multiprocessing for throughput; crash-safe chunked writes (JSONL shards).

### 4.2 Data pipeline (`data/`)

- **Trajectory schema v1** (structured): one record per decision.
  ```
  {game_id, decision_idx, seed, map_type, player_color, turn, phase,
   state: {...engine state...}, valid_actions: [...], action_taken,
   expert_policy: "alphabeta|valuefunction|weightedrandom|llm",
   outcome: {winner, vps, turns}}
  ```
- **Renderers** (unstructured): deterministic templates that turn state into a compact natural-language board/hand/development description (the same rendering the LLM will see at play time — train/test consistency is critical).
- **Rationale generation** (unstructured, two tiers):
  - *Tier A (free):* template rationales derived from the expert bot's own features (e.g., value-function deltas: "this settlement maximizes ore+wheat pips and blocks BLUE's road network").
  - *Tier B (sampled):* a frontier teacher model annotates a small subset (e.g., 5–10%) with richer strategic commentary — used for reasoning-style SFT and for the dataset-quality story.
- **Dataset builder:** dedup, filter (drop corrupted/unfinished games), balance (by phase: initial placement / early / mid / late game; by action type so rare actions like `PLAY_MONOPOLY` aren't swamped), split by `game_id` to prevent leakage, and emit chat-format JSONL (system/user/assistant) plus the raw structured records for RL.
- **Versioning:** every dataset carries a manifest (generator versions, bot mix, seeds, counts, checksums).

### 4.3 Training (`training/`)

Two stages (details in §7), driven by config files:

- **SFT:** behavior-clone expert decisions (+ Tier A/B rationales) into an 8–12B instruct model. QLoRA path for iteration; LoRA/full-FT configs for scale.
- **RL (GRPO with verifiable rewards):** decision-level prompts with group rollouts scored by the engine (legality, immediate VP/shape heuristics) plus full-game outcome rewards (win, VP margin). Framework target: TRL or verl with vLLM rollouts.

### 4.4 Evaluation arena (`eval/`)

- Plays N-game matches of a checkpoint vs fixed opponents (`Random`, `WeightedRandom`, `ValueFunction`, `AlphaBeta`) and vs other checkpoints (round-robin), seeded, 4-player and 1v1 formats.
- Metrics: win rate with confidence intervals, VP margin, turns-to-win, action legality/parse-failure rate, per-action-type error breakdown, Elo across the checkpoint pool.
- Regression gate: a checkpoint must not lose ground on legality and must beat the previous champion by a margin before promotion.
- Baselines to contextualize: catan-bench's prompt-only results and HexMachina's ~54% vs AlphaBeta.

### 4.5 Serving + live play (`serving/`, `play/`)

- `LLMPlayer`: implements Catanatron's `Player.decide(game, playable_actions)` by rendering state (same renderer as training), querying an OpenAI-compatible endpoint (vLLM server hosting the checkpoint), parsing the chosen action, with constrained output (JSON schema / grammar) and a safe fallback (highest-pip legal action) on parse failure.
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

- **Leakage control:** splits by game/seed, never by decision.
- **Curriculum:** start on `MINI` maps / lower `vps_to_win` (shorter games, cleaner signal), graduate to full `BASE` maps — Catanatron supports this natively via Gym config.
- **Class balance:** Catan decisions are dominated by `ROLL`/build/end-turn; rare high-leverage actions (dev cards, monopoly, year-of-plenty, 4:1/3:1 maritime trades, robber placement) are oversampled.
- **Train/play consistency:** one canonical state renderer + action indexing used by data gen, training, and live play alike.

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

### Stage 0 — Spike (prove the loop)

- Tiny run: a small instruct model (e.g., 1–3B) through SFT on ~10k expert decisions → plays legal moves vs `RandomPlayer`. Validates schema, renderer, parser, eval harness before real compute.

### Stage 1 — SFT (behavior cloning)

- Data: `expert-trajectories` + `expert-commentary` (assistant = short rationale + chosen action in strict JSON).
- Loss on assistant tokens only; 2–3 epochs; cosine LR ~1e-5 (full) / ~1e-4 (LoRA).
- Exit criteria: ≥99% parse+legality rate on holdout; beats `WeightedRandomPlayer` convincingly; competitive with `ValueFunctionPlayer`.

### Stage 2 — RL (GRPO with verifiable rewards)

- **Prompts:** decision points sampled from games (mix of bot-generated states and on-policy states).
- **Rollouts:** G=8–16 completions per decision via vLLM; temperature ~0.8–1.0.
- **Rewards (all engine-verifiable):**
  - Format/legality: parseable + legal action (gates everything else).
  - Outcome: win/loss and VP margin for the full game containing the decision (group-shared credit assignment — simple and effective at this scale).
  - Shaping (small weights): immediate VP gain, avoiding hand >7 before a roll (discard risk), longest-road/army threats — all computable from state.
  - Penalties: illegal action, parse failure, degenerate repetition.
- **Stability:** KL penalty to the SFT reference policy; standard GRPO clipping.
- **Self-play loop (optional phase 2b):** refresh opponent pool with recent checkpoints; regenerate `self-play-rollouts`; iterate.
- **Hardware path:** small-scale GRPO is feasible on the local 16GB card (QLoRA policy + vLLM colocate mode, small group sizes, short sequences) but slow; meaningful RL iterations assume a burst-rented A100/H100 (see §9).
- Exit criteria: beats `AlphaBetaPlayer` head-to-head (target ≥55% over ≥1k games, CI-checked), then enters the champion/challenger pool.

### What we deliberately avoid (for now)

- PPO with a learned value model (GRPO removes the critic — roughly half the memory, cleaner with binary-ish game rewards).
- Training on full-game single-context rollouts initially (context bloat); decision-level training first, full-game trajectories as a later extension.

## 8. Evaluation plan

| Metric | Definition | Target |
|---|---|---|
| Win rate vs bot ladder | Seeded N-game matches vs Random / WeightedRandom / ValueFunction / AlphaBeta | ≥ AlphaBeta parity, then exceed |
| Elo (checkpoint pool) | Round-robin between all checkpoints + bots | monotonic improvement per iteration |
| Legality & parse rate | % decisions with parseable, engine-legal action | ≥ 99.5% (fallback covers the rest) |
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

**Toolchain note:** the 5060 Ti is a Blackwell card (sm_120) — it needs recent PyTorch (cu128+ builds) and a current vLLM; these get pinned in the env setup. The 16GB VRAM ceiling is the binding local constraint: no full fine-tuning and no 12B training locally; both are rental-only paths.

Data generation itself is CPU-only and cheap; the GPU budget is dominated by RL rollouts, which is why Stage 2 is scoped at 8B first.

### 9.2 Software stack

- Python 3.11+, `catanatron` + `catanatron_gym` (pinned), `gymnasium`.
- Training: PyTorch, transformers, TRL (SFT + GRPO), PEFT (LoRA/QLoRA), bitsandbytes; optional verl/OpenRLHF for scaled RL; DeepSpeed/FSDP for full-FT.
- Serving: vLLM (OpenAI-compatible endpoint).
- Storage: datasets as JSONL/Parquet shards + manifests (DVC or plain checksums); checkpoints on object storage.
- Experiment tracking: W&B or MLflow (decision in Phase 0).
- CI: unit tests for schema/renderer/parser, fast smoke tests (mini games on CPU).

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
| R9 | **Catanatron API drift** | Breakage | Pin versions; thin adapter layer isolates the rest of the codebase |
| R10 | **Partial observability handled wrong** (opponents' hands hidden) | Unfair/incorrect training signal | Renderer hides hidden info exactly as catan-bench does; verifier in CI |
| R11 | **16GB VRAM ceiling on the local card** | No local full-FT, 12B, or heavy RL | 8B-first + QLoRA is the default path; burst rentals gated on Stage-1 success; Blackwell (sm_120) toolchain pinned (cu128+ PyTorch, current vLLM) |

## 11. Phased roadmap (deliverable-gated, not time-gated)

- **Phase 0 — Foundations & spike.** Repo scaffold (`catan-llm/` package), pinned deps, simulator adapter, trajectory schema, canonical renderer + action parser, tiny SFT smoke run, eval arena v0 (bot-vs-bot), CI. **Exit:** end-to-end loop proven with a small model on CPU/single GPU.
- **Phase 1 — Data engine at scale.** Bulk generation (bot ladder), dataset builder + manifests, Tier A rationales, holdout set, dataset quality report. **Exit:** v1 dataset (≥100k decisions) with quality sign-off.
- **Phase 2 — SFT model v1.** 8B QLoRA SFT, eval vs bot ladder, failure taxonomy v1. **Exit:** passes Stage-1 exit criteria.
- **Phase 3 — RL v1.** GRPO loop on 8B, reward stack, first champion checkpoint. **Exit:** ≥ AlphaBeta parity in arena.
- **Phase 4 — Scale & polish.** 12B-class run (rental), self-play iteration loop, optional teacher-model commentary subset, live-spectate UX, docs, write-up for the owner's blog. **Exit:** champion beats AlphaBeta decisively; reproducible from README; benchmark artifacts (charts, game logs, Elo table) ready to publish.

## 12. Decisions (locked with the owner, 2026-07-30)

1. **Repo strategy → split.** The project lives in its own `catan-llm` repository, not inside `dataversen`. The `dataversen` copy of this document is a staging copy until the new repo is created.
2. **Base model → Qwen3-8B** (Apache-2.0, instruct checkpoint). 12B-class step-up deferred to Phase 4 and rental-only.
3. **Compute → owner's RTX 5060 Ti 16GB is the primary box.** This makes QLoRA-first mandatory (not just preferred), keeps SFT + data + eval + serving at ~zero marginal cost, and puts full-FT / 12B / serious GRPO behind small burst rentals.
4. **Teacher-model commentary → deferred.** Tier A (free template rationales) only through Phase 2; revisit Tier B (frontier-annotated subset) once SFT results are in.
5. **Trading → later.** Strictly Catanatron's native action space for the initial project; the negotiation layer is a documented future extension.
6. **Endgame → semi-public.** Results and write-up go on the owner's blog; keep benchmark artifacts (charts, game logs, Elo tables) publishable from the start.

## 13. References

- Catanatron: [github.com/bcollazo/catanatron](https://github.com/bcollazo/catanatron), [docs.catanatron.com](https://docs.catanatron.com)
- catan-bench: [github.com/vmmadathil/catan-bench](https://github.com/vmmadathil/catan-bench), [write-up](https://visakhmadathil.com/blog/2026-02-09-can-llms-play-catan)
- HexMachina / Agents of Change: [arXiv:2506.04651](https://arxiv.org/abs/2506.04651)
- GRPO / RLVR background: [arXiv:2503.06639](https://arxiv.org/html/2503.06639v3), [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)
- Fine-tuning VRAM/cost planning: [computecomparison.com Llama fine-tuning guide](https://computecomparison.com/guides/fine-tuning-llama-cost-guide)
