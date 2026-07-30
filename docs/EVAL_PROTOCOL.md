# Evaluation Protocol v1

Normative evaluation and promotion rules for catan-llm.
Implements SCOPE §4.4, §7 exit criteria, and §11.

Smoke parse/legality alone is **not** skill evidence.

## 1. Pinned fixture identity

Every published number must record:

| Field | Example / rule |
|---|---|
| `catanatron_commit` | `82aae93ab1f7c267218be0566df573ce477ec3d8` (update when bumping) |
| `source_commit` | git SHA of `catan-llm` |
| `map_type` / `map_hash` | usually `BASE` for headline claims |
| `vps_to_win` | `10` for headline; `6`/`8` allowed for fast smoke only |
| `seats` | ordered bot/model kinds + params |
| `alphabeta` | **depth=2**, default value weights, `prunning=false` unless noted |
| `seed_start` / `num_games` | contiguous range from [`SEED_REGISTRY.md`](SEED_REGISTRY.md) |
| `seat_rotation` | rotate seats each game (`game_i % n_seats`) |
| `timeout_turns` | engine `TURNS_LIMIT` (currently 1000); unfinished counted separately |
| `decoding` | temperature (0.0 for headline), max tokens (128), constrained decoding on/off |
| `fallback_policy` | **`first_legal`** (locked — SCOPE §12.12) |
| `prompt_version` | renderer constant used by the candidate |

Headline “vs AlphaBeta” claims use this exact AlphaBeta config unless a report title says otherwise.

## 2. Match formats

### 2.1 Ladder (4-player) — primary SFT gate

Seats (before rotation):

1. `candidate` (LLM checkpoint)
2. `random`
3. `weightedrandom`
4. `valuefunction`

Optional fifth report: replace `valuefunction` with `alphabeta` for a harder ladder (still 4p).

Minimum for Stage-1 “beats WeightedRandom”: **≥200 finished games**, seeds declared in advance.

### 2.2 AlphaBeta head-to-head — primary RL / champion gate

Preferred formats (both reported when possible):

| Format | Seats | Games |
|---|---|---|
| `ab-4p` | candidate + AlphaBeta + ValueFunction + Random | ≥1000 finished |
| `ab-1v1-mirror` | candidate vs AlphaBeta, mirrored seat swap on same seeds | ≥1000 finished pairs (500 seeds × 2 seats) |

Promotion target (SCOPE Stage 2): **≥55% win rate** with **95% Wilson lower bound > 50%** on the pre-registered fixture, plus one independent reproducibility rerun with the same config hash.

## 3. Metrics (required in every report)

| Metric | Definition |
|---|---|
| `games` / `finished` / `unfinished` | counts |
| `win_rate[name]` | wins / finished; include **all seats**, even 0-win |
| `wilson95[name]` | Wilson score interval |
| `avg_turns` | mean turns over finished games |
| `vp_margin` | candidate final VP − best opponent VP (finished only) |
| `parse_rate_model` | parse OK / model calls (**exclude** auto-play single-action turns) |
| `legality_rate_model` | legal chosen action / model calls **before** fallback |
| `fallback_rate` | fallback used / model calls |
| `action_error_hist` | parse/illegal counts by `action_type` when available |

Optional later: Elo across checkpoint pool.

## 4. Accounting rules

1. Auto-played singleton actions (e.g. only `ROLL`) do **not** count as model calls.
2. If fallback plays, the game continues, but the decision counts against `parse_rate_model` / `legality_rate_model` as appropriate and increments `fallback_rate`.
3. Unfinished games do not grant wins; report them explicitly.
4. Do not hide zero-win seats from `win_rates`.

## 5. Promotion gates

### Gate A — Plumbing smoke (Phase 0)

- Small model or stub can finish games without exceptions.
- Useful for CI only.

### Gate B — Stage-1 SFT promotion (Phase 2)

All required:

1. `parse_rate_model ≥ 0.995`
2. `legality_rate_model ≥ 0.995`
3. Ladder vs WeightedRandom: Wilson LB(`candidate`) > 0.50 at ≥200 finished games
4. Failure taxonomy v1 published (top illegal/parse modes)
5. Config hash + seeds recorded

### Gate C — Champion vs AlphaBeta (Phase 3)

All required:

1. Still satisfies Gate B legality/parse floors on the champion fixture
2. Pre-registered AlphaBeta fixture (§2.2)
3. Win rate ≥ 0.55 and Wilson LB > 0.50 over ≥1000 finished games
4. Independent reproducibility rerun agrees (same side of the threshold)
5. No severe reward-hacking flags (turn-length outliers / pass-heavy degeneracy) in the RL report

## 6. Holdout policy

- Training data seeds and eval seeds are disjoint ranges recorded in manifests.
- Champion evaluation seeds are immutable once Phase-3 begins; new seeds require a new protocol version.

## 7. Report artifact shape

Write JSON under `outputs/arena/<run_id>.json`:

```json
{
  "protocol_version": "v1",
  "fixture": { "format": "ladder-4p", "...": "..." },
  "candidate": { "checkpoint": "...", "decoding": {} },
  "results": { "games": 0, "win_rates": {}, "parse_rate_model": 0.0 }
}
```

Human summary may be Markdown, but the JSON is authoritative.
