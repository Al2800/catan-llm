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
| `map_type` / `map_hash` | **BASE** for headline claims (record hash) |
| `vps_to_win` | `10` for headline; `6`/`8` allowed for fast smoke only |
| `seats` | ordered bot/model kinds + params |
| `alphabeta` | **depth=2**, default value weights, `prunning=false` unless noted |
| `seed_range_name` | from [`SEED_REGISTRY.md`](SEED_REGISTRY.md) |
| `seed_start` / `num_games` | contiguous range used |
| `seat_rotation` | rotate seats each game (`game_i % n_seats`) |
| `timeout_turns` | engine `TURNS_LIMIT` (currently 1000); unfinished counted separately |
| `decoding` | temperature (0.0 for headline), max tokens (128), constrained decoding on/off |
| `fallback_policy` | **`first_legal`** (locked — SCOPE §12.12) |
| `prompt_version` | renderer constant used by the candidate |

Headline claims must pre-register the full fixture **before** the training run they evaluate.

## 2. Match formats

### 2.1 `ladder-4p` — primary SFT gate (headline)

Seats (before rotation):

1. `candidate` (LLM checkpoint)
2. `random`
3. `weightedrandom`
4. `valuefunction`

Map: BASE, `vps_to_win=10`, seeds from `ladder_sft_gate`.

Minimum: **≥200 finished games**.

### 2.2 `ab-4p` — primary RL / champion gate (headline)

Seats (before rotation):

1. `candidate`
2. `alphabeta` (depth=2)
3. `valuefunction`
4. `random`

Map: BASE, `vps_to_win=10`, seeds from `champion_ab` (rerun on `champion_ab_rerun`).

Minimum: **≥1000 finished games**.

### 2.3 Secondary diagnostics (optional)

| Format | Use |
|---|---|
| `ladder-4p-hard` | Replace `valuefunction` with `alphabeta` in the SFT ladder |
| `ab-1v1-mirror` | candidate vs AlphaBeta, seat-swapped pairs — **diagnostic only**; not the headline Catan claim (2p is non-standard) |
| `wr-1v1-mirror` | candidate vs WeightedRandom pairs — optional extra evidence |

## 3. Metrics (required in every report)

| Metric | Definition |
|---|---|
| `games` / `finished` / `unfinished` | counts |
| `win_rate[name]` | wins / finished; include **all seats**, even 0-win |
| `wilson95[name]` | Wilson score interval on `win_rate[name]` |
| `win_share_gap[a,b]` | `win_rate[a] - win_rate[b]` in the **same** finished games |
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
5. Never interpret absolute candidate win rate > 50% in a 4-player table as “beats WeightedRandom.”

## 5. Promotion gates

### Gate A — Plumbing smoke (Phase 0)

- Small model or stub can finish games without exceptions.
- Useful for CI only.

### Gate B — Stage-1 SFT promotion (Phase 2)

All required on fixture `ladder-4p`:

1. `parse_rate_model ≥ 0.995`
2. `legality_rate_model ≥ 0.995`
3. **Beats WeightedRandom (same-fixture):** `win_rate[candidate] > win_rate[weightedrandom]` over ≥200 finished games. Report both rates + `win_share_gap[candidate,weightedrandom]`. (Stretch: bootstrap/CI on the gap > 0.)
4. Failure taxonomy v1 published (top illegal/parse modes)
5. Full fixture identity (§1) recorded

“Competitive with ValueFunction” is qualitative for Stage-1: candidate win share should not collapse to ~Random levels vs VF; exceeding VF is **not** required to promote.

### Gate C — Champion vs AlphaBeta (Phase 3)

All required on fixture `ab-4p` (headline):

1. Still satisfies Gate B legality/parse floors on the champion fixture
2. Pre-registered AlphaBeta fixture (§2.2) with map hash + seeds declared before training
3. `win_rate[candidate] ≥ 0.55` and Wilson LB > 0.50 over ≥1000 finished games
4. Independent reproducibility rerun on `champion_ab_rerun` agrees (same side of the threshold)
5. No severe reward-hacking flags (turn-length outliers / pass-heavy degeneracy) in the RL report

Secondary `ab-1v1-mirror` may be published alongside but **does not** substitute for Gate C.

## 6. Holdout policy

- Training data seeds and eval seeds are disjoint ranges from [`SEED_REGISTRY.md`](SEED_REGISTRY.md).
- Champion evaluation seeds are immutable once Phase-3 begins; new seeds require a new protocol version.

## 7. Report artifact shape

Write JSON under `outputs/arena/<run_id>.json`:

```json
{
  "protocol_version": "v1",
  "fixture": {
    "format": "ladder-4p",
    "seed_range_name": "ladder_sft_gate",
    "map_type": "BASE",
    "map_hash": "...",
    "fallback_policy": "first_legal"
  },
  "candidate": { "checkpoint": "...", "prompt_version": "...", "decoding": {} },
  "results": {
    "games": 0,
    "finished": 0,
    "win_rates": {},
    "win_share_gap": {"candidate_vs_weightedrandom": 0.0},
    "parse_rate_model": 0.0,
    "legality_rate_model": 0.0,
    "fallback_rate": 0.0
  }
}
```

Human summary may be Markdown, but the JSON is authoritative.
