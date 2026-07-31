# 18 — Failure taxonomy v1

**What to build:** From Gate B (and holdout) logs, publish per-action-type
parse/illegal breakdown and top failure modes to drive the next data/training fix.

**Blocked by:** 17 (final sign-off waits on Gate B report; tooling ready now)

**Status:** ready-for-agent (scaffolding landed; fill from Gate B JSON after 17)

**Phase:** 2

## Entrypoints

- Arena now records `action_error_hist` / `phase_error_hist` from `LLMPlayer`
- `catan-taxonomy --report outputs/arena/gate_b_ladder4p.json`
- Writes `docs/reports/failure_taxonomy_v1.{json,md}`

```bash
catan-gate-b --adapter … --games 200 --out outputs/arena/gate_b_ladder4p.json
catan-taxonomy --report outputs/arena/gate_b_ladder4p.json
```

- [ ] Taxonomy artifact checked in (from real Gate B run)
- [ ] Top failure modes linked to proposed data or decoding fixes
- [ ] Stage-1 / Phase-2 exit explicitly signed
