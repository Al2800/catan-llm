# 06 — Eval Gate-B metrics + 4p fixtures

**What to build:** Arena reports the metrics required for Stage-1 promotion,
including same-fixture candidate vs WeightedRandom win-share gap. Headline
fixtures `ladder-4p` and `ab-4p` are selectable.

**Blocked by:** 01

**Status:** ready-for-agent

**Phase:** 0.5 (T6)

- [ ] Reports `parse_rate_model`, `legality_rate_model`, `fallback_rate`, `vp_margin`
- [ ] Reports `win_share_gap[candidate,weightedrandom]`
- [ ] Fallback policy constant is `first_legal`
- [ ] Zero-win seats still appear in `win_rates`
- [ ] `ladder-4p` and `ab-4p` fixtures can be run from the CLI
