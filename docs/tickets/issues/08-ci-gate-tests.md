# 08 — CI gate tests

**What to build:** Continuous integration fails when contract gates regress
(parity, MINI, schema v2, no-truncation, POV rationales, assistant mask). Agents
see red, not prose.

**Blocked by:** 02, 04, 05, 06, 07

**Status:** ready-for-agent

**Phase:** 0.5 (T7)

- [ ] Parity / MINI / schema-v2 / no-truncation / POV / mask tests run in CI
- [ ] Default CI still avoids downloading 8B models
- [ ] Any remaining unfinished checks use `xfail(strict=True)` only with a ticket id
