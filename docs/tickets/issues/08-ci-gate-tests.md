# 08 — CI gate tests

**What to build:** Continuous integration fails when contract gates regress
(parity, MINI, schema v2, no-truncation, POV rationales, assistant mask). Agents
see red, not prose.

**Blocked by:** 02, 04, 05, 06, 07

**Status:** done

**Phase:** 0.5 (T7)

- [x] Parity / MINI / schema-v2 / no-truncation / POV / mask tests run in CI
- [x] Default CI still avoids downloading 8B models
- [x] Any remaining unfinished checks use `xfail(strict=True)` only with a ticket id

**Notes:** Explicit gate pytest step in `.github/workflows/ci.yml`. Qwen 8B
one-batch is a clear `pytest.skip` pending ticket 09 (not a silent pass).
