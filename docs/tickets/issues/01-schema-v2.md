# 01 — Schema v2 + identity fields

**What to build:** Every new trajectory decision record carries contract v2
identity fields so splits, manifests, and prompt parity are reproducible.
Legacy v1 plumbing rows are rejected on Phase-1 dataset paths.

**Blocked by:** None — can start immediately.

**Status:** done

**Phase:** 0.5 (T1)

- [x] Records include `schema_version="v2"`, `game_key`, `map_hash`, `bot_config`, `bot_config_hash`, `prompt_version`, `catanatron_commit`, `source_commit`
- [x] Unit tests cover serialize/deserialize round-trip
- [x] Phase-1 dataset builder rejects non-v2 rows
- [x] Docs and code agree on schema v2 in the same commit
