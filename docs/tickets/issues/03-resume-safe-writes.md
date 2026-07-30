# 03 — Resume-safe shard writes

**What to build:** Trajectory generation can crash and resume without deleting
completed work. Writes are atomic or append-only with a completed-`game_key` journal.

**Blocked by:** None — can start immediately.

**Status:** done

**Phase:** 0.5 (T5)

- [x] Generation does not `unlink()` completed outputs at job start
- [x] Atomic rename and/or append-only shards + journal of completed `game_key`s
- [x] Documented resume flag / behavior in CLI help or PHASE0 notes
