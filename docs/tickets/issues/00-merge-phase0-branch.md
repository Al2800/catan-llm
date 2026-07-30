# 00 — Merge Phase-0 branch to main (or pin agents to it)

**What to build:** Make the current plan + Phase-0 code the default base for every
agent by landing it on `main` and requiring subsequent work on `main` only.

**Blocked by:** None.

**Status:** done

**Phase:** Meta

- [x] PR #1 reviewed and merged to `main`
- [x] New agents cloning the default branch see Phase 0.5 docs (`DATA_CONTRACT`, tickets, etc.)
- [x] BACKLOG.md / AGENTS.md / SCOPE.md updated: work on `main` only (no long-lived feature branches)

**Done notes:** Merged `cursor/phase-0-foundations-9ca9` → `main` (PR [#1](https://github.com/Al2800/catan-llm/pull/1)). Owner policy: do not use separate branches going forward.
