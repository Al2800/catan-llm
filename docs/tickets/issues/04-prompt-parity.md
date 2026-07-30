# 04 — Canonical train/play prompt parity

**What to build:** Training chat JSONL is produced by the exact same renderer
functions used at live play. Compact alternate training prompts are removed for
labeled SFT. Every record stores `prompt_version`.

**Blocked by:** 01

**Status:** ready-for-agent

**Phase:** 0.5 (T2)

- [ ] Labeled SFT examples use live `render_system_prompt` / `render_user_prompt`
- [ ] `PROMPT_VERSION` constant exists and is written on records
- [ ] Parity test over ≥20 seeded games: stored/dataset prompts == live renders
- [ ] Bumping renderer text requires bumping `PROMPT_VERSION`
