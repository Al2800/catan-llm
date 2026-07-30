# 04 — Canonical train/play prompt parity

**What to build:** Training chat JSONL is produced by the exact same renderer
functions used at live play. Compact alternate training prompts are removed for
labeled SFT. Every record stores `prompt_version`.

**Blocked by:** 01

**Status:** done

**Phase:** 0.5 (T2)

- [x] Labeled SFT examples use live `render_system_prompt` / `render_user_prompt`
- [x] `PROMPT_VERSION` constant exists and is written on records
- [x] Parity test over ≥20 seeded games: stored/dataset prompts == live renders
- [x] Bumping renderer text requires bumping `PROMPT_VERSION`
