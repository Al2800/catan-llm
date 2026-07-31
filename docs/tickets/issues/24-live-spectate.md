# 24 — Live spectate UX

**What to build:** Make matches watchable live via Catanatron’s web UI (or a
thin wrapper CLI `--watch`) using the served checkpoint.

**Blocked by:** 16 (done)

**Status:** done (2026-07-31)

**Phase:** 4 (pulled forward — serving plumbing already landed)

## Entrypoints

- `catan-spectate --watch` → terminal action stream + `outputs/spectate/replay.json`
- Docs: [`docs/SPECTATE.md`](../../SPECTATE.md) (incl. Catanatron Docker UI recipe)
- Training curves: `train_history.md` / `train_history.json` from `catan-qlora-train`

## Acceptance criteria

- [x] Documented one-command or short recipe to spectate a game
- [x] Works with the OpenAI-compatible served model
- [x] README updated
