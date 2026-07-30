# Issue tracker: Local Markdown (docs/tickets)

Issues for this repo live as markdown under [`docs/tickets/`](../tickets/).

GitHub Issues are not created automatically from this cloud environment (`gh` is
read-only here). The committed backlog is the source of truth until someone
mirrors tickets to GitHub Issues manually if desired.

## Conventions

- Index: [`docs/tickets/BACKLOG.md`](../tickets/BACKLOG.md)
- One file per ticket: `docs/tickets/issues/<NN>-<slug>.md`, numbered from `01`
- Status line uses: `ready-for-agent` | `claimed` | `done` | `blocked` | `wontfix`
- Blocking edges: `Blocked by:` near the top (ticket numbers)
- Frontier = open tickets whose blockers are all `done`

## Normative parents

- [`docs/SCOPE.md`](../SCOPE.md)
- [`docs/PHASE0_5_TASKS.md`](../PHASE0_5_TASKS.md)
- [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md)
- [`docs/EVAL_PROTOCOL.md`](../EVAL_PROTOCOL.md)
