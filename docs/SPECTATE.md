# Live spectate (ticket 24)

Watch a served (or mock) LLM play Catan in the terminal, and optionally use
Catanatron’s web UI for board visualization.

## One-command terminal watch (recommended)

```bash
# Terminal A — OpenAI-compatible server
catan-serve --mock --port 8000          # CPU plumbing
# or: vLLM / transformers serve of Qwen3.5-9B + adapter (docs/SERVING.md)

# Terminal B — spectate one game (prints every action; writes replay JSON)
catan-spectate \
  --base-url http://127.0.0.1:8000/v1 \
  --model local-model \
  --watch \
  --vps 8 \
  --seed 1007 \
  --out outputs/spectate/replay.json
```

Human-paced:

```bash
catan-spectate --base-url http://127.0.0.1:8000/v1 --watch --delay 0.25
```

Bot-only dry run (no server):

```bash
catan-spectate --bots-only --watch --seed 7 --vps 6
```

In-process PEFT adapter (rental GPU; no HTTP server):

```bash
# after downloading adapter from AlCampbell/catan-llm-sft-v1 or local train out
catan-spectate --adapter outputs/sft/qwen3.5-9b-qlora/adapter --watch --vps 8
```

Replay artifact: `outputs/spectate/replay.json` (turn, seat, action, VPs, timing).

## Catanatron web UI (full board)

Upstream Catanatron ships a React UI + Flask API (optional `[web]` extra + Docker).
High-level recipe from [bcollazo/catanatron](https://github.com/bcollazo/catanatron):

1. Start their `docker-compose` stack (Postgres + UI on `localhost:3000`).
2. Persist games with `catanatron-play … --db` / `--step-db`, **or** upsert via
   `catanatron.web.utils.ensure_link(game)`.
3. Open the printed `/games/<uuid>/states/…` link.

Wiring an `LLMPlayer` into that Docker path is the same serve endpoint as
`catan-spectate` / `catan-play-endpoint` (`docs/SERVING.md`). Terminal `--watch`
is the zero-Docker default for this repo.

## Training visualization

QLoRA runs write:

| File | Contents |
|---|---|
| `outputs/sft/…/train_report.json` | peak VRAM, step time, mask check |
| `outputs/sft/…/train_history.json` | TRL `log_history` rows |
| `outputs/sft/…/train_history.md` | loss table + sparkline |

## Acceptance

- [x] Documented one-command / short recipe (`catan-spectate --watch`)
- [x] Works with OpenAI-compatible served model (`--base-url`)
- [x] README linked
