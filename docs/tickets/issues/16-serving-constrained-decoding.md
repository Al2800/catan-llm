# 16 — Serving + constrained decoding path

**What to build:** Serve the checkpoint on an OpenAI-compatible endpoint (vLLM
preferred) with constrained/structured JSON when available, and `LLMPlayer`
using `first_legal` fallback.

**Blocked by:** 09 (done)

**Status:** done (2026-07-31) — plumbing + docs; real 9B serve remains rental/vLLM

**Phase:** 2

## Entrypoints

- Docs: [`docs/SERVING.md`](../../SERVING.md)
- `catan-serve --mock` (CPU stub) / vLLM recipe for rental
- `catan-play-endpoint --base-url …` (structured JSON with graceful degrade)
- Helpers: `catan_llm.serve.decoding`, `openai_client`

## Acceptance criteria

- [x] Documented serve command for local 4-bit (or rental) path (`docs/SERVING.md`)
- [x] Constrained decoding enabled when supported; graceful degrade documented
- [x] `LLMPlayer` plays a full game via the endpoint (mock server coverage in CI)
- [x] Fallback policy remains `first_legal`
