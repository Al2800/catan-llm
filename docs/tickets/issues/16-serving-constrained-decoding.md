# 16 — Serving + constrained decoding path

**What to build:** Serve the checkpoint on an OpenAI-compatible endpoint (vLLM
preferred) with constrained/structured JSON when available, and `LLMPlayer`
using `first_legal` fallback.

**Blocked by:** 09

**Status:** blocked

**Phase:** 2

- [ ] Documented serve command for local 4-bit (or rental) path
- [ ] Constrained decoding enabled when supported; graceful degrade documented
- [ ] `LLMPlayer` plays a full game via the endpoint
- [ ] Fallback policy remains `first_legal`
