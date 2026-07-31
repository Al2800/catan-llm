# Serving + constrained decoding (ticket 16)

OpenAI-compatible serving for `LLMPlayer` after Qwen3.5-9B QLoRA SFT.
Fallback policy stays **`first_legal`** (locked).

## Prerequisites

- Pinned base: `Qwen/Qwen3.5-9B` @ revision in `configs/qwen3.5-9b-qlora.yaml`
- Adapter checkpoint from Phase-2 SFT (or base-only for plumbing smoke)
- GPU: rental ≥24GB for 4-bit 9B (local 16GB train is no-go; inference-only may fit — measure)

## Recommended: vLLM (rental)

```bash
# Example: merge adapter into base first, or pass LoRA modules if your vLLM build supports it.
export MODEL=Qwen/Qwen3.5-9B
export REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REVISION" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --port 8000
```

4-bit / AWQ paths depend on the image; if VRAM is tight on rental, prefer an
AWQ/GPTQ export or bitsandbytes load in a transformers+uvicorn stack (below).

## Fallback stack: transformers + OpenAI-compatible shim

For plumbing without vLLM:

```bash
# Terminal A — serve (rental GPU)
catan-serve --model Qwen/Qwen3.5-9B --port 8000

# Terminal B — one game via endpoint
catan-play-endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B \
  --games 1 --vps 8 --seed 1007
```

`LLMPlayer` posts to `{base_url}/chat/completions` and uses **`first_legal`**
whenever parse/legality fails.

## Constrained / structured JSON

When the server supports it, request JSON-shaped completions:

| Backend | Mechanism | Status |
|---|---|---|
| vLLM | guided decoding / `guided_json` (version-dependent) | preferred when available |
| OpenAI-compatible | `response_format={"type":"json_object"}` | use if server accepts it |
| None | free-form + parser | always works; rely on `first_legal` |

Client helper: `catan_llm.serve.decoding.completion_kwargs()` adds
`response_format` when `structured=True`. If the server rejects it, the play
CLI retries once without constraints (graceful degrade).

JSON schema target (assistant content):

```json
{"action": <int>, "rationale": "<short string>"}
```

Exact guided-json schema wiring is backend-specific; see
`src/catan_llm/serve/decoding.py`.

## Acceptance smoke

1. Start server (vLLM or `catan-serve`).
2. `catan-play-endpoint --games 1` completes a finished game.
3. Report includes `fallback_policy: first_legal` and parse/fallback rates.
4. With `structured=True`, either constrained decoding is used or the log notes
   a graceful degrade.
