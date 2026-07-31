"""CLI: serve OpenAI-compatible endpoint (mock or print vLLM recipe)."""

from __future__ import annotations

import click


@click.command()
@click.option("--mock", is_flag=True, help="Run CPU mock server for plumbing tests")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option(
    "--model",
    default="Qwen/Qwen3.5-9B",
    show_default=True,
    help="Model id (documentation / future real serve)",
)
def main(mock: bool, host: str, port: int, model: str) -> None:
    """Ticket 16 serve entrypoint.

    Real 9B serving should use vLLM on a rental GPU — see docs/SERVING.md.
    ``--mock`` starts a tiny OpenAI-compatible stub for CI / laptop wiring.
    """
    if mock:
        from catan_llm.serve.mock_server import serve_forever

        # Paths under /v1/... — LLMPlayer appends /chat/completions to base_url.
        click.echo(f"Starting mock server; use --base-url http://{host}:{port}/v1")
        serve_forever(host=host, port=port)
        return

    click.echo("Real GPU serve is documented in docs/SERVING.md (vLLM preferred).")
    click.echo("Example:")
    click.echo(
        "  python -m vllm.entrypoints.openai.api_server "
        f"--model {model} --trust-remote-code --max-model-len 4096 --port {port}"
    )
    click.echo("For local plumbing without a GPU, re-run with --mock.")


if __name__ == "__main__":
    main()
