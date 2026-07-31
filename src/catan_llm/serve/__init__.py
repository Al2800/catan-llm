"""Serving helpers for OpenAI-compatible play (ticket 16)."""

from catan_llm.serve.decoding import ACTION_JSON_SCHEMA, completion_kwargs

__all__ = ["ACTION_JSON_SCHEMA", "completion_kwargs"]
