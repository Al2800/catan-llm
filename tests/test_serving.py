"""Ticket 16: structured decoding helpers + mock endpoint play."""

from __future__ import annotations

import threading
import time
from http.server import ThreadingHTTPServer

from catanatron.models.player import Color, RandomPlayer

from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.eval.arena import SeatSpec, run_match
from catan_llm.play.llm_player import LLMPlayer
from catan_llm.serve.decoding import ACTION_JSON_SCHEMA, completion_kwargs
from catan_llm.serve.mock_server import _Handler
from catan_llm.serve.openai_client import chat_complete


def test_fallback_policy_locked():
    assert FALLBACK_POLICY == "first_legal"


def test_completion_kwargs_structured_and_plain():
    body = completion_kwargs(structured=True, backend="openai")
    assert body["response_format"]["type"] == "json_object"
    vllm = completion_kwargs(structured=True, backend="vllm")
    assert vllm["guided_json"] == ACTION_JSON_SCHEMA
    plain = completion_kwargs(structured=False, backend="none")
    assert "response_format" not in plain


def test_mock_endpoint_one_game():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    base = f"http://127.0.0.1:{port}/v1"

    def complete(system: str, user: str) -> str:
        text, meta = chat_complete(
            base_url=base,
            model="mock",
            system=system,
            user=user,
            structured=True,
            backend="openai",
        )
        assert meta["structured_applied"] is True
        return text

    llm = LLMPlayer(Color.RED, complete_fn=complete, model="mock")
    seats = [
        SeatSpec(name="llm", kind="llm", player=llm),
        SeatSpec(name="random", kind="random", player=RandomPlayer(Color.BLUE)),
        SeatSpec(name="random2", kind="random", player=RandomPlayer(Color.ORANGE)),
        SeatSpec(name="random3", kind="random", player=RandomPlayer(Color.WHITE)),
    ]
    stats = run_match(
        seats, num_games=1, seed=42, vps_to_win=6, candidate_name="llm", versus_name="random"
    )
    summary = stats.summary(candidate="llm", versus="random")
    assert summary["games"] == 1
    assert summary["finished"] == 1
    assert summary["fallback_policy"] == "first_legal"
    httpd.shutdown()
