"""LLMPlayer action_error_hist feeds Gate B taxonomy."""

from __future__ import annotations

from catanatron.models.player import Color, RandomPlayer

from catan_llm.eval.arena import SeatSpec, run_match
from catan_llm.play.llm_player import LLMPlayer


def test_bad_completions_populate_action_error_hist():
    def complete(_system: str, _user: str) -> str:
        return "not-json-at-all"

    llm = LLMPlayer(Color.RED, complete_fn=complete, model="stub")
    seats = [
        SeatSpec(name="candidate", kind="llm", player=llm),
        SeatSpec(name="random", kind="random", player=RandomPlayer(Color.BLUE)),
        SeatSpec(name="random2", kind="random", player=RandomPlayer(Color.ORANGE)),
        SeatSpec(name="random3", kind="random", player=RandomPlayer(Color.WHITE)),
    ]
    stats = run_match(
        seats,
        num_games=1,
        seed=3,
        vps_to_win=6,
        candidate_name="candidate",
        versus_name="random",
    )
    summary = stats.summary(candidate="candidate", versus="random")
    assert summary["fallback_rate"] and summary["fallback_rate"] > 0
    assert summary["action_error_hist"].get("json_parse_failed", 0) > 0
