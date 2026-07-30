from catanatron import Game
from catanatron.models.player import Color, RandomPlayer

from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.play.llm_player import LLMPlayer


def test_llm_player_uses_parser_and_fallback():
    def complete(_system, user):
        # Always pick action 0 if present.
        assert "AVAILABLE ACTIONS:" in user
        return '{"action": 0, "reasoning": "first legal"}'

    llm = LLMPlayer(Color.RED, complete_fn=complete)
    assert llm.fallback_policy == FALLBACK_POLICY == "first_legal"
    players = [
        llm,
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.ORANGE),
        RandomPlayer(Color.WHITE),
    ]
    game = Game(players, seed=2, vps_to_win=6)
    winner = game.play()
    # Game should complete without raising.
    assert winner is None or winner in game.state.colors
    counters = llm.consume_eval_counters()
    # Single-option / ROLL auto-plays may mean zero model calls; that's ok.
    assert counters["parse_total"] == counters["legal_total"]
    assert counters["parse_ok"] <= counters["parse_total"]
    assert counters["legal_ok"] <= counters["legal_total"]
    assert counters["fallback_count"] <= counters["parse_total"]


def test_llm_player_fallback_on_bad_json():
    llm = LLMPlayer(Color.RED, complete_fn=lambda *_: "not-json")
    players = [llm, RandomPlayer(Color.BLUE), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
    game = Game(players, seed=3, vps_to_win=5)
    game.play()
    counters = llm.consume_eval_counters()
    assert counters["parse_total"] == counters["legal_total"]
    if counters["parse_total"] > 0:
        assert counters["fallback_count"] == counters["parse_total"]
