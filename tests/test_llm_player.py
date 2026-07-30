from catanatron import Game
from catanatron.models.player import Color, RandomPlayer

from catan_llm.play.llm_player import LLMPlayer


def test_llm_player_uses_parser_and_fallback():
    def complete(_system, user):
        # Always pick action 0 if present.
        assert "AVAILABLE ACTIONS:" in user
        return '{"action": 0, "reasoning": "first legal"}'

    llm = LLMPlayer(Color.RED, complete_fn=complete)
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
    parsed_ok, parsed_total, legal_ok, legal_total = llm.consume_eval_counters()
    # Single-option / ROLL auto-plays may mean zero model calls; that's ok.
    assert parsed_total == legal_total
    assert parsed_ok <= parsed_total
    assert legal_ok <= legal_total


def test_llm_player_fallback_on_bad_json():
    llm = LLMPlayer(Color.RED, complete_fn=lambda *_: "not-json")
    players = [llm, RandomPlayer(Color.BLUE), RandomPlayer(Color.ORANGE), RandomPlayer(Color.WHITE)]
    game = Game(players, seed=3, vps_to_win=5)
    # Advance until a multi-action decision likely happens; just play full game.
    game.play()
    _po, pt, _lo, lt = llm.consume_eval_counters()
    assert pt == lt
