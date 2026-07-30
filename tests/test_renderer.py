from catanatron import Game
from catanatron.models.player import Color, RandomPlayer

from catan_llm.data.renderer import (
    compact_state_dict,
    render_system_prompt,
    render_user_prompt,
    serialize_actions,
)


def test_renderer_hides_opponent_hands():
    players = [RandomPlayer(c) for c in (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)]
    game = Game(players, seed=7)
    # Play a few ticks so hands are non-trivial eventually; even at start structure holds.
    for _ in range(8):
        if game.winning_color() is not None:
            break
        game.play_tick()

    system = render_system_prompt(game, Color.RED)
    assert "BOARD LAYOUT" in system
    assert "RED" in system

    user = render_user_prompt(game, Color.RED, game.playable_actions)
    assert "AVAILABLE ACTIONS:" in user
    assert "(YOU)" in user
    # Opponent private resources should not appear as hand=[...]
    for line in user.splitlines():
        if line.strip().startswith("BLUE") or "BLUE:" in line:
            assert "hand=" not in line

    state = compact_state_dict(game, Color.RED)
    you = next(p for p in state["players"] if p["color"] == "RED")
    opp = next(p for p in state["players"] if p["color"] == "BLUE")
    assert "hand" in you
    assert "hand" not in opp
    assert "cards" in opp

    actions_txt = serialize_actions(game.playable_actions)
    assert actions_txt.startswith("AVAILABLE ACTIONS:")
