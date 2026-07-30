from catan_llm.eval.arena import bot_ladder_arena
from catan_llm.eval.metrics import wilson_interval


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_bot_ladder_smoke():
    report = bot_ladder_arena(
        num_games=4,
        seed=1,
        vps_to_win=6,
        include_alphabeta=False,
    )
    assert report["results"]["games"] == 4
    assert report["results"]["finished"] >= 1
    assert "win_rates" in report["results"]
