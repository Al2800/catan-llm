from catan_llm.data.parser import FALLBACK_POLICY
from catan_llm.eval.arena import bot_ladder_arena, run_fixture
from catan_llm.eval.metrics import MatchStats, wilson_interval


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
    assert report["results"]["fallback_policy"] == FALLBACK_POLICY


def test_zero_win_seats_appear():
    stats = MatchStats()
    stats.register_seats(["candidate", "random", "weightedrandom", "valuefunction"])
    stats.add_game("random", turns=40, vp_margin=-1.0)
    stats.add_game("random", turns=40, vp_margin=-2.0)
    summary = stats.summary(candidate="candidate", versus="weightedrandom")
    assert "candidate" in summary["win_rates"]
    assert summary["win_rates"]["candidate"]["wins"] == 0
    assert "candidate,weightedrandom" in summary["win_share_gap"]
    assert summary["vp_margin"] == -1.5
    assert summary["fallback_policy"] == "first_legal"


def test_ladder_4p_fixture():
    report = run_fixture(
        "ladder-4p",
        num_games=4,
        vps_to_win=6,
        candidate_kind="random",
    )
    assert report["fixture"]["format"] == "ladder-4p"
    assert report["fixture"]["seed_range_name"] == "ladder_sft_gate"
    assert report["fixture"]["fallback_policy"] == "first_legal"
    results = report["results"]
    assert results["games"] == 4
    for seat in ("candidate", "random", "weightedrandom", "valuefunction"):
        assert seat in results["win_rates"]
    assert "candidate,weightedrandom" in results["win_share_gap"]
    assert "parse_rate_model" in results
    assert "legality_rate_model" in results
    assert "fallback_rate" in results
    assert "vp_margin" in results


def test_ab_4p_fixture():
    report = run_fixture(
        "ab-4p",
        num_games=2,
        vps_to_win=6,
        candidate_kind="random",
    )
    assert report["fixture"]["format"] == "ab-4p"
    assert report["fixture"]["seed_range_name"] == "champion_ab"
    assert "candidate" in report["results"]["win_rates"]
    assert "alphabeta" in report["results"]["win_rates"]
