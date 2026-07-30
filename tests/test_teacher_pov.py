"""Teacher POV audit + Tier A leakage gates (SCOPE §5.1 / ticket 07).

Audit summary (see also docs/TEACHER_POV.md):
- Teachers may use the full engine Game to choose actions (privileged distillation).
- Learner prompts stay POV-limited (own hand only; opponent card counts).
- Tier A rationale text must be POV-safe (no opponent private-hand literals).
"""

from pathlib import Path

import pytest

from catan_llm.data.dataset import decision_to_chat, tier_a_rationale
from catan_llm.data.pov import (
    assert_state_pov_safe,
    assert_tier_a_pov_safe,
    assert_user_prompt_pov_safe,
    find_tier_a_leaks,
)
from catan_llm.sim.adapter import play_one


def test_audit_doc_exists():
    path = Path(__file__).resolve().parents[1] / "docs" / "TEACHER_POV.md"
    text = path.read_text(encoding="utf-8")
    assert "privileged distillation" in text.lower()
    assert "POV-limited" in text
    assert "Tier A" in text


def test_leak_detector_flags_opponent_hand_literals():
    bad = "steal from BLUE hand=[W=2,B=1] then road"
    assert find_tier_a_leaks(bad)
    with pytest.raises(AssertionError, match="private-hand"):
        assert_tier_a_pov_safe(bad)

    bad2 = "ORANGE has W=3 H=2 so block the wheat"
    assert find_tier_a_leaks(bad2)

    # Card counts are allowed.
    good = "robber on 6-pip wheat; steal from ORANGE (7 cards)"
    assert find_tier_a_leaks(good) == []
    assert_tier_a_pov_safe(good)


def test_live_records_are_pov_safe_and_tier_a_clean():
    result = play_one(
        ["alphabeta", "random", "weightedrandom", "valuefunction"],
        seed=11,
        vps_to_win=6,
    )
    assert result.records
    for record in result.records:
        assert_state_pov_safe(record.state, record.player_color)
        assert_user_prompt_pov_safe(record.user_prompt, record.player_color)
        rationale = tier_a_rationale(record)
        assert_tier_a_pov_safe(rationale)
        chat = decision_to_chat(record)
        assert_tier_a_pov_safe(chat["messages"][-1]["content"])


def test_injected_leak_fails_gate():
    """Gate must fail closed if a future Tier A template leaks hands."""
    with pytest.raises(AssertionError):
        assert_tier_a_pov_safe("valuefunction copies BLUE hand=[W=1,S=2,H=1]")
