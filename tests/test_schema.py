from catan_llm.data.schema import DecisionRecord, ExpertPolicy, GameOutcome


def test_decision_record_roundtrip():
    record = DecisionRecord(
        game_id="g1",
        decision_idx=0,
        seed=1,
        player_color="RED",
        turn=0,
        phase="BUILD_INITIAL_SETTLEMENT",
        state={"turn": 0},
        valid_actions=[{"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3}],
        action_taken={"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3},
        action_index=0,
        expert_policy=ExpertPolicy.RANDOM,
        outcome=GameOutcome(winner="RED", vps={"RED": 10}, turns=40),
    )
    raw = record.model_dump_json()
    again = DecisionRecord.model_validate_json(raw)
    assert again.game_id == "g1"
    assert again.expert_policy == ExpertPolicy.RANDOM
    assert again.outcome is not None
    assert again.outcome.winner == "RED"
