import json

import pytest
from pydantic import ValidationError

from catan_llm.data.identity import (
    CATANATRON_COMMIT,
    PROMPT_VERSION,
    bot_config_hash,
    make_game_key,
)
from catan_llm.data.schema import (
    SCHEMA_VERSION,
    DecisionRecord,
    ExpertPolicy,
    GameOutcome,
    require_schema_v2,
)


def _sample_record(**overrides) -> DecisionRecord:
    bot_config = [{"name": "random", "params": {}}]
    cfg_hash = bot_config_hash(bot_config)
    map_hash = "a" * 64
    defaults = dict(
        game_key=make_game_key(1, map_hash, cfg_hash),
        game_id="g1",
        decision_idx=0,
        seed=1,
        map_type="BASE",
        map_hash=map_hash,
        bot_config=bot_config,
        bot_config_hash=cfg_hash,
        catanatron_commit=CATANATRON_COMMIT,
        source_commit="deadbeef",
        player_color="RED",
        turn=0,
        phase="BUILD_INITIAL_SETTLEMENT",
        board={"tiles": []},
        state={"turn": 0},
        system_prompt="SYSTEM",
        user_prompt="USER\nAVAILABLE ACTIONS:\n  [0] BUILD_SETTLEMENT",
        valid_actions=[{"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3}],
        action_taken={"color": "RED", "action_type": "BUILD_SETTLEMENT", "value": 3},
        action_index=0,
        expert_policy=ExpertPolicy.RANDOM,
        outcome=GameOutcome(winner="RED", vps={"RED": 10}, turns=40),
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def test_decision_record_v2_roundtrip():
    record = _sample_record()
    raw = record.model_dump_json()
    again = DecisionRecord.model_validate_json(raw)
    assert again.schema_version == SCHEMA_VERSION
    assert again.prompt_version == PROMPT_VERSION
    assert again.game_key == record.game_key
    assert again.map_hash == record.map_hash
    assert again.bot_config_hash == record.bot_config_hash
    assert again.catanatron_commit == CATANATRON_COMMIT
    assert again.source_commit == "deadbeef"
    assert again.expert_policy == ExpertPolicy.RANDOM
    assert again.outcome is not None
    assert again.outcome.winner == "RED"


def test_schema_v1_rejected():
    bot_config = [{"name": "random", "params": {}}]
    payload = {
        "schema_version": "v1",
        "game_id": "g1",
        "decision_idx": 0,
        "seed": 1,
        "player_color": "RED",
        "turn": 0,
        "phase": "BUILD_INITIAL_SETTLEMENT",
        "state": {},
        "valid_actions": [],
        "action_taken": {"color": "RED", "action_type": "END_TURN", "value": None},
        "action_index": 0,
        "expert_policy": "random",
        "game_key": "x",
        "map_hash": "y",
        "bot_config": bot_config,
        "bot_config_hash": "z",
    }
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(payload)


def test_require_schema_v2_rejects_unknown_prompt():
    record = _sample_record(prompt_version="not-a-real-prompt")
    with pytest.raises(ValueError, match="unknown prompt_version"):
        require_schema_v2([record])


def test_game_key_formula():
    bot_config = [{"name": "alphabeta", "params": {"depth": 2}}]
    cfg_hash = bot_config_hash(bot_config)
    key = make_game_key(42, "maphash", cfg_hash)
    assert len(key) == 64
    assert key == make_game_key(42, "maphash", cfg_hash)
    assert json.loads(json.dumps(bot_config)) == bot_config
