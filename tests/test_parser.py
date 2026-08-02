from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color

from catan_llm.data.parser import (
    fallback_action,
    format_assistant_target,
    parse_action_response,
    strip_thinking,
)


def _actions():
    return [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.END_TURN, None),
    ]


def test_parse_valid_json():
    result = parse_action_response('{"action": 1, "reasoning": "done"}', _actions())
    assert result.ok
    assert result.action_index == 1
    assert result.action is not None
    assert result.action.action_type == ActionType.END_TURN


def test_parse_embedded_json():
    text = 'Sure.\n{"action": 0, "reasoning": "roll"}\n'
    result = parse_action_response(text, _actions())
    assert result.ok
    assert result.action_index == 0


def test_parse_out_of_range():
    result = parse_action_response('{"action": 9}', _actions())
    assert not result.ok
    assert result.error == "action_out_of_range"


def test_fallback_and_target_format():
    actions = _actions()
    assert fallback_action(actions) == actions[0]
    assert '"action": 1' in format_assistant_target(1, "x")


def test_parse_strips_qwen_think_blocks():
    text = '<think>\nponder\n</think>\n\n{"action": 1, "reasoning": "done"}'
    assert '"action": 1' in strip_thinking(text)
    result = parse_action_response(text, _actions())
    assert result.ok
    assert result.action_index == 1


def test_parse_unclosed_think_without_json_fails():
    result = parse_action_response("<think>\nstill thinking", _actions())
    assert not result.ok
    assert result.error == "json_parse_failed"
