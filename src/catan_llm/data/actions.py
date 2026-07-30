"""Action serialization / deserialization helpers shared by sim, renderer, parser."""

from __future__ import annotations

from typing import Any

from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color

from catan_llm.data.schema import ActionRecord

RESOURCE_ABBREV = {
    "WOOD": "W",
    "BRICK": "B",
    "SHEEP": "S",
    "WHEAT": "H",
    "ORE": "O",
}
ABBREV_TO_RESOURCE = {v: k for k, v in RESOURCE_ABBREV.items()}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Color):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return str(value)


def action_to_record(action: Action) -> ActionRecord:
    return ActionRecord(
        color=action.color.value,
        action_type=action.action_type.value,
        value=_jsonable(action.value),
    )


def record_to_action(record: ActionRecord | dict[str, Any]) -> Action:
    if isinstance(record, dict):
        record = ActionRecord.model_validate(record)
    color = Color[record.color]
    action_type = ActionType[record.action_type]
    value = _revive_value(action_type, record.value)
    return Action(color, action_type, value)


def _revive_value(action_type: ActionType, value: Any) -> Any:
    if value is None:
        return None
    if action_type == ActionType.BUILD_ROAD:
        return tuple(value)
    if action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        return tuple(value)
    if action_type == ActionType.MOVE_ROBBER:
        # Catanatron 3.3 uses (coordinate, victim); older forks may include a 3rd slot.
        coord = tuple(value[0])
        victim = Color[value[1]] if value[1] else None
        if len(value) >= 3:
            return (coord, victim, value[2])
        return (coord, victim)
    if action_type == ActionType.MARITIME_TRADE:
        return tuple(value)
    if action_type == ActionType.DISCARD and isinstance(value, list):
        return value
    return value


def format_action(action: Action) -> str:
    """Compact human-readable action string for prompts."""
    at = action.action_type
    val = action.value

    if at == ActionType.ROLL:
        return "ROLL"
    if at == ActionType.END_TURN:
        return "END_TURN"
    if at == ActionType.BUILD_SETTLEMENT:
        return f"BUILD_SETTLEMENT node={val}"
    if at == ActionType.BUILD_ROAD:
        return f"BUILD_ROAD edge={val}"
    if at == ActionType.BUILD_CITY:
        return f"BUILD_CITY node={val}"
    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return "BUY_DEV_CARD"
    if at == ActionType.PLAY_KNIGHT_CARD:
        return "PLAY_KNIGHT"
    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        resources = [RESOURCE_ABBREV.get(str(r), str(r)) for r in val]
        return f"YEAR_OF_PLENTY [{','.join(resources)}]"
    if at == ActionType.PLAY_MONOPOLY:
        return f"MONOPOLY {RESOURCE_ABBREV.get(str(val), str(val))}"
    if at == ActionType.PLAY_ROAD_BUILDING:
        return "ROAD_BUILDING"
    if at == ActionType.MARITIME_TRADE:
        giving = [RESOURCE_ABBREV.get(str(r), "?") for r in val[:-1] if r is not None]
        receiving = RESOURCE_ABBREV.get(str(val[-1]), str(val[-1]))
        return f"MARITIME_TRADE give=[{','.join(giving)}] get={receiving}"
    if at == ActionType.MOVE_ROBBER:
        coord = val[0]
        target_color = val[1]
        target = target_color.value if target_color else "nobody"
        return f"MOVE_ROBBER to={coord} steal_from={target}"
    if at == ActionType.DISCARD:
        if val:
            resources = [RESOURCE_ABBREV.get(str(r), str(r)) for r in val]
            return f"DISCARD [{','.join(resources)}]"
        return "DISCARD (random)"
    return f"{at.value} {val}"
