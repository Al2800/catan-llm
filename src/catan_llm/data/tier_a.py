"""Feature-aware Tier A rationale templates (SCOPE §7.4 / DATA_CONTRACT §8).

All features are learner-observable (POV-safe). Teachers may still choose actions
with full Game state; these strings must not cite opponent private hands.
"""

from __future__ import annotations

from typing import Any

from catan_llm.data.actions import RESOURCE_ABBREV
from catan_llm.data.pov import assert_tier_a_pov_safe
from catan_llm.data.schema import DecisionRecord

_PIP_WEIGHT = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _abbrev(resource: str | None) -> str:
    if resource is None:
        return "DESERT"
    return RESOURCE_ABBREV.get(str(resource), str(resource)[:1])


def _ego(record: DecisionRecord) -> dict[str, Any]:
    for player in record.state.get("players", []):
        if player.get("color") == record.player_color:
            return player
    return {}


def _tiles_for_node(board: dict[str, Any], node_id: int) -> list[dict[str, Any]]:
    out = []
    for tile in board.get("tiles", []):
        if node_id in tile.get("nodes", []):
            out.append(tile)
    return out


def _tile_at_coord(board: dict[str, Any], coord: list | tuple) -> dict[str, Any] | None:
    coord_list = list(coord)
    for tile in board.get("tiles", []):
        if list(tile.get("coord", [])) == coord_list:
            return tile
    return None


def _node_pips(board: dict[str, Any], node_id: int) -> tuple[int, list[str]]:
    tiles = _tiles_for_node(board, node_id)
    pips = 0
    resources: list[str] = []
    for tile in tiles:
        number = tile.get("number")
        if number is not None:
            pips += _PIP_WEIGHT.get(int(number), 0)
        res = tile.get("resource")
        if res is not None:
            resources.append(_abbrev(res))
    # Stable unique resource order by first appearance.
    seen: list[str] = []
    for r in resources:
        if r not in seen:
            seen.append(r)
    return pips, seen


def _ports_at_node(board: dict[str, Any], node_id: int) -> list[str]:
    labels = []
    for port in board.get("ports", []):
        if node_id in port.get("nodes", []):
            res = port.get("resource")
            if res is None:
                labels.append("3:1")
            else:
                labels.append(f"2:1 {_abbrev(res)}")
    return labels


def _blockers_near_node(
    board: dict[str, Any], state: dict[str, Any], node_id: int, ego: str
) -> list[str]:
    nearby_nodes: set[int] = set()
    for tile in _tiles_for_node(board, node_id):
        nearby_nodes.update(int(n) for n in tile.get("nodes", []))
    colors: list[str] = []
    for nid, building in state.get("buildings", {}).items():
        try:
            n = int(nid)
        except (TypeError, ValueError):
            continue
        if n in nearby_nodes and building.get("color") and building["color"] != ego:
            colors.append(building["color"])
    return sorted(set(colors))


def _race_cues(ego: dict[str, Any], state: dict[str, Any]) -> list[str]:
    cues = []
    if ego.get("has_longest_road"):
        cues.append("holds_longest_road")
    if ego.get("has_largest_army"):
        cues.append("holds_largest_army")
    lr = ego.get("longest_road_length")
    if lr is not None:
        # Best opponent public road length.
        best_opp = 0
        for player in state.get("players", []):
            if player.get("color") == ego.get("color"):
                continue
            best_opp = max(best_opp, int(player.get("longest_road_length") or 0))
        cues.append(f"road_len={lr} (best_opp={best_opp})")
    return cues


def _hand_imbalance(hand: dict[str, int]) -> str | None:
    if not hand:
        return None
    items = sorted(((k, int(v)) for k, v in hand.items() if int(v) > 0), key=lambda kv: -kv[1])
    if not items:
        return "hand empty"
    top = ",".join(f"{k}={v}" for k, v in items[:3])
    return f"hand[{top}]"


def _rationale_settlement_or_city(record: DecisionRecord, *, kind: str) -> str:
    node = int(record.action_taken.value)
    pips, resources = _node_pips(record.board, node)
    ports = _ports_at_node(record.board, node)
    blockers = _blockers_near_node(
        record.board, record.state, node, record.player_color
    )
    res_str = "+".join(resources) if resources else "no-res"
    parts = [f"{kind} node={node}", f"pips={pips} ({res_str})"]
    if ports:
        parts.append("port " + "/".join(ports))
    if blockers:
        parts.append("blocks " + ",".join(blockers))
    if record.phase.startswith("BUILD_INITIAL"):
        parts.append("initial_placement")
    return "; ".join(parts)


def _rationale_road(record: DecisionRecord) -> str:
    edge = record.action_taken.value
    a, b = int(edge[0]), int(edge[1])
    ego = _ego(record)
    buildings = record.state.get("buildings", {})
    open_ends = []
    for node in (a, b):
        if str(node) not in buildings:
            pips, resources = _node_pips(record.board, node)
            res_str = "+".join(resources) if resources else "no-res"
            open_ends.append(f"node {node} pips={pips} ({res_str})")
    parts = [f"road edge=({a},{b})"]
    if open_ends:
        parts.append("toward open " + " / ".join(open_ends))
    else:
        parts.append("connects network")
    lr = ego.get("longest_road_length")
    if lr is not None:
        parts.append(f"longest_road threat={lr}")
    return "; ".join(parts)


def _rationale_robber(record: DecisionRecord, *, via_knight: bool) -> str:
    value = record.action_taken.value
    coord = value[0]
    victim = value[1] if len(value) > 1 else None
    tile = _tile_at_coord(record.board, coord)
    if tile is None:
        tile_desc = f"coord={coord}"
    else:
        number = tile.get("number")
        pips = _PIP_WEIGHT.get(int(number), 0) if number is not None else 0
        tile_desc = f"pips={pips} {_abbrev(tile.get('resource'))}"
    prefix = "knight then robber" if via_knight else "robber"
    parts = [f"{prefix} on {tile_desc}"]
    if victim:
        # Card *count* only — never private composition.
        cards = None
        for player in record.state.get("players", []):
            if player.get("color") == victim:
                cards = player.get("cards")
                break
        if cards is not None:
            parts.append(f"steal from {victim} ({cards} cards)")
        else:
            parts.append(f"steal from {victim}")
    return "; ".join(parts)


def _rationale_maritime(record: DecisionRecord) -> str:
    trade = list(record.action_taken.value or [])
    # Catanatron maritime: offered resources then requested (last element is gain).
    if len(trade) >= 2:
        give = [_abbrev(x) for x in trade[:-1]]
        get = _abbrev(trade[-1])
        give_str = "".join(give) if give else "?"
        imbalance = _hand_imbalance(_ego(record).get("hand") or {})
        parts = [f"maritime {give_str}→{get}"]
        if imbalance:
            parts.append(f"corrects {imbalance}")
        return "; ".join(parts)
    return "maritime trade rebalances hand"


def _rationale_dev(record: DecisionRecord, *, bought: bool) -> str:
    ego = _ego(record)
    cues = _race_cues(ego, record.state)
    if bought:
        parts = ["buy_dev"]
    else:
        at = record.action_taken.action_type
        parts = [f"play_{at.replace('PLAY_', '').lower()}"]
    if cues:
        parts.append(", ".join(cues))
    vp = ego.get("vp")
    if vp is not None:
        parts.append(f"vp={vp}")
    return "; ".join(parts)


def _rationale_end_turn(record: DecisionRecord) -> str:
    ego = _ego(record)
    imbalance = _hand_imbalance(ego.get("hand") or {})
    parts = ["end_turn; no higher-value legal build"]
    if imbalance:
        parts.append(imbalance)
    return "; ".join(parts)


def _optional_value_delta(record: DecisionRecord) -> str | None:
    """Include valueΔ only when the record exposes a POV-safe numeric signal."""
    raw = record.state.get("value_delta")
    if raw is None:
        return None
    try:
        delta = float(raw)
    except (TypeError, ValueError):
        return None
    policy = record.expert_policy.value
    sign = "+" if delta >= 0 else ""
    return f"valueΔ={sign}{delta:.2f} (expert {policy})"


def render_tier_a_rationale(record: DecisionRecord) -> str:
    """One-line feature-aware rationale for the taken action."""
    at = record.action_taken.action_type
    if at == "ROLL":
        text = "must roll to start the turn"
    elif at == "BUILD_SETTLEMENT":
        text = _rationale_settlement_or_city(record, kind="settlement")
    elif at == "BUILD_CITY":
        text = _rationale_settlement_or_city(record, kind="city")
    elif at == "BUILD_ROAD":
        text = _rationale_road(record)
    elif at == "MOVE_ROBBER":
        text = _rationale_robber(record, via_knight=False)
    elif at == "PLAY_KNIGHT_CARD":
        # Knight play may be separate from MOVE_ROBBER; describe army race if present.
        ego = _ego(record)
        cues = _race_cues(ego, record.state)
        text = "play_knight; " + (", ".join(cues) if cues else "contest largest_army")
    elif at == "MARITIME_TRADE":
        text = _rationale_maritime(record)
    elif at == "BUY_DEVELOPMENT_CARD":
        text = _rationale_dev(record, bought=True)
    elif at.startswith("PLAY_"):
        text = _rationale_dev(record, bought=False)
    elif at == "END_TURN":
        text = _rationale_end_turn(record)
    elif at == "DISCARD":
        hand = _hand_imbalance(_ego(record).get("hand") or {})
        text = f"discard to 7; {hand}" if hand else "discard to 7"
    else:
        text = f"{at.lower().replace('_', ' ')}; public board features only"

    delta = _optional_value_delta(record)
    if delta:
        text = f"{text}; {delta}"

    assert_tier_a_pov_safe(text, context="tier_a")
    return text


def is_feature_aware(text: str) -> bool:
    """Heuristic: reject pure policy restatements banned by DATA_CONTRACT §8."""
    banned_substrings = (
        "policy selects",
        "expands board position with",
        "initial placement via",
        "applies robber pressure via",
        "uses development-card line",
        "balances hand via maritime trade",
    )
    lowered = text.lower()
    if any(b in lowered for b in banned_substrings):
        return False
    # Must mention at least one concrete feature cue for non-trivial actions.
    feature_cues = (
        "pips=",
        "port ",
        "blocks ",
        "toward open",
        "longest_road",
        "steal from",
        "maritime ",
        "hand[",
        "road_len=",
        "buy_dev",
        "play_",
        "must roll",
        "end_turn",
        "discard",
        "robber on",
        "connects network",
        "initial_placement",
        "valueδ=",
        "valueΔ=",
    )
    return any(cue.lower() in lowered for cue in feature_cues)
