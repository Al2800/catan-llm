"""Canonical game-state renderer for train / play consistency.

Adapted from the catan-bench style (indexed legal actions, hidden opponent hands).
Domestic negotiation trading is intentionally omitted (locked decision §12.5).
"""

from __future__ import annotations

from catanatron.models.enums import (
    CITY,
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
)
from catanatron.models.player import Color
from catanatron.state_functions import player_key

from catan_llm.data.actions import RESOURCE_ABBREV, format_action

SYSTEM_RULES = """You are an expert Settlers of Catan player.

GAME RULES:
Victory: First player to reach the VP target on their turn wins immediately.
VP sources: settlement=1, city=2, longest road (5+ segments)=2, largest army (3+ knights)=2, VP dev cards=1 each.

Resources: W=Wood, B=Brick, S=Sheep, H=Wheat, O=Ore
Build costs: Road=W+B, Settlement=W+B+S+H, City=3O+2H, Dev card=S+H+O

Dice & Production:
- Each turn starts with a dice roll (2d6). Every tile matching the number produces resources.
- Settlements on a producing tile get 1 resource; cities get 2.
- Tile probability (pips): 2→1, 3→2, 4→3, 5→4, 6→5, 8→5, 9→4, 10→3, 11→2, 12→1.

Rolling a 7 / Robber:
- Any player with more than 7 resource cards must discard half (rounded down).
- The active player moves the robber to any tile, blocking its production, and may steal 1 card.

Building Rules:
- Settlements must be at least 2 edges away from any other settlement or city.
- Roads must connect to your existing network.
- Cities upgrade existing settlements.

Development Cards:
- Knight: move robber and steal; counts toward Largest Army.
- Year of Plenty / Monopoly / Road Building as usual.
- Victory Point cards stay hidden and count automatically.
- At most 1 dev card per turn; not the turn it was bought.

Trading: maritime trades only (4:1, 3:1 ports, 2:1 resource ports). No player-to-player negotiation.

RESPONSE FORMAT:
Respond with ONLY valid JSON:
{"action": <index>, "reasoning": "<brief explanation>"}
The "action" value must be the [index] of one of the AVAILABLE ACTIONS.
"""


def serialize_board(game) -> str:
    """Static board layout (safe for system-prompt caching)."""
    board = game.state.board
    catan_map = board.map
    lines = ["=== BOARD LAYOUT ===", "Tiles (id: resource number, nodes):"]
    for _coord, tile in sorted(catan_map.land_tiles.items(), key=lambda x: x[1].id):
        if tile.resource is None:
            res_str = "DESERT"
            num_str = ""
        else:
            res_str = RESOURCE_ABBREV.get(str(tile.resource), str(tile.resource))
            num_str = f" #{tile.number}"
        node_ids = sorted(tile.nodes.values())
        lines.append(f"  T{tile.id}: {res_str}{num_str} nodes={node_ids}")

    lines.append("Ports:")
    for resource, node_ids in catan_map.port_nodes.items():
        if resource is None:
            lines.append(f"  3:1 port at nodes {sorted(node_ids)}")
        else:
            abbrev = RESOURCE_ABBREV.get(str(resource), str(resource))
            lines.append(f"  2:1 {abbrev} port at nodes {sorted(node_ids)}")

    lines.append(f"Robber starts at: {board.robber_coordinate}")
    return "\n".join(lines)


def render_system_prompt(game, player_color: Color) -> str:
    return (
        f"{SYSTEM_RULES}\n"
        f"You are playing as {player_color.value}.\n\n"
        f"{serialize_board(game)}"
    )


def serialize_dynamic_state(game, player_color: Color) -> str:
    """Partial-observability-aware dynamic state (own hand visible only)."""
    state = game.state
    lines = [
        f"=== TURN {state.num_turns} ===",
        f"Current player: {state.current_color().value}",
        f"Phase: {state.current_prompt.value}",
        "\nPLAYER STATUS:",
    ]

    for color in state.colors:
        key = player_key(state, color)
        ps = state.player_state
        is_you = " (YOU)" if color == player_color else ""

        vp = ps[f"{key}_VICTORY_POINTS"]
        settlements = 5 - ps[f"{key}_SETTLEMENTS_AVAILABLE"]
        cities = 4 - ps[f"{key}_CITIES_AVAILABLE"]
        roads = 15 - ps[f"{key}_ROADS_AVAILABLE"]
        has_road = ps[f"{key}_HAS_ROAD"]
        has_army = ps[f"{key}_HAS_ARMY"]
        longest_road_len = ps[f"{key}_LONGEST_ROAD_LENGTH"]

        line = (
            f"  {color.value}{is_you}: VP={vp} settlements={settlements} "
            f"cities={cities} roads={roads} longest_road={longest_road_len}"
        )
        if has_road:
            line += " [LONGEST ROAD]"
        if has_army:
            line += " [LARGEST ARMY]"

        if color == player_color:
            hand_parts = []
            for res in RESOURCES:
                count = ps[f"{key}_{res}_IN_HAND"]
                if count > 0:
                    hand_parts.append(f"{RESOURCE_ABBREV[str(res)]}={count}")
            hand_str = ",".join(hand_parts) if hand_parts else "empty"
            line += f" hand=[{hand_str}]"

            dev_parts = []
            for dev in DEVELOPMENT_CARDS:
                count = ps[f"{key}_{dev}_IN_HAND"]
                if count > 0:
                    dev_parts.append(f"{dev}={count}")
            if dev_parts:
                line += f" dev_cards=[{','.join(dev_parts)}]"
        else:
            total_resources = sum(ps[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
            total_dev = sum(ps[f"{key}_{dev}_IN_HAND"] for dev in DEVELOPMENT_CARDS)
            line += f" cards={total_resources} dev_cards={total_dev}"

        lines.append(line)

    lines.append("\nBUILDINGS:")
    for node_id, (color, building_type) in sorted(state.board.buildings.items()):
        btype = "S" if building_type == SETTLEMENT else "C"
        lines.append(f"  node {node_id}: {color.value} {btype}")

    seen_edges: set[tuple[int, int]] = set()
    road_lines: list[str] = []
    for edge, color in state.board.roads.items():
        canonical = tuple(sorted(edge))
        if canonical not in seen_edges:
            seen_edges.add(canonical)
            road_lines.append(f"  edge {canonical}: {color.value}")
    if road_lines:
        lines.append("\nROADS:")
        lines.extend(road_lines)

    lines.append(f"\nRobber at: {state.board.robber_coordinate}")
    return "\n".join(lines)


def serialize_actions(playable_actions) -> str:
    lines = ["AVAILABLE ACTIONS:"]
    for i, action in enumerate(playable_actions):
        lines.append(f"  [{i}] {format_action(action)}")
    return "\n".join(lines)


def render_user_prompt(game, player_color: Color, playable_actions) -> str:
    return "\n\n".join(
        [
            serialize_dynamic_state(game, player_color),
            serialize_actions(playable_actions),
            (
                "Respond with ONLY a JSON object, no other text. "
                'Example: {"action": 0, "reasoning": "highest pip settlement"}'
            ),
        ]
    )


def compact_state_dict(game, player_color: Color) -> dict:
    """Structured (JSON) state used in trajectory records — POV-aware."""
    state = game.state
    players = []
    for color in state.colors:
        key = player_key(state, color)
        ps = state.player_state
        entry = {
            "color": color.value,
            "vp": ps[f"{key}_VICTORY_POINTS"],
            "settlements": 5 - ps[f"{key}_SETTLEMENTS_AVAILABLE"],
            "cities": 4 - ps[f"{key}_CITIES_AVAILABLE"],
            "roads": 15 - ps[f"{key}_ROADS_AVAILABLE"],
            "has_longest_road": bool(ps[f"{key}_HAS_ROAD"]),
            "has_largest_army": bool(ps[f"{key}_HAS_ARMY"]),
            "longest_road_length": ps[f"{key}_LONGEST_ROAD_LENGTH"],
        }
        if color == player_color:
            entry["hand"] = {
                RESOURCE_ABBREV[str(res)]: ps[f"{key}_{res}_IN_HAND"] for res in RESOURCES
            }
            entry["dev_cards"] = {
                str(dev): ps[f"{key}_{dev}_IN_HAND"] for dev in DEVELOPMENT_CARDS
            }
        else:
            entry["cards"] = sum(ps[f"{key}_{res}_IN_HAND"] for res in RESOURCES)
            entry["dev_cards_count"] = sum(
                ps[f"{key}_{dev}_IN_HAND"] for dev in DEVELOPMENT_CARDS
            )
        players.append(entry)

    buildings = {
        str(node_id): {"color": color.value, "type": building_type}
        for node_id, (color, building_type) in state.board.buildings.items()
    }
    roads = {
        str(tuple(sorted(edge))): color.value for edge, color in state.board.roads.items()
    }

    return {
        "turn": state.num_turns,
        "phase": state.current_prompt.value,
        "current_color": state.current_color().value,
        "robber": list(state.board.robber_coordinate)
        if isinstance(state.board.robber_coordinate, tuple)
        else state.board.robber_coordinate,
        "players": players,
        "buildings": buildings,
        "roads": roads,
    }
