"""Bot factory for Catanatron built-ins used across sim / eval."""

from __future__ import annotations

from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_llm.data.schema import ExpertPolicy

BOT_ALIASES = {
    "random": (RandomPlayer, ExpertPolicy.RANDOM, {}),
    "weightedrandom": (WeightedRandomPlayer, ExpertPolicy.WEIGHTEDRANDOM, {}),
    "valuefunction": (ValueFunctionPlayer, ExpertPolicy.VALUEFUNCTION, {}),
    "alphabeta": (AlphaBetaPlayer, ExpertPolicy.ALPHABETA, {"depth": 2}),
}


def make_player(name: str, color: Color, **overrides) -> tuple[Player, ExpertPolicy]:
    key = name.strip().lower()
    if key not in BOT_ALIASES:
        raise ValueError(f"Unknown bot '{name}'. Choose from: {sorted(BOT_ALIASES)}")
    cls, policy, defaults = BOT_ALIASES[key]
    kwargs = {**defaults, **overrides}
    return cls(color, **kwargs), policy


def make_seat(bot_names: list[str], colors: list[Color] | None = None) -> list[Player]:
    colors = colors or [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    if len(bot_names) > len(colors):
        raise ValueError("At most 4 players")
    players = []
    for name, color in zip(bot_names, colors, strict=False):
        player, _policy = make_player(name, color)
        players.append(player)
    return players


def policy_for_player(player: Player) -> ExpertPolicy:
    mapping = {
        "RandomPlayer": ExpertPolicy.RANDOM,
        "WeightedRandomPlayer": ExpertPolicy.WEIGHTEDRANDOM,
        "ValueFunctionPlayer": ExpertPolicy.VALUEFUNCTION,
        "AlphaBetaPlayer": ExpertPolicy.ALPHABETA,
        "LLMPlayer": ExpertPolicy.LLM,
    }
    return mapping.get(type(player).__name__, ExpertPolicy.OTHER)
