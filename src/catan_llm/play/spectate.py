"""Live spectate helpers — terminal watch + replay JSON (ticket 24)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catanatron import Game
from catanatron.game import GameAccumulator
from catanatron.models.player import Color, Player
from catanatron.state_functions import get_actual_victory_points

from catan_llm.sim.players import make_player


@dataclass
class SpectateEvent:
    turn: int
    color: str
    action_type: str
    value: Any
    vps: dict[str, int]
    t_rel_s: float


@dataclass
class SpectateResult:
    game_id: str
    seed: int
    winner: str | None
    turns: int
    events: list[SpectateEvent] = field(default_factory=list)
    seat_labels: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "winner": self.winner,
            "turns": self.turns,
            "seat_labels": self.seat_labels,
            "events": [
                {
                    "turn": e.turn,
                    "color": e.color,
                    "action_type": e.action_type,
                    "value": e.value,
                    "vps": e.vps,
                    "t_rel_s": e.t_rel_s,
                }
                for e in self.events
            ],
        }


class WatchAccumulator(GameAccumulator):
    """Print / record each action for terminal spectating."""

    def __init__(
        self,
        *,
        watch: bool = True,
        seat_labels: dict[str, str] | None = None,
        delay_s: float = 0.0,
    ):
        self.watch = watch
        self.seat_labels = seat_labels or {}
        self.delay_s = delay_s
        self.events: list[SpectateEvent] = []
        self._t0 = 0.0
        self.game_id = ""
        self.seed = 0

    def before(self, game: Game) -> None:
        self._t0 = time.time()
        self.game_id = str(game.id)
        self.seed = int(game.seed)
        if self.watch:
            seats = ", ".join(
                f"{c.value}:{self.seat_labels.get(c.value, c.value)}"
                for c in game.state.colors
            )
            print(f"[spectate] game={self.game_id} seed={self.seed} seats=[{seats}]")

    def step(self, game_before_action: Game, action) -> None:
        color = action.color.value
        atype = action.action_type.value if hasattr(action.action_type, "value") else str(
            action.action_type
        )
        vps = {
            c.value: int(get_actual_victory_points(game_before_action.state, c))
            for c in game_before_action.state.colors
        }
        event = SpectateEvent(
            turn=int(game_before_action.state.num_turns),
            color=color,
            action_type=atype,
            value=action.value,
            vps=vps,
            t_rel_s=round(time.time() - self._t0, 3),
        )
        self.events.append(event)
        if self.watch:
            label = self.seat_labels.get(color, color)
            print(
                f"t={event.turn:3d}  {label:16s}  {atype:22s}  "
                f"value={action.value!r}  vps={vps}"
            )
            if self.delay_s > 0:
                time.sleep(self.delay_s)

    def after(self, game: Game) -> None:
        winner = game.winning_color()
        if self.watch:
            w = winner.value if winner else None
            print(
                f"[spectate] finished winner={w} turns={game.state.num_turns} "
                f"events={len(self.events)}"
            )


def play_spectate_game(
    players: list[Player],
    *,
    seed: int,
    vps_to_win: int = 10,
    watch: bool = True,
    delay_s: float = 0.0,
    seat_labels: dict[str, str] | None = None,
) -> SpectateResult:
    """Play one game with optional terminal watch + replay events."""
    labels = seat_labels or {
        p.color.value: getattr(p, "model", None) or type(p).__name__ for p in players
    }
    acc = WatchAccumulator(watch=watch, seat_labels=labels, delay_s=delay_s)
    game = Game(players, seed=seed, vps_to_win=vps_to_win)
    winner = game.play(accumulators=[acc])
    return SpectateResult(
        game_id=str(game.id),
        seed=int(game.seed),
        winner=winner.value if winner else None,
        turns=int(game.state.num_turns),
        events=acc.events,
        seat_labels=labels,
    )


def write_replay(result: SpectateResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")


def bot_seat(kind: str, color: Color) -> Player:
    player, _ = make_player(kind, color)
    return player
