"""Ticket 24 — terminal spectate + replay."""

from __future__ import annotations

import json
from pathlib import Path

from catanatron.models.player import Color

from catan_llm.play.spectate import play_spectate_game, write_replay
from catan_llm.sim.players import make_player


def test_bots_only_spectate_writes_replay(tmp_path: Path):
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    kinds = ["random", "weightedrandom", "random", "random"]
    players = [make_player(k, c)[0] for k, c in zip(kinds, colors, strict=True)]
    result = play_spectate_game(
        players,
        seed=21,
        vps_to_win=6,
        watch=False,
        seat_labels={c.value: k for c, k in zip(colors, kinds, strict=True)},
    )
    assert result.turns > 0
    assert len(result.events) > 0
    out = tmp_path / "replay.json"
    write_replay(result, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed"] == 21
    assert payload["events"]
    assert "action_type" in payload["events"][0]
