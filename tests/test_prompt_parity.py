"""Train/play prompt parity gate (DATA_CONTRACT §4 / ticket 04)."""

from catanatron.game import Game, GameAccumulator
from catanatron.models.player import Color

from catan_llm.data.dataset import decision_to_chat
from catan_llm.data.identity import PROMPT_VERSION
from catan_llm.data.renderer import render_system_prompt, render_user_prompt
from catan_llm.sim.adapter import play_one
from catan_llm.sim.players import make_player


class _LivePromptCapture(GameAccumulator):
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def step(self, game_before_action, action):
        playable = list(game_before_action.playable_actions)
        system = render_system_prompt(game_before_action, action.color)
        user = render_user_prompt(game_before_action, action.color, playable)
        self.prompts.append((system, user))


def test_prompt_parity_over_20_seeded_games():
    checked = 0
    for seed in range(20):
        result = play_one(
            ["random", "weightedrandom", "random", "random"],
            seed=seed,
            vps_to_win=6,
        )
        assert result.records
        assert result.records[0].prompt_version == PROMPT_VERSION

        # Independent live capture on a fresh identical game.
        capture = _LivePromptCapture()
        colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
        players = [make_player(n, c)[0] for n, c in zip(
            ["random", "weightedrandom", "random", "random"], colors, strict=True
        )]
        from catan_llm.sim.adapter import map_for_seed

        game = Game(
            players,
            seed=seed,
            vps_to_win=6,
            catan_map=map_for_seed("BASE", seed, n_players=4),
        )
        game.play(accumulators=[capture])

        assert len(capture.prompts) == len(result.records)
        for record, (system, user) in zip(result.records, capture.prompts, strict=True):
            assert record.system_prompt == system
            assert record.user_prompt == user
            chat = decision_to_chat(record)
            assert chat["messages"][0]["content"] == system
            assert chat["messages"][1]["content"] == user
            assert chat["prompt_version"] == PROMPT_VERSION
            checked += 1

    assert checked >= 20
