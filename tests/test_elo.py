from types import SimpleNamespace

from t8_agent.train import elo


def test_play_match_treats_timeout_as_draw_for_rating(monkeypatch) -> None:
    class TimeoutEnv:
        def __init__(self, seed: int) -> None:
            self.state = SimpleNamespace(winner=1)

        def reset(self, seed: int):
            return None

        def step(self, _p1_action, _p2_action):
            return SimpleNamespace(
                terminated=True,
                truncated=False,
                info={"timed_out": True, "stalemate": False},
            )

    monkeypatch.setattr(elo, "TekkenLiteEnv", TimeoutEnv)

    score = elo.play_match(lambda _env, _player: None, lambda _env, _player: None, seed=1, max_decisions=1)

    assert score == 0.5


def test_play_match_counts_clean_p1_win(monkeypatch) -> None:
    class WinEnv:
        def __init__(self, seed: int) -> None:
            self.state = SimpleNamespace(winner=1)

        def reset(self, seed: int):
            return None

        def step(self, _p1_action, _p2_action):
            return SimpleNamespace(
                terminated=True,
                truncated=False,
                info={"timed_out": False, "stalemate": False},
            )

    monkeypatch.setattr(elo, "TekkenLiteEnv", WinEnv)

    score = elo.play_match(lambda _env, _player: None, lambda _env, _player: None, seed=1, max_decisions=1)

    assert score == 1.0
