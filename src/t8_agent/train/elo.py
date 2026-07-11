from __future__ import annotations

from pathlib import Path

from t8_agent.sim.tekken_lite import TekkenLiteEnv
from t8_agent.train.ppo_opponents import PpoCheckpointOpponent


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: float) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    return rating_a + k * (score_a - expected_a), rating_b + k * ((1.0 - score_a) - expected_b)


def play_match(policy_a, policy_b, seed: int, max_decisions: int) -> float:
    env = TekkenLiteEnv(seed=seed)
    env.reset(seed=seed)
    last_info = {}
    for _ in range(max_decisions):
        result = env.step(policy_a(env, 1), policy_b(env, 2))
        last_info = result.info
        if result.terminated or result.truncated:
            break
    if last_info.get("timed_out") or last_info.get("stalemate"):
        return 0.5
    if env.state.winner == 1:
        return 1.0
    if env.state.winner == 2:
        return 0.0
    return 0.5


def rank_checkpoints(
    checkpoint_paths: list[str | Path],
    episodes_per_pair: int,
    seed: int,
    max_decisions: int,
    k: float = 32.0,
) -> dict[str, float]:
    checkpoints = [Path(path) for path in checkpoint_paths]
    if len(checkpoints) < 2:
        return {str(path): 1000.0 for path in checkpoints}

    policies = {path: PpoCheckpointOpponent(path) for path in checkpoints}
    ratings = {path: 1000.0 for path in checkpoints}
    match_idx = 0
    for i, path_a in enumerate(checkpoints):
        for path_b in checkpoints[i + 1 :]:
            for episode in range(episodes_per_pair):
                score_a = play_match(
                    policies[path_a],
                    policies[path_b],
                    seed=seed + match_idx * 100 + episode,
                    max_decisions=max_decisions,
                )
                ratings[path_a], ratings[path_b] = update_elo(ratings[path_a], ratings[path_b], score_a, k)
            match_idx += 1
    return {str(path): rating for path, rating in ratings.items()}


def update_checkpoint_ratings(
    checkpoint_paths: list[str | Path],
    current_ratings: dict[str, float] | None,
    episodes_per_pair: int,
    seed: int,
    max_decisions: int,
    k: float = 32.0,
) -> dict[str, float]:
    """Incrementally rate newly-added checkpoints against the existing pool."""

    checkpoints = [Path(path) for path in checkpoint_paths]
    ratings = dict(current_ratings or {})
    if not checkpoints:
        return ratings

    policies: dict[Path, PpoCheckpointOpponent] = {}
    match_idx = 0
    for path in checkpoints:
        path_key = str(path)
        if path_key in ratings:
            continue

        ratings[path_key] = 1000.0
        opponents = [opponent for opponent in checkpoints if opponent != path and str(opponent) in ratings]
        if not opponents:
            continue

        policies[path] = PpoCheckpointOpponent(path)
        for opponent in opponents:
            if opponent not in policies:
                policies[opponent] = PpoCheckpointOpponent(opponent)
            opponent_key = str(opponent)
            for episode in range(episodes_per_pair):
                score_new = play_match(
                    policies[path],
                    policies[opponent],
                    seed=seed + match_idx * 100 + episode,
                    max_decisions=max_decisions,
                )
                ratings[path_key], ratings[opponent_key] = update_elo(
                    ratings[path_key],
                    ratings[opponent_key],
                    score_new,
                    k,
                )
            match_idx += 1

    return ratings
