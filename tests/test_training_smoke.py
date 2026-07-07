from t8_agent.train.cem import evaluate_policy, train_cem
from t8_agent.train.linear_policy import LinearPolicy


def test_linear_policy_evaluates() -> None:
    policy = LinearPolicy.zeros()
    result = evaluate_policy(
        policy=policy,
        episodes=2,
        seed=10,
        max_decisions=50,
        opponent_names=["random"],
    )

    assert result.avg_frames > 0


def test_cem_training_smoke() -> None:
    policy, history = train_cem(
        generations=1,
        population=3,
        elite_fraction=0.34,
        noise_std=0.5,
        episodes_per_candidate=1,
        seed=11,
        max_decisions=60,
        opponent_names=["random"],
    )

    assert policy.weights.shape == LinearPolicy.zeros().weights.shape
    assert len(history) == 1
