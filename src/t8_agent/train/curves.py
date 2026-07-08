from __future__ import annotations

import json
import os
from pathlib import Path


def plot_selfplay_metrics(metrics_path: str | Path, output_path: str | Path | None = None) -> Path:
    metrics_path = Path(metrics_path)
    output_path = Path(output_path) if output_path else metrics_path.with_name("curves.png")
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))

    import matplotlib.pyplot as plt
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = metrics.get("history", [])
    if not history:
        raise ValueError(f"no history entries found in {metrics_path}")

    iterations = [item["iteration"] for item in history]
    scripted_win_rate = [item.get("scripted_eval_win_rate", item.get("eval_win_rate", 0.0)) for item in history]
    scripted_reward = [item.get("scripted_eval_avg_reward", item.get("eval_avg_reward", 0.0)) for item in history]
    pool_win_rate = [item.get("checkpoint_eval_win_rate") for item in history]
    pool_reward = [item.get("checkpoint_eval_avg_reward") for item in history]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(iterations, scripted_win_rate, marker="o", label="scripted")
    if any(value is not None for value in pool_win_rate):
        axes[0].plot(iterations, _none_to_nan(pool_win_rate), marker="o", label="checkpoint pool")
    axes[0].set_ylabel("Win rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(iterations, scripted_reward, marker="o", label="scripted")
    if any(value is not None for value in pool_reward):
        axes[1].plot(iterations, _none_to_nan(pool_reward), marker="o", label="checkpoint pool")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Average reward")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Tekken-lite PPO Self-Play Curves")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _none_to_nan(values: list[float | None]) -> list[float]:
    return [float("nan") if value is None else float(value) for value in values]
