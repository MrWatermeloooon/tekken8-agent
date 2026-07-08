from __future__ import annotations

import argparse

from t8_agent.train.curves import plot_selfplay_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PPO self-play training curves.")
    parser.add_argument("--metrics", default="runs/ppo_selfplay_latest/metrics.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output = plot_selfplay_metrics(args.metrics, args.out)
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
