"""Generate a small synthetic offline RL dataset matching data.csv schema.

Used by unit/smoke tests and local quickstart verification when the full
competition dataset (14 MB CSV) is not bundled in git.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def generate_synthetic_csv(
    output_path: str | Path,
    num_episodes: int = 12,
    steps_per_episode: int = 45,
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "",
        "index",
        "obs_1",
        "obs_2",
        "obs_3",
        "obs_4",
        "obs_5",
        "action_1",
        "action_2",
        "action_3",
        "reward",
    ]
    row_id = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ep in range(num_episodes):
            state = rng.uniform(-0.5, 0.5, size=5).astype(np.float64)
            target = np.zeros(5, dtype=np.float64)
            for _ in range(steps_per_episode):
                # Simple stabilizing feedback controller + exploration noise
                action = np.clip(-0.6 * state[:3] + rng.normal(0.0, 0.08, size=3), -1.0, 1.0)
                err = float(np.linalg.norm(state - target))
                reward = float(1.5 - err)
                writer.writerow(
                    [row_id, ep]
                    + [round(float(x), 6) for x in state]
                    + [round(float(a), 6) for a in action]
                    + [round(reward, 6)]
                )
                next_state = state.copy()
                next_state[:3] = 0.92 * state[:3] + 0.12 * action + rng.normal(0.0, 0.01, size=3)
                next_state[3:] = 0.88 * state[3:] + 0.05 * action[:2] + rng.normal(0.0, 0.01, size=2)
                state = next_state
                row_id += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic data.csv for smoke tests")
    parser.add_argument("--output", default="data/demo_data.csv", help="Output CSV file path")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = generate_synthetic_csv(args.output, args.episodes, args.steps, args.seed)
    print(f"Generated synthetic offline RL trajectories -> {out} ({args.episodes} episodes x {args.steps} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
