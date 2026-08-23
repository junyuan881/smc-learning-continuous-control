from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting import plot_learning_curves
from src.training import TrainConfig, evaluate_policy, train_continuous_q, train_sarsa, train_smc, train_tile
from src.utils import save_json


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-scale boat experiment (default: 100,000 episodes).")
    p.add_argument("--episodes", type=int, default=100_000)
    p.add_argument("--seeds", type=int, default=3, help="Independent runs per algorithm; raise for publication-quality statistics.")
    p.add_argument("--base-seed", type=int, default=114024511)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "paper_scale")
    p.add_argument("--skip-slow-baselines", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(episodes=args.episodes, state_bins=10)
    specs = [
        ("SMC-5", lambda seed: train_smc(seed, 5, cfg)[0]),
        ("SMC-10", lambda seed: train_smc(seed, 10, cfg)[0]),
        ("SARSA-5", lambda seed: train_sarsa(seed, 5, cfg)[0]),
        ("SARSA-10", lambda seed: train_sarsa(seed, 10, cfg)[0]),
        ("SARSA-20", lambda seed: train_sarsa(seed, 20, cfg)[0]),
        ("SARSA-40", lambda seed: train_sarsa(seed, 40, cfg)[0]),
    ]
    if not args.skip_slow_baselines:
        specs += [
            ("Tile-80", lambda seed: train_tile(seed, cfg)[0]),
            ("Continuous-Q-40", lambda seed: train_continuous_q(seed, 40, cfg)[0]),
        ]

    means: dict[str, np.ndarray] = {}
    summaries = {}
    for alg_i, (label, fn) in enumerate(specs):
        reward_runs = []
        for run in range(args.seeds):
            seed = args.base_seed + 1000 * alg_i + run
            result = fn(seed)
            reward_runs.append(result.rewards)
        arr = np.vstack(reward_runs)
        means[label] = np.mean(arr, axis=0)
        summaries[label] = {
            "final_5000_mean_reward": float(np.mean(arr[:, -min(5000, args.episodes):])),
            "runs": args.seeds,
        }
        np.save(args.out / f"{label.lower().replace('-', '_')}_rewards.npy", arr)

    plot_learning_curves(means, args.out / "paper_scale_learning_curves.png", window=max(100, args.episodes // 100))
    save_json(args.out / "summary.json", {
        "student": {"name": "謝濬遠", "student_id": "114024511"},
        "config": cfg.__dict__,
        "results": summaries,
        "reproduction_status": "faithful core reconstruction; see docs/REPRODUCTION_NOTES.md for ambiguities",
    })
    print(json.dumps(summaries, indent=2))
    print(f"Saved paper-scale results to {args.out}")


if __name__ == "__main__":
    main()
