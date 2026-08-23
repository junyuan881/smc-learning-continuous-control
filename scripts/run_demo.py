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

from src.environment import StateDiscretizer
from src.plotting import plot_learning_curves, plot_particles, plot_trajectories
from src.training import TrainConfig, evaluate_policy, train_continuous_q, train_sarsa, train_smc, train_tile
from src.utils import save_json


def _save_rewards(path: Path, rows: dict[str, np.ndarray]) -> None:
    names = list(rows)
    n = max(len(v) for v in rows.values())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", *names])
        for i in range(n):
            writer.writerow([i + 1, *[rows[name][i] if i < len(rows[name]) else "" for name in names]])


def main() -> None:
    p = argparse.ArgumentParser(description="Run a compact, CPU-friendly SMC-learning reproduction demo.")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--state-bins", type=int, default=6, help="Demo default is coarser than paper's 10 bins.")
    p.add_argument("--seed", type=int, default=114024511)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / "demo")
    p.add_argument("--all-baselines", action="store_true", help="Also run tile coding and Continuous-Q baselines.")
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(episodes=args.episodes, state_bins=args.state_bins)
    eval_every = max(1, args.episodes // 6)

    runs = {}
    evaluations = {}

    smc5, smc5_agent = train_smc(args.seed, 5, cfg, evaluation_every=eval_every)
    smc10, smc10_agent = train_smc(args.seed + 1, 10, cfg, evaluation_every=eval_every)
    sarsa5, sarsa5_agent = train_sarsa(args.seed + 2, 5, cfg, evaluation_every=eval_every)
    sarsa40, sarsa40_agent = train_sarsa(args.seed + 3, 40, cfg, evaluation_every=eval_every)

    runs["SMC-5"] = smc5.rewards
    runs["SMC-10"] = smc10.rewards
    runs["SARSA-5"] = sarsa5.rewards
    runs["SARSA-40"] = sarsa40.rewards

    evaluations["SMC-5"] = evaluate_policy(smc5_agent, "smc", cfg, seed=args.seed + 100, episodes=50)
    evaluations["SMC-10"] = evaluate_policy(smc10_agent, "smc", cfg, seed=args.seed + 101, episodes=50)
    evaluations["SARSA-5"] = evaluate_policy(sarsa5_agent, "sarsa", cfg, seed=args.seed + 102, episodes=50)
    evaluations["SARSA-40"] = evaluate_policy(sarsa40_agent, "sarsa", cfg, seed=args.seed + 103, episodes=50)

    if args.all_baselines:
        tile, tile_agent = train_tile(args.seed + 4, cfg)
        cont, cont_agent = train_continuous_q(args.seed + 5, 40, cfg)
        runs["Tile-80"] = tile.rewards
        runs["Continuous-Q-40"] = cont.rewards
        evaluations["Tile-80"] = evaluate_policy(tile_agent, "tile", cfg, seed=args.seed + 104, episodes=50)
        evaluations["Continuous-Q-40"] = evaluate_policy(cont_agent, "continuous_q", cfg, seed=args.seed + 105, episodes=50)

    plot_learning_curves(runs, out / "learning_curves.png", window=max(20, args.episodes // 30))
    plot_trajectories(evaluations["SMC-10"]["trajectories"], out / "smc10_trajectories.png", "SMC-learning (10 particles): greedy evaluation")

    # Inspect actor particles in a representative central state.
    disc = StateDiscretizer(args.state_bins, cfg.state_features)
    representative = disc.encode(np.array([100.0, 110.0, 0.0, 0.0, 1.75]))
    plot_particles(smc10_agent, representative, out / "smc_particles.png", "SMC actor particles at a representative state")

    serializable_eval = {
        k: {kk: vv for kk, vv in v.items() if kk != "trajectories"}
        for k, v in evaluations.items()
    }
    summary = {
        "student": {"name": "謝濬遠", "student_id": "114024511"},
        "config": cfg.__dict__,
        "seed": args.seed,
        "evaluation": serializable_eval,
        "notes": [
            "This is a compact demo, not a claim of exact numerical replication of Figure 2.",
            "Use scripts/run_paper_scale.py for the paper's 10 state bins and 100,000 episodes.",
        ],
    }
    save_json(out / "summary.json", summary)
    _save_rewards(out / "rewards.csv", runs)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved demo outputs to: {out}")


if __name__ == "__main__":
    main()
