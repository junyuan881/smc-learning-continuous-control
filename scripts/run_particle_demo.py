from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import SMCLearningAgent
from src.utils import save_json


def reward_fn(a: np.ndarray | float) -> np.ndarray | float:
    """A two-peak continuous reward surface with a narrow global optimum."""
    x = np.asarray(a, dtype=float)
    y = 10.0 * np.exp(-0.5 * ((x + 35.0) / 7.0) ** 2)
    y += 7.0 * np.exp(-0.5 * ((x - 45.0) / 11.0) ** 2)
    return y if isinstance(a, np.ndarray) else float(y)


def main() -> None:
    p = argparse.ArgumentParser(description="Fast visual demo of the SMC actor mechanism.")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--particles", type=int, default=20)
    p.add_argument("--seed", type=int, default=114024511)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / "particle_demo")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    agent = SMCLearningAgent(args.particles, rng, ess_ratio=0.85)
    state = (0, 0)
    checkpoints = {0, 20, 50, 100, 250, 500, args.steps - 1}
    history: list[tuple[int, np.ndarray, np.ndarray]] = []

    for t in range(args.steps):
        idx, action = agent.select(state)
        reward = reward_fn(action)
        tau = max(0.15, 2.0 / (1.0 + 0.01 * t))
        agent.update(state, idx, reward, None, None, alpha=0.3, gamma=0.0, temperature=tau)
        if t in checkpoints:
            ps = agent.get(state)
            history.append((t + 1, ps.actions.copy(), ps.weights.copy()))

    grid = np.linspace(-90, 90, 800)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grid, reward_fn(grid), linewidth=2, label="reward surface")
    ps = agent.get(state)
    ax.scatter(ps.actions, reward_fn(ps.actions), s=50 + 700 * ps.weights, alpha=0.75, label="final SMC particles")
    ax.axvline(-35, linestyle="--", linewidth=1, label="global optimum")
    ax.set_xlabel("continuous action")
    ax.set_ylabel("reward")
    ax.set_title("SMC particles concentrate near the high-value action region")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "particle_convergence.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for step, actions, weights in history:
        ax.scatter(actions, np.full_like(actions, step), s=20 + 250 * weights, alpha=0.65)
    ax.set_xlabel("continuous action")
    ax.set_ylabel("training step")
    ax.set_title("Particle movement across SMC resampling checkpoints")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out / "particle_evolution.png", dpi=170)
    plt.close(fig)

    best_idx = int(np.argmax(ps.q))
    summary = {
        "student": {"name": "謝濬遠", "student_id": "114024511"},
        "steps": args.steps,
        "particles": args.particles,
        "best_particle": float(ps.actions[best_idx]),
        "weighted_mean_action": float(np.sum(ps.actions * ps.weights)),
        "global_optimum": -35.0,
        "absolute_error_of_best_particle": float(abs(ps.actions[best_idx] + 35.0)),
        "resampling_events": int(agent.total_resamples),
    }
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
