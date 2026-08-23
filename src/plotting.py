from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import moving_average


def plot_learning_curves(series: dict[str, np.ndarray], out: str | Path, window: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, rewards in series.items():
        rewards = np.asarray(rewards, dtype=float)
        y = moving_average(rewards, window)
        ax.plot(np.arange(1, rewards.size + 1), y, label=label, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Boat task learning curves (moving average = {window})")
    ax.set_ylim(-10.5, 10.5)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_trajectories(trajectories: list[np.ndarray], out: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 200, 400)
    current = 1.25 * (x / 50.0 - (x / 100.0) ** 2)
    # Visual river-current cue inspired by the paper; it is not a physical bank boundary.
    current_curve = 55.0 * np.maximum(current / max(np.max(current), 1e-9), 0.0)
    ax.plot(x, current_curve, linestyle="--", linewidth=1.0, label="current profile (scaled)")
    ax.axhspan(100, 120, alpha=0.08, label="viability zone")
    ax.axhspan(109.9, 110.1, alpha=0.18, label="success zone")
    for i, tr in enumerate(trajectories):
        ax.plot(tr[:, 0], tr[:, 1], linewidth=1.3, alpha=0.8, label="trajectory" if i == 0 else None)
    ax.scatter([200], [110], marker="x", s=70, label="quay")
    ax.set_xlim(0, 202)
    ax.set_ylim(0, 200)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def plot_particles(agent, state: tuple[int, ...], out: str | Path, title: str = "SMC actor particles") -> None:
    ps = agent.get(state)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    size = 50 + 700 * ps.weights
    ax.scatter(ps.actions, ps.q, s=size, alpha=0.7)
    for a, q, w in zip(ps.actions, ps.q, ps.weights):
        ax.annotate(f"w={w:.2f}", (a, q), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("Continuous action U (degrees)")
    ax.set_ylabel("Q estimate")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
