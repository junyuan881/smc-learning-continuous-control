from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.utils import stable_softmax


class SarsaAgent:
    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self.n_actions = int(n_actions)
        width = 180.0 / self.n_actions
        self.actions = -90.0 + width * (np.arange(self.n_actions, dtype=float) + 0.5)
        self.q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))
        self.rng = rng

    def select(self, state: tuple[int, ...], temperature: float, greedy: bool = False) -> tuple[int, float]:
        values = self.q[state]
        if greedy:
            best = np.flatnonzero(np.isclose(values, np.max(values)))
            idx = int(self.rng.choice(best))
        else:
            probs = stable_softmax(values, temperature)
            idx = int(self.rng.choice(self.n_actions, p=probs))
        return idx, float(self.actions[idx])

    def update(
        self,
        state: tuple[int, ...],
        action_idx: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        next_action_idx: int | None,
        alpha: float,
        gamma: float,
    ) -> float:
        old = float(self.q[state][action_idx])
        bootstrap = 0.0 if next_state is None else float(self.q[next_state][int(next_action_idx)])
        target = reward + gamma * bootstrap
        self.q[state][action_idx] = (1.0 - alpha) * old + alpha * target
        return float(self.q[state][action_idx] - old)
