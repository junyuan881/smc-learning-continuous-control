from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.utils import stable_softmax


class ActionTileSarsaAgent:
    """CMAC-style baseline over the action coordinate.

    The SMC paper states two tilings at 2.25 degree effective resolution, equivalent
    to 80 actions, but does not provide the complete CMAC specification. This
    implementation keeps the paper's tabular state discretization and uses two
    offset action tilings to generalize across neighboring actions.
    """

    def __init__(self, rng: np.random.Generator, resolution: float = 2.25, n_tilings: int = 2) -> None:
        self.rng = rng
        self.resolution = float(resolution)
        self.n_tilings = int(n_tilings)
        self.candidates = -90.0 + self.resolution * (np.arange(80, dtype=float) + 0.5)
        self.tile_width = self.resolution * self.n_tilings
        self.weights = defaultdict(dict)

    def _features(self, state: tuple[int, ...], action: float) -> list[tuple[int, int]]:
        feats = []
        for t in range(self.n_tilings):
            offset = t * self.resolution
            idx = int(np.floor((action + 90.0 + offset) / self.tile_width))
            feats.append((t, idx))
        return feats

    def q_value(self, state: tuple[int, ...], action: float) -> float:
        table = self.weights[state]
        return float(sum(table.get(f, 0.0) for f in self._features(state, action)) / self.n_tilings)

    def select(self, state: tuple[int, ...], temperature: float, greedy: bool = False) -> tuple[int, float]:
        q = np.array([self.q_value(state, a) for a in self.candidates], dtype=float)
        if greedy:
            best = np.flatnonzero(np.isclose(q, np.max(q)))
            idx = int(self.rng.choice(best))
        else:
            idx = int(self.rng.choice(q.size, p=stable_softmax(q, temperature)))
        return idx, float(self.candidates[idx])

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
        action = float(self.candidates[action_idx])
        old = self.q_value(state, action)
        bootstrap = 0.0 if next_state is None else self.q_value(next_state, float(self.candidates[int(next_action_idx)]))
        td = reward + gamma * bootstrap - old
        table = self.weights[state]
        for f in self._features(state, action):
            table[f] = table.get(f, 0.0) + alpha * td / self.n_tilings
        return float(td)
