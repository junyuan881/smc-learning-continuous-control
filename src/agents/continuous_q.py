from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class ContinuousActionChoice:
    action: float
    anchor: int
    left_weight: float
    center_weight: float
    right_weight: float
    q_value: float


class ContinuousQAgent:
    """Continuous-action Q-learning interpolation baseline.

    It implements the local three-anchor action interpolation described by Millan,
    Posenato & Dedieu (2002), while using the boat paper's discrete state coding.
    The original method used ITPM state representation and eligibility traces; those
    details are not specified for the boat comparison, so this reproduction uses TD(0).
    """

    def __init__(self, n_anchors: int, rng: np.random.Generator) -> None:
        self.n_anchors = int(n_anchors)
        width = 180.0 / self.n_anchors
        self.actions = -90.0 + width * (np.arange(self.n_anchors, dtype=float) + 0.5)
        self.q = defaultdict(lambda: np.zeros(self.n_anchors, dtype=float))
        self.rng = rng

    def _interpolate(self, state: tuple[int, ...], anchor: int) -> ContinuousActionChoice:
        q = self.q[state]
        l = int(anchor)
        ql = float(q[l])
        left_idx = max(0, l - 1)
        right_idx = min(self.n_anchors - 1, l + 1)
        left = 0.0 if left_idx == l else 1.0 / (2.0 + (ql - float(q[left_idx])) ** 2)
        right = 0.0 if right_idx == l else 1.0 / (2.0 + (ql - float(q[right_idx])) ** 2)
        action = float(
            self.actions[l]
            + right * (self.actions[right_idx] - self.actions[l])
            + left * (self.actions[left_idx] - self.actions[l])
        )
        denom = 1.0 + left + right
        q_value = float((ql + right * q[right_idx] + left * q[left_idx]) / denom)
        return ContinuousActionChoice(
            action=action,
            anchor=l,
            left_weight=left / denom,
            center_weight=1.0 / denom,
            right_weight=right / denom,
            q_value=q_value,
        )

    def select(self, state: tuple[int, ...], epsilon: float, greedy: bool = False) -> ContinuousActionChoice:
        q = self.q[state]
        if (not greedy) and self.rng.random() < epsilon:
            anchor = int(self.rng.integers(0, self.n_anchors))
        else:
            best = np.flatnonzero(np.isclose(q, np.max(q)))
            anchor = int(self.rng.choice(best))
        return self._interpolate(state, anchor)

    def update(
        self,
        state: tuple[int, ...],
        choice: ContinuousActionChoice,
        reward: float,
        next_state: tuple[int, ...] | None,
        alpha: float,
        gamma: float,
    ) -> float:
        if next_state is None:
            bootstrap = 0.0
        else:
            nq = self.q[next_state]
            next_best = np.flatnonzero(np.isclose(nq, np.max(nq)))
            next_anchor = int(self.rng.choice(next_best))
            bootstrap = self._interpolate(next_state, next_anchor).q_value
        td = float(reward + gamma * bootstrap - choice.q_value)

        q = self.q[state]
        l = choice.anchor
        q[l] += alpha * td * choice.center_weight
        if l > 0:
            q[l - 1] += alpha * td * choice.left_weight
        if l < self.n_anchors - 1:
            q[l + 1] += alpha * td * choice.right_weight
        return td
