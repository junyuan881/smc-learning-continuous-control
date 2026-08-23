from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import stable_softmax


@dataclass
class ParticleSet:
    actions: np.ndarray
    weights: np.ndarray
    q: np.ndarray
    resamples: int = 0


class SMCLearningAgent:
    """Sequential Monte Carlo actor with a tabular SARSA critic.

    Each discrete state owns N continuous action particles. Importance weights are
    updated with exp(Delta Q / tau), ESS triggers systematic resampling, and a local
    uniform-kernel move step keeps the particle set continuous.
    """

    def __init__(
        self,
        n_particles: int,
        rng: np.random.Generator,
        *,
        action_low: float = -90.0,
        action_high: float = 90.0,
        ess_ratio: float = 0.95,
    ) -> None:
        self.n_particles = int(n_particles)
        self.rng = rng
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.ess_ratio = float(ess_ratio)
        self.states: dict[tuple[int, ...], ParticleSet] = {}
        self.total_resamples = 0

    def _new_particles(self) -> ParticleSet:
        actions = np.sort(self.rng.uniform(self.action_low, self.action_high, size=self.n_particles))
        return ParticleSet(
            actions=actions,
            weights=np.full(self.n_particles, 1.0 / self.n_particles, dtype=float),
            q=np.zeros(self.n_particles, dtype=float),
        )

    def get(self, state: tuple[int, ...]) -> ParticleSet:
        if state not in self.states:
            self.states[state] = self._new_particles()
        return self.states[state]

    def select(self, state: tuple[int, ...], greedy: bool = False) -> tuple[int, float]:
        ps = self.get(state)
        if greedy:
            score = ps.q + 1e-9 * ps.weights
            best = np.flatnonzero(np.isclose(score, np.max(score), atol=1e-12, rtol=1e-12))
            idx = int(self.rng.choice(best))
        else:
            idx = int(self.rng.choice(self.n_particles, p=ps.weights))
        return idx, float(ps.actions[idx])

    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=float)
        return float(1.0 / np.sum(np.square(weights)))

    def _systematic_indices(self, weights: np.ndarray) -> np.ndarray:
        n = weights.size
        positions = (self.rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="right")

    def _move_from_parent_cells(self, old_actions: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
        """Uniform-kernel smoothing using local cells around pre-resampling particles."""
        order = np.argsort(old_actions)
        sorted_actions = old_actions[order]
        rank_of = np.empty_like(order)
        rank_of[order] = np.arange(order.size)
        moved = np.empty(parent_idx.size, dtype=float)

        for k, parent in enumerate(parent_idx):
            r = int(rank_of[parent])
            a = float(sorted_actions[r])
            if self.n_particles == 1:
                lo, hi = self.action_low - a, self.action_high - a
            elif r == 0:
                d = float(sorted_actions[1] - a)
                lo, hi = -d, d / 2.0
            elif r == self.n_particles - 1:
                d = float(a - sorted_actions[-2])
                lo, hi = -d / 2.0, d
            else:
                lo = float((sorted_actions[r - 1] - a) / 2.0)
                hi = float((sorted_actions[r + 1] - a) / 2.0)
            if hi < lo:
                lo, hi = hi, lo
            moved[k] = np.clip(a + self.rng.uniform(lo, hi), self.action_low, self.action_high)
        return moved

    def _resample(self, ps: ParticleSet, temperature: float) -> None:
        old_actions = ps.actions.copy()
        old_q = ps.q.copy()
        parents = self._systematic_indices(ps.weights)
        new_actions = self._move_from_parent_cells(old_actions, parents)
        new_q = old_q[parents].copy()

        order = np.argsort(new_actions)
        ps.actions = new_actions[order]
        ps.q = new_q[order]
        ps.weights = stable_softmax(ps.q, temperature)
        ps.resamples += 1
        self.total_resamples += 1

    def update(
        self,
        state: tuple[int, ...],
        action_idx: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        next_action_idx: int | None,
        alpha: float,
        gamma: float,
        temperature: float,
    ) -> dict[str, float | bool]:
        ps = self.get(state)
        old_q = float(ps.q[action_idx])
        if next_state is None:
            bootstrap = 0.0
        else:
            next_ps = self.get(next_state)
            bootstrap = float(next_ps.q[int(next_action_idx)])
        target = reward + gamma * bootstrap
        new_q = (1.0 - alpha) * old_q + alpha * target
        delta_q = float(new_q - old_q)
        ps.q[action_idx] = new_q

        # Eq. (2): w <- w * exp(Delta Q / tau), then normalize.
        log_factor = float(np.clip(delta_q / max(temperature, 1e-8), -60.0, 60.0))
        ps.weights[action_idx] *= np.exp(log_factor)
        total = float(np.sum(ps.weights))
        if not np.isfinite(total) or total <= 0.0:
            ps.weights[:] = 1.0 / self.n_particles
        else:
            ps.weights /= total

        ess = self.effective_sample_size(ps.weights)
        did_resample = bool(ess / self.n_particles < self.ess_ratio)
        if did_resample:
            self._resample(ps, temperature)

        return {"delta_q": delta_q, "ess": ess, "resampled": did_resample}
