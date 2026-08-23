from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class BoatState:
    x: float
    y: float
    delta: float  # boat angle, degrees
    omega: float  # angular/rudder response state, degrees
    speed: float
    prev_speed: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.delta, self.omega, self.speed], dtype=float)


class BoatEnv:
    """Boat control task reconstructed from Section 4.1 of Lazaric et al.

    The uploaded paper leaves a few implementation details unspecified (exact start-bank
    points, initial dynamic state, and whether the y update's s_{t-1} is intentional).
    Those choices are explicit constructor arguments instead of hidden assumptions.
    """

    def __init__(
        self,
        *,
        fc: float = 1.25,
        inertia: float = 0.1,
        s_max: float = 2.5,
        s_desired: float = 1.75,
        p: float = 0.9,
        quay_y: float = 110.0,
        success_width: float = 0.2,
        viability_width: float = 20.0,
        action_low: float = -90.0,
        action_high: float = 90.0,
        max_steps: int = 260,
        start_y_values: Iterable[float] = (20, 40, 60, 80, 100, 120, 140, 160, 180),
        initial_speed: float | None = None,
        dynamics_variant: str = "paper",
        seed: int = 0,
    ) -> None:
        self.fc = float(fc)
        self.inertia = float(inertia)
        self.s_max = float(s_max)
        self.s_desired = float(s_desired)
        self.p = float(p)
        self.quay_y = float(quay_y)
        self.success_width = float(success_width)
        self.viability_width = float(viability_width)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.max_steps = int(max_steps)
        self.start_y_values = np.asarray(tuple(start_y_values), dtype=float)
        self.initial_speed = self.s_desired if initial_speed is None else float(initial_speed)
        if dynamics_variant not in {"paper", "s_next"}:
            raise ValueError("dynamics_variant must be 'paper' or 's_next'")
        self.dynamics_variant = dynamics_variant
        self.rng = np.random.default_rng(seed)
        self.state: BoatState | None = None
        self.steps = 0
        self.trajectory: list[BoatState] = []

    def reset(self, *, seed: int | None = None, start_y: float | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if start_y is None:
            start_y = float(self.rng.choice(self.start_y_values))
        self.state = BoatState(
            x=0.0,
            y=float(np.clip(start_y, 0.0, 200.0)),
            delta=0.0,
            omega=0.0,
            speed=self.initial_speed,
            prev_speed=self.initial_speed,
        )
        self.steps = 0
        self.trajectory = [self._copy_state(self.state)]
        return self.state.as_array()

    @staticmethod
    def _copy_state(s: BoatState) -> BoatState:
        return BoatState(s.x, s.y, s.delta, s.omega, s.speed, s.prev_speed)

    def current_effect(self, x: float) -> float:
        return self.fc * (x / 50.0 - (x / 100.0) ** 2)

    def terminal_reward(self, y: float) -> float:
        """Equation (7), interpreting published zone widths as total widths.

        D decreases linearly from +10 at the success-zone boundary to -10 at the
        viability-zone boundary. The exact convention for 'width' is not stated in
        the paper; this project uses total width, hence half-width around quay_y.
        """
        d = abs(float(y) - self.quay_y)
        success_half = self.success_width / 2.0
        viability_half = self.viability_width / 2.0
        if d <= success_half:
            return 10.0
        if d <= viability_half:
            frac = (d - success_half) / max(viability_half - success_half, 1e-12)
            return float(10.0 - 20.0 * frac)
        return -10.0

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        s = self.state
        u = float(np.clip(action, self.action_low, self.action_high))

        # Equations from the paper. Angles are represented in degrees and converted
        # to radians only for trigonometric functions.
        speed_next = s.speed + (self.s_desired - s.speed) * self.inertia
        rudder = float(np.clip(self.p * (u - s.delta), -45.0, 45.0))
        omega_next = s.omega + ((rudder - s.omega) * (speed_next / self.s_max))
        delta_next = s.delta + self.inertia * omega_next

        x_next = float(np.clip(s.x + speed_next * np.cos(np.deg2rad(delta_next)), 0.0, 200.0))
        vertical_speed = s.prev_speed if self.dynamics_variant == "paper" else speed_next
        y_next = float(
            np.clip(
                s.y - vertical_speed * np.sin(np.deg2rad(delta_next)) - self.current_effect(x_next),
                0.0,
                200.0,
            )
        )

        next_state = BoatState(
            x=x_next,
            y=y_next,
            delta=delta_next,
            omega=omega_next,
            speed=speed_next,
            prev_speed=s.speed,
        )
        self.state = next_state
        self.steps += 1
        self.trajectory.append(self._copy_state(next_state))

        reached_bank = bool(x_next >= 200.0 - 1e-9)
        timed_out = bool(self.steps >= self.max_steps)
        done = reached_bank or timed_out
        reward = self.terminal_reward(y_next) if reached_bank else (-10.0 if timed_out else 0.0)
        info = {
            "reached_bank": reached_bank,
            "timed_out": timed_out,
            "success": reached_bank and abs(y_next - self.quay_y) <= self.success_width / 2.0,
            "final_y": y_next if done else None,
        }
        return next_state.as_array(), float(reward), done, info


class StateDiscretizer:
    """Sparse tabular state coding.

    ``features="xy"`` uses the two boat coordinates explicitly presented as the
    task state in Section 4.1 and is the default reproduction choice.
    ``features="full"`` additionally includes internal dynamic variables as a
    diagnostic alternative.
    """

    def __init__(self, bins: int = 10, features: str = "xy") -> None:
        self.bins = int(bins)
        if features not in {"xy", "full"}:
            raise ValueError("features must be 'xy' or 'full'")
        self.features = features
        if features == "xy":
            self.lows = np.array([0.0, 0.0], dtype=float)
            self.highs = np.array([200.0, 200.0], dtype=float)
        else:
            self.lows = np.array([0.0, 0.0, -180.0, -45.0, 0.0], dtype=float)
            self.highs = np.array([200.0, 200.0, 180.0, 45.0, 2.5], dtype=float)

    def encode(self, observation: np.ndarray) -> tuple[int, ...]:
        x = np.asarray(observation, dtype=float)
        if self.features == "xy":
            x = x[:2]
        clipped = np.clip(x, self.lows, self.highs)
        scaled = (clipped - self.lows) / np.maximum(self.highs - self.lows, 1e-12)
        idx = np.floor(scaled * self.bins).astype(int)
        idx = np.clip(idx, 0, self.bins - 1)
        return tuple(int(i) for i in idx)
