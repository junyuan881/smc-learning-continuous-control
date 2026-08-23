from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def decayed(initial: float, decay: float, episode: int) -> float:
    """Paper footnote schedule: x(N) = x(0) / (1 + delta_x N)."""
    return float(initial / (1.0 + decay * episode))


def stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    temperature = max(float(temperature), 1e-8)
    z = values / temperature
    z = z - np.max(z)
    e = np.exp(np.clip(z, -60.0, 60.0))
    total = float(np.sum(e))
    if not np.isfinite(total) or total <= 0:
        return np.full(values.shape, 1.0 / values.size, dtype=float)
    return e / total


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(window, x.size)
    kernel = np.ones(window, dtype=float) / window
    y = np.convolve(x, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, y])


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
