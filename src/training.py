from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.agents import ActionTileSarsaAgent, ContinuousQAgent, SMCLearningAgent, SarsaAgent
from src.environment import BoatEnv, StateDiscretizer
from src.utils import decayed


@dataclass
class TrainConfig:
    episodes: int = 3000
    state_bins: int = 6
    state_features: str = "xy"
    gamma: float = 0.99
    alpha0: float = 0.5
    alpha_decay: float = 0.01
    sarsa_tau0: float = 3.0
    sarsa_tau_decay: float = 0.0001
    smc_tau0: float = 25.0
    smc_tau_decay: float = 0.0005
    smc_sigma: float = 0.95
    cont_epsilon0: float = 0.4
    cont_epsilon_decay: float = 0.005
    max_steps: int = 260
    dynamics_variant: str = "paper"


@dataclass
class RunResult:
    rewards: np.ndarray
    final_y: np.ndarray
    success: np.ndarray
    reached_bank: np.ndarray
    lengths: np.ndarray
    trajectories: list[np.ndarray]
    metadata: dict


def make_env(seed: int, cfg: TrainConfig) -> BoatEnv:
    return BoatEnv(seed=seed, max_steps=cfg.max_steps, dynamics_variant=cfg.dynamics_variant)


def _rollout_sarsa(
    agent: SarsaAgent | ActionTileSarsaAgent,
    env: BoatEnv,
    disc: StateDiscretizer,
    cfg: TrainConfig,
    rng: np.random.Generator,
    *,
    evaluation_every: int,
) -> RunResult:
    rewards, final_y, success, reached, lengths = [], [], [], [], []
    snapshots: list[np.ndarray] = []
    for ep in range(cfg.episodes):
        obs = env.reset()
        state = disc.encode(obs)
        alpha = decayed(cfg.alpha0, cfg.alpha_decay, ep)
        tau = decayed(cfg.sarsa_tau0, cfg.sarsa_tau_decay, ep)
        aidx, action = agent.select(state, tau)
        total = 0.0
        info = {"success": False, "reached_bank": False, "final_y": np.nan}
        for step in range(cfg.max_steps):
            next_obs, reward, done, info = env.step(action)
            total += reward
            next_state = None if done else disc.encode(next_obs)
            if done:
                next_idx = None
                next_action = None
            else:
                next_idx, next_action = agent.select(next_state, tau)
            agent.update(state, aidx, reward, next_state, next_idx, alpha, cfg.gamma)
            if done:
                break
            state, aidx, action = next_state, int(next_idx), float(next_action)
        rewards.append(total)
        final_y.append(info.get("final_y") if info.get("final_y") is not None else np.nan)
        success.append(bool(info.get("success", False)))
        reached.append(bool(info.get("reached_bank", False)))
        lengths.append(step + 1)
        if evaluation_every and (ep == 0 or (ep + 1) % evaluation_every == 0 or ep == cfg.episodes - 1):
            snapshots.append(np.array([[s.x, s.y] for s in env.trajectory], dtype=float))
    return RunResult(
        rewards=np.asarray(rewards),
        final_y=np.asarray(final_y),
        success=np.asarray(success),
        reached_bank=np.asarray(reached),
        lengths=np.asarray(lengths),
        trajectories=snapshots,
        metadata={},
    )


def train_sarsa(seed: int, n_actions: int, cfg: TrainConfig, evaluation_every: int = 0) -> tuple[RunResult, SarsaAgent]:
    rng = np.random.default_rng(seed)
    env = make_env(seed, cfg)
    disc = StateDiscretizer(cfg.state_bins, cfg.state_features)
    agent = SarsaAgent(n_actions, rng)
    result = _rollout_sarsa(agent, env, disc, cfg, rng, evaluation_every=evaluation_every)
    result.metadata = {"algorithm": "SARSA", "n_actions": n_actions, "seed": seed}
    return result, agent


def train_tile(seed: int, cfg: TrainConfig, evaluation_every: int = 0) -> tuple[RunResult, ActionTileSarsaAgent]:
    rng = np.random.default_rng(seed)
    env = make_env(seed, cfg)
    disc = StateDiscretizer(cfg.state_bins, cfg.state_features)
    agent = ActionTileSarsaAgent(rng)
    result = _rollout_sarsa(agent, env, disc, cfg, rng, evaluation_every=evaluation_every)
    result.metadata = {"algorithm": "Tile-SARSA", "effective_actions": 80, "seed": seed}
    return result, agent


def train_smc(seed: int, n_particles: int, cfg: TrainConfig, evaluation_every: int = 0) -> tuple[RunResult, SMCLearningAgent]:
    rng = np.random.default_rng(seed)
    env = make_env(seed, cfg)
    disc = StateDiscretizer(cfg.state_bins, cfg.state_features)
    agent = SMCLearningAgent(n_particles, rng, ess_ratio=cfg.smc_sigma)

    rewards, final_y, success, reached, lengths = [], [], [], [], []
    snapshots: list[np.ndarray] = []
    ess_values: list[float] = []
    for ep in range(cfg.episodes):
        obs = env.reset()
        state = disc.encode(obs)
        alpha = decayed(cfg.alpha0, cfg.alpha_decay, ep)
        tau = decayed(cfg.smc_tau0, cfg.smc_tau_decay, ep)
        aidx, action = agent.select(state)
        total = 0.0
        info = {"success": False, "reached_bank": False, "final_y": np.nan}
        for step in range(cfg.max_steps):
            next_obs, reward, done, info = env.step(action)
            total += reward
            next_state = None if done else disc.encode(next_obs)
            if done:
                next_idx = None
                next_action = None
            else:
                next_idx, next_action = agent.select(next_state)
            stat = agent.update(state, aidx, reward, next_state, next_idx, alpha, cfg.gamma, tau)
            ess_values.append(float(stat["ess"]))
            if done:
                break
            state, aidx, action = next_state, int(next_idx), float(next_action)
        rewards.append(total)
        final_y.append(info.get("final_y") if info.get("final_y") is not None else np.nan)
        success.append(bool(info.get("success", False)))
        reached.append(bool(info.get("reached_bank", False)))
        lengths.append(step + 1)
        if evaluation_every and (ep == 0 or (ep + 1) % evaluation_every == 0 or ep == cfg.episodes - 1):
            snapshots.append(np.array([[s.x, s.y] for s in env.trajectory], dtype=float))

    result = RunResult(
        rewards=np.asarray(rewards),
        final_y=np.asarray(final_y),
        success=np.asarray(success),
        reached_bank=np.asarray(reached),
        lengths=np.asarray(lengths),
        trajectories=snapshots,
        metadata={
            "algorithm": "SMC-learning",
            "n_particles": n_particles,
            "seed": seed,
            "total_resamples": agent.total_resamples,
            "mean_ess": float(np.mean(ess_values)) if ess_values else np.nan,
        },
    )
    return result, agent


def train_continuous_q(seed: int, n_anchors: int, cfg: TrainConfig, evaluation_every: int = 0) -> tuple[RunResult, ContinuousQAgent]:
    rng = np.random.default_rng(seed)
    env = make_env(seed, cfg)
    disc = StateDiscretizer(cfg.state_bins, cfg.state_features)
    agent = ContinuousQAgent(n_anchors, rng)
    rewards, final_y, success, reached, lengths = [], [], [], [], []
    snapshots: list[np.ndarray] = []

    for ep in range(cfg.episodes):
        obs = env.reset()
        alpha = decayed(cfg.alpha0, cfg.alpha_decay, ep)
        epsilon = decayed(cfg.cont_epsilon0, cfg.cont_epsilon_decay, ep)
        total = 0.0
        info = {"success": False, "reached_bank": False, "final_y": np.nan}
        for step in range(cfg.max_steps):
            state = disc.encode(obs)
            choice = agent.select(state, epsilon)
            next_obs, reward, done, info = env.step(choice.action)
            next_state = None if done else disc.encode(next_obs)
            agent.update(state, choice, reward, next_state, alpha, cfg.gamma)
            total += reward
            obs = next_obs
            if done:
                break
        rewards.append(total)
        final_y.append(info.get("final_y") if info.get("final_y") is not None else np.nan)
        success.append(bool(info.get("success", False)))
        reached.append(bool(info.get("reached_bank", False)))
        lengths.append(step + 1)
        if evaluation_every and (ep == 0 or (ep + 1) % evaluation_every == 0 or ep == cfg.episodes - 1):
            snapshots.append(np.array([[s.x, s.y] for s in env.trajectory], dtype=float))

    result = RunResult(
        rewards=np.asarray(rewards),
        final_y=np.asarray(final_y),
        success=np.asarray(success),
        reached_bank=np.asarray(reached),
        lengths=np.asarray(lengths),
        trajectories=snapshots,
        metadata={"algorithm": "Continuous-Q", "n_anchors": n_anchors, "seed": seed},
    )
    return result, agent


def evaluate_policy(
    agent,
    algorithm: str,
    cfg: TrainConfig,
    *,
    seed: int = 999,
    episodes: int = 50,
) -> dict:
    env = make_env(seed, cfg)
    disc = StateDiscretizer(cfg.state_bins, cfg.state_features)
    rewards: list[float] = []
    ys: list[float] = []
    successes: list[bool] = []
    trajectories: list[np.ndarray] = []

    for ep in range(episodes):
        obs = env.reset()
        total = 0.0
        for _ in range(cfg.max_steps):
            state = disc.encode(obs)
            if algorithm == "smc":
                _, action = agent.select(state, greedy=True)
            elif algorithm in {"sarsa", "tile"}:
                _, action = agent.select(state, temperature=1e-6, greedy=True)
            elif algorithm == "continuous_q":
                action = agent.select(state, epsilon=0.0, greedy=True).action
            else:
                raise ValueError(algorithm)
            obs, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
        ys.append(float(info["final_y"]) if info.get("final_y") is not None else np.nan)
        successes.append(bool(info.get("success", False)))
        if ep < 9:
            trajectories.append(np.array([[s.x, s.y] for s in env.trajectory], dtype=float))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_final_y": float(np.nanmean(ys)),
        "trajectories": trajectories,
    }
