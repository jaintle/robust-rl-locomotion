"""
Deterministic clean evaluation harness.

Phase 2 — no PPO, no wrappers, no perturbations.
File I/O is the caller's responsibility; this module is pure evaluation logic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from robust_rl_locomotion.envs.make_env import make_env
from robust_rl_locomotion.eval.metrics import (
    eval_seed_list,
    hash_seed_list,
    summarize_episodes,
)


def evaluate_policy(
    env_id: str,
    seed: int,
    episodes: int,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    render_mode: str | None = None,
) -> dict:
    """Run a deterministic episode loop and return aggregate metrics.

    Evaluation seeds are derived from ``seed`` and are fixed for every call
    with the same arguments, ensuring reproducibility across runs.

    Args:
        env_id:     Gymnasium environment identifier (e.g. "HalfCheetah-v4").
        seed:       Base seed; used both to initialise the environment and to
                    derive the per-episode seed list.
        episodes:   Number of episodes to evaluate.
        policy_fn:  Callable that accepts an observation (np.ndarray) and
                    returns an action (np.ndarray, dtype float32).
        render_mode: Passed through to make_env; None for headless evaluation.

    Returns:
        Dict with keys:
            env_id, seed, episodes, eval_seed_list_hash,
            return_mean, return_std, episode_len_mean, episode_len_std.
    """
    # Build fixed seed list and its audit hash.
    seeds = eval_seed_list(base_seed=seed + 1000, episodes=episodes)
    seed_hash = hash_seed_list(seeds)

    env = make_env(env_id, seed=seed, render_mode=render_mode)

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_shape = env.action_space.shape

    returns: list[float] = []
    lengths: list[int] = []

    for episode_seed in seeds:
        obs, _ = env.reset(seed=episode_seed)
        ep_return = 0.0
        ep_length = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            raw_action = policy_fn(obs)
            # Guarantee correct shape, dtype, and bounds.
            action = np.asarray(raw_action, dtype=np.float32).reshape(action_shape)
            action = np.clip(action, action_low, action_high)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1

        returns.append(ep_return)
        lengths.append(ep_length)

    env.close()

    summary = summarize_episodes(returns, lengths)

    return {
        "env_id": env_id,
        "seed": seed,
        "episodes": episodes,
        "eval_seed_list_hash": seed_hash,
        **summary,
    }
