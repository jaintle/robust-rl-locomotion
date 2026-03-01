"""
Deterministic evaluation harness.

Phase 2 — core episode loop and metric aggregation.
Phase 5 — extended with optional observation wrapper and caller-supplied
           episode seed list for exact cross-severity comparability.

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
    wrapper_fn: Callable | None = None,
    wrapper_kwargs: dict | None = None,
    episode_seeds: list[int] | None = None,
) -> dict:
    """Run a deterministic episode loop and return aggregate metrics.

    Args:
        env_id:        Gymnasium environment identifier (e.g. "HalfCheetah-v4").
        seed:          Base seed for environment initialisation. Also used to
                       derive the episode seed list when ``episode_seeds`` is
                       not provided.
        episodes:      Number of episodes to evaluate.
        policy_fn:     Callable ``obs -> action`` (np.ndarray, dtype float32).
        render_mode:   Passed through to make_env; None for headless evaluation.
        wrapper_fn:    Optional callable ``(env, **kwargs) -> wrapped_env``.
                       Applied to the bare environment after make_env().
                       Use this to inject observation or dynamics wrappers.
        wrapper_kwargs: Keyword arguments forwarded to ``wrapper_fn``.
                        Ignored when ``wrapper_fn`` is None.
        episode_seeds: Optional explicit per-episode seed list.  When provided,
                       it is used directly and the hash is computed from it.
                       When None, the seed list is generated internally as
                       ``eval_seed_list(base_seed=seed+1000, episodes=episodes)``.
                       Pass a precomputed list to guarantee identical seeds across
                       clean and shifted evaluation runs.

    Returns:
        Dict with keys:
            env_id, seed, episodes, eval_seed_list_hash,
            return_mean, return_std, episode_len_mean, episode_len_std.
        The dict does NOT include shift metadata (shift_type, severity, etc.);
        the caller attaches those when writing JSONL.
    """
    # --- Episode seed list ---------------------------------------------------
    if episode_seeds is not None:
        seeds = episode_seeds
    else:
        seeds = eval_seed_list(base_seed=seed + 1000, episodes=episodes)
    seed_hash = hash_seed_list(seeds)

    # --- Environment ---------------------------------------------------------
    env = make_env(env_id, seed=seed, render_mode=render_mode)

    if wrapper_fn is not None:
        env = wrapper_fn(env, **(wrapper_kwargs or {}))

    action_low   = env.action_space.low
    action_high  = env.action_space.high
    action_shape = env.action_space.shape

    # --- Episode loop --------------------------------------------------------
    returns: list[float] = []
    lengths: list[int]   = []

    for episode_seed in seeds:
        obs, _ = env.reset(seed=episode_seed)
        ep_return = 0.0
        ep_length = 0

        terminated = False
        truncated  = False

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
        "env_id":              env_id,
        "seed":                seed,
        "episodes":            episodes,
        "eval_seed_list_hash": seed_hash,
        **summary,
    }
