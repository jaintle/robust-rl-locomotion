"""
Deterministic environment creation helper.

Phase 1 — no wrappers, no normalization, no perturbations.
"""

from __future__ import annotations

import gymnasium as gym

from robust_rl_locomotion.utils.seeding import seed_env_spaces, set_global_seeds


def make_env(
    env_id: str,
    seed: int,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a Gymnasium environment with deterministic seeding applied.

    Steps performed (in order):
    1. Create the raw environment via gym.make.
    2. Call set_global_seeds to fix Python / NumPy / Torch state.
    3. Call env.reset(seed=seed) to initialise the environment RNG.
    4. Call seed_env_spaces to seed action and observation spaces.

    No wrappers, normalization, or perturbations are applied here.

    Args:
        env_id:      Gymnasium environment identifier (e.g. "HalfCheetah-v4").
        seed:        Integer seed for all random number generators.
        render_mode: Optional render mode passed to gym.make (e.g. "human",
                     "rgb_array"). Pass None to create a headless environment.

    Returns:
        A seeded Gymnasium environment ready for use.
    """
    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

    set_global_seeds(seed)
    env.reset(seed=seed)
    seed_env_spaces(env, seed)

    return env
