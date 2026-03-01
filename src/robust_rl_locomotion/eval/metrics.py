"""
Pure metric utilities for evaluation.

Phase 2 — no PPO, no wrappers, no perturbations.
"""

from __future__ import annotations

import hashlib

import numpy as np


def eval_seed_list(base_seed: int, episodes: int) -> list[int]:
    """Return a deterministic list of per-episode seeds.

    Args:
        base_seed: Starting seed value.
        episodes:  Number of episodes (length of returned list).

    Returns:
        List of integers [base_seed, base_seed+1, ..., base_seed+episodes-1].
    """
    return [base_seed + i for i in range(episodes)]


def hash_seed_list(seed_list: list[int]) -> str:
    """Return a SHA-256 hex digest of the seed list for audit logging.

    The list is serialised as comma-joined integers (no spaces) before hashing,
    so the result is stable across Python versions and platforms.

    Args:
        seed_list: List of integer seeds.

    Returns:
        64-character lowercase hex string.
    """
    payload = ",".join(str(s) for s in seed_list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_episodes(
    returns: list[float],
    lengths: list[int],
) -> dict:
    """Compute summary statistics over a set of episodes.

    Args:
        returns: List of per-episode cumulative rewards.
        lengths: List of per-episode step counts.

    Returns:
        Dict with keys: return_mean, return_std, episode_len_mean, episode_len_std.
        All values are Python floats.
    """
    r = np.array(returns, dtype=np.float64)
    l = np.array(lengths, dtype=np.float64)
    return {
        "return_mean": float(np.mean(r)),
        "return_std": float(np.std(r)),
        "episode_len_mean": float(np.mean(l)),
        "episode_len_std": float(np.std(l)),
    }
