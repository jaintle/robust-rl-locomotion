"""
Deterministic seeding utilities.

Phase 1 — no PPO, no wrappers, no perturbations.
"""

import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:
        # Some ops do not yet have deterministic implementations.
        # We warn but do not crash — the caller must be aware that
        # full op-level determinism may not be guaranteed on this platform.
        print(
            f"[seeding] WARNING: torch.use_deterministic_algorithms(True) "
            f"raised RuntimeError: {exc}. "
            "Deterministic mode is best-effort on this platform."
        )


def seed_env_spaces(env, seed: int) -> None:
    """Seed the action and observation spaces of a Gymnasium environment.

    Args:
        env:  A Gymnasium environment instance.
        seed: Integer seed value.
    """
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
