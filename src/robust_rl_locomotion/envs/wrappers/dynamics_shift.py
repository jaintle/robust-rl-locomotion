"""
Mass-scale dynamics shift wrapper (sim2real-style gap).

Phase 6 — evaluation-time perturbation only.
Training code is never modified; this wrapper is applied post-hoc.

Threat model (from docs/threat_model.md):
  Perturbation type: Dynamics parameter shift
  Injection point:   Per-episode environment reset
  Parameter:         body_mass (MuJoCo model)
  Severity:          alpha -- uniform half-width of mass scaling factor
  Shift formula:     mass_scale ~ Uniform(1 - alpha, 1 + alpha)
                     body_mass[:] = base_body_mass * mass_scale

Only mass scaling is applied in Phase 6 (no friction, inertia, or other params).
No training-time domain randomisation is introduced.
"""

from __future__ import annotations

import copy

import numpy as np
import gymnasium


class MassScaleWrapper(gymnasium.Wrapper):
    """Uniformly scale all MuJoCo body masses at the start of each episode.

    On every ``reset()``:
      1. Call the underlying env reset (seeding the env RNG as usual).
      2. Sample a scalar ``mass_scale ~ Uniform(1 - alpha, 1 + alpha)`` from
         a private, isolated NumPy RandomState.
      3. Apply: ``env.unwrapped.model.body_mass[:] = base_body_mass * mass_scale``
      4. Expose ``self.last_mass_scale`` for the caller to record.

    The private RNG (``self.rng``) does NOT share state with global NumPy or
    Torch RNGs, so noise is fully deterministic for a given ``seed`` regardless
    of external RNG activity.

    When ``alpha == 0.0`` the wrapper still samples (always returning 1.0),
    which gives uniform code-path treatment across the severity grid while
    having no practical effect on dynamics.

    Args:
        env:   Wrapped Gymnasium MuJoCo environment.
        alpha: Half-width of the uniform distribution over mass scales.
               Must satisfy 0.0 <= alpha <= 0.5.
               E.g. alpha=0.2 → mass_scale in [0.8, 1.2].
        seed:  Integer seed for the private mass-scale RNG.

    Raises:
        ValueError: If ``alpha`` is outside [0.0, 0.5].
        RuntimeError: If the underlying environment does not expose
                      ``env.unwrapped.model.body_mass`` (non-MuJoCo env).
    """

    def __init__(self, env: gymnasium.Env, alpha: float, seed: int) -> None:
        super().__init__(env)

        alpha = float(alpha)
        if not (0.0 <= alpha <= 0.5):
            raise ValueError(
                f"MassScaleWrapper: alpha must be in [0.0, 0.5], got {alpha!r}. "
                "Values above 0.5 would allow zero or negative masses."
            )
        self.alpha = alpha
        self.rng   = np.random.RandomState(seed)

        # Verify and cache original body masses at wrapper construction time.
        # Raises RuntimeError early if the env doesn't support MuJoCo body_mass.
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "model") or not hasattr(unwrapped.model, "body_mass"):
            raise RuntimeError(
                "MassScaleWrapper requires a MuJoCo Gymnasium environment that "
                "exposes env.unwrapped.model.body_mass (numpy array). "
                f"Got env type: {type(unwrapped).__name__!r}. "
                "Ensure the environment is a standard MuJoCo locomotion env "
                "(e.g. HalfCheetah-v4, Hopper-v4) and MuJoCo is installed."
            )

        # Store a pristine copy; never mutated after init.
        self._base_body_mass: np.ndarray = copy.deepcopy(unwrapped.model.body_mass)

        # Initialise to neutral; will be overwritten on first reset.
        self.last_mass_scale: float = 1.0

    # ------------------------------------------------------------------
    # Gymnasium Wrapper interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Reset env, sample mass_scale, apply to model, return (obs, info)."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Sample mass scale from private RNG (isolated from global state).
        if self.alpha == 0.0:
            mass_scale = 1.0
        else:
            low  = 1.0 - self.alpha
            high = 1.0 + self.alpha
            mass_scale = float(self.rng.uniform(low, high))

        self.last_mass_scale = mass_scale

        # Apply scaling to the live MuJoCo model.
        self.env.unwrapped.model.body_mass[:] = self._base_body_mass * mass_scale

        return obs, info
