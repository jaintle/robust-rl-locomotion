"""
Phase 1 smoke test: determinism verification.

Creates two independent environment instances with the same seed and verifies
that identical action sequences produce identical trajectories (observations,
rewards, terminated, truncated) at every step.

Usage:
    python scripts/smoke_determinism.py --env_id HalfCheetah-v4 --seed 0 --steps 50
"""

import argparse
import sys

import numpy as np

from robust_rl_locomotion.envs.make_env import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that make_env produces identical trajectories for the same seed."
    )
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Smoke test  env={args.env_id}  seed={args.seed}  steps={args.steps}")

    # --- Create two independent environments with the same seed ---
    env_a = make_env(args.env_id, seed=args.seed)
    env_b = make_env(args.env_id, seed=args.seed)

    # --- Reset both with the same seed and collect initial observations ---
    obs_a, _ = env_a.reset(seed=args.seed)
    obs_b, _ = env_b.reset(seed=args.seed)

    obs_a = np.asarray(obs_a, dtype=np.float64)
    obs_b = np.asarray(obs_b, dtype=np.float64)

    diff = np.max(np.abs(obs_a - obs_b))
    if not np.allclose(obs_a, obs_b, atol=1e-8):
        print(f"FAIL at reset: initial observations differ (max abs diff={diff:.3e})")
        sys.exit(1)

    # --- Fixed action sequence via a local NumPy RNG (not env.action_space.sample) ---
    rng = np.random.RandomState(args.seed + 12345)
    low = env_a.action_space.low
    high = env_a.action_space.high

    # --- Step both environments and assert trajectory equality ---
    for step in range(args.steps):
        action = rng.uniform(low, high).astype(np.float32)

        obs_a, rew_a, term_a, trunc_a, _ = env_a.step(action)
        obs_b, rew_b, term_b, trunc_b, _ = env_b.step(action)

        obs_a = np.asarray(obs_a, dtype=np.float64)
        obs_b = np.asarray(obs_b, dtype=np.float64)
        rew_a = float(rew_a)
        rew_b = float(rew_b)

        obs_diff = np.max(np.abs(obs_a - obs_b))
        rew_diff = abs(rew_a - rew_b)

        if not np.allclose(obs_a, obs_b, atol=1e-8):
            print(
                f"FAIL at step {step}: observations differ "
                f"(max abs diff={obs_diff:.3e})"
            )
            sys.exit(1)

        if rew_diff > 1e-10:
            print(
                f"FAIL at step {step}: rewards differ "
                f"(|{rew_a:.6f} - {rew_b:.6f}| = {rew_diff:.3e})"
            )
            sys.exit(1)

        if term_a != term_b:
            print(
                f"FAIL at step {step}: terminated flags differ "
                f"({term_a} vs {term_b})"
            )
            sys.exit(1)

        if trunc_a != trunc_b:
            print(
                f"FAIL at step {step}: truncated flags differ "
                f"({trunc_a} vs {trunc_b})"
            )
            sys.exit(1)

        if term_a or trunc_a:
            print(f"Episode ended at step {step}; stopping early.")
            break

    env_a.close()
    env_b.close()

    print("Determinism smoke test PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
