"""
Phase 2 smoke script: deterministic clean evaluation with a random policy.

Runs evaluate_policy with a seeded random policy (no env.action_space.sample()),
writes results to a JSON file, and prints to stdout.

Usage:
    python scripts/eval_clean_random.py --seed 0 --episodes 10 --out results/eval_clean_random.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from robust_rl_locomotion.eval.evaluate import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic clean evaluation with a random policy."
    )
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="results/eval_clean_random.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Deterministic random policy — does NOT use env.action_space.sample().
    # The RandomState is created once and closed over; it advances identically
    # across runs because it is seeded from args.seed (a fixed CLI argument).
    rng = np.random.RandomState(args.seed + 999)

    # We need the action space bounds to sample; retrieve them by peeking at
    # the environment without running any episodes.
    import gymnasium as gym
    _env = gym.make(args.env_id)
    action_low = _env.action_space.low.copy()
    action_high = _env.action_space.high.copy()
    _env.close()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return rng.uniform(action_low, action_high).astype(np.float32)

    print(f"Evaluating  env={args.env_id}  seed={args.seed}  episodes={args.episodes}")

    results = evaluate_policy(
        env_id=args.env_id,
        seed=args.seed,
        episodes=args.episodes,
        policy_fn=policy_fn,
    )

    # Ensure output directory exists.
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    json_str = json.dumps(results, indent=2)
    print(json_str)

    with open(args.out, "w") as f:
        f.write(json_str)
        f.write("\n")  # trailing newline for POSIX compliance

    print(f"\nResults written to {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
