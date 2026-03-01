"""
Phase 5: Shifted evaluation under Gaussian observation noise.

Loads a trained PPO checkpoint from a run_dir (produced by train_ppo_state.py),
then evaluates the deterministic policy across a σ severity grid.
Writes one JSON line per σ value to eval_shifted.jsonl.

The evaluation episode seed list is computed ONCE and shared across all
severities, so eval_seed_list_hash is identical for every JSONL entry and
matches the clean eval (eval_clean.json) from the same run.

Usage:
    python scripts/eval_shifted_noise.py \\
        --run_dir runs/smoke_seed0 \\
        --episodes 5 \\
        --sigmas 0.0,0.01,0.05,0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from robust_rl_locomotion.algo.ppo import PolicyNetwork
from robust_rl_locomotion.envs.wrappers.obs_noise import GaussianObsNoiseWrapper
from robust_rl_locomotion.eval.evaluate import evaluate_policy
from robust_rl_locomotion.eval.metrics import eval_seed_list


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved PPO checkpoint under Gaussian observation noise."
    )
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Run directory containing checkpoint.pt and config.json.",
    )
    p.add_argument(
        "--env_id", type=str, default=None,
        help="Override env_id (default: read from config.json).",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override seed (default: read from config.json).",
    )
    p.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes per severity.",
    )
    p.add_argument(
        "--sigmas", type=str, default="0.0,0.01,0.05,0.1",
        help="Comma-separated σ values (e.g. '0.0,0.01,0.05,0.1').",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output JSONL path (default: <run_dir>/eval_shifted.jsonl).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(path):
        print(f"ERROR: config.json not found in {run_dir!r}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _load_policy(run_dir: str, obs_dim: int, action_dim: int) -> PolicyNetwork:
    """Reconstruct PolicyNetwork and load saved weights from checkpoint.pt.

    Architecture is fixed to the default used by PPOAgent (hidden=(64, 64)).
    This matches the default set in train_ppo_state.py.
    """
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint.pt not found in {run_dir!r}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden=(64, 64))
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def _make_policy_fn(policy: PolicyNetwork):
    """Return a deterministic policy_fn compatible with evaluate_policy."""
    @torch.no_grad()
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t  = torch.FloatTensor(obs)
        action, _ = policy.act(obs_t, deterministic=True)
        return action.cpu().numpy()
    return policy_fn


def _noise_seed(base_seed: int, sigma: float) -> int:
    """Deterministic noise RNG seed derived from training seed and sigma."""
    return base_seed + 4242 + int(sigma * 1_000_000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load config ---------------------------------------------------------
    config = _load_config(args.run_dir)
    env_id = args.env_id if args.env_id is not None else config["env_id"]
    seed   = args.seed   if args.seed   is not None else config["seed"]
    obs_dim    = config["obs_dim"]
    action_dim = config["action_dim"]

    # --- Parse sigma grid ----------------------------------------------------
    sigmas = [float(s.strip()) for s in args.sigmas.split(",")]
    if not sigmas:
        print("ERROR: --sigmas produced an empty list.", file=sys.stderr)
        sys.exit(1)

    # --- Output path ---------------------------------------------------------
    out_path = args.out or os.path.join(args.run_dir, "eval_shifted.jsonl")
    out_dir  = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- Load policy ---------------------------------------------------------
    policy    = _load_policy(args.run_dir, obs_dim, action_dim)
    policy_fn = _make_policy_fn(policy)

    # --- Precompute shared episode seed list ---------------------------------
    # The SAME list is used for ALL sigma values so that eval_seed_list_hash
    # is identical across every JSONL entry and matches eval_clean.json.
    episode_seeds = eval_seed_list(base_seed=seed + 1000, episodes=args.episodes)

    print(f"Shifted eval  env={env_id}  seed={seed}  episodes={args.episodes}")
    print(f"Sigma grid:   {sigmas}")
    print(f"Output:       {out_path}\n")

    # --- Clean baseline (printed only; does not overwrite eval_clean.json) ---
    clean_result = evaluate_policy(
        env_id=env_id,
        seed=seed,
        episodes=args.episodes,
        policy_fn=policy_fn,
        episode_seeds=episode_seeds,
    )
    print(
        f"[baseline σ=0.0] "
        f"return_mean={clean_result['return_mean']:.3f}  "
        f"return_std={clean_result['return_std']:.3f}  "
        f"hash={clean_result['eval_seed_list_hash'][:12]}…"
    )

    # --- Shifted evaluation across sigma grid --------------------------------
    with open(out_path, "w") as f:
        for sigma in sigmas:
            ns = _noise_seed(seed, sigma)

            result = evaluate_policy(
                env_id=env_id,
                seed=seed,
                episodes=args.episodes,
                policy_fn=policy_fn,
                episode_seeds=episode_seeds,
                wrapper_fn=GaussianObsNoiseWrapper,
                wrapper_kwargs={"sigma": sigma, "seed": ns},
            )

            line: dict = {
                **result,
                "shift_type": "gaussian",
                "severity":   sigma,
                "noise_seed": ns,
            }
            f.write(json.dumps(line) + "\n")

            print(
                f"  σ={sigma:.4f}  noise_seed={ns}  "
                f"return_mean={result['return_mean']:.3f}  "
                f"return_std={result['return_std']:.3f}"
            )

    print(f"\nWrote {len(sigmas)} entries → {out_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
