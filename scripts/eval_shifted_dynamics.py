"""
Phase 6: Shifted evaluation under mass-scale dynamics perturbation.

Loads a trained PPO checkpoint from a run_dir (produced by train_ppo_state.py),
then evaluates the deterministic policy across an alpha severity grid.
Writes one JSON line per alpha value to eval_shifted_dynamics.jsonl.

The evaluation episode seed list is computed ONCE and shared across all
severities, so eval_seed_list_hash is identical for every JSONL entry.

Because we need per-episode mass_scale values (recorded after each reset),
this script uses a local episode loop rather than evaluate_policy(), which
does not expose wrapper internals.  The loop logic mirrors evaluate_policy()
exactly — same seed list, same deterministic policy, same episode structure.

Usage:
    python scripts/eval_shifted_dynamics.py \\
        --run_dir runs/smoke_seed0 \\
        --episodes 10 \\
        --alphas 0.0,0.1,0.2,0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from robust_rl_locomotion.algo.ppo import PolicyNetwork
from robust_rl_locomotion.envs.make_env import make_env
from robust_rl_locomotion.envs.wrappers.dynamics_shift import MassScaleWrapper
from robust_rl_locomotion.eval.metrics import eval_seed_list, hash_seed_list, summarize_episodes


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a saved PPO checkpoint under mass-scale dynamics shift."
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
        "--alphas", type=str, default="0.0,0.1,0.2,0.3",
        help="Comma-separated alpha values (e.g. '0.0,0.1,0.2,0.3').",
    )
    p.add_argument(
        "--out", type=str, default=None,
        help="Output JSONL path (default: <run_dir>/eval_shifted_dynamics.jsonl).",
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

    Architecture fixed to default used by PPOAgent (hidden=(64, 64)),
    matching the default set in train_ppo_state.py.
    """
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: checkpoint.pt not found in {run_dir!r}", file=sys.stderr)
        sys.exit(1)

    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden=(64, 64))
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def _mass_seed(base_seed: int, alpha: float) -> int:
    """Deterministic mass-scale RNG seed derived from training seed and alpha."""
    return base_seed + 9000 + int(alpha * 1_000_000)


# ---------------------------------------------------------------------------
# Local episode loop (needed to capture wrapper.last_mass_scale per episode)
# ---------------------------------------------------------------------------

def _run_episodes(
    env_id: str,
    seed: int,
    episode_seeds: list[int],
    alpha: float,
    mass_seed: int,
    policy: PolicyNetwork,
) -> dict:
    """Run one full evaluation under a given alpha and collect per-episode stats.

    Returns a summary dict matching the fields produced by evaluate_policy(),
    plus mass_scale_mean and mass_scale_std.

    The episode loop mirrors evaluate_policy() exactly:
      - make_env called once per alpha
      - MassScaleWrapper applied immediately after
      - env.reset(seed=episode_seed) called per episode
      - deterministic policy (mean action) throughout
    """
    seed_hash = hash_seed_list(episode_seeds)

    env = make_env(env_id, seed=seed)
    env = MassScaleWrapper(env, alpha=alpha, seed=mass_seed)

    action_low   = env.action_space.low
    action_high  = env.action_space.high
    action_shape = env.action_space.shape

    returns:     list[float] = []
    lengths:     list[int]   = []
    mass_scales: list[float] = []

    for episode_seed in episode_seeds:
        obs, _ = env.reset(seed=episode_seed)
        # Record the mass scale that was sampled at this reset.
        mass_scales.append(env.last_mass_scale)

        ep_return = 0.0
        ep_length = 0
        terminated = False
        truncated  = False

        while not (terminated or truncated):
            with torch.no_grad():
                obs_t  = torch.FloatTensor(obs)
                action_t, _ = policy.act(obs_t, deterministic=True)
                raw_action  = action_t.cpu().numpy()

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
        "eval_seed_list_hash": seed_hash,
        "mass_scale_mean":     float(np.mean(mass_scales)),
        "mass_scale_std":      float(np.std(mass_scales)),
        **summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load config ---------------------------------------------------------
    config     = _load_config(args.run_dir)
    env_id     = args.env_id if args.env_id is not None else config["env_id"]
    seed       = args.seed   if args.seed   is not None else config["seed"]
    obs_dim    = config["obs_dim"]
    action_dim = config["action_dim"]

    # --- Parse alpha grid ----------------------------------------------------
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    if not alphas:
        print("ERROR: --alphas produced an empty list.", file=sys.stderr)
        sys.exit(1)

    # --- Output path ---------------------------------------------------------
    out_path = args.out or os.path.join(args.run_dir, "eval_shifted_dynamics.jsonl")
    out_dir  = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # --- Load policy ---------------------------------------------------------
    policy = _load_policy(args.run_dir, obs_dim, action_dim)

    # --- Precompute shared episode seed list ---------------------------------
    # The SAME list is used for ALL alpha values so that eval_seed_list_hash
    # is identical across every JSONL entry.
    episode_seeds = eval_seed_list(base_seed=seed + 1000, episodes=args.episodes)

    print(f"Shifted eval  env={env_id}  seed={seed}  episodes={args.episodes}")
    print(f"Alpha grid:   {alphas}")
    print(f"Output:       {out_path}\n")

    # --- Evaluate across alpha grid ------------------------------------------
    with open(out_path, "w") as f:
        for alpha in alphas:
            ms = _mass_seed(seed, alpha)

            result = _run_episodes(
                env_id=env_id,
                seed=seed,
                episode_seeds=episode_seeds,
                alpha=alpha,
                mass_seed=ms,
                policy=policy,
            )

            line: dict = {
                "env_id":    env_id,
                "seed":      seed,
                "episodes":  args.episodes,
                "shift_type": "mass_scale",
                "severity":  alpha,
                "mass_seed": ms,
                **result,
            }
            f.write(json.dumps(line) + "\n")

            print(
                f"  alpha={alpha:.3f}  mass_seed={ms}  "
                f"mass_scale_mean={result['mass_scale_mean']:.4f}  "
                f"mass_scale_std={result['mass_scale_std']:.4f}  "
                f"return_mean={result['return_mean']:.3f}  "
                f"return_std={result['return_std']:.3f}"
            )

    print(f"\nWrote {len(alphas)} entries → {out_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
