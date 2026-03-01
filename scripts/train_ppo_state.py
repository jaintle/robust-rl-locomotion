"""
Phase 3 training script: PPO on state observations.

Runs a deterministic PPO training loop, saves required artifacts, and
runs a final clean evaluation using the Phase 2 evaluation harness.

Saved artifacts (in --save_dir):
  config.json        – all CLI args + derived dims
  metrics.csv        – per-update training metrics
  checkpoint.pt      – policy + value state dicts + optimiser
  train_summary.json – summary stats over all completed training episodes
  eval_clean.json    – deterministic clean eval result from evaluate_policy

Usage (smoke):
  python scripts/train_ppo_state.py --total_timesteps 5000 --seed 0 --save_dir runs/smoke_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

from robust_rl_locomotion.algo.ppo import PPOAgent, RolloutBuffer
from robust_rl_locomotion.envs.make_env import make_env
from robust_rl_locomotion.eval.evaluate import evaluate_policy
from robust_rl_locomotion.utils.seeding import set_global_seeds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO (state obs) on a Gymnasium env.")
    p.add_argument("--env_id",           type=str,   default="HalfCheetah-v4")
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--total_timesteps",  type=int,   default=5000)
    p.add_argument("--num_envs",         type=int,   default=1,    help="Fixed at 1; kept for API consistency.")
    p.add_argument("--rollout_steps",    type=int,   default=1024)
    p.add_argument("--update_epochs",    type=int,   default=4)
    p.add_argument("--minibatch_size",   type=int,   default=256)
    p.add_argument("--learning_rate",    type=float, default=3e-4)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--gae_lambda",       type=float, default=0.95)
    p.add_argument("--clip_coef",        type=float, default=0.2)
    p.add_argument("--ent_coef",         type=float, default=0.0)
    p.add_argument("--vf_coef",          type=float, default=0.5)
    p.add_argument("--max_grad_norm",    type=float, default=0.5)
    p.add_argument("--eval_episodes",    type=int,   default=10)
    p.add_argument("--eval_every",       type=int,   default=0,    help="Run mid-training eval every N steps (0 = final only).")
    p.add_argument("--save_dir",         type=str,   default="runs/smoke")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Policy wrapper for evaluate_policy
# ---------------------------------------------------------------------------

def make_deterministic_policy(agent: PPOAgent, device: torch.device):
    """Return a deterministic policy_fn compatible with evaluate_policy."""
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).to(device)
        action, _, _ = agent.act(obs_t, deterministic=True)
        return action.cpu().numpy()
    return policy_fn


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Global seeding (Python / NumPy / Torch) ---
    set_global_seeds(args.seed)
    device = torch.device("cpu")

    # --- Environment ---
    env = make_env(args.env_id, seed=args.seed)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low  = env.action_space.low.copy()
    action_high = env.action_space.high.copy()

    # --- Agent + optimiser ---
    agent     = PPOAgent(obs_dim, action_dim, device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Rollout buffer ---
    buffer = RolloutBuffer(args.rollout_steps, obs_dim, action_dim, device)

    # --- Output directory ---
    os.makedirs(args.save_dir, exist_ok=True)

    # --- config.json ---
    config: dict = vars(args).copy()
    config["obs_dim"]    = obs_dim
    config["action_dim"] = action_dim
    config["device"]     = str(device)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {args.save_dir}/config.json")

    # --- metrics.csv ---
    csv_path = os.path.join(args.save_dir, "metrics.csv")
    CSV_COLS  = [
        "global_step", "episode_return", "episode_length",
        "policy_loss", "value_loss", "entropy", "approx_kl",
    ]
    csv_fh     = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLS)
    csv_writer.writeheader()

    # --- Training state ---
    global_step    = 0
    all_ep_returns: list[float] = []
    cur_ep_return  = 0.0
    cur_ep_length  = 0
    last_done      = False

    obs, _ = env.reset(seed=args.seed)
    obs_t  = torch.FloatTensor(obs).to(device)

    print(f"\nTraining  env={args.env_id}  seed={args.seed}  "
          f"total_timesteps={args.total_timesteps}\n")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while global_step < args.total_timesteps:
        buffer.reset()
        rollout_ep_returns: list[float] = []
        rollout_ep_lengths: list[int]   = []

        # --- Collect rollout ---
        for _ in range(args.rollout_steps):
            if global_step >= args.total_timesteps:
                break

            action_t, logprob_t, value_t = agent.act(obs_t, deterministic=False)

            # Clip to action bounds before stepping.
            action_np = np.clip(action_t.cpu().numpy(), action_low, action_high)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            last_done = bool(terminated or truncated)

            buffer.store(obs_t, action_t, logprob_t, float(reward), last_done, value_t)
            global_step   += 1
            cur_ep_return += float(reward)
            cur_ep_length += 1

            if last_done:
                rollout_ep_returns.append(cur_ep_return)
                rollout_ep_lengths.append(cur_ep_length)
                all_ep_returns.append(cur_ep_return)
                cur_ep_return = 0.0
                cur_ep_length = 0
                next_obs, _ = env.reset()

            obs   = next_obs
            obs_t = torch.FloatTensor(obs).to(device)

        n = buffer.ptr
        if n == 0:
            break

        # --- Bootstrap value for the last state ---
        with torch.no_grad():
            next_value = agent.value(obs_t)

        # --- GAE ---
        advantages, returns = buffer.compute_gae(
            next_value, last_done, args.gamma, args.gae_lambda
        )
        # Normalise advantages over the full rollout (before minibatch split).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO update epochs ---
        flat_batch = {
            "obs":        buffer.obs[:n],
            "actions":    buffer.actions[:n],
            "logprobs":   buffer.logprobs[:n],
            "advantages": advantages,
            "returns":    returns,
        }
        update_losses: list[dict] = []
        for _ in range(args.update_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch_size):
                mb_idx = perm[start : start + args.minibatch_size]
                mb     = {k: v[mb_idx] for k, v in flat_batch.items()}
                losses = agent.update(
                    mb, optimizer,
                    args.clip_coef, args.vf_coef, args.ent_coef, args.max_grad_norm,
                )
                update_losses.append(losses)

        avg = {k: float(np.mean([l[k] for l in update_losses])) for k in update_losses[0]}

        # Rollout-level episode aggregates (NaN if no episode completed this rollout).
        ep_ret = float(np.mean(rollout_ep_returns)) if rollout_ep_returns else float("nan")
        ep_len = float(np.mean(rollout_ep_lengths)) if rollout_ep_lengths else float("nan")

        csv_writer.writerow({
            "global_step":    global_step,
            "episode_return": ep_ret,
            "episode_length": ep_len,
            **avg,
        })
        csv_fh.flush()

        print(
            f"step={global_step:>7d} | "
            f"ep_ret={ep_ret:>9.2f} | "
            f"pol_loss={avg['policy_loss']:>8.4f} | "
            f"val_loss={avg['value_loss']:>8.4f} | "
            f"kl={avg['approx_kl']:>7.4f}"
        )

        # --- Optional mid-training eval ---
        if args.eval_every > 0 and global_step % args.eval_every == 0:
            # NOTE: evaluate_policy calls make_env → set_global_seeds internally,
            # which modifies the global RNG. Training is best-effort deterministic
            # when eval_every > 0. Use eval_every=0 for fully reproducible runs.
            mid_eval = evaluate_policy(
                env_id=args.env_id,
                seed=args.seed,
                episodes=args.eval_episodes,
                policy_fn=make_deterministic_policy(agent, device),
            )
            print(f"  [mid-eval step={global_step}] "
                  f"return_mean={mid_eval['return_mean']:.2f} ± {mid_eval['return_std']:.2f}")

    csv_fh.close()
    env.close()

    # -----------------------------------------------------------------------
    # Artifacts
    # -----------------------------------------------------------------------

    # checkpoint.pt
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
    torch.save(
        {
            "policy_state_dict":    agent.policy.state_dict(),
            "value_state_dict":     agent.value.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":          global_step,
            "seed":                 args.seed,
            "env_id":               args.env_id,
        },
        ckpt_path,
    )
    print(f"\nCheckpoint saved → {ckpt_path}")

    # train_summary.json
    last_n = all_ep_returns[-100:] if all_ep_returns else []
    train_summary = {
        "env_id":                    args.env_id,
        "seed":                      args.seed,
        "total_timesteps":           global_step,
        "episodes_completed":        len(all_ep_returns),
        "train_return_mean_last_100": float(np.mean(last_n)) if last_n else None,
        "train_return_std_last_100":  float(np.std(last_n))  if last_n else None,
    }
    with open(os.path.join(args.save_dir, "train_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2)
    print(f"Train summary saved → {args.save_dir}/train_summary.json")

    # eval_clean.json  (deterministic, policy in eval mode)
    print(f"\nRunning final clean eval ({args.eval_episodes} episodes)…")
    eval_results = evaluate_policy(
        env_id=args.env_id,
        seed=args.seed,
        episodes=args.eval_episodes,
        policy_fn=make_deterministic_policy(agent, device),
    )
    with open(os.path.join(args.save_dir, "eval_clean.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Eval saved → {args.save_dir}/eval_clean.json")

    # -----------------------------------------------------------------------
    # Summary print
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Training complete  ({global_step} steps)")
    if last_n:
        print(f"  train return (last {len(last_n)} eps):  "
              f"mean={train_summary['train_return_mean_last_100']:.2f}  "
              f"std={train_summary['train_return_std_last_100']:.2f}")
    print(f"  clean eval return:  "
          f"mean={eval_results['return_mean']:.2f}  "
          f"std={eval_results['return_std']:.2f}")
    print(f"  artifacts → {args.save_dir}/")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
