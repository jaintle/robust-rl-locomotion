"""
Multi-seed experiment runner.

Phase 8 — runs training + both shifted evaluations for a list of seeds
sequentially.  Stops immediately on any failure.  No multiprocessing.

For each seed the following three steps are executed in order:
  1. train_ppo_state.py   → <base_save_dir>/seed_<seed>/
  2. eval_shifted_noise.py   --run_dir <seed_dir>
  3. eval_shifted_dynamics.py --run_dir <seed_dir>

Steps 2 and 3 use their own default severity grids so that multi-seed results
are directly comparable to single-seed smoke runs.

Usage:
    python scripts/run_multiseed.py \\
        --seeds 0,1,2 \\
        --base_save_dir runs/multiseed_exp \\
        --total_timesteps 20000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run training + shifted eval for multiple seeds sequentially."
    )
    p.add_argument(
        "--seeds", type=str, default="0,1,2",
        help="Comma-separated list of integer seeds (e.g. '0,1,2').",
    )
    p.add_argument(
        "--base_save_dir", type=str, required=True,
        help="Parent directory; each seed is saved to <base_save_dir>/seed_<seed>/.",
    )
    p.add_argument(
        "--total_timesteps", type=int, default=20000,
        help="Total training timesteps passed to train_ppo_state.py.",
    )
    p.add_argument(
        "--env_id", type=str, default=None,
        help="Optional env_id override forwarded to train_ppo_state.py.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command.  Exit with its return code on failure."""
    print(f"\n[{label}] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"\nERROR: step '{label}' failed with exit code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("ERROR: --seeds produced an empty list.", file=sys.stderr)
        sys.exit(1)

    python      = sys.executable
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    train_script    = os.path.join(scripts_dir, "train_ppo_state.py")
    noise_script    = os.path.join(scripts_dir, "eval_shifted_noise.py")
    dynamics_script = os.path.join(scripts_dir, "eval_shifted_dynamics.py")

    for script in (train_script, noise_script, dynamics_script):
        if not os.path.isfile(script):
            print(f"ERROR: script not found: {script!r}", file=sys.stderr)
            sys.exit(1)

    print(f"Multi-seed run  seeds={seeds}  total_timesteps={args.total_timesteps}")
    print(f"base_save_dir:  {args.base_save_dir}")

    for idx, seed in enumerate(seeds):
        seed_dir = os.path.join(args.base_save_dir, f"seed_{seed}")

        width = 60
        print(f"\n{'=' * width}")
        print(f"  Seed {seed}  ({idx + 1}/{len(seeds)})  →  {seed_dir}")
        print(f"{'=' * width}")

        # Step 1 — train
        train_cmd = [
            python, train_script,
            "--seed",             str(seed),
            "--total_timesteps",  str(args.total_timesteps),
            "--save_dir",         seed_dir,
        ]
        if args.env_id is not None:
            train_cmd += ["--env_id", args.env_id]

        _run(train_cmd, f"seed={seed} train")

        # Step 2 — gaussian observation noise eval
        _run(
            [python, noise_script, "--run_dir", seed_dir],
            f"seed={seed} eval_shifted_noise",
        )

        # Step 3 — mass-scale dynamics eval
        _run(
            [python, dynamics_script, "--run_dir", seed_dir],
            f"seed={seed} eval_shifted_dynamics",
        )

        print(f"\n  [done] seed={seed}", flush=True)

    print(f"\n{'=' * 60}")
    print(f"All {len(seeds)} seed(s) completed successfully.")
    print(f"Run dirs: {args.base_save_dir}/seed_{{" + ",".join(str(s) for s in seeds) + "}}")
    sys.exit(0)


if __name__ == "__main__":
    main()
