#!/usr/bin/env bash
# reproduce_single_seed.sh
# End-to-end single-seed reproduction for robust-rl-locomotion.
#
# Runs the full pipeline for one seed:
#   1. Train PPO
#   2. Evaluate under Gaussian observation noise
#   3. Evaluate under mass-scale dynamics shift
#   4. Validate run directory
#   5. Aggregate per-run metrics
#   6. Generate plots
#
# Usage:
#   bash scripts/reproduce_single_seed.sh
#   SEED=1 TIMESTEPS=500000 bash scripts/reproduce_single_seed.sh
#
# Environment variables (all optional; defaults shown):
#   SEED        Training seed (default: 0)
#   TIMESTEPS   Total training timesteps (default: 1000000)
#   ENV_ID      Gymnasium environment ID (default: HalfCheetah-v4)
#   RUN_DIR     Run directory (default: runs/seed_${SEED})
#   OUT_DIR     Aggregation output directory (default: results/agg_seed_${SEED})
#   SIGMAS      Comma-separated sigma values (default: 0.0,0.01,0.05,0.1)
#   ALPHAS      Comma-separated alpha values (default: 0.0,0.1,0.2,0.3)
#   EPISODES    Evaluation episodes per severity (default: 10)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED="${SEED:-0}"
TIMESTEPS="${TIMESTEPS:-1000000}"
ENV_ID="${ENV_ID:-HalfCheetah-v4}"
RUN_DIR="${RUN_DIR:-runs/seed_${SEED}}"
OUT_DIR="${OUT_DIR:-results/agg_seed_${SEED}}"
SIGMAS="${SIGMAS:-0.0,0.01,0.05,0.1}"
ALPHAS="${ALPHAS:-0.0,0.1,0.2,0.3}"
EPISODES="${EPISODES:-10}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

echo "================================================================"
echo "  robust-rl-locomotion — single-seed reproduction"
echo "================================================================"
echo "  SEED        : ${SEED}"
echo "  TIMESTEPS   : ${TIMESTEPS}"
echo "  ENV_ID      : ${ENV_ID}"
echo "  RUN_DIR     : ${RUN_DIR}"
echo "  OUT_DIR     : ${OUT_DIR}"
echo "  SIGMAS      : ${SIGMAS}"
echo "  ALPHAS      : ${ALPHAS}"
echo "  EPISODES    : ${EPISODES}"
echo "================================================================"
echo ""

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1 — Train
# ---------------------------------------------------------------------------

echo "[1/6] Training PPO  (seed=${SEED}, timesteps=${TIMESTEPS})"
python scripts/train_ppo_state.py \
    --env_id          "${ENV_ID}" \
    --seed            "${SEED}" \
    --total_timesteps "${TIMESTEPS}" \
    --save_dir        "${RUN_DIR}"

echo ""

# ---------------------------------------------------------------------------
# Step 2 — Gaussian shift evaluation
# ---------------------------------------------------------------------------

echo "[2/6] Evaluating under Gaussian observation noise"
python scripts/eval_shifted_noise.py \
    --run_dir  "${RUN_DIR}" \
    --sigmas   "${SIGMAS}" \
    --episodes "${EPISODES}"

echo ""

# ---------------------------------------------------------------------------
# Step 3 — Mass-scale dynamics evaluation
# ---------------------------------------------------------------------------

echo "[3/6] Evaluating under mass-scale dynamics shift"
python scripts/eval_shifted_dynamics.py \
    --run_dir  "${RUN_DIR}" \
    --alphas   "${ALPHAS}" \
    --episodes "${EPISODES}"

echo ""

# ---------------------------------------------------------------------------
# Step 4 — Validate
# ---------------------------------------------------------------------------

echo "[4/6] Validating run directory"
python tools/validate_results.py --run_dir "${RUN_DIR}"

echo ""

# ---------------------------------------------------------------------------
# Step 5 — Aggregate
# ---------------------------------------------------------------------------

echo "[5/6] Aggregating per-run metrics"
python tools/aggregate.py \
    --run_dir "${RUN_DIR}" \
    --out_dir "${OUT_DIR}"

echo ""

# ---------------------------------------------------------------------------
# Step 6 — Plot
# ---------------------------------------------------------------------------

echo "[6/6] Generating plots"
python tools/plot_curves.py \
    --curves_csv "${OUT_DIR}/curves.csv" \
    --out_dir    "${OUT_DIR}/plots"

echo ""
echo "================================================================"
echo "  Reproduction complete."
echo "  Run directory  : ${RUN_DIR}"
echo "  Aggregation    : ${OUT_DIR}"
echo "  Plots          : ${OUT_DIR}/plots"
echo "================================================================"
