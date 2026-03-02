#!/usr/bin/env bash
# reproduce_multiseed.sh
# End-to-end multi-seed reproduction for robust-rl-locomotion.
#
# Runs the full pipeline for three seeds sequentially:
#   1. Train + evaluate all seeds (via scripts/run_multiseed.py)
#   2. Cross-seed aggregation + multi-seed plots (via tools/aggregate_multiseed.py)
#
# run_multiseed.py handles per-seed training, Gaussian shift eval, and
# mass-scale dynamics eval in sequence.  It stops on the first failure.
#
# Usage:
#   bash scripts/reproduce_multiseed.sh
#   SEEDS=0,1,2,3 TIMESTEPS=500000 bash scripts/reproduce_multiseed.sh
#
# Environment variables (all optional; defaults shown):
#   SEEDS           Comma-separated seed list (default: 0,1,2)
#   TIMESTEPS       Total training timesteps per seed (default: 1000000)
#   ENV_ID          Gymnasium environment ID (default: HalfCheetah-v4)
#   BASE_SAVE_DIR   Parent run directory (default: runs/multiseed_exp)
#   OUT_DIR         Aggregation output directory (default: results/agg_multiseed_exp)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS="${SEEDS:-0,1,2}"
TIMESTEPS="${TIMESTEPS:-1000000}"
ENV_ID="${ENV_ID:-HalfCheetah-v4}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-runs/multiseed_exp}"
OUT_DIR="${OUT_DIR:-results/agg_multiseed_exp}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

echo "================================================================"
echo "  robust-rl-locomotion — multi-seed reproduction"
echo "================================================================"
echo "  SEEDS         : ${SEEDS}"
echo "  TIMESTEPS     : ${TIMESTEPS}"
echo "  ENV_ID        : ${ENV_ID}"
echo "  BASE_SAVE_DIR : ${BASE_SAVE_DIR}"
echo "  OUT_DIR       : ${OUT_DIR}"
echo "================================================================"
echo ""

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 1 — Train and evaluate all seeds
# ---------------------------------------------------------------------------

echo "[1/2] Running multi-seed training + evaluation"
echo "      Seeds: ${SEEDS}  |  Timesteps per seed: ${TIMESTEPS}"
echo ""

python scripts/run_multiseed.py \
    --seeds           "${SEEDS}" \
    --base_save_dir   "${BASE_SAVE_DIR}" \
    --total_timesteps "${TIMESTEPS}" \
    --env_id          "${ENV_ID}"

echo ""

# ---------------------------------------------------------------------------
# Step 2 — Cross-seed aggregation and plots
# ---------------------------------------------------------------------------

echo "[2/2] Cross-seed aggregation and plot generation"
python tools/aggregate_multiseed.py \
    --base_dir "${BASE_SAVE_DIR}" \
    --out_dir  "${OUT_DIR}"

echo ""
echo "================================================================"
echo "  Reproduction complete."
echo "  Seed run dirs   : ${BASE_SAVE_DIR}/seed_*"
echo "  Cross-seed CSVs : ${OUT_DIR}/cross_seed_summary.csv"
echo "                    ${OUT_DIR}/cross_seed_curves.csv"
echo "  Plots           : ${OUT_DIR}/plots/"
echo "================================================================"
