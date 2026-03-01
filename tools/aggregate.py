"""
Robustness metrics aggregation.

Phase 7 — reads eval_clean.json and any shifted JSONL files from a run_dir
and produces two CSV files:

  curves.csv   — one row per (shift_type, severity)
  summary.csv  — one row per shift_type with AUC and worst-case stats

Metrics computed (from CLAUDE.md):

  relative_drop_pct = 100 * (clean_return_mean - return_mean)
                            / max(|clean_return_mean|, 1e-8)

  AUC              = Σ (s[i+1] - s[i]) * (R[i] + R[i+1]) / 2  (trapezoidal)

  AUC_norm         = AUC / (s_max - s_min)   if (s_max - s_min) > 0
                   = AUC                       otherwise

No external dependencies — stdlib + numpy only (no pandas).

Usage:
    python tools/aggregate.py --run_dir runs/smoke_seed0 \\
                               --out_dir results/agg_smoke_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8  # denominator guard for relative_drop_pct

# Filenames that carry shifted eval data, mapped to their expected shift_type.
# The script reads whichever are present; absence is not an error.
_SHIFTED_FILES: dict[str, str] = {
    "eval_shifted.jsonl":          "gaussian",
    "eval_shifted_dynamics.jsonl": "mass_scale",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate robustness metrics from a run_dir into curves.csv "
            "and summary.csv."
        )
    )
    p.add_argument(
        "--run_dir", type=str, required=True,
        help="Run directory containing eval_clean.json and optional JSONL files.",
    )
    p.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for curves.csv and summary.csv.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping blank lines."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _relative_drop_pct(clean_return_mean: float, return_mean: float) -> float:
    """Compute relative degradation (% drop) for a single severity point."""
    denom = max(abs(clean_return_mean), _EPS)
    return 100.0 * (clean_return_mean - return_mean) / denom


def _auc_trapezoidal(severities: list[float], returns: list[float]) -> float:
    """AUC over the severity grid using the trapezoidal rule.

    Severities must be ascending and at least length 2.
    Returns a scalar AUC value.
    """
    s = np.array(severities, dtype=np.float64)
    r = np.array(returns,    dtype=np.float64)
    return float(np.sum((s[1:] - s[:-1]) * (r[1:] + r[:-1]) / 2.0))


def _auc_normalised(auc: float, severities: list[float]) -> float:
    """Normalise AUC by the severity range (AUC per unit severity)."""
    s_range = max(severities) - min(severities)
    if s_range > 0.0:
        return auc / s_range
    return auc


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate(run_dir: str, out_dir: str) -> None:
    """Run the full aggregation pipeline and write CSVs to out_dir."""

    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Clean baseline -------------------------------------------------------
    clean_path = os.path.join(run_dir, "eval_clean.json")
    if not os.path.isfile(clean_path):
        print(f"ERROR: eval_clean.json not found in {run_dir!r}", file=sys.stderr)
        sys.exit(1)

    clean = _load_json(clean_path)
    clean_return_mean = float(clean["return_mean"])

    print(f"Clean baseline: return_mean={clean_return_mean:.4f}")

    # --- Collect shifted data per shift_type ----------------------------------
    # shift_type -> records sorted ascending by severity
    per_shift: dict[str, list[dict]] = {}

    for fname in _SHIFTED_FILES:
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath):
            print(f"  [--] {fname} not found — skipping.")
            continue

        records = _load_jsonl(fpath)
        if not records:
            print(f"  [--] {fname} is empty — skipping.", file=sys.stderr)
            continue

        # Validate shift_type consistency.
        shift_types_found = {r.get("shift_type") for r in records}
        if len(shift_types_found) != 1:
            print(
                f"ERROR: {fname} contains mixed shift_types {shift_types_found}. "
                "Expected exactly one.",
                file=sys.stderr,
            )
            sys.exit(1)

        actual_shift_type = next(iter(shift_types_found))
        records.sort(key=lambda r: float(r["severity"]))
        per_shift[actual_shift_type] = records
        print(f"  [ok] {fname}  shift_type={actual_shift_type!r}  n={len(records)}")

    if not per_shift:
        print(
            "WARNING: no shifted evaluation files found. "
            "curves.csv and summary.csv will be empty.",
            file=sys.stderr,
        )

    # --- Build curves rows ----------------------------------------------------
    curves_rows: list[dict] = []

    for shift_type, records in per_shift.items():
        for rec in records:
            sev      = float(rec["severity"])
            ret_mean = float(rec["return_mean"])
            ret_std  = float(rec["return_std"])
            drop_pct = _relative_drop_pct(clean_return_mean, ret_mean)
            hash_val = str(rec.get("eval_seed_list_hash", ""))

            curves_rows.append({
                "run_dir":             run_dir,
                "shift_type":          shift_type,
                "severity":            sev,
                "return_mean":         ret_mean,
                "return_std":          ret_std,
                "clean_return_mean":   clean_return_mean,
                "relative_drop_pct":   drop_pct,
                "eval_seed_list_hash": hash_val,
            })

    # Stable sort: shift_type then severity.
    curves_rows.sort(key=lambda r: (r["shift_type"], r["severity"]))

    # --- Build summary rows ---------------------------------------------------
    summary_rows: list[dict] = []

    for shift_type, records in per_shift.items():
        severities = [float(r["severity"]) for r in records]
        ret_means  = [float(r["return_mean"]) for r in records]

        if len(severities) < 2:
            print(
                f"WARNING: {shift_type} has only {len(severities)} severity "
                "point(s). AUC requires >= 2 points; reporting NaN.",
                file=sys.stderr,
            )
            auc      = float("nan")
            auc_norm = float("nan")
        else:
            auc      = _auc_trapezoidal(severities, ret_means)
            auc_norm = _auc_normalised(auc, severities)

        # Records are sorted ascending; worst = highest severity.
        worst_sev  = severities[-1]
        worst_ret  = ret_means[-1]
        drop_worst = _relative_drop_pct(clean_return_mean, worst_ret)

        summary_rows.append({
            "run_dir":                 run_dir,
            "shift_type":              shift_type,
            "clean_return_mean":       clean_return_mean,
            "auc_return":              auc,
            "auc_return_norm":         auc_norm,
            "worst_severity":          worst_sev,
            "worst_return_mean":       worst_ret,
            "relative_drop_worst_pct": drop_worst,
            "n_points":                len(records),
        })

    # Stable sort: shift_type.
    summary_rows.sort(key=lambda r: r["shift_type"])

    # --- Write CSVs -----------------------------------------------------------
    curves_path  = os.path.join(out_dir, "curves.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    _write_csv(
        curves_path,
        fieldnames=[
            "run_dir", "shift_type", "severity",
            "return_mean", "return_std",
            "clean_return_mean", "relative_drop_pct",
            "eval_seed_list_hash",
        ],
        rows=curves_rows,
    )

    _write_csv(
        summary_path,
        fieldnames=[
            "run_dir", "shift_type",
            "clean_return_mean",
            "auc_return", "auc_return_norm",
            "worst_severity", "worst_return_mean",
            "relative_drop_worst_pct",
            "n_points",
        ],
        rows=summary_rows,
    )

    print(f"\nWrote: {curves_path}")
    print(f"Wrote: {summary_path}")


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    aggregate(run_dir=args.run_dir, out_dir=args.out_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
