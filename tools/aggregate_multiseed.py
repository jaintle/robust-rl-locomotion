"""
Cross-seed robustness aggregation.

Phase 8 — discovers seed_* subdirectories under base_dir, ensures each has
per-seed curves.csv + summary.csv (running tools/aggregate.py automatically
if they are absent), then aggregates across seeds to produce:

  cross_seed_curves.csv   — mean ± std of return and %drop per
                            (shift_type, severity)
  cross_seed_summary.csv  — mean ± std of clean return, AUC, and worst-case
                            drop per shift_type

Also generates two PNG plots per shift_type:
  plots/return_vs_severity_<shift_type>_multiseed.png
  plots/drop_vs_severity_<shift_type>_multiseed.png

std is computed across seeds (population std, ddof=0).  With a single seed the
std columns will be 0.0; that is correct and expected.

No pandas.  stdlib + numpy only.

Usage:
    python tools/aggregate_multiseed.py \\
        --base_dir runs/multiseed_exp \\
        --out_dir  results/agg_multiseed_exp
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate robustness metrics across multiple training seeds."
    )
    p.add_argument(
        "--base_dir", type=str, required=True,
        help="Parent directory containing seed_* run subdirectories.",
    )
    p.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for cross-seed CSVs and plots.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> list[dict]:
    """Load a CSV file into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Seed-dir discovery
# ---------------------------------------------------------------------------

def _discover_seed_dirs(base_dir: str) -> list[str]:
    """Return sorted list of absolute paths to seed_* subdirs."""
    if not os.path.isdir(base_dir):
        print(f"ERROR: base_dir does not exist: {base_dir!r}", file=sys.stderr)
        sys.exit(1)

    entries = sorted(
        e for e in os.listdir(base_dir) if e.startswith("seed_")
    )
    dirs = [
        os.path.join(base_dir, e)
        for e in entries
        if os.path.isdir(os.path.join(base_dir, e))
    ]

    if not dirs:
        print(
            f"ERROR: no seed_* subdirectories found in {base_dir!r}. "
            "Run scripts/run_multiseed.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    return dirs


# ---------------------------------------------------------------------------
# Per-seed CSV loader (runs aggregate.py if CSVs are absent)
# ---------------------------------------------------------------------------

def _ensure_per_seed_csvs(seed_dir: str) -> tuple[str, str]:
    """Return (curves_path, summary_path) for a seed dir.

    If curves.csv / summary.csv are absent under seed_dir/agg/, run
    tools/aggregate.py to produce them first.
    """
    agg_dir     = os.path.join(seed_dir, "agg")
    curves_path  = os.path.join(agg_dir, "curves.csv")
    summary_path = os.path.join(agg_dir, "summary.csv")

    if os.path.isfile(curves_path) and os.path.isfile(summary_path):
        return curves_path, summary_path

    # Locate tools/aggregate.py relative to this file.
    tools_dir    = os.path.dirname(os.path.abspath(__file__))
    agg_script   = os.path.join(tools_dir, "aggregate.py")

    if not os.path.isfile(agg_script):
        print(
            f"ERROR: tools/aggregate.py not found at {agg_script!r}. "
            "Cannot produce per-seed CSVs.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  [auto] running aggregate.py for {seed_dir!r} …")
    result = subprocess.run(
        [sys.executable, agg_script, "--run_dir", seed_dir, "--out_dir", agg_dir]
    )
    if result.returncode != 0:
        print(
            f"ERROR: aggregate.py failed for {seed_dir!r} (exit {result.returncode}).",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    return curves_path, summary_path


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------

def _aggregate_curves(
    seed_curves: list[list[dict]],
) -> list[dict]:
    """Compute mean ± std of return and %drop across seeds per (shift_type, severity).

    Args:
        seed_curves: list (one per seed) of curve-row lists.

    Returns:
        List of cross-seed curve row dicts, sorted by (shift_type, severity).
    """
    # Index: (shift_type, severity) -> list of (return_mean, relative_drop_pct)
    from collections import defaultdict
    bucket: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)

    for rows in seed_curves:
        for row in rows:
            key = (row["shift_type"], float(row["severity"]))
            bucket[key].append(
                (float(row["return_mean"]), float(row["relative_drop_pct"]))
            )

    out_rows: list[dict] = []
    for (shift_type, severity), vals in sorted(bucket.items()):
        returns = np.array([v[0] for v in vals], dtype=np.float64)
        drops   = np.array([v[1] for v in vals], dtype=np.float64)
        out_rows.append({
            "shift_type":              shift_type,
            "severity":                severity,
            "mean_return":             float(np.mean(returns)),
            "std_return":              float(np.std(returns)),
            "mean_relative_drop_pct":  float(np.mean(drops)),
            "std_relative_drop_pct":   float(np.std(drops)),
            "n_seeds":                 len(vals),
        })

    return out_rows


def _aggregate_summary(
    seed_summaries: list[list[dict]],
) -> list[dict]:
    """Compute mean ± std of key metrics across seeds per shift_type.

    Args:
        seed_summaries: list (one per seed) of summary-row lists.

    Returns:
        List of cross-seed summary row dicts, sorted by shift_type.
    """
    from collections import defaultdict
    # (shift_type) -> list of (clean_return_mean, auc_return_norm, relative_drop_worst_pct)
    bucket: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    for rows in seed_summaries:
        for row in rows:
            shift_type = row["shift_type"]
            try:
                auc_norm  = float(row["auc_return_norm"])
                drop_worst = float(row["relative_drop_worst_pct"])
                clean_ret  = float(row["clean_return_mean"])
            except (ValueError, KeyError):
                # NaN-safe: if AUC is "nan" keep it as nan
                auc_norm   = float("nan")
                drop_worst = float("nan")
                clean_ret  = float("nan")
            bucket[shift_type].append((clean_ret, auc_norm, drop_worst))

    out_rows: list[dict] = []
    for shift_type, vals in sorted(bucket.items()):
        clean_rets  = np.array([v[0] for v in vals], dtype=np.float64)
        auc_norms   = np.array([v[1] for v in vals], dtype=np.float64)
        drop_worsts = np.array([v[2] for v in vals], dtype=np.float64)
        out_rows.append({
            "shift_type":                    shift_type,
            "mean_clean_return":             float(np.nanmean(clean_rets)),
            "std_clean_return":              float(np.nanstd(clean_rets)),
            "mean_auc_return_norm":          float(np.nanmean(auc_norms)),
            "std_auc_return_norm":           float(np.nanstd(auc_norms)),
            "mean_relative_drop_worst_pct":  float(np.nanmean(drop_worsts)),
            "std_relative_drop_worst_pct":   float(np.nanstd(drop_worsts)),
            "n_seeds":                       len(vals),
        })

    return out_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_multiseed_return(
    rows: list[dict],
    shift_type: str,
    out_dir: str,
) -> str:
    """Plot mean_return ± std_return vs severity across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = [r["severity"]    for r in rows]
    means      = [r["mean_return"] for r in rows]
    stds       = [r["std_return"]  for r in rows]
    n_seeds    = rows[0]["n_seeds"] if rows else 0

    fig, ax = plt.subplots()
    ax.plot(severities, means, marker="o", label=f"mean return (n={n_seeds} seeds)")
    ax.fill_between(
        severities,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="± std (across seeds)",
    )
    ax.set_xlabel("severity")
    ax.set_ylabel("return mean")
    ax.set_title(f"Return vs severity — {shift_type} (multi-seed)")
    ax.legend()
    fig.tight_layout()

    fname    = f"return_vs_severity_{shift_type}_multiseed.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_multiseed_drop(
    rows: list[dict],
    shift_type: str,
    out_dir: str,
) -> str:
    """Plot mean_relative_drop_pct ± std vs severity across seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = [r["severity"]             for r in rows]
    means      = [r["mean_relative_drop_pct"] for r in rows]
    stds       = [r["std_relative_drop_pct"]  for r in rows]
    n_seeds    = rows[0]["n_seeds"] if rows else 0

    fig, ax = plt.subplots()
    ax.plot(severities, means, marker="o", label=f"mean %drop (n={n_seeds} seeds)")
    ax.fill_between(
        severities,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="± std (across seeds)",
    )
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("severity")
    ax.set_ylabel("relative drop (%)")
    ax.set_title(f"% degradation vs severity — {shift_type} (multi-seed)")
    ax.legend()
    fig.tight_layout()

    fname    = f"drop_vs_severity_{shift_type}_multiseed.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    out_dir  = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Discover seed dirs ---------------------------------------------------
    seed_dirs = _discover_seed_dirs(base_dir)
    print(f"Found {len(seed_dirs)} seed dir(s) under {base_dir!r}:")
    for d in seed_dirs:
        print(f"  {d}")

    # --- Load per-seed CSVs (auto-aggregating if needed) ---------------------
    all_curves:   list[list[dict]] = []
    all_summaries: list[list[dict]] = []

    for seed_dir in seed_dirs:
        curves_path, summary_path = _ensure_per_seed_csvs(seed_dir)

        curves_rows   = _load_csv(curves_path)
        summary_rows  = _load_csv(summary_path)

        if not curves_rows:
            print(
                f"WARNING: curves.csv is empty for {seed_dir!r} — "
                "no shifted eval data found.",
                file=sys.stderr,
            )
        if not summary_rows:
            print(
                f"WARNING: summary.csv is empty for {seed_dir!r}.",
                file=sys.stderr,
            )

        all_curves.append(curves_rows)
        all_summaries.append(summary_rows)
        print(
            f"  [ok] {os.path.basename(seed_dir)}: "
            f"{len(curves_rows)} curve rows, {len(summary_rows)} summary rows"
        )

    # --- Cross-seed aggregation -----------------------------------------------
    cross_curves  = _aggregate_curves(all_curves)
    cross_summary = _aggregate_summary(all_summaries)

    # --- Write cross-seed CSVs ------------------------------------------------
    curves_out  = os.path.join(out_dir, "cross_seed_curves.csv")
    summary_out = os.path.join(out_dir, "cross_seed_summary.csv")

    _write_csv(
        curves_out,
        fieldnames=[
            "shift_type", "severity",
            "mean_return", "std_return",
            "mean_relative_drop_pct", "std_relative_drop_pct",
            "n_seeds",
        ],
        rows=cross_curves,
    )

    _write_csv(
        summary_out,
        fieldnames=[
            "shift_type",
            "mean_clean_return", "std_clean_return",
            "mean_auc_return_norm", "std_auc_return_norm",
            "mean_relative_drop_worst_pct", "std_relative_drop_worst_pct",
            "n_seeds",
        ],
        rows=cross_summary,
    )

    print(f"\nWrote: {curves_out}")
    print(f"Wrote: {summary_out}")

    # --- Plots ----------------------------------------------------------------
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Group cross_curves by shift_type.
    grouped: dict[str, list[dict]] = {}
    for row in cross_curves:
        st = row["shift_type"]
        if st not in grouped:
            grouped[st] = []
        grouped[st].append(row)

    # Sort each group by severity.
    for rows in grouped.values():
        rows.sort(key=lambda r: r["severity"])

    generated: list[str] = []
    for shift_type, rows in sorted(grouped.items()):
        print(f"\nPlotting {shift_type!r}  n_severities={len(rows)}")
        p1 = _plot_multiseed_return(rows, shift_type, plots_dir)
        p2 = _plot_multiseed_drop(rows,   shift_type, plots_dir)
        generated.extend([p1, p2])
        print(f"  → {p1}")
        print(f"  → {p2}")

    print(f"\nGenerated {len(generated)} plot(s) in {plots_dir!r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
