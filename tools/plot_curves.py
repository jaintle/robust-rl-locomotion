"""
Robustness curve plotter.

Phase 7 — reads curves.csv (produced by tools/aggregate.py) and generates
one pair of PNG plots per shift_type:

  return_vs_severity_<shift_type>.png  — return_mean (± std) vs severity
  drop_vs_severity_<shift_type>.png    — relative_drop_pct vs severity

Uses matplotlib only (no seaborn, no pandas).

Usage:
    python tools/plot_curves.py \\
        --curves_csv results/agg_smoke_seed0/curves.csv \\
        --out_dir    results/agg_smoke_seed0/plots
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot return and %drop vs severity from curves.csv."
    )
    p.add_argument(
        "--curves_csv", type=str, required=True,
        help="Path to curves.csv produced by tools/aggregate.py.",
    )
    p.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for PNG files.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_curves(path: str) -> dict[str, list[dict]]:
    """Load curves.csv and group rows by shift_type.

    Returns:
        Dict mapping shift_type -> list of row dicts, sorted ascending by severity.
    """
    if not os.path.isfile(path):
        print(f"ERROR: curves.csv not found: {path!r}", file=sys.stderr)
        sys.exit(1)

    grouped: dict[str, list[dict]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shift_type = row["shift_type"]
            if shift_type not in grouped:
                grouped[shift_type] = []
            grouped[shift_type].append({
                "severity":          float(row["severity"]),
                "return_mean":       float(row["return_mean"]),
                "return_std":        float(row["return_std"]),
                "clean_return_mean": float(row["clean_return_mean"]),
                "relative_drop_pct": float(row["relative_drop_pct"]),
            })

    # Sort each group by severity ascending.
    for rows in grouped.values():
        rows.sort(key=lambda r: r["severity"])

    return grouped


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_return(
    rows: list[dict],
    shift_type: str,
    out_dir: str,
) -> str:
    """Plot return_mean (± std) vs severity.  Returns output path."""
    import matplotlib
    matplotlib.use("Agg")          # headless; must be set before pyplot import
    import matplotlib.pyplot as plt

    severities = [r["severity"]    for r in rows]
    means      = [r["return_mean"] for r in rows]
    stds       = [r["return_std"]  for r in rows]
    clean_mean = rows[0]["clean_return_mean"]

    fig, ax = plt.subplots()

    ax.plot(severities, means, marker="o", label="shifted return")
    ax.fill_between(
        severities,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25,
        label="± std",
    )
    ax.axhline(clean_mean, linestyle="--", label=f"clean baseline ({clean_mean:.2f})")

    ax.set_xlabel("severity")
    ax.set_ylabel("return mean")
    ax.set_title(f"Return vs severity — {shift_type}")
    ax.legend()
    fig.tight_layout()

    fname = f"return_vs_severity_{shift_type}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _plot_drop(
    rows: list[dict],
    shift_type: str,
    out_dir: str,
) -> str:
    """Plot relative_drop_pct vs severity.  Returns output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severities = [r["severity"]          for r in rows]
    drops      = [r["relative_drop_pct"] for r in rows]

    fig, ax = plt.subplots()

    ax.plot(severities, drops, marker="o")
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=0.8)

    ax.set_xlabel("severity")
    ax.set_ylabel("relative drop (%)")
    ax.set_title(f"% degradation vs severity — {shift_type}")
    fig.tight_layout()

    fname = f"drop_vs_severity_{shift_type}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    grouped = _load_curves(args.curves_csv)

    if not grouped:
        print("WARNING: curves.csv contains no rows — nothing to plot.", file=sys.stderr)
        sys.exit(0)

    os.makedirs(args.out_dir, exist_ok=True)

    generated: list[str] = []

    for shift_type, rows in sorted(grouped.items()):
        print(f"Plotting shift_type={shift_type!r}  n_points={len(rows)}")

        p1 = _plot_return(rows, shift_type, args.out_dir)
        p2 = _plot_drop(rows,   shift_type, args.out_dir)

        generated.extend([p1, p2])
        print(f"  → {p1}")
        print(f"  → {p2}")

    print(f"\nGenerated {len(generated)} plot(s) in {args.out_dir!r}")
    sys.exit(0)


if __name__ == "__main__":
    main()
