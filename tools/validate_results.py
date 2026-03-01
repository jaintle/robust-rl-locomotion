"""
Validate a training run directory against results/schema.json.

Usage:
    python tools/validate_results.py --run_dir runs/smoke_seed0

Exits 0 and prints VALIDATION PASSED on success.
Exits 1 and prints a clear error message on the first failure.

Phase 4: validates required files + JSON key/type checks.
Phase 5: also validates eval_shifted.jsonl if present (optional file).
Phase 6: also validates eval_shifted_dynamics.jsonl if present (optional file).

No external dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Schema location (relative to this file -> repo root -> results/schema.json)
# ---------------------------------------------------------------------------

_TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_TOOLS_DIR)
_SCHEMA_PATH = os.path.join(_REPO_ROOT, "results", "schema.json")


# ---------------------------------------------------------------------------
# Type checkers (map schema type-string -> predicate)
# ---------------------------------------------------------------------------

def _is_str(v) -> bool:
    return isinstance(v, str)


def _is_int(v) -> bool:
    # Explicitly exclude bool, which is a subclass of int in Python.
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_TYPE_CHECK = {
    "str":    _is_str,
    "int":    _is_int,
    "number": _is_number,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f"VALIDATION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _fail(f"Could not parse {path!r}: {exc}")


def _check_keys(source: str, data: dict, key_specs: dict[str, str]) -> None:
    """Validate presence and types of required keys in a dict."""
    for key, expected_type in key_specs.items():
        if key not in data:
            _fail(f"{source}: missing required key {key!r}")

        value = data[key]

        if value is None:
            _fail(
                f"{source}: key {key!r} is null "
                f"(expected type {expected_type!r})"
            )

        checker = _TYPE_CHECK.get(expected_type)
        if checker is None:
            _fail(f"Schema error: unknown type tag {expected_type!r} for key {key!r}")

        if not checker(value):
            _fail(
                f"{source}: key {key!r} = {value!r} "
                f"failed type check (expected {expected_type!r}, "
                f"got {type(value).__name__!r})"
            )

        print(f"  [ok] {source}: {key!r} = {value!r} ({expected_type})")


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate(run_dir: str, schema: dict) -> None:
    """Run all schema checks against a run directory. Calls _fail on error."""

    run_dir = os.path.abspath(run_dir)

    if not os.path.isdir(run_dir):
        _fail(f"Run directory does not exist: {run_dir!r}")

    # 1. Required files -------------------------------------------------------
    required_files: list[str] = schema["required_files"]
    for fname in required_files:
        fpath = os.path.join(run_dir, fname)
        if not os.path.isfile(fpath):
            _fail(f"Missing required file: {fname!r} (expected at {fpath!r})")
        print(f"  [ok] file exists: {fname}")

    # 2. Required keys + type checks on JSON files ----------------------------
    required_keys: dict[str, dict[str, str]] = schema["required_keys"]

    for json_file, key_specs in required_keys.items():
        fpath = os.path.join(run_dir, json_file)
        data  = _load_json(fpath)
        _check_keys(json_file, data, key_specs)

    # 3. eval_clean.json: eval_seed_list_hash must be non-empty --------------
    eval_clean_path = os.path.join(run_dir, "eval_clean.json")
    eval_clean_data = _load_json(eval_clean_path)
    clean_hash = eval_clean_data.get("eval_seed_list_hash", "")
    if not isinstance(clean_hash, str) or len(clean_hash) == 0:
        _fail(
            "eval_clean.json: eval_seed_list_hash is missing or empty "
            f"(got {clean_hash!r})"
        )
    clean_episodes = eval_clean_data.get("episodes")
    print(
        f"  [ok] eval_clean.json: eval_seed_list_hash is non-empty "
        f"({len(clean_hash)} chars)"
    )

    # 4. Optional: eval_shifted.jsonl (Phase 5 — Gaussian noise) -------------
    shifted_path = os.path.join(run_dir, "eval_shifted.jsonl")
    if not os.path.isfile(shifted_path):
        print("  [--] eval_shifted.jsonl not present (optional — skipping)")
    else:
        _validate_shifted_jsonl(shifted_path, schema, clean_hash, clean_episodes)

    # 5. Optional: eval_shifted_dynamics.jsonl (Phase 6 — mass scale) ---------
    dynamics_path = os.path.join(run_dir, "eval_shifted_dynamics.jsonl")
    if not os.path.isfile(dynamics_path):
        print("  [--] eval_shifted_dynamics.jsonl not present (optional — skipping)")
    else:
        _validate_shifted_dynamics_jsonl(dynamics_path, schema, clean_hash, clean_episodes)

    # 6. All checks passed ----------------------------------------------------
    print("\nVALIDATION PASSED")


def _validate_shifted_jsonl(
    shifted_path: str,
    schema: dict,
    clean_hash: str,
    clean_episodes: int | None,
) -> None:
    """Validate eval_shifted.jsonl line by line.

    Key invariants enforced:
    1. All lines must carry the SAME eval_seed_list_hash (the shifted-eval
       script precomputes one seed list and reuses it across all severities).
    2. If the episode count in eval_shifted.jsonl matches eval_clean.json,
       the hash must also match eval_clean.json (same episodes ↔ same seeds).
       When episode counts differ this check is skipped and a note is printed.
    """

    jsonl_schema  = schema.get("optional_files", {}).get("eval_shifted.jsonl", {})
    line_key_spec = jsonl_schema.get("per_line_required_keys", {})

    with open(shifted_path) as f:
        raw_lines = f.readlines()

    non_empty = [l.strip() for l in raw_lines if l.strip()]
    if len(non_empty) == 0:
        _fail("eval_shifted.jsonl: file exists but contains no non-empty lines")

    seen_hashes:   set[str] = set()
    seen_episodes: set[int] = set()

    for i, line in enumerate(non_empty, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"eval_shifted.jsonl line {i}: not valid JSON — {exc}")

        source = f"eval_shifted.jsonl line {i}"

        # Required keys and types.
        _check_keys(source, entry, line_key_spec)

        # shift_type must be a non-empty string.
        shift_type = entry.get("shift_type")
        if not isinstance(shift_type, str) or not shift_type:
            _fail(f"{source}: shift_type must be a non-empty string (got {shift_type!r})")

        # severity must be a number.
        severity = entry.get("severity")
        if not _is_number(severity):
            _fail(
                f"{source}: severity must be a number "
                f"(got {type(severity).__name__!r}: {severity!r})"
            )

        line_hash = entry.get("eval_seed_list_hash", "")
        seen_hashes.add(line_hash)

        line_eps = entry.get("episodes")
        if _is_int(line_eps):
            seen_episodes.add(line_eps)

        print(
            f"  [ok] eval_shifted.jsonl line {i}: "
            f"shift_type={shift_type!r}  severity={severity}"
        )

    # Invariant 1: all lines must share the same seed-list hash.
    if len(seen_hashes) > 1:
        _fail(
            f"eval_shifted.jsonl: inconsistent eval_seed_list_hash across lines — "
            f"found {len(seen_hashes)} distinct values: {seen_hashes}. "
            "All severities must use the same episode seed list."
        )
    shifted_hash = next(iter(seen_hashes))
    print(f"  [ok] eval_shifted.jsonl: all {len(non_empty)} lines share hash {shifted_hash[:16]}…")

    # Invariant 2: if episode count matches eval_clean.json, hashes must agree.
    if len(seen_episodes) == 1:
        shifted_eps = next(iter(seen_episodes))
        if clean_episodes is not None and shifted_eps == clean_episodes:
            if shifted_hash != clean_hash:
                _fail(
                    f"eval_shifted.jsonl: eval_seed_list_hash {shifted_hash!r} "
                    f"does not match eval_clean.json hash {clean_hash!r}, "
                    f"but both use episodes={shifted_eps}. "
                    "Identical episode counts must produce identical seed lists."
                )
            print(
                f"  [ok] eval_shifted.jsonl: hash matches eval_clean.json "
                f"(episodes={shifted_eps})"
            )
        else:
            print(
                f"  [--] eval_shifted.jsonl uses episodes={shifted_eps}, "
                f"eval_clean.json uses episodes={clean_episodes} — "
                "hash cross-check skipped (different episode counts)."
            )

    print(f"  [ok] eval_shifted.jsonl: {len(non_empty)} entries validated")


def _validate_shifted_dynamics_jsonl(
    dynamics_path: str,
    schema: dict,
    clean_hash: str,
    clean_episodes: int | None,
) -> None:
    """Validate eval_shifted_dynamics.jsonl line by line.

    Key invariants enforced:
    1. All lines must carry the SAME eval_seed_list_hash (the shifted-eval
       script precomputes one seed list and reuses it across all severities).
    2. If the episode count in eval_shifted_dynamics.jsonl matches eval_clean.json,
       the hash must also match eval_clean.json (same episodes ↔ same seeds).
       When episode counts differ this check is skipped and a note is printed.
    3. shift_type must be 'mass_scale' for every line.
    """

    jsonl_schema  = schema.get("optional_files", {}).get("eval_shifted_dynamics.jsonl", {})
    line_key_spec = jsonl_schema.get("per_line_required_keys", {})

    with open(dynamics_path) as f:
        raw_lines = f.readlines()

    non_empty = [line.strip() for line in raw_lines if line.strip()]
    if len(non_empty) == 0:
        _fail("eval_shifted_dynamics.jsonl: file exists but contains no non-empty lines")

    seen_hashes:   set[str] = set()
    seen_episodes: set[int] = set()

    for i, line in enumerate(non_empty, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"eval_shifted_dynamics.jsonl line {i}: not valid JSON — {exc}")

        source = f"eval_shifted_dynamics.jsonl line {i}"

        # Required keys and types.
        _check_keys(source, entry, line_key_spec)

        # shift_type must be 'mass_scale'.
        shift_type = entry.get("shift_type")
        if shift_type != "mass_scale":
            _fail(
                f"{source}: shift_type must be 'mass_scale' for Phase 6 runs "
                f"(got {shift_type!r})"
            )

        # severity must be a number.
        severity = entry.get("severity")
        if not _is_number(severity):
            _fail(
                f"{source}: severity must be a number "
                f"(got {type(severity).__name__!r}: {severity!r})"
            )

        line_hash = entry.get("eval_seed_list_hash", "")
        seen_hashes.add(line_hash)

        line_eps = entry.get("episodes")
        if _is_int(line_eps):
            seen_episodes.add(line_eps)

        print(
            f"  [ok] eval_shifted_dynamics.jsonl line {i}: "
            f"shift_type={shift_type!r}  severity={severity}  "
            f"mass_scale_mean={entry.get('mass_scale_mean')}"
        )

    # Invariant 1: all lines must share the same seed-list hash.
    if len(seen_hashes) > 1:
        _fail(
            f"eval_shifted_dynamics.jsonl: inconsistent eval_seed_list_hash across "
            f"lines — found {len(seen_hashes)} distinct values: {seen_hashes}. "
            "All severities must use the same episode seed list."
        )
    shifted_hash = next(iter(seen_hashes))
    print(
        f"  [ok] eval_shifted_dynamics.jsonl: all {len(non_empty)} lines "
        f"share hash {shifted_hash[:16]}…"
    )

    # Invariant 2: conditional hash cross-check against eval_clean.json.
    if len(seen_episodes) == 1:
        shifted_eps = next(iter(seen_episodes))
        if clean_episodes is not None and shifted_eps == clean_episodes:
            if shifted_hash != clean_hash:
                _fail(
                    f"eval_shifted_dynamics.jsonl: eval_seed_list_hash {shifted_hash!r} "
                    f"does not match eval_clean.json hash {clean_hash!r}, "
                    f"but both use episodes={shifted_eps}. "
                    "Identical episode counts must produce identical seed lists."
                )
            print(
                f"  [ok] eval_shifted_dynamics.jsonl: hash matches eval_clean.json "
                f"(episodes={shifted_eps})"
            )
        else:
            print(
                f"  [--] eval_shifted_dynamics.jsonl uses episodes={shifted_eps}, "
                f"eval_clean.json uses episodes={clean_episodes} — "
                "hash cross-check skipped (different episode counts)."
            )

    print(f"  [ok] eval_shifted_dynamics.jsonl: {len(non_empty)} entries validated")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a training run directory against results/schema.json."
    )
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the run directory to validate (e.g. runs/smoke_seed0).",
    )
    p.add_argument(
        "--schema",
        type=str,
        default=_SCHEMA_PATH,
        help=f"Path to schema.json (default: {_SCHEMA_PATH}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.schema):
        _fail(f"Schema file not found: {args.schema!r}")

    schema = _load_json(args.schema)
    print(f"Validating run directory: {args.run_dir!r}")
    print(f"Schema: {args.schema!r}\n")

    validate(args.run_dir, schema)
    sys.exit(0)


if __name__ == "__main__":
    main()
