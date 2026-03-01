"""
Validate a training run directory against results/schema.json.

Usage:
    python tools/validate_results.py --run_dir runs/smoke_seed0

Exits 0 and prints VALIDATION PASSED on success.
Exits 1 and prints a clear error message on the first failure.

No external dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Schema location (relative to this file → repo root → results/schema.json)
# ---------------------------------------------------------------------------

_TOOLS_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_TOOLS_DIR)
_SCHEMA_PATH = os.path.join(_REPO_ROOT, "results", "schema.json")


# ---------------------------------------------------------------------------
# Type checkers (map schema type-string → predicate)
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

    # 2. Required keys + type checks ------------------------------------------
    required_keys: dict[str, dict[str, str]] = schema["required_keys"]

    for json_file, key_specs in required_keys.items():
        fpath = os.path.join(run_dir, json_file)
        data  = _load_json(fpath)

        for key, expected_type in key_specs.items():
            # Key presence.
            if key not in data:
                _fail(f"{json_file}: missing required key {key!r}")

            value = data[key]

            # Null guard — null is never acceptable for any typed field.
            if value is None:
                _fail(
                    f"{json_file}: key {key!r} is null "
                    f"(expected type {expected_type!r})"
                )

            # Type check.
            checker = _TYPE_CHECK.get(expected_type)
            if checker is None:
                _fail(f"Schema error: unknown type tag {expected_type!r} for key {key!r}")

            if not checker(value):
                _fail(
                    f"{json_file}: key {key!r} = {value!r} "
                    f"failed type check (expected {expected_type!r}, "
                    f"got {type(value).__name__!r})"
                )

            print(f"  [ok] {json_file}: {key!r} = {value!r} ({expected_type})")

    # 3. Extra check: eval_seed_list_hash must be a non-empty string ----------
    eval_path = os.path.join(run_dir, "eval_clean.json")
    eval_data = _load_json(eval_path)
    seed_hash = eval_data.get("eval_seed_list_hash", "")
    if not isinstance(seed_hash, str) or len(seed_hash) == 0:
        _fail(
            "eval_clean.json: eval_seed_list_hash is missing or empty "
            f"(got {seed_hash!r})"
        )
    print(
        f"  [ok] eval_clean.json: eval_seed_list_hash is non-empty "
        f"({len(seed_hash)} chars)"
    )

    # 4. All checks passed ----------------------------------------------------
    print("\nVALIDATION PASSED")


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
