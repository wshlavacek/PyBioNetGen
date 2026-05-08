"""Correctness sweep: run BNGL-Models/models through BNGsim path and diff vs reference.

Usage:
    .venv312/bin/python scripts/bngl_models_sweep.py [--models-dir PATH] [--only NAME]

For each .bngl file discovered, runs it via bionetgen.run(simulator='bngsim'),
locates the produced .gdat/.scan files, and compares them column-by-column
against the committed reference/ data. Emits a markdown table to stdout and a
JSON blob to ``dev/bngl_models_sweep_results.json``.

A model counts as OK only when every output file is within the abs/rel
tolerances (default ``abs < 1.0``, ``rel < 1e-2``). Per-model overrides
go in ``scripts/bngl_models_sweep_tolerances.yaml`` next to this
script, keyed by ``<model_dir>/<bngl_name>``:

    tolerances:
      tlbr_solution_macken1982/tlbr_solution_macken1982.bngl:
        abs: 1e3
        rel: 1e13

The JSON blob carries each file's ``status`` and the cutoffs used, so
later runs can compare apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml


DEFAULT_MODELS_DIR = Path.home() / "Code" / "BNGL-Models" / "models"
DEFAULT_TOL_FILE = Path(__file__).with_name("bngl_models_sweep_tolerances.yaml")
DEFAULT_ABS_TOL = 1.0
DEFAULT_REL_TOL = 1e-2

# Statuses that count as a quiet failure (DIFF) rather than OK. ``error``
# (exception/timeout) and ``shape-mismatch``/``missing-*`` are loud
# failures already; ``large-abs``/``large-rel`` are the new gates that
# previously slipped through as OK.
_DIFF_STATUSES = frozenset({
    "shape-mismatch", "missing-reference", "missing-output",
    "large-abs", "large-rel",
})


@dataclass
class FileDiff:
    output: str
    reference: str
    n_rows_out: int
    n_rows_ref: int
    n_cols: int
    max_abs_err: float
    max_rel_err: float
    abs_tol: float
    rel_tol: float
    # ok | large-abs | large-rel | shape-mismatch | missing-reference | missing-output
    status: str


@dataclass
class ModelResult:
    model: str
    bngl: str
    simulator: str
    wall_seconds: float
    ok: bool
    error: str = ""
    files: list[FileDiff] = field(default_factory=list)


def load_tolerances(path: Path) -> dict[str, dict[str, float]]:
    """Load per-model tolerance overrides from YAML.

    Returns a dict keyed by ``<model_dir>/<bngl_name>`` with values
    ``{"abs": ..., "rel": ...}``. Missing file → empty dict.
    """
    if not path.is_file():
        return {}
    with path.open() as f:
        doc = yaml.safe_load(f) or {}
    tols = doc.get("tolerances", {}) or {}
    out: dict[str, dict[str, float]] = {}
    for key, spec in tols.items():
        if not isinstance(spec, dict):
            continue
        entry: dict[str, float] = {}
        if "abs" in spec:
            entry["abs"] = float(spec["abs"])
        if "rel" in spec:
            entry["rel"] = float(spec["rel"])
        if entry:
            out[str(key)] = entry
    return out


def load_bng_data(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a BNG .gdat or .scan file. First line is '#' + whitespace-separated headers."""
    with path.open() as f:
        header = f.readline().strip()
    if header.startswith("#"):
        header = header[1:]
    cols = header.split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    return cols, data


def diff_file(out_path: Path, ref_path: Path, abs_tol: float, rel_tol: float) -> FileDiff:
    if not ref_path.exists():
        return FileDiff(
            output=out_path.name,
            reference=ref_path.name,
            n_rows_out=0,
            n_rows_ref=0,
            n_cols=0,
            max_abs_err=float("nan"),
            max_rel_err=float("nan"),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            status="missing-reference",
        )
    out_cols, out_data = load_bng_data(out_path)
    ref_cols, ref_data = load_bng_data(ref_path)
    if out_data.shape != ref_data.shape:
        return FileDiff(
            output=out_path.name,
            reference=ref_path.name,
            n_rows_out=out_data.shape[0],
            n_rows_ref=ref_data.shape[0],
            n_cols=min(out_data.shape[1], ref_data.shape[1]),
            max_abs_err=float("nan"),
            max_rel_err=float("nan"),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            status="shape-mismatch",
        )
    abs_err = np.abs(out_data - ref_data)
    denom = np.maximum(np.abs(ref_data), 1e-12)
    rel_err = abs_err / denom
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    # Pass if EITHER metric is within its tolerance, mirroring isclose
    # semantics: an observable on the scale of 1e+8 with abs err 5 has
    # negligible rel err (5e-8) and shouldn't be flagged just because
    # the abs cutoff is 1.0. Both metrics must blow past their cutoff
    # to call this a real divergence. When that happens, classify by
    # the dominant violator (largest cutoff overshoot ratio).
    if max_abs <= abs_tol or max_rel <= rel_tol:
        status = "ok"
    elif max_rel / rel_tol >= max_abs / abs_tol:
        status = "large-rel"
    else:
        status = "large-abs"
    return FileDiff(
        output=out_path.name,
        reference=ref_path.name,
        n_rows_out=out_data.shape[0],
        n_rows_ref=ref_data.shape[0],
        n_cols=out_data.shape[1],
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        status=status,
    )


RUNNER_SNIPPET = """
import sys, bionetgen
try:
    bionetgen.run(sys.argv[1], out=sys.argv[2], simulator=sys.argv[3], suppress=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(2)
"""


def run_one(
    bngl: Path, simulator: str, out_dir: Path, timeout: int,
    abs_tol: float, rel_tol: float,
) -> ModelResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ModelResult(
        model=bngl.parent.name,
        bngl=bngl.name,
        simulator=simulator,
        wall_seconds=0.0,
        ok=False,
    )
    t0 = time.monotonic()
    env = os.environ.copy()
    if simulator == "subprocess":
        env["BIONETGEN_NO_BNGSIM"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER_SNIPPET, str(bngl), str(out_dir), simulator],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        result.wall_seconds = time.monotonic() - t0
        result.error = f"TIMEOUT after {timeout}s"
        return result
    result.wall_seconds = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        result.error = "\n".join(tail) or f"exit {proc.returncode}"
        return result

    ref_dir = bngl.parent / "reference"
    produced: list[Path] = []
    for pattern in ("*.gdat", "*.scan"):
        produced.extend(sorted(out_dir.rglob(pattern)))

    for out_file in produced:
        ref_file = ref_dir / out_file.name
        result.files.append(diff_file(out_file, ref_file, abs_tol, rel_tol))

    if not produced:
        result.error = "no .gdat or .scan files produced"
        return result

    result.ok = all(f.status == "ok" for f in result.files)
    return result


def format_row(r: ModelResult) -> str:
    if r.error and not r.files:
        return f"| {r.model}/{r.bngl} | {r.wall_seconds:.1f}s | ERROR | {r.error.splitlines()[0][:60]} |"
    # Report worst err across all numeric files (whether or not they
    # passed the gate) so the table shows what the gate is reacting to.
    numeric = [f for f in r.files if f.status in ("ok", "large-abs", "large-rel")]
    worst_abs = max((f.max_abs_err for f in numeric), default=float("nan"))
    worst_rel = max((f.max_rel_err for f in numeric), default=float("nan"))
    bad = [f for f in r.files if f.status != "ok"]
    note = ""
    if bad:
        # Count by status so a model with 2 large-rel and 1 missing-ref
        # reads as "2 large-rel, 1 missing-reference".
        by_status: dict[str, int] = {}
        for f in bad:
            by_status[f.status] = by_status.get(f.status, 0) + 1
        note = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    return (
        f"| {r.model}/{r.bngl} | {r.wall_seconds:.1f}s | "
        f"{worst_abs:.2e} | {worst_rel:.2e} | {len(r.files)} | {note} |"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--only", help="substring filter on model directory name")
    p.add_argument("--simulator", default="bngsim", choices=["bngsim", "subprocess", "auto"])
    p.add_argument("--out-root", type=Path, default=Path("dev/bngl_models_sweep_out"))
    p.add_argument("--timeout", type=int, default=300, help="per-model timeout in seconds")
    p.add_argument(
        "--abs-tol", type=float, default=DEFAULT_ABS_TOL,
        help=f"default max abs err allowed (default {DEFAULT_ABS_TOL}; per-model overrides via --tolerances)",
    )
    p.add_argument(
        "--rel-tol", type=float, default=DEFAULT_REL_TOL,
        help=f"default max rel err allowed (default {DEFAULT_REL_TOL})",
    )
    p.add_argument(
        "--tolerances", type=Path, default=DEFAULT_TOL_FILE,
        help=f"YAML with per-model tolerance overrides (default {DEFAULT_TOL_FILE.name})",
    )
    args = p.parse_args()

    bngls = sorted(args.models_dir.glob("*/*.bngl"))
    if args.only:
        bngls = [b for b in bngls if args.only in b.parent.name]
    if not bngls:
        print(f"no .bngl files found under {args.models_dir}", file=sys.stderr)
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)

    overrides = load_tolerances(args.tolerances)
    if overrides:
        print(f"[tolerances] {len(overrides)} per-model overrides from {args.tolerances}")

    results: list[ModelResult] = []
    for bngl in bngls:
        slug = f"{bngl.parent.name}__{bngl.stem}"
        out_dir = args.out_root / args.simulator / slug
        # Override key uses the same identifier shown in the table.
        key = f"{bngl.parent.name}/{bngl.name}"
        ov = overrides.get(key, {})
        abs_tol = ov.get("abs", args.abs_tol)
        rel_tol = ov.get("rel", args.rel_tol)
        print(f"[run] {slug} ...", flush=True)
        r = run_one(bngl, args.simulator, out_dir, args.timeout, abs_tol, rel_tol)
        results.append(r)
        status = "OK" if r.ok else ("FAIL" if r.error else "DIFF")
        print(f"  -> {status} ({r.wall_seconds:.1f}s, {len(r.files)} output files)")

    print()
    print(f"## Sweep results ({args.simulator})")
    print()
    print(f"_Default cutoffs: abs < {args.abs_tol:g}, rel < {args.rel_tol:g}._")
    print()
    print("| model/bngl | wall | max abs err | max rel err | files | note |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(format_row(r))

    ok = sum(1 for r in results if r.ok)
    err = sum(1 for r in results if r.error)
    diff = len(results) - ok - err
    print()
    print(f"**Totals:** {ok} ok, {diff} diff, {err} errored, {len(results)} total")

    results_json = Path("dev/bngl_models_sweep_results.json")
    results_json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nJSON written to {results_json}")
    return 0 if err == 0 and diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
