#!/usr/bin/env python3
"""Sweep the BNGL corpus through bionetgen.run() with an explicit simulator.

For each .bngl, classifies the model as deterministic (only ode/cvode actions)
or stochastic (any nf/ssa/pla/psa action). Deterministic models are run once
per side. Stochastic models are patched to inject seeds 1..N (default N=10) and
run once per seed per side, into per-seed output directories so the diff
script can compute ensemble means and stds.

Designed to be called twice: once with --simulator subprocess, once with
--simulator bngsim, into different --out roots. The diff script then
ensemble-compares the two roots.

Inherits the gotcha-fixes from seeded_sweep.py:
  * cwd=tempfile.gettempdir() so the venv install isn't shadowed by a
    PyBioNetGen source dir.
  * regex `\\g<1>` not `\\1` for backref-then-digits substitution.
  * TEND_OVERRIDES for slow models.
  * Spaces-in-filenames safe (Path-based, no bash globbing).
  * Doesn't depend on bionetgen.bngmodel() — uses regex over raw text so
    parser-rejecting BNGL like prion2_YTLedits.bngl can still be classified.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

OUTPUT_EXTENSIONS = {".gdat", ".cdat", ".net", ".scan", ".xml", ".species"}
DETERMINISTIC_METHODS = {"ode", "cvode"}

TEND_OVERRIDES = {
    "AD 3 State FREE Expanding nfs.bngl": 100,  # default 650 -> 180s timeout
    # B6: bngsim codegen falls back to interpreted ODE RHS for this model
    # ("Starred arguments in lambda not supported"), and the default
    # t_end=730 over 10 NF seeds blows the 180s budget. Cap to keep the
    # parity check exercising both deterministic and stochastic segments
    # within the timeout. The cap is a sweep-side workaround for a
    # documented bngsim codegen limitation; not a model-correctness change.
    "scaling_example.bngl": 50,
}


def parse_simulate_methods(text):
    """Return list of (suffix_or_None, method) for each simulate-style action.

    Strips comments first. Handles both simulate({method=>X,...}) and
    simulate_<method>({...}).
    """
    text = re.sub(r"#.*", "", text)
    out = []
    for blob in re.findall(r"simulate\s*\(\s*\{([^}]*)\}", text, re.DOTALL):
        method_m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        suffix_m = re.search(r"suffix\s*=>\s*['\"]?([^'\",}\s]+)['\"]?", blob)
        method = (method_m.group(1) if method_m else "ode").lower()
        suffix = suffix_m.group(1) if suffix_m else None
        out.append((suffix, method))
    for m_method, blob in re.findall(
        r"simulate_(\w+)\s*\(\s*\{([^}]*)\}", text, re.DOTALL,
    ):
        suffix_m = re.search(r"suffix\s*=>\s*['\"]?([^'\",}\s]+)['\"]?", blob)
        suffix = suffix_m.group(1) if suffix_m else None
        out.append((suffix, m_method.lower()))
    return out


def is_stochastic(text):
    for _, method in parse_simulate_methods(text):
        if method not in DETERMINISTIC_METHODS:
            return True
    return False


def patch_bngl(text, seed, tend_override=None):
    """Inject seed=>K into every active simulate action; optional t_end override."""
    out_lines = []
    seed_inject = re.compile(r"(simulate(?:_\w+)?\s*\(\s*\{)")
    tend_re = re.compile(r"(t_end\s*=>\s*)([0-9eE.+\-*/() ]+)")
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out_lines.append(line)
            continue
        new_line = seed_inject.sub(rf"\g<1>seed=>{seed},", line)
        if tend_override is not None:
            new_line = tend_re.sub(rf"\g<1>{tend_override}", new_line)
        out_lines.append(new_line)
    return "".join(out_lines)


def patch_bngl_tend_only(text, tend_override):
    """For deterministic models we only need the t_end override (no seed)."""
    if tend_override is None:
        return text
    out_lines = []
    tend_re = re.compile(r"(t_end\s*=>\s*)([0-9eE.+\-*/() ]+)")
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out_lines.append(line)
            continue
        out_lines.append(tend_re.sub(rf"\g<1>{tend_override}", line))
    return "".join(out_lines)


def run_one(simulator, bngl_path, run_path, out_dir, timeout):
    """Run a (possibly patched) .bngl through bionetgen.run() in a subprocess.

    `bngl_path` is the original (just for logging / summary identity).
    `run_path` is what bionetgen.run() actually loads (patched copy or original).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_run.log"
    inner = (
        "import sys, bionetgen\n"
        f"bionetgen.run({str(run_path)!r}, out={str(out_dir)!r}, "
        f"timeout={timeout}, suppress=True, simulator={simulator!r})\n"
    )
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", inner],
            capture_output=True, text=True,
            timeout=timeout + 30,
            cwd=tempfile.gettempdir(),
        )
        elapsed = time.monotonic() - start
        artifacts = sorted(
            f.name for f in out_dir.iterdir()
            if f.is_file() and f.suffix in OUTPUT_EXTENSIONS
        ) if out_dir.exists() else []
        if proc.returncode == 0:
            log_path.write_text(
                f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
                f"# status=ok\n# wall_seconds={elapsed:.2f}\n"
                f"# artifacts={artifacts}\n\n"
                f"--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n"
            )
            return {
                "bngl": str(bngl_path),
                "run": str(run_path),
                "category": Path(bngl_path).parent.name,
                "status": "ok",
                "wall_seconds": elapsed,
                "artifacts": artifacts,
                "out_dir": str(out_dir),
                "error": "",
            }
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=crash\n# returncode={proc.returncode}\n"
            f"# wall_seconds={elapsed:.2f}\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "crash",
            "wall_seconds": elapsed,
            "artifacts": artifacts,
            "out_dir": str(out_dir),
            "error": (proc.stderr.strip().splitlines() or [""])[-1][:500],
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=timeout\n# wall_seconds={elapsed:.2f}\n\n"
            f"--- STDOUT ---\n{e.stdout or ''}\n\n--- STDERR ---\n{e.stderr or ''}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "timeout",
            "wall_seconds": elapsed,
            "artifacts": [],
            "out_dir": str(out_dir),
            "error": f"timed out after {timeout}s",
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        log_path.write_text(
            f"# {bngl_path}\n# run={run_path}\n# simulator={simulator}\n"
            f"# status=error\n# wall_seconds={elapsed:.2f}\n\n"
            f"--- TRACEBACK ---\n{traceback.format_exc()}\n"
        )
        return {
            "bngl": str(bngl_path),
            "run": str(run_path),
            "category": Path(bngl_path).parent.name,
            "status": "error",
            "wall_seconds": elapsed,
            "artifacts": [],
            "out_dir": str(out_dir),
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory tree with .bngl files")
    ap.add_argument("--out", required=True, help="Output root for patched copies + artifacts")
    ap.add_argument("--simulator", required=True, choices=("subprocess", "bngsim"),
                    help="Simulator to pass to bionetgen.run()")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="Seeds 1..N for stochastic models")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180, help="Per-model timeout (s)")
    ap.add_argument("--limit", type=int, default=0, help="Max .bngl files (0=all)")
    ap.add_argument("--include", default="",
                    help="Substring filter on file path (debugging)")
    ap.add_argument("--exclude", default="",
                    help="Substring filter — drop matching file paths")
    ap.add_argument("--models", default="",
                    help="Comma-separated model basenames (.bngl optional); "
                         "restrict the sweep to exactly these — for selective "
                         "high-seed re-runs of models flagged DIFF")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    patch_root = out_root / "_patched"
    patch_root.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(root.rglob("*.bngl"))
    if args.include:
        candidate_files = [f for f in candidate_files if args.include in str(f)]
    if args.exclude:
        candidate_files = [f for f in candidate_files if args.exclude not in str(f)]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        wanted |= {m + ".bngl" for m in set(wanted) if not m.endswith(".bngl")}
        candidate_files = [f for f in candidate_files if f.name in wanted]

    # For each model, decide regime + emit (run_path, out_dir, role) units.
    # role is "deterministic" or seed_K.
    units = []     # list of dicts: bngl, run, out_dir, role, regime
    n_det = 0
    n_stoch = 0
    n_unreadable = 0
    for src in candidate_files:
        try:
            text = src.read_text(errors="replace")
        except Exception:
            n_unreadable += 1
            continue
        rel = src.relative_to(root)
        tend_override = TEND_OVERRIDES.get(src.name)
        if is_stochastic(text):
            n_stoch += 1
            for seed in range(1, args.n_seeds + 1):
                patched_dir = patch_root / rel.parent / f"{rel.stem}__seed{seed}"
                patched_dir.mkdir(parents=True, exist_ok=True)
                patched_path = patched_dir / rel.name
                patched_path.write_text(patch_bngl(text, seed, tend_override))
                out_dir = out_root / rel.parent / rel.stem / f"seed{seed}"
                units.append({
                    "bngl": str(src),
                    "run": str(patched_path),
                    "out_dir": str(out_dir),
                    "role": f"seed{seed}",
                    "regime": "stochastic",
                })
        else:
            n_det += 1
            # Apply t_end override if needed (none of the deterministic
            # models in current corpus need it, but kept for safety).
            if tend_override is not None:
                patched_dir = patch_root / rel.parent
                patched_dir.mkdir(parents=True, exist_ok=True)
                patched_path = patched_dir / rel.name
                patched_path.write_text(patch_bngl_tend_only(text, tend_override))
                run_path = patched_path
            else:
                run_path = src
            out_dir = out_root / rel.parent / rel.stem / "det"
            units.append({
                "bngl": str(src),
                "run": str(run_path),
                "out_dir": str(out_dir),
                "role": "det",
                "regime": "deterministic",
            })

    if args.limit:
        units = units[: args.limit]

    print(f"sweep root:    {root}")
    print(f"sweep out:     {out_root}")
    print(f"simulator:     {args.simulator}")
    print(f"n_seeds:       {args.n_seeds}")
    print(f"deterministic: {n_det}")
    print(f"stochastic:    {n_stoch}  (× {args.n_seeds} seeds = {n_stoch * args.n_seeds} runs)")
    print(f"unreadable:    {n_unreadable}")
    print(f"total runs:    {len(units)}")
    print(f"workers:       {args.workers}, per-model timeout: {args.timeout}s")
    print(f"python:        {sys.executable}")

    probe = subprocess.run(
        [sys.executable, "-c",
         "import bionetgen; print(bionetgen.__file__)"],
        capture_output=True, text=True, cwd=tempfile.gettempdir(),
    )
    bionetgen_path = (probe.stdout.strip().splitlines() or ["<unresolved>"])[-1]
    print(f"bionetgen:     {bionetgen_path}")
    bngsim_path = None
    if args.simulator == "bngsim":
        probe2 = subprocess.run(
            [sys.executable, "-c",
             "import bngsim; print(bngsim.__file__, getattr(bngsim, '__version__', '?'))"],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
        )
        bngsim_path = (probe2.stdout.strip().splitlines() or [""])[-1]
        print(f"bngsim:        {bngsim_path}")

    summary = []
    started_at = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                run_one,
                args.simulator,
                u["bngl"], u["run"], u["out_dir"],
                args.timeout,
            ): u for u in units
        }
        done = 0
        for fut in cf.as_completed(futs):
            done += 1
            u = futs[fut]
            res = fut.result()
            res["role"] = u["role"]
            res["regime"] = u["regime"]
            summary.append(res)
            if done % 50 == 0 or done == len(units):
                print(f"[{done}/{len(units)}] {res['status']:8s} "
                      f"{res['wall_seconds']:6.1f}s  {u['role']:8s} {res['bngl']}")

    elapsed_total = time.time() - started_at
    by_status = {}
    for r in summary:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1

    summary_path = out_root / "_summary.json"
    summary_path.write_text(json.dumps({
        "root": str(root),
        "out": str(out_root),
        "python": sys.executable,
        "bionetgen_path": bionetgen_path,
        "bngsim_path": bngsim_path,
        "simulator": args.simulator,
        "n_seeds": args.n_seeds,
        "tend_overrides": TEND_OVERRIDES,
        "n_deterministic_models": n_det,
        "n_stochastic_models": n_stoch,
        "n_units": len(units),
        "elapsed_total_seconds": elapsed_total,
        "by_status": by_status,
        "results": sorted(summary, key=lambda r: (r["bngl"], r.get("role", ""))),
    }, indent=2))

    print(f"\nDone in {elapsed_total/60:.1f}m. By status: {by_status}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
