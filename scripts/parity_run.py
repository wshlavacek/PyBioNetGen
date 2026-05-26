#!/usr/bin/env python3
"""End-to-end BNGsim parity sweep with adaptive replicate counts.

One command runs the whole parity check:

  1. Base sweep — every model through ``parity_sweep.py`` twice
     (``--simulator subprocess`` then ``--simulator bngsim``) at the base
     seed count (default 10 for stochastic models).
  2. Base diff — ``parity_diff.py`` buckets every model PASS/DIFF/...
  3. Escalation — every *stochastic* model bucketed DIFF whose failure is
     an ensemble-noise failure (not a structural shape/load mismatch) is
     re-run at a higher seed count (default 50) on both sides.
  4. Final diff — ``parity_diff.py`` again, with the escalated sweeps as
     an overlay so those models are re-judged at the escalated seed count.

Why escalate? A stochastic model's ensemble verdict at N=10 seeds is
noisy: the per-seed std estimate that feeds the 3-sigma test is itself
unreliable at small N, so genuinely-consistent models routinely flag
DIFF purely from small-sample scatter (proven — v06/v09/v13 etc. DIFF at
10, PASS at 50). More seeds *sharpens* the verdict (it does not loosen
the test — for a consistent pair the test is N-invariant in expectation,
the noise just shrinks). Escalating to ~50 seeds and re-judging gives a
report whose DIFFs are real, not sampling noise.

Structural stochastic DIFFs (a .gdat/.cdat shape mismatch, a load error,
inconsistent shapes across seeds) are NOT escalated: those are real,
seed-count-independent discrepancies and re-running at 50 seeds only
burns time. They stay DIFF.

Layout under ``--out``:
    subprocess/            base subprocess sweep
    bngsim/                base bngsim sweep
    escalated/subprocess/  escalated subprocess re-run (selected models)
    escalated/bngsim/      escalated bngsim re-run
    parity_report_base.*   base verdict (before escalation)
    parity_report.{md,json} final verdict (escalated models re-judged)

Run from the .venv venv so the spawned sweeps and the backend
helper see the installed bngsim. Cap --workers at the core count.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SWEEP = SCRIPTS / "parity_sweep.py"
DIFF = SCRIPTS / "parity_diff.py"

# A stochastic DIFF whose only per-file failures are these is structural —
# a real, seed-count-independent discrepancy. Escalating it wastes runs.
_STRUCTURAL_FAIL_TOKENS = ("shape", "load_error")


def run(cmd):
    """Run a subprocess step, streaming output; abort the orchestrator on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    proc = subprocess.run([str(c) for c in cmd])
    if proc.returncode != 0:
        sys.exit(f"step failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")


def sweep(
    simulator,
    root,
    out,
    n_seeds,
    workers,
    timeout,
    limit,
    include,
    exclude,
    models=None,
    manifest="",
):
    cmd = [
        sys.executable,
        SWEEP,
        "--root",
        root,
        "--out",
        out,
        "--simulator",
        simulator,
        "--n-seeds",
        n_seeds,
        "--workers",
        workers,
        "--timeout",
        timeout,
    ]
    if manifest:
        cmd += ["--manifest", manifest]
    if limit:
        cmd += ["--limit", limit]
    if include:
        cmd += ["--include", include]
    if exclude:
        cmd += ["--exclude", exclude]
    if models:
        cmd += ["--models", ",".join(models)]
    run(cmd)


def diff(sub, bng, md_out, json_out, overlays=()):
    cmd = [
        sys.executable,
        DIFF,
        "--subprocess",
        sub,
        "--bngsim",
        bng,
        "--out",
        md_out,
        "--json-out",
        json_out,
    ]
    for ov_sub, ov_bng in overlays:
        cmd += ["--overlay-subprocess", ov_sub, "--overlay-bngsim", ov_bng]
    run(cmd)


def is_escalatable_stochastic_diff(info):
    """True if a DIFF per-model entry is a stochastic *ensemble-noise* failure.

    Escalate only stochastic DIFFs whose failures are ensemble-test
    failures — those are the ones more seeds can resolve. A structural
    failure (a shape mismatch, a load error) is seed-count-independent
    and stays DIFF.
    """
    if info.get("bucket") != "DIFF" or info.get("regime") != "stochastic":
        return False
    per_file = (info.get("details") or {}).get("per_file", {})
    saw_failure = False
    for stats in per_file.values():
        if "load_error" in stats:
            return False
        fail = str(stats.get("fail", ""))
        if not fail:
            continue
        saw_failure = True
        if any(tok in fail for tok in _STRUCTURAL_FAIL_TOKENS):
            return False
    # Escalate if there was at least one (non-structural) failure. If no
    # per-file fail is recorded the DIFF came from elsewhere (e.g. a
    # side-status issue) — not something seeds fix.
    return saw_failure


def main():
    ap = argparse.ArgumentParser(description="Adaptive BNGsim parity sweep (base + escalation).")
    ap.add_argument("--root", required=True, help="Directory tree with .bngl files")
    ap.add_argument("--out", required=True, help="Output root")
    ap.add_argument(
        "--n-seeds", type=int, default=10, help="Base seed count for stochastic models (default 10)"
    )
    ap.add_argument(
        "--escalate-seeds",
        type=int,
        default=150,
        help="Seed count to re-judge stochastic DIFFs at (default "
        "150; some slow-tier oscillator/rare-species ensemble "
        "means need ~100-150 seeds to settle above the 0.99 bar)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=180, help="Per-model timeout (s)")
    ap.add_argument("--limit", type=int, default=0, help="Max .bngl files (0=all)")
    ap.add_argument("--include", default="", help="Substring path filter")
    ap.add_argument("--exclude", default="", help="Substring path filter (drop)")
    ap.add_argument(
        "--manifest",
        default="",
        help="parity-corpus manifest.json forwarded to the sweeps "
        "for per-model overrides (relpath-keyed).",
    )
    ap.add_argument(
        "--models",
        default="",
        help="Comma-separated model basenames to restrict the base "
        "sweep to (a selected smaller suite).",
    )
    ap.add_argument(
        "--no-escalate",
        action="store_true",
        help="Run only the base sweep + diff; skip escalation.",
    )
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_sub = out / "subprocess"
    base_bng = out / "bngsim"
    esc_sub = out / "escalated" / "subprocess"
    esc_bng = out / "escalated" / "bngsim"
    base_report_md = out / "parity_report_base.md"
    base_report_json = out / "parity_report_base.json"
    final_report_md = out / "parity_report.md"
    final_report_json = out / "parity_report.json"

    started = time.time()

    # 1. Base sweeps.
    base_models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    common = dict(
        root=args.root,
        n_seeds=str(args.n_seeds),
        workers=str(args.workers),
        timeout=str(args.timeout),
        limit=str(args.limit) if args.limit else "",
        include=args.include,
        exclude=args.exclude,
        manifest=args.manifest,
        models=base_models,
    )
    print("=== [1/4] base sweep — subprocess ===")
    sweep("subprocess", out=str(base_sub), **common)
    print("=== [1/4] base sweep — bngsim ===")
    sweep("bngsim", out=str(base_bng), **common)

    # 2. Base diff.
    print("=== [2/4] base diff ===")
    diff(str(base_sub), str(base_bng), str(base_report_md), str(base_report_json))
    base = json.loads(base_report_json.read_text())

    # 3. Decide escalation set.
    to_escalate = sorted(
        bngl for bngl, info in base["per_model"].items() if is_escalatable_stochastic_diff(info)
    )
    structural = sorted(
        Path(bngl).name
        for bngl, info in base["per_model"].items()
        if info.get("bucket") == "DIFF"
        and info.get("regime") == "stochastic"
        and not is_escalatable_stochastic_diff(info)
    )

    print(f"\n=== base buckets: { {b: len(v) for b, v in base['buckets'].items()} } ===")
    print(
        f"stochastic DIFFs to escalate "
        f"({args.n_seeds}->{args.escalate_seeds} seeds): "
        f"{[Path(m).name for m in to_escalate] or 'none'}"
    )
    if structural:
        print(f"stochastic DIFFs left as-is (structural, seed-independent): {structural}")

    overlays = ()
    if args.no_escalate or not to_escalate:
        if args.no_escalate:
            print("=== [3/4] escalation skipped (--no-escalate) ===")
        else:
            print("=== [3/4] no escalatable stochastic DIFFs — skipping ===")
        # Final report == base report; re-diff so the file names are stable.
    else:
        print(
            f"=== [3/4] escalation re-run "
            f"({len(to_escalate)} models @ {args.escalate_seeds} seeds) ==="
        )
        models = [Path(m).name for m in to_escalate]
        esc_common = dict(
            root=args.root,
            n_seeds=str(args.escalate_seeds),
            workers=str(args.workers),
            timeout=str(args.timeout),
            limit="",
            include="",
            exclude="",
            manifest=args.manifest,
        )
        sweep("subprocess", out=str(esc_sub), models=models, **esc_common)
        sweep("bngsim", out=str(esc_bng), models=models, **esc_common)
        overlays = ((str(esc_sub), str(esc_bng)),)

    # 4. Final diff (with overlay if escalation ran).
    print("=== [4/4] final diff ===")
    diff(
        str(base_sub),
        str(base_bng),
        str(final_report_md),
        str(final_report_json),
        overlays=overlays,
    )
    final = json.loads(final_report_json.read_text())

    # Summary.
    elapsed = time.time() - started
    base_counts = {b: len(v) for b, v in base["buckets"].items()}
    final_counts = {b: len(v) for b, v in final["buckets"].items()}
    flipped = sorted(set(base["buckets"]["DIFF"]) - set(final["buckets"]["DIFF"]))
    print("\n" + "=" * 64)
    print(f"adaptive parity sweep done in {elapsed / 60:.1f}m")
    print(f"  base : {base_counts}")
    print(f"  final: {final_counts}")
    if overlays:
        print(
            f"  escalated {len(to_escalate)} stochastic DIFF(s) to "
            f"{args.escalate_seeds} seeds; "
            f"{len(flipped)} flipped DIFF->PASS (small-sample noise):"
        )
        for m in flipped:
            print(f"    - {Path(m).name}")
        still = sorted(set(final["buckets"]["DIFF"]) & set(to_escalate))
        if still:
            print(f"  {len(still)} escalated model(s) still DIFF (real divergence):")
            for m in still:
                print(f"    - {Path(m).name}")
    print(f"  report: {final_report_md}")
    print("=" * 64)


if __name__ == "__main__":
    main()
