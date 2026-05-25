#!/usr/bin/env python3
"""One command: run the vendored parity corpus end-to-end and assert verdicts.

This is the single entrypoint that makes the BNGsim-vs-subprocess parity
result reproducible. It drives the whole pipeline against the tracked corpus
under ``tests/parity/`` and checks the outcome against the manifest:

    1. SELECT a subset of the corpus  (--tier / --models / --all)
    2. SWEEP + ESCALATE + DIFF        (delegates to parity_run.py, which runs
       parity_sweep.py on each simulator and parity_diff.py to bucket models;
       stochastic DIFFs are re-judged at a higher seed count)
    3. ASSERT each model's bucket equals the manifest's expected bucket, and
       that the DIFF / ERROR buckets are empty within the selection.

The manifest (tests/parity/manifest.json, built by build_parity_corpus.py) is
the source of truth: per-model overrides are read from it (forwarded to the
sweeps) and the expected verdict bucket is asserted against it.

Examples
--------
    # Full suite (all tiers) with bngsim pinned, fail on any new DIFF:
    python scripts/parity_validate.py --all

    # Fast subset only (CI-friendly):
    python scripts/parity_validate.py --tier fast

    # A hand-picked smaller suite:
    python scripts/parity_validate.py --models egfr_net,fceri_ji

    # Re-establish the expected verdicts after an intended change (bless):
    python scripts/parity_validate.py --all --update-baseline

Run from the venv that has the pinned bngsim wheel installed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
RUN = SCRIPTS / "parity_run.py"
DEFAULT_MANIFEST = REPO / "tests" / "parity" / "manifest.json"
DEFAULT_MODELS_ROOT = REPO / "tests" / "parity" / "models"

# bngsim version this corpus's expected verdicts were blessed against. A
# different version may shift trailing digits or bucket membership; we warn
# (or fail with --strict-version) so a verdict change is never silently a
# version artifact. Keep in sync with the pinned wheel / README.
PINNED_BNGSIM = "0.9.7"

TIERS = ("fast", "slow", "glacial", "original")


def bngsim_version():
    try:
        import bngsim

        return getattr(bngsim, "__version__", None)
    except Exception:
        return None


def load_manifest(path):
    return json.loads(Path(path).read_text())


def select(manifest, tier, models):
    """Return (selected_records, include_filter, models_arg).

    ``include_filter`` is the path substring handed to the sweep (tier dir);
    ``models_arg`` restricts the base sweep to specific basenames.
    """
    recs = manifest["models"]
    if models:
        wanted = {m if m.endswith(".bngl") else m + ".bngl" for m in models}
        sel = [r for r in recs if r["basename"] in wanted]
        return sel, "", sorted({r["basename"] for r in sel})
    if tier and tier != "all":
        sel = [r for r in recs if r["tier"] == tier]
        return sel, f"/{tier}/", None
    return list(recs), "", None


def report_bucket_by_id(report, models_root):
    """Map manifest id (relpath under models_root) -> actual bucket."""
    root = Path(models_root).resolve()
    out = {}
    for key, info in report["per_model"].items():
        try:
            rid = Path(key).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        out[rid] = info.get("bucket", "?")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tier", choices=TIERS + ("all",), help="run one tier")
    g.add_argument("--all", action="store_true", help="run every tier (default)")
    g.add_argument("--models", default="", help="comma-separated basenames")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT))
    ap.add_argument(
        "--out",
        default=str(REPO / "dev" / "parity_validate_out"),
        help="work dir for sweep/diff artifacts + reports",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument(
        "--escalate-seeds",
        type=int,
        default=150,
        help="seeds to re-judge stochastic DIFFs at (default 150). "
        "50 leaves a few slow-tier oscillator / rare-species "
        "NF means just under the 0.99 ensemble bar — verified "
        "small-sample noise (they reach >=0.99 by 150), not a "
        "divergence; see tests/parity/README.md.",
    )
    ap.add_argument("--no-escalate", action="store_true")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="write the observed buckets back into the manifest "
        "'expected' fields (bless an intended change)",
    )
    ap.add_argument(
        "--strict-version",
        action="store_true",
        help="hard-fail if the installed bngsim != the pinned version (default: warn)",
    )
    args = ap.parse_args()

    # --- bngsim version gate -------------------------------------------------
    ver = bngsim_version()
    if ver != PINNED_BNGSIM:
        msg = (
            f"bngsim version is {ver!r}, expected pinned {PINNED_BNGSIM!r}. "
            f"Verdicts were blessed against {PINNED_BNGSIM}."
        )
        if ver is None:
            msg = (
                "bngsim is not importable in this interpreter "
                f"({sys.executable}). Run from the venv with the pinned wheel."
            )
        if args.strict_version:
            sys.exit(f"ERROR: {msg}")
        print(f"WARNING: {msg}")

    manifest = load_manifest(args.manifest)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tier = args.tier or ("all" if (args.all or not models) else None)
    sel, include, models_arg = select(manifest, tier, models)
    if not sel:
        sys.exit(f"no models selected (tier={tier!r}, models={models!r})")

    label = f"models={models}" if models else f"tier={tier}"
    print(f"=== parity_validate: {label} -> {len(sel)} models (bngsim {ver}) ===")

    # --- run the pipeline ----------------------------------------------------
    cmd = [
        sys.executable,
        str(RUN),
        "--root",
        args.models_root,
        "--manifest",
        args.manifest,
        "--out",
        args.out,
        "--workers",
        str(args.workers),
        "--timeout",
        str(args.timeout),
        "--n-seeds",
        str(args.n_seeds),
        "--escalate-seeds",
        str(args.escalate_seeds),
    ]
    if include:
        cmd += ["--include", include]
    if models_arg:
        cmd += ["--models", ",".join(models_arg)]
    if args.no_escalate:
        cmd += ["--no-escalate"]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit("pipeline (parity_run.py) failed")

    # --- assert verdicts against the manifest --------------------------------
    report = json.loads((Path(args.out) / "parity_report.json").read_text())
    actual = report_bucket_by_id(report, args.models_root)

    rows = []  # (id, expected, got)
    missing = []  # selected but absent from report
    for r in sel:
        rid = r["id"]
        got = actual.get(rid)
        if got is None:
            missing.append(rid)
            continue
        rows.append((rid, r["expected"], got))

    mismatches = [(rid, exp, got) for rid, exp, got in rows if exp != got]
    diffs = [rid for rid, _, got in rows if got == "DIFF"]
    errors = [rid for rid, _, got in rows if got == "ERROR"]

    if args.update_baseline:
        got_by_id = {rid: got for rid, _, got in rows}
        changed = 0
        for rec in manifest["models"]:
            if rec["id"] in got_by_id and rec["expected"] != got_by_id[rec["id"]]:
                rec["expected"] = got_by_id[rec["id"]]
                changed += 1
        Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"\n=== --update-baseline: rewrote {changed} expected bucket(s) in {args.manifest} ==="
        )
        _print_bucket_summary(rows, missing)
        return

    _print_bucket_summary(rows, missing)
    ok = not (mismatches or missing or diffs or errors)
    if missing:
        print(
            f"\nMISSING from report ({len(missing)}): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )
    if mismatches:
        print(f"\nVERDICT MISMATCHES ({len(mismatches)}):")
        for rid, exp, got in mismatches:
            print(f"  {rid}\n      expected {exp}, got {got}")
    if ok:
        print(f"\nOK — {len(rows)} models all match expected buckets; DIFF and ERROR empty.")
        sys.exit(0)
    sys.exit(
        f"\nFAIL — {len(mismatches)} mismatch, {len(missing)} missing, "
        f"{len(diffs)} DIFF, {len(errors)} ERROR. "
        f"If this change is intended, re-bless with --update-baseline."
    )


def _print_bucket_summary(rows, missing):
    counts = {}
    for _, _, got in rows:
        counts[got] = counts.get(got, 0) + 1
    print(f"\nobserved buckets: {counts}" + (f"  (+{len(missing)} missing)" if missing else ""))


if __name__ == "__main__":
    main()
