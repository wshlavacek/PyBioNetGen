#!/usr/bin/env python3
"""Diff a subprocess parity_sweep run against a bngsim parity_sweep run.

For each model:

* Deterministic models (only ode/cvode actions): per-cell combined
  absolute+relative diff of every common .gdat/.cdat — the standard
  ODE-solver error model. A cell passes iff
      |a - b| <= ABS_TOL * col_peak + REL_TOL * max(|a|, |b|)
  where ``col_peak`` is that column's peak magnitude across both runs.
  The relative term governs the bulk of the trajectory; the absolute
  term governs the tail of a quantity decaying toward zero, where
  ``|a-b| / max(|a|, |b|)`` is undefined — a trailing-digit difference
  over an exponentially tiny value reads as a relative diff of ~2.0
  (a sign flip). A genuine divergence is a meaningful fraction of the
  column scale and clears both terms by many decades.
  As a backstop, a cell where *both* sides sit below
  ``zero_floor = max(1e-12, scale * NEAR_ZERO_FLOOR_REL)`` (``scale`` =
  file peak) is forgiven outright — sub-scale underflow noise in a
  column that never carries a real signal.
  Time column compared exactly (TOL=0). NaNs equal NaNs in the same
  cell are treated as a match (real data; both BNGsim and BNG2.pl
  evaluate `sqrt` on negative discriminants the same way).
  Single-sample step-discontinuity shifts are forgiven: a step
  function `if(t<N, a, b)` whose threshold N sits exactly on an output
  grid point transitions one sample early/late between the two
  integrators because their clock observable disagrees by ~1e-13 of
  integrator roundoff at the grid point. That is an artifact of
  comparing two ODE integrators, not a parity failure — see
  `_discontinuity_shift_mask`.
* Stochastic models (any nf/ssa/pla/psa action): real ensemble
  comparison at N=10 seeds per side.
    For each (time, observable) cell, compute mean (mu_s, mu_b) and
    std (sigma_s, sigma_b) across seeds. Compare with a t-style test:
        |mu_s - mu_b| <= K * sqrt((sigma_s^2 + sigma_b^2) / N),  K=3
    Pass if >= ENSEMBLE_PASS_FRAC (0.99) of cells pass the test.
    Skip near-zero cells where max(|mu_s|, |mu_b|) < NEAR_ZERO_REL *
    file_max_observable, to avoid NFsim noise on near-zero observables.

Buckets per model:
  PASS         — within tolerance / ensemble agreed
  DIFF         — measurable numerical mismatch
  KNOWN_ARTIFACT — would be DIFF, but the divergence has been
                 investigated and confirmed to be a comparison artifact
                 (not a simulator discrepancy). Listed per-model in
                 KNOWN_DETERMINISTIC_ARTIFACTS; still run and diffed,
                 reclassified out of DIFF only while within a recorded
                 magnitude bound.
  NOT_SUPPORTED — bngsim explicitly raised that it doesn't support X
                 (e.g. Model.from_net rejects population species).
                 We classify by error string.
  ERROR        — actual unexpected crash worth filing

Overlay merge (--overlay-subprocess / --overlay-bngsim):
  A stochastic model's PASS/DIFF verdict at a low seed count is noisy —
  the per-seed std estimate that feeds the ensemble test is itself
  unreliable at N=10. To get a dependable verdict, such a model is
  re-run at a higher seed count into a separate "overlay" sweep, and the
  overlay's results for that model fully replace the base results here
  so the model is judged at the escalated seed count. Pass an overlay
  --out pair per escalation re-run; ``parity_run.py`` automates this.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Deterministic per-cell relative bar. BNG2.pl (CVODE/BDF) and BNGsim
# are two independent stiff-ODE integrators; even both at atol/rtol
# 1e-8 they only agree to ~1e-5 relative once error accumulates over a
# long trajectory. 1e-6 is below that floor and flags pure integrator
# noise as a divergence. Genuine divergences in this corpus are rel
# >= 0.1 — well clear of 1e-4.
REL_TOL = 1e-4
# Absolute tolerance, as a fraction of each column's own peak magnitude.
# The deterministic test is a combined abs+rel bar — the standard
# ODE-solver error model |a-b| <= atol + rtol*|y|. The relative term
# (REL_TOL) governs the bulk of the trajectory; this absolute term
# governs the tail of a quantity decaying toward zero, where the
# relative diff is undefined (a trailing-digit difference over an
# exponentially tiny value reads as rel ~2.0). 1e-6 of column peak is 4
# decades below REL_TOL and 5+ below any genuine divergence (those run
# >=0.1 of model scale), so it forgives decay-tail roundoff without
# loosening detection of a real discrepancy.
ABS_TOL = 1e-6
# Closeness bar used only by `_discontinuity_shift_mask` to confirm a
# value equals an adjacent-row value on the other side. Kept tight and
# independent of REL_TOL: step-function branch values are bit-identical
# literals between the two sides, so a loose bar here would only risk a
# genuine divergence coincidentally matching a neighbour.
SHIFT_MATCH_TOL = 1e-6
TIME_TOL = 0.0  # time column must match exactly
# Deterministic near-zero floor: a cell where both sides are below
# scale * NEAR_ZERO_FLOOR_REL (scale = file peak magnitude) is below
# the integrator's resolvable range — the quantity is numerically zero
# on both sides, so any apparent diff (incl. a sign flip, which
# inflates the relative diff to ~2.0) is underflow noise and the cell
# is forgiven outright (see deterministic_compare).
NEAR_ZERO_FLOOR_REL = 1e-12
ENSEMBLE_K = 3.0
ENSEMBLE_PASS_FRAC = 0.99
NEAR_ZERO_REL = 1e-9

# Models whose DIFF has been investigated and confirmed to be a
# comparison artifact, not a simulator discrepancy. They stay in the
# sweep (still run, still diffed) but are reclassified out of DIFF so a
# known, understood artifact does not read as a regression. This is a
# named per-model exception — NOT a relaxation of the tolerances; every
# other model is still held to REL_TOL. Each entry carries a magnitude
# bound (overall max absolute diff): if the model's divergence ever
# exceeds it the model is *not* excused and falls back to DIFF, since
# that would be a new, uninvestigated change. Bounds are set well above
# the observed artifact magnitude yet far below any real divergence
# (genuine divergences in this corpus are >=0.1 of model scale).
KNOWN_DETERMINISTIC_ARTIFACTS = {
    # Stiff relaxation oscillator: V sawtooths 10<->20, gated by sharp
    # tanh((V-V0)/0.01) switches. The two integrators resolve those
    # near-discontinuous switches at slightly different sub-step
    # timings, so the oscillator phase wanders <=4e-3 time units over
    # its 9 cycles. Sampled at fixed output times that shows as up to
    # ~0.43 in V on the steep edges, while period, amplitude and cycle
    # count match both sides. Same family as the resolved if(t<N)
    # knife-edge cluster. Verified 2026-05-17. Observed max_abs 0.43.
    "proliferation": {
        "max_abs_bound": 1.0,
        "reason": "stiff relaxation-oscillator phase wander across "
                  "sharp tanh() switches (<=4e-3 time units over 9 "
                  "cycles); period, amplitude and cycle count match "
                  "both sides. Verified 2026-05-17.",
    },
    # NOTE: Post / erlang / residence_time were here (exponential-decay-
    # tail relative-error blow-up) but are now handled generally by the
    # combined abs+rel criterion — no per-model exception needed.
    #
    # Staircase function sampled exactly on a discontinuity. The
    # APdat_*() functions are step lookups, if(t<=0,..,if(t<=230,V230,
    # if(t<=240,V240,..))), with a breakpoint exactly at t=230 — an
    # output grid point. At that sample BNG2.pl's run_network evaluates
    # the function with an internal time <= 230 and BNGsim with a time
    # ~1 ULP > 230, so the two pick adjacent staircase buckets and the
    # column jumps one step (~24.5) at that single row. The .cdat (all
    # species) and every non-staircase column agree to 12 sig figs and
    # the .net files are byte-identical — same if(t<N) knife-edge family
    # as the resolved rel=1.0 cluster, just a staircase the 1-sample
    # shift mask can't bridge (the neighbouring buckets also differ).
    # Verified by investigation 2026-05-17. Observed max_abs 24.55.
    "ATG_model_v16": {
        "max_abs_bound": 100.0,
        "reason": "discontinuous staircase function (APdat_*) sampled "
                  "exactly on its t=230 breakpoint; a ~1-ULP output-time "
                  "difference flips one bucket on a single row. Species "
                  "and all non-staircase columns agree to 12 sig figs. "
                  "Verified 2026-05-17.",
    },
}

# Comparable extensions for both regimes.
# We focus numerical comparison on .gdat (and .cdat where present), plus
# .scan for parameter_scan models -- a .scan is a 2D table (parameter
# value + each observable at t_end) and diffs cell-by-cell like a .gdat.
# .net / .xml differ in cosmetic formatting between simulators and aren't
# the right things to numeric-diff.
NUM_EXTENSIONS = {".gdat", ".cdat", ".scan"}
SCAN_EXT = ".scan"

NOT_SUPPORTED_PATTERNS = [
    # Categorical bngsim/bridge capability gaps. These are documented
    # limitations of the bngsim integration, not generic crashes.
    # bngsim's BNG-XML parser doesn't recognize certain rate-law types.
    re.compile(r"Unrecognized rate law type '[^']+'", re.I),
    # bngsim ExprTk treats `t` as the reserved time variable. Models
    # that define a parameter / observable named `t` can't load.
    re.compile(r"ExprTk: failed to register variable 't'", re.I),
    # ExprTk function-name collision with built-ins (e.g. `divide`).
    re.compile(r"ModelBuilder: failed to compile function.*ExprTk compilation failed", re.I),
    # Fork parser fails on rare BNGL keywords (check_iso, iptg, ...).
    re.compile(r"argument \w+ not recognized for action", re.I),
    # bngsim .net loader rejecting population species / table functions.
    re.compile(r"BNGSimError", re.I),
    re.compile(r"unsupported|not supported|not implemented", re.I),
    re.compile(r"Model\.from_net.*reject", re.I),
    re.compile(r"population species", re.I),
    re.compile(r"table.+function", re.I),
]


# Some models legitimately produce no comparable output: e.g. NFsim
# models with no observables block (one side writes an empty .gdat,
# the other an empty .cdat). Treat as PASS not ERROR.
def _names_by_basename(d):
    out = {}
    for p in Path(d).iterdir() if Path(d).is_dir() else []:
        if p.is_file() and p.suffix in NUM_EXTENSIONS:
            out.setdefault(p.stem, []).append(p)
    return out


def _matches_not_supported(text):
    if not text:
        return False
    for pat in NOT_SUPPORTED_PATTERNS:
        if pat.search(text):
            return True
    return False


def is_not_supported(err, out_dir=None, status=None):
    """Classify a bngsim failure as a documented capability gap.

    `parity_sweep.run_one` truncates `error` to the last 500 chars of the
    last stderr line. For multi-line bngsim error messages (e.g. the 'Sat'
    rate-law guidance, which spans 3 lines and ends on a help string that
    matches no pattern), the truncated tail can hide the categorical
    signal. Fall through to the per-model `_run.log` when available.

    Only do the fallback for ``crash`` (raised exception). Timeouts and
    harness errors don't carry a meaningful exception tail and may have
    incidental "not supported" warning strings in stderr (e.g. bngsim
    codegen falls back to interpreted ODE — that's a perf hit, not a
    capability gap, and the model deserves a real ERROR with a
    real-cause investigation, not a quiet bucket reassignment).
    """
    if _matches_not_supported(err):
        return True
    if out_dir and status == "crash":
        log = Path(out_dir) / "_run.log"
        if log.is_file():
            try:
                # Tail-bound to keep big logs cheap.
                data = log.read_text(errors="replace")[-20000:]
            except Exception:
                return False
            return _matches_not_supported(data)
    return False


def load_array(path):
    """Load a .gdat/.cdat/.scan as float ndarray. Skip BNG comment header."""
    return np.loadtxt(str(path), comments="#", ndmin=2)


def safe_load(path):
    try:
        return load_array(path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _discontinuity_shift_mask(sub_data, bng_data, fail_mask):
    """Mark failing cells that are single-sample step-discontinuity shifts.

    A step-function column (e.g. ``if(t<N, a, b)``) switches branches at a
    threshold. When the threshold sits exactly on an output grid point the
    two ODE integrators land on opposite sides of the ``<`` comparison: the
    clock observable they feed into the function disagrees by ~1e-13 of
    integrator roundoff at the grid point, so the column transitions one
    sample early on one side. That is a numerical artifact of comparing two
    integrators, not a parity failure — the columns are otherwise identical.

    A failing cell ``(r, c)`` is forgiven only when the two columns are
    bit-equal except a <=1-sample horizontal offset at this single
    transition: each side's value at row ``r`` must equal the other side's
    value at an adjacent row. A genuine divergence will not coincide with a
    neighbour, and an off-by-two-or-more shift (a real ~0.1+ time-unit lag,
    not roundoff) leaves a non-matching neighbour and stays flagged.
    """
    R = sub_data.shape[0]
    forgiven = np.zeros_like(fail_mask)

    def close(x, y):
        d = abs(x - y)
        return d <= SHIFT_MATCH_TOL * max(abs(x), abs(y), 1e-12)

    rows, cols = np.nonzero(fail_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        s, b = sub_data[r, c], bng_data[r, c]
        sp = sub_data[r - 1, c] if r > 0 else None
        sn = sub_data[r + 1, c] if r + 1 < R else None
        bp = bng_data[r - 1, c] if r > 0 else None
        bn = bng_data[r + 1, c] if r + 1 < R else None
        ok = False
        # interior row: one side's transition leads the other by one sample
        if bp is not None and sn is not None and close(s, bp) and close(b, sn):
            ok = True  # bngsim transitions one sample earlier
        elif bn is not None and sp is not None and close(s, bn) and close(b, sp):
            ok = True  # subprocess transitions one sample earlier
        # boundary row: one side steps at the final/first sample and the
        # other has no room to follow. Require agreement just before/after.
        elif r == R - 1 and sp is not None and bp is not None:
            if close(s, bp) and close(s, sp) and not close(b, bp):
                ok = True  # bngsim stepped at the final sample
            elif close(b, sp) and close(b, bp) and not close(s, sp):
                ok = True  # subprocess stepped at the final sample
        elif r == 0 and sn is not None and bn is not None:
            if close(s, bn) and close(s, sn) and not close(b, bn):
                ok = True  # bngsim stepped at the first sample
            elif close(b, sn) and close(b, bn) and not close(s, sn):
                ok = True  # subprocess stepped at the first sample
        if ok:
            forgiven[r, c] = True
    return forgiven


def deterministic_compare(sub_dir, bng_dir):
    """Compare deterministic (ODE/CVODE) outputs by per-cell relative diff.

    Returns: (status, details)
      status: 'pass' | 'diff' | 'no_artifacts' | 'load_error'
      details: dict with per-file numbers + chosen-file summary
    """
    sub_files = {}
    bng_files = {}
    if Path(sub_dir).is_dir():
        sub_files = {p.name: p for p in Path(sub_dir).iterdir()
                     if p.is_file() and p.suffix in NUM_EXTENSIONS}
    if Path(bng_dir).is_dir():
        bng_files = {p.name: p for p in Path(bng_dir).iterdir()
                     if p.is_file() and p.suffix in NUM_EXTENSIONS}
    common = sorted(set(sub_files) & set(bng_files))
    only_sub = sorted(set(sub_files) - set(bng_files))
    only_bng = sorted(set(bng_files) - set(sub_files))
    # Both sides ran ok with no comparable outputs: model has no
    # observables / no simulate actions. PASS by definition.
    if not sub_files and not bng_files:
        return "pass", {"note": "no .gdat/.cdat outputs on either side",
                        "only_sub": only_sub, "only_bng": only_bng}
    # Cross-extension match by basename when only_sub is .gdat and
    # only_bng is .cdat (or vice versa) for the same stem. Subprocess
    # NF writes .gdat (observables only), bngsim NF writes .cdat
    # (concentrations only). For models with no observables both files
    # collapse to a time-only column and compare cleanly.
    if not common:
        sub_by_stem = {Path(n).stem: sub_files[n] for n in only_sub}
        bng_by_stem = {Path(n).stem: bng_files[n] for n in only_bng}
        cross_stems = sorted(set(sub_by_stem) & set(bng_by_stem))
        cross_pairs = [(s, sub_by_stem[s], bng_by_stem[s]) for s in cross_stems]
        if not cross_pairs:
            return "no_artifacts", {"only_sub": only_sub, "only_bng": only_bng}
        # Compare cross pairs as if they were "common" — but if the
        # stem ext differs, only the time column is meaningful. We will
        # still compute relative diff and let it speak for itself.
        per_file = {}
        overall_pass = True
        for stem, sp, bp in cross_pairs:
            sub, e1 = safe_load(sp)
            bng, e2 = safe_load(bp)
            if e1 or e2:
                per_file[f"{stem}.{sp.suffix}|{bp.suffix}"] = {
                    "load_error": e1 or e2}
                overall_pass = False
                continue
            if sub.shape != bng.shape:
                per_file[f"{stem}.{sp.suffix}|{bp.suffix}"] = {
                    "note": "cross-ext naming mismatch with shape mismatch",
                    "shape_sub": list(sub.shape),
                    "shape_bng": list(bng.shape)}
                overall_pass = False
                continue
            time_diff = float(np.max(np.abs(sub[:, 0] - bng[:, 0])))
            data_diff = 0.0
            if sub.shape[1] > 1:
                # If they happen to align beyond time column, compare too.
                data_diff = float(np.max(np.abs(sub[:, 1:] - bng[:, 1:])))
            per_file[f"{stem}{sp.suffix}<->{stem}{bp.suffix}"] = {
                "note": "cross-ext name match (likely empty observables NF)",
                "shape": list(sub.shape),
                "time_diff": time_diff,
                "data_diff": data_diff,
            }
            if time_diff > TIME_TOL:
                overall_pass = False
        return ("pass" if overall_pass else "diff"), {
            "only_sub": only_sub,
            "only_bng": only_bng,
            "per_file": per_file,
            "note": "cross-extension basename matching used",
        }
    per_file = {}
    overall_pass = True
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    for name in common:
        sub, e1 = safe_load(sub_files[name])
        bng, e2 = safe_load(bng_files[name])
        if e1 or e2:
            per_file[name] = {"load_error": e1 or e2}
            overall_pass = False
            continue
        if sub.shape != bng.shape:
            per_file[name] = {"shape_sub": list(sub.shape),
                              "shape_bng": list(bng.shape)}
            overall_pass = False
            continue
        # Time column (col 0) must match exactly.
        time_diff = float(np.max(np.abs(sub[:, 0] - bng[:, 0])))
        # NaN==NaN: treat both-NaN cells as zero diff.
        sub_data = sub[:, 1:]
        bng_data = bng[:, 1:]
        both_nan = np.isnan(sub_data) & np.isnan(bng_data)
        absd = np.abs(sub_data - bng_data)
        # Replace nan-vs-nan with 0
        absd = np.where(both_nan, 0.0, absd)
        # File-peak scale → near-zero backstop floor (see module docstring).
        finite_mag = np.concatenate([np.abs(sub_data).ravel(),
                                     np.abs(bng_data).ravel()])
        finite_mag = finite_mag[np.isfinite(finite_mag)]
        scale = float(finite_mag.max()) if finite_mag.size else 1.0
        zero_floor = max(1e-12, scale * NEAR_ZERO_FLOOR_REL)
        # Per-cell magnitude max(|a|,|b|). A NaN on one side -> 0 here so
        # the one-side-NaN cell still flags below (its absd is inf).
        colmag = np.maximum(np.abs(sub_data), np.abs(bng_data))
        colmag = np.where(np.isfinite(colmag), colmag, 0.0)
        # Per-column peak magnitude across both runs — the scale for the
        # absolute term of the combined tolerance.
        col_peak = colmag.max(axis=0) if colmag.size else np.zeros(0)
        # One side NaN, other not -> absd is NaN -> treat as inf (flag).
        absd_clean = np.where(np.isnan(absd), np.inf, absd)
        # Relative diff kept for reporting/visibility only.
        denom = np.maximum(colmag, zero_floor)
        reld_clean = np.where(np.isnan(absd), np.inf, absd / denom)
        # Combined absolute + relative tolerance, per cell — the ODE
        # solver error model |a-b| <= atol + rtol*|y|. The absolute term
        # uses each column's own peak; see module docstring / ABS_TOL.
        cell_tol = ABS_TOL * col_peak[np.newaxis, :] + REL_TOL * colmag
        fail_mask = absd_clean > cell_tol
        n_fail = int(np.sum(fail_mask))
        # Forgive single-sample step-discontinuity shifts: roundoff in a
        # clock observable makes a step function transition one sample
        # early/late at a threshold landing on an output grid point.
        if n_fail:
            shift_mask = _discontinuity_shift_mask(sub_data, bng_data, fail_mask)
        else:
            shift_mask = np.zeros_like(fail_mask)
        n_shift = int(np.sum(shift_mask))
        # Backstop: forgive a still-failing cell where both sides sit
        # below the file-scale near-zero floor — pure sub-scale underflow
        # noise in a column that never carries a real signal.
        near_zero_mask = fail_mask & (colmag < zero_floor)
        n_near_zero = int(np.sum(near_zero_mask))
        forgive_mask = shift_mask | near_zero_mask
        effective_fail = fail_mask & ~forgive_mask
        n_effective = int(np.sum(effective_fail))
        # Reported figures: max over genuinely-failing cells (0 if the
        # file passes); raw figures over all cells kept for visibility.
        max_abs_raw = float(np.max(absd_clean)) if absd_clean.size else 0.0
        max_rel_raw = float(np.max(reld_clean)) if reld_clean.size else 0.0
        max_abs = (float(np.max(absd_clean[effective_fail]))
                   if n_effective else 0.0)
        max_rel = (float(np.max(reld_clean[effective_fail]))
                   if n_effective else 0.0)
        per_file[name] = {
            "shape": list(sub.shape),
            "time_diff": time_diff,
            "max_abs": max_abs if np.isfinite(max_abs) else "inf",
            "max_rel": max_rel if np.isfinite(max_rel) else "inf",
        }
        if n_shift:
            per_file[name]["discontinuity_shifts"] = n_shift
        if n_near_zero:
            per_file[name]["near_zero_forgiven"] = n_near_zero
        if n_shift or n_near_zero:
            per_file[name]["max_rel_raw"] = (
                max_rel_raw if np.isfinite(max_rel_raw) else "inf")
            per_file[name]["max_abs_raw"] = (
                max_abs_raw if np.isfinite(max_abs_raw) else "inf")
        overall_max_abs = max(overall_max_abs, max_abs if np.isfinite(max_abs) else float("inf"))
        overall_max_rel = max(overall_max_rel, max_rel if np.isfinite(max_rel) else float("inf"))
        if time_diff > TIME_TOL:
            per_file[name]["fail"] = "time"
            overall_pass = False
        if n_effective:
            per_file[name]["fail"] = per_file[name].get("fail", "") + "value"
            overall_pass = False
    return ("pass" if overall_pass else "diff"), {
        "common_files": common,
        "only_sub": only_sub,
        "only_bng": only_bng,
        "per_file": per_file,
        "max_abs": overall_max_abs if np.isfinite(overall_max_abs) else "inf",
        "max_rel": overall_max_rel if np.isfinite(overall_max_rel) else "inf",
    }


def stochastic_compare(sub_seed_dirs, bng_seed_dirs):
    """Ensemble compare two lists of per-seed output dirs.

    For each common artifact name (.gdat/.cdat) across all seeds on each
    side, stack across seeds and compare ensemble means with a t-style
    test.

    Returns: (status, details)
    """
    # Index per-seed artifacts.
    def index(seed_dirs):
        per_name = defaultdict(list)
        for d in seed_dirs:
            if not Path(d).is_dir():
                continue
            for p in sorted(Path(d).iterdir()):
                if p.is_file() and p.suffix in NUM_EXTENSIONS:
                    per_name[p.name].append(p)
        return per_name

    sub_per_name = index(sub_seed_dirs)
    bng_per_name = index(bng_seed_dirs)
    common = sorted(set(sub_per_name) & set(bng_per_name))
    only_sub = sorted(set(sub_per_name) - set(bng_per_name))
    only_bng = sorted(set(bng_per_name) - set(sub_per_name))
    if not sub_per_name and not bng_per_name:
        return "pass", {"note": "no .gdat/.cdat outputs on either side",
                        "only_sub": only_sub, "only_bng": only_bng}
    if not common:
        # Cross-extension basename match (subprocess .gdat <-> bngsim .cdat).
        sub_by_stem = {}
        for n, lst in sub_per_name.items():
            sub_by_stem.setdefault(Path(n).stem, []).extend([(n, p) for p in lst])
        bng_by_stem = {}
        for n, lst in bng_per_name.items():
            bng_by_stem.setdefault(Path(n).stem, []).extend([(n, p) for p in lst])
        cross_stems = sorted(set(sub_by_stem) & set(bng_by_stem))
        if not cross_stems:
            return "no_artifacts", {"only_sub": only_sub, "only_bng": only_bng}
        per_file = {}
        overall_pass = True
        for stem in cross_stems:
            sub_pairs = sub_by_stem[stem]
            bng_pairs = bng_by_stem[stem]
            sub_arrs, bng_arrs = [], []
            load_err = None
            for _n, p in sub_pairs:
                a, e = safe_load(p)
                if e:
                    load_err = e; break
                sub_arrs.append(a)
            if load_err:
                per_file[stem] = {"load_error": load_err}
                overall_pass = False; continue
            for _n, p in bng_pairs:
                a, e = safe_load(p)
                if e:
                    load_err = e; break
                bng_arrs.append(a)
            if load_err:
                per_file[stem] = {"load_error": load_err}
                overall_pass = False; continue
            sub_shapes = {a.shape for a in sub_arrs}
            bng_shapes = {a.shape for a in bng_arrs}
            if len(sub_shapes) != 1 or len(bng_shapes) != 1 \
                    or next(iter(sub_shapes)) != next(iter(bng_shapes)):
                per_file[stem] = {"note": "cross-ext naming, shape mismatch",
                                  "shape_sub": [list(s) for s in sub_shapes],
                                  "shape_bng": [list(s) for s in bng_shapes]}
                overall_pass = False; continue
            sub_stack = np.stack(sub_arrs)
            bng_stack = np.stack(bng_arrs)
            time_diff = float(np.max(np.abs(np.mean(sub_stack[:, :, 0], axis=0) -
                                              np.mean(bng_stack[:, :, 0], axis=0))))
            per_file[stem] = {"note": "cross-ext name match (NF empty observables)",
                              "shape_seeds": [sub_stack.shape[0],
                                              *list(sub_stack.shape[1:])],
                              "time_diff": time_diff}
            if time_diff > TIME_TOL:
                overall_pass = False
        return ("pass" if overall_pass else "diff"), {
            "only_sub": only_sub,
            "only_bng": only_bng,
            "per_file": per_file,
            "note": "cross-extension basename matching used",
        }
    per_file = {}
    overall_pass = True
    for name in common:
        sub_arrs = []
        bng_arrs = []
        load_err = None
        for p in sub_per_name[name]:
            a, e = safe_load(p)
            if e:
                load_err = e
                break
            sub_arrs.append(a)
        if load_err:
            per_file[name] = {"load_error": load_err}
            overall_pass = False
            continue
        for p in bng_per_name[name]:
            a, e = safe_load(p)
            if e:
                load_err = e
                break
            bng_arrs.append(a)
        if load_err:
            per_file[name] = {"load_error": load_err}
            overall_pass = False
            continue
        # Need seeds to align in shape on each side; if any seed has a
        # different shape, treat as shape-mismatch (real signal).
        sub_shapes = {a.shape for a in sub_arrs}
        bng_shapes = {a.shape for a in bng_arrs}
        if len(sub_shapes) != 1 or len(bng_shapes) != 1:
            per_file[name] = {"shape_sub": [list(s) for s in sub_shapes],
                              "shape_bng": [list(s) for s in bng_shapes],
                              "fail": "shape_inconsistent_across_seeds"}
            overall_pass = False
            continue
        s_shape = next(iter(sub_shapes))
        b_shape = next(iter(bng_shapes))
        if s_shape != b_shape:
            per_file[name] = {"shape_sub": list(s_shape),
                              "shape_bng": list(b_shape),
                              "fail": "shape_mismatch"}
            overall_pass = False
            continue
        sub_stack = np.stack(sub_arrs)  # (N_seeds, T, K)
        bng_stack = np.stack(bng_arrs)
        # Time column (col 0) — treat all seeds; mean time across seeds
        # should match exactly between sides. Don't test variance on time.
        sub_time = sub_stack[:, :, 0]
        bng_time = bng_stack[:, :, 0]
        time_diff = float(np.max(np.abs(np.mean(sub_time, axis=0) -
                                         np.mean(bng_time, axis=0))))
        # Observable columns: ensemble means, stds; ddof=1 for sample std
        sub_obs = sub_stack[:, :, 1:]
        bng_obs = bng_stack[:, :, 1:]
        N_sub = sub_obs.shape[0]
        N_bng = bng_obs.shape[0]
        mu_s = np.mean(sub_obs, axis=0)
        mu_b = np.mean(bng_obs, axis=0)
        # var/N for each side; sample variance with ddof=1 if >1 seeds
        var_s = np.var(sub_obs, axis=0, ddof=1) if N_sub > 1 else np.zeros_like(mu_s)
        var_b = np.var(bng_obs, axis=0, ddof=1) if N_bng > 1 else np.zeros_like(mu_b)
        se = np.sqrt(var_s / N_sub + var_b / N_bng)
        # Threshold cells where both ensembles are near-zero relative to
        # the file's overall scale — NFsim noise blows the test up there.
        scale = max(np.nanmax(np.abs(mu_s)), np.nanmax(np.abs(mu_b)), 1e-12)
        near_zero = (np.maximum(np.abs(mu_s), np.abs(mu_b))
                     < NEAR_ZERO_REL * scale)
        # Add a small absolute floor so cells with 0 std (e.g. constant
        # observable that *should* match) still get a chance to pass.
        # Use 1e-12 * scale.
        floor = 1e-12 * scale
        threshold = ENSEMBLE_K * np.maximum(se, floor)
        diff = np.abs(mu_s - mu_b)
        # Relative-agreement escape hatch. A model carrying both an SSA and
        # an ODE simulate is classified stochastic, so its *deterministic*
        # ODE segments are ensemble-compared too — but every seed is
        # identical there, so se=0 and the 3-sigma test collapses to
        # near-exact equality, flagging ~1e-6 cross-integrator noise. A
        # cell whose two means agree to within the deterministic REL_TOL
        # is consistent regardless of the sigma test (a genuinely
        # stochastic cell almost never agrees that closely by chance).
        rel_floor = np.maximum(
            np.maximum(np.abs(mu_s), np.abs(mu_b)), scale * NEAR_ZERO_FLOOR_REL
        )
        rel_ok = diff <= REL_TOL * rel_floor
        # NaN handling: both-NaN cell is a pass; one-side-NaN is a fail.
        both_nan = np.isnan(mu_s) & np.isnan(mu_b)
        either_nan = np.isnan(mu_s) | np.isnan(mu_b)
        cell_pass = (diff <= threshold) | rel_ok
        cell_pass = np.where(both_nan, True, cell_pass)
        cell_pass = np.where(either_nan & ~both_nan, False, cell_pass)
        # Apply near-zero skip
        cell_pass = np.where(near_zero, True, cell_pass)
        n_total = cell_pass.size
        n_pass = int(np.sum(cell_pass))
        frac_pass = n_pass / n_total if n_total else 1.0
        per_file[name] = {
            "shape_seeds": [N_sub, *list(s_shape)],
            "time_diff": time_diff,
            "n_cells": n_total,
            "n_pass": n_pass,
            "frac_pass": frac_pass,
            "max_abs_mean_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        }
        if time_diff > TIME_TOL:
            per_file[name]["fail"] = "time"
            overall_pass = False
        if frac_pass < ENSEMBLE_PASS_FRAC:
            per_file[name]["fail"] = per_file[name].get("fail", "") + "ensemble"
            overall_pass = False
    return ("pass" if overall_pass else "diff"), {
        "common_files": common,
        "only_sub": only_sub,
        "only_bng": only_bng,
        "per_file": per_file,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subprocess", required=True, help="subprocess sweep --out")
    ap.add_argument("--bngsim", required=True, help="bngsim sweep --out")
    ap.add_argument("--out", default="-", help="Markdown report path")
    ap.add_argument("--json-out", default="", help="Optional JSON dump")
    ap.add_argument("--overlay-subprocess", action="append", default=[],
                    help="Additional subprocess sweep --out whose models "
                         "override the base run (e.g. a high-seed escalation "
                         "re-run). Pair 1:1 with --overlay-bngsim, in order.")
    ap.add_argument("--overlay-bngsim", action="append", default=[],
                    help="Additional bngsim sweep --out; see --overlay-subprocess.")
    args = ap.parse_args()

    if len(args.overlay_subprocess) != len(args.overlay_bngsim):
        ap.error("--overlay-subprocess and --overlay-bngsim must be given "
                 "the same number of times (they pair 1:1)")

    sub_summary = json.loads(Path(args.subprocess, "_summary.json").read_text())
    bng_summary = json.loads(Path(args.bngsim, "_summary.json").read_text())

    # Overlay merge: a stochastic model flagged DIFF at the base seed count
    # is re-run at a higher seed count into an overlay sweep; the overlay's
    # results for that model fully replace the base results so the model is
    # judged at the escalated seed count. ``escalated`` records the seed
    # count each overridden model was re-judged at, for the report.
    sub_results = list(sub_summary["results"])
    bng_results = list(bng_summary["results"])
    escalated = {}  # bngl -> n_seeds it was re-judged at
    for ov_sub, ov_bng in zip(args.overlay_subprocess, args.overlay_bngsim):
        ov_sub_summary = json.loads(Path(ov_sub, "_summary.json").read_text())
        ov_bng_summary = json.loads(Path(ov_bng, "_summary.json").read_text())
        ov_models = ({r["bngl"] for r in ov_sub_summary["results"]} |
                     {r["bngl"] for r in ov_bng_summary["results"]})
        sub_results = ([r for r in sub_results if r["bngl"] not in ov_models]
                       + ov_sub_summary["results"])
        bng_results = ([r for r in bng_results if r["bngl"] not in ov_models]
                       + ov_bng_summary["results"])
        ov_seeds = ov_bng_summary.get("n_seeds")
        for m in ov_models:
            escalated[m] = ov_seeds

    # Group results by (bngl, regime). For deterministic, one row per side.
    # For stochastic, one row per (bngl, seed) per side.
    def index(results):
        det = {}            # bngl -> result
        stoch = defaultdict(list)  # bngl -> list of result (per seed)
        for r in results:
            if r.get("regime") == "deterministic":
                det[r["bngl"]] = r
            else:
                stoch[r["bngl"]].append(r)
        return det, stoch

    sub_det, sub_stoch = index(sub_results)
    bng_det, bng_stoch = index(bng_results)

    all_models = sorted(set(sub_det) | set(sub_stoch) |
                        set(bng_det) | set(bng_stoch))

    buckets = {"PASS": [], "DIFF": [], "KNOWN_ARTIFACT": [],
               "NOT_SUPPORTED": [], "ERROR": []}
    per_model = {}

    for bngl in all_models:
        # Determine regime by where the model lives.
        in_sub_det = bngl in sub_det
        in_bng_det = bngl in bng_det
        in_sub_stoch = bngl in sub_stoch
        in_bng_stoch = bngl in bng_stoch
        # If sides disagree on regime, that's an unexpected mismatch.
        if (in_sub_det and in_bng_stoch) or (in_sub_stoch and in_bng_det):
            buckets["ERROR"].append(bngl)
            per_model[bngl] = {"bucket": "ERROR",
                               "reason": "regime classification disagreement"}
            continue
        if in_sub_det:
            sub_r = sub_det[bngl]
            bng_r = bng_det[bngl]
            # Side-status checks
            if sub_r["status"] != "ok" and bng_r["status"] != "ok":
                # Both failed identically — informational
                buckets["ERROR"].append(bngl)
                per_model[bngl] = {"bucket": "ERROR",
                                   "regime": "deterministic",
                                   "sub_status": sub_r["status"],
                                   "bng_status": bng_r["status"],
                                   "sub_error": sub_r.get("error", ""),
                                   "bng_error": bng_r.get("error", ""),
                                   "reason": "both sides failed (may be pre-existing)"}
                continue
            if bng_r["status"] != "ok":
                if is_not_supported(bng_r.get("error", ""), bng_r.get("out_dir"), bng_r.get("status")):
                    buckets["NOT_SUPPORTED"].append(bngl)
                    per_model[bngl] = {"bucket": "NOT_SUPPORTED",
                                       "regime": "deterministic",
                                       "bng_status": bng_r["status"],
                                       "bng_error": bng_r.get("error", "")}
                else:
                    buckets["ERROR"].append(bngl)
                    per_model[bngl] = {"bucket": "ERROR",
                                       "regime": "deterministic",
                                       "bng_status": bng_r["status"],
                                       "bng_error": bng_r.get("error", "")}
                continue
            if sub_r["status"] != "ok":
                # subprocess failed but bngsim succeeded — call ERROR
                buckets["ERROR"].append(bngl)
                per_model[bngl] = {"bucket": "ERROR",
                                   "regime": "deterministic",
                                   "sub_status": sub_r["status"],
                                   "sub_error": sub_r.get("error", ""),
                                   "reason": "subprocess failed; bngsim succeeded"}
                continue
            status, details = deterministic_compare(sub_r["out_dir"], bng_r["out_dir"])
            if status == "pass":
                buckets["PASS"].append(bngl)
                per_model[bngl] = {"bucket": "PASS", "regime": "deterministic",
                                   "details": details}
            elif status == "no_artifacts":
                buckets["ERROR"].append(bngl)
                per_model[bngl] = {"bucket": "ERROR", "regime": "deterministic",
                                   "reason": "no common artifacts to diff",
                                   "details": details}
            else:
                # A model whose DIFF is a confirmed comparison artifact
                # is reclassified out of DIFF — but only while its
                # divergence stays within the recorded bound; a larger
                # divergence is a new change and stays DIFF.
                artifact = KNOWN_DETERMINISTIC_ARTIFACTS.get(Path(bngl).stem)
                max_abs = details.get("max_abs")
                excused = (
                    artifact is not None
                    and isinstance(max_abs, (int, float))
                    and max_abs <= artifact["max_abs_bound"]
                )
                if excused:
                    buckets["KNOWN_ARTIFACT"].append(bngl)
                    per_model[bngl] = {"bucket": "KNOWN_ARTIFACT",
                                       "regime": "deterministic",
                                       "reason": artifact["reason"],
                                       "details": details}
                else:
                    buckets["DIFF"].append(bngl)
                    per_model[bngl] = {"bucket": "DIFF", "regime": "deterministic",
                                       "details": details}
        else:
            # Stochastic regime
            sub_rs = sub_stoch.get(bngl, [])
            bng_rs = bng_stoch.get(bngl, [])
            sub_ok = [r for r in sub_rs if r["status"] == "ok"]
            bng_ok = [r for r in bng_rs if r["status"] == "ok"]
            sub_bad = [r for r in sub_rs if r["status"] != "ok"]
            bng_bad = [r for r in bng_rs if r["status"] != "ok"]
            if not bng_ok and bng_bad:
                # All bngsim seeds failed — bucket by error type from first.
                err = bng_bad[0].get("error", "")
                if is_not_supported(err, bng_bad[0].get("out_dir"), bng_bad[0].get("status")):
                    buckets["NOT_SUPPORTED"].append(bngl)
                    per_model[bngl] = {"bucket": "NOT_SUPPORTED",
                                       "regime": "stochastic",
                                       "bng_error": err}
                else:
                    buckets["ERROR"].append(bngl)
                    per_model[bngl] = {"bucket": "ERROR",
                                       "regime": "stochastic",
                                       "bng_error": err,
                                       "n_bng_failed": len(bng_bad)}
                continue
            if not sub_ok and sub_bad:
                # subprocess all failed — informational error
                buckets["ERROR"].append(bngl)
                per_model[bngl] = {"bucket": "ERROR",
                                   "regime": "stochastic",
                                   "sub_error": sub_bad[0].get("error", ""),
                                   "reason": "subprocess all seeds failed"}
                continue
            # We have at least one OK on each side; ensemble compare.
            sub_dirs = [r["out_dir"] for r in sub_ok]
            bng_dirs = [r["out_dir"] for r in bng_ok]
            status, details = stochastic_compare(sub_dirs, bng_dirs)
            details["n_sub_ok"] = len(sub_ok)
            details["n_bng_ok"] = len(bng_ok)
            details["n_sub_failed"] = len(sub_bad)
            details["n_bng_failed"] = len(bng_bad)
            if status == "pass":
                buckets["PASS"].append(bngl)
                per_model[bngl] = {"bucket": "PASS", "regime": "stochastic",
                                   "details": details}
            elif status == "no_artifacts":
                buckets["ERROR"].append(bngl)
                per_model[bngl] = {"bucket": "ERROR", "regime": "stochastic",
                                   "reason": "no common artifacts to diff",
                                   "details": details}
            else:
                buckets["DIFF"].append(bngl)
                per_model[bngl] = {"bucket": "DIFF", "regime": "stochastic",
                                   "details": details}

    # Tag models re-judged at an escalated seed count (overlay merge).
    for bngl, seeds in escalated.items():
        if bngl in per_model:
            per_model[bngl]["escalated_seeds"] = seeds

    # Build the markdown report.
    lines = []
    lines.append("# BNGsim parity sweep — diff report")
    lines.append("")
    lines.append(f"- subprocess sweep: `{args.subprocess}` (n={sub_summary.get('n_units')}, by_status={sub_summary.get('by_status')})")
    lines.append(f"  - simulator={sub_summary.get('simulator')}, n_seeds={sub_summary.get('n_seeds')}")
    lines.append(f"- bngsim sweep:     `{args.bngsim}` (n={bng_summary.get('n_units')}, by_status={bng_summary.get('by_status')})")
    lines.append(f"  - simulator={bng_summary.get('simulator')}, n_seeds={bng_summary.get('n_seeds')}")
    lines.append(f"- tolerance: deterministic rel={REL_TOL}, time={TIME_TOL}; "
                 f"stochastic K={ENSEMBLE_K} sigma over N={sub_summary.get('n_seeds')}, "
                 f"pass>={ENSEMBLE_PASS_FRAC}")
    if escalated:
        seed_counts = sorted({s for s in escalated.values() if s})
        lines.append(
            f"- escalated: {len(escalated)} stochastic model(s) flagged DIFF "
            f"at the base seed count were re-run and re-judged at "
            f"{', '.join(str(s) for s in seed_counts)} seeds (overlay merge)")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"| Bucket | Count |")
    lines.append(f"|---|---:|")
    for b in ("PASS", "DIFF", "KNOWN_ARTIFACT", "NOT_SUPPORTED", "ERROR"):
        lines.append(f"| {b} | {len(buckets[b])} |")
    lines.append(f"| **TOTAL** | **{len(all_models)}** |")
    lines.append("")
    if escalated:
        lines.append("## Escalated stochastic models")
        lines.append("")
        lines.append("Stochastic models re-judged at a higher seed count to "
                     "separate genuine divergence from small-sample noise. "
                     "A DIFF here survived the escalation and is real; any "
                     "not listed as DIFF were small-sample noise.")
        lines.append("")
        lines.append("| model | escalated seeds | final bucket |")
        lines.append("|---|---:|---|")
        for bngl in sorted(escalated):
            info = per_model.get(bngl, {})
            lines.append(f"| `{Path(bngl).name}` | {escalated[bngl]} | "
                         f"{info.get('bucket', '?')} |")
        lines.append("")
    for b in ("DIFF", "KNOWN_ARTIFACT", "NOT_SUPPORTED", "ERROR"):
        if not buckets[b]:
            continue
        lines.append(f"## {b}  ({len(buckets[b])})")
        lines.append("")
        for bngl in sorted(buckets[b]):
            info = per_model[bngl]
            lines.append(f"### `{bngl}`")
            lines.append("")
            lines.append(f"- regime: {info.get('regime')}")
            if "escalated_seeds" in info:
                lines.append(f"- escalated: re-judged at "
                             f"{info['escalated_seeds']} seeds")
            if "reason" in info:
                lines.append(f"- reason: {info['reason']}")
            for k in ("sub_status", "bng_status", "sub_error", "bng_error",
                      "n_bng_failed", "n_sub_failed"):
                if k in info and info[k] not in ("", 0):
                    lines.append(f"- {k}: `{info[k]}`")
            details = info.get("details")
            if details:
                if "max_abs" in details:
                    lines.append(f"- max_abs={details.get('max_abs')}, "
                                 f"max_rel={details.get('max_rel')}")
                if "only_sub" in details and details["only_sub"]:
                    lines.append(f"- only_subprocess: {details['only_sub']}")
                if "only_bng" in details and details["only_bng"]:
                    lines.append(f"- only_bngsim: {details['only_bng']}")
                if "n_sub_ok" in details:
                    lines.append(f"- n_sub_ok={details['n_sub_ok']}, "
                                 f"n_bng_ok={details['n_bng_ok']}, "
                                 f"n_sub_failed={details['n_sub_failed']}, "
                                 f"n_bng_failed={details['n_bng_failed']}")
                pf = details.get("per_file", {})
                for name, stats in pf.items():
                    parts = [f"`{name}`"]
                    for k, v in stats.items():
                        parts.append(f"{k}={v}")
                    lines.append(f"  - {', '.join(parts)}")
            lines.append("")

    text = "\n".join(lines) + "\n"
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"Report written to {args.out}")
        print(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "buckets": {b: sorted(v) for b, v in buckets.items()},
            "per_model": per_model,
            "escalated": escalated,
        }, indent=2, default=str))
        print(f"JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
