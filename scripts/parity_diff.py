#!/usr/bin/env python3
"""Diff a subprocess parity_sweep run against a bngsim parity_sweep run.

For each model:

* Deterministic models (only ode/cvode actions): per-cell combined
  absolute+relative diff of every common .gdat/.cdat — the standard
  ODE-solver error model with a small fail-fraction budget.
  A cell passes the per-cell bar iff
      |a - b| <= ABS_TOL_FILE * file_scale
                 + ABS_TOL_COL * col_peak
                 + REL_TOL * max(|a|, |b|)
  where ``col_peak`` is that column's peak magnitude across both runs
  and ``file_scale`` is the maximum magnitude over the whole file.
  A file passes iff (1) no cell exceeds the hard ceilings
  ``HARD_REL_CEILING`` (per-cell relative) or
  ``HARD_ABS_CEILING_FILE * file_scale`` (file-scale absolute), AND
  (2) at most ``FAIL_FRAC_BUDGET`` of cells fail the per-cell bar
  (after the shift / near-zero forgiveness rules below) — the budget
  forgives a handful of isolated stiff-transient cells while the hard
  ceilings catch any concentrated divergence.
  The relative term governs the bulk of the trajectory; the column-
  relative absolute term governs the tail of a quantity decaying toward
  zero (a trailing-digit difference over an exponentially tiny value
  reads as rel ~2.0); the file-relative absolute term forgives sub-scale
  columns where the column peak is itself many decades below the model
  scale (otherwise the column-relative term holds those to atto-precision).
  A genuine divergence is a meaningful fraction of model scale and clears
  all three terms by many decades.
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
# Absolute tolerances. The deterministic per-cell bar is the standard
# ODE-solver error model |a-b| <= atol + rtol*|y|, with the absolute
# term split into two scale-relative pieces:
#
#   ABS_TOL_COL * col_peak — column-relative, the original ABS_TOL.
#     Forgives the tail of a quantity decaying toward zero within its
#     own column (where |a-b|/max(|a|,|b|) is undefined and rel reads
#     ~2.0 on a sign flip).
#   ABS_TOL_FILE * file_scale — file-relative, NEW. Forgives sub-scale
#     columns: a species that lives at 1e-8 of model scale contributes
#     a col_peak of 1e-8, so the column-relative term alone holds it to
#     1e-14 — tighter than any integrator's intrinsic precision.
#     The file-scale term keeps "the diff is insignificant on the model
#     scale" available as a backstop. 1e-9 is 8 decades below any real
#     divergence (>=0.1 of model scale) and 3 decades below the
#     column-relative term, so it acts only on tiny-column noise.
#
# Either term clearing alone is enough; the bar is their sum.
ABS_TOL_COL = 1e-6
ABS_TOL_FILE = 1e-9
# Closeness bar used only by `_discontinuity_shift_mask` to confirm a
# value equals an adjacent-row value on the other side. Kept tight and
# independent of REL_TOL: step-function branch values are bit-identical
# literals between the two sides, so a loose bar here would only risk a
# genuine divergence coincidentally matching a neighbour.
SHIFT_MATCH_TOL = 1e-6
# Maximum sample-offset the step/phase shift mask will search. The
# original mask handled only single-sample step discontinuities (k=1);
# k=3 also forgives stiff relaxation oscillators whose phase wanders a
# few samples over many cycles, and multi-bucket staircase functions
# sampled at their discontinuities. The two cell-match constraints (a
# matches the other side's r-dr row AND b matches the r+dr row) keep
# the bar tight: a real divergence does not coincidentally match a
# neighbour to 1ppm.
SHIFT_MASK_K = 3
# Time-column tolerance is *scale-relative* — see `_time_tol`. Independently
# writing the same float on two engines won't yield bit equality; demanding
# it is wrong. Floor (TIME_TOL_FLOOR) protects sub-second models; relative
# term (TIME_TOL_REL) scales with t_max so the bar stays meaningful on long
# trajectories.
#
# TIME_TOL_REL is 1 ppb of t_max. Two CVODE runs agree on output times to
# ~1e-13 relative, but bngsim's NfsimSession and BNG2.pl's NFsim build their
# sample-time grids with different float arithmetic and drift up to ~1 ppb of
# the trajectory length (observed deterministically, same across all seeds,
# on `scaling_example`: ~49 ns over a 50-unit span). 1 ppb still sits 5-6
# decades below any real time-axis misalignment — an off-by-one-sample bug is
# a whole sample step, i.e. ~1/n_steps of t_max (e.g. ~1.4e-3 of t_max for a
# 730-step run), millions of ppb. Tracked upstream (NFsim sample-time grid).
TIME_TOL_FLOOR = 1e-9
TIME_TOL_REL = 1e-9
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

# Deterministic cell-fraction budget. Two independent stiff-ODE integrators
# at the same atol/rtol can disagree above the soft per-cell bar on a few
# isolated cells (a sharp transient, a single stiff sub-step) while the
# trajectory as a whole is the same dynamical solution. Allow up to
# FAIL_FRAC_BUDGET of cells to fail the soft tolerance ONLY if no cell
# blows past the hard ceilings — the budget catches "the simulations agree
# overall but one row has integrator noise", the ceilings guard against
# "a tiny region of the trajectory has a real engine bug".
#
# Sizing: corpus-real engine bugs (e.g. bngsim #41 compartmental clamp:
# 1.66% of cells in Motivating_example_cBNGL_2, 55% in catalysis) fail at
# >=1.5% of cells, so 0.5% is a 3-10x safety margin under the minimum
# observed real-bug fail rate. The smallest real bug we expect would be
# one wrong species column on a 20-column model (1/20 = 5% of cells),
# well above the budget. Hard ceilings: 5% per-cell relative and 1% of
# file-peak absolute are 50-100x the soft per-cell tol but still 10-20x
# smaller than the magnitude of any real engine bug we've filed.
FAIL_FRAC_BUDGET = 5e-3
HARD_REL_CEILING = 0.05
HARD_ABS_CEILING_FILE = 1e-2

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

# Models where the *subprocess* reference is the wrong oracle for the
# network-free (nf) segment: BNG2.pl bundles NFsim v1.14.3, which lacks
# `block_same_complex_binding` (-bscb) support and wrongly applies a
# two-product dissociation rule to a bond *inside a connected/cyclic
# complex* whose removal doesn't actually dissociate it. bngsim defaults
# bscb=True (and RuleMonkey agrees), so bngsim's NF matches the ODE
# network result — the network generator drops that same molecularity-
# violating reaction. So a bngsim-vs-subprocess nf DIFF here means
# bngsim is the *correct* engine, not a regression (PyBNF-Private#54).
#
# For these models there is no genuine ODE-vs-NF physics gap, so we
# revalidate bngsim's nf ensemble against the trusted ODE oracle (the
# subprocess run_network result, which both engines reproduce exactly)
# instead of against the buggy subprocess nf. Passing means "bngsim's nf
# tracks ODE", which is positive evidence of correctness and still
# catches a future bngsim regression (it would no longer track ODE).
# Each entry names the model's ODE-segment and nf-segment suffixes.
SUBPROCESS_NF_UNRELIABLE = {
    "ode_vs_nf_discrepancy": {
        "ode_suffix": "A_ODE", "nf_suffix": "B_NFsim",
        "issue": "PyBNF-Private#54",
        "reason": "subprocess NFsim v1.14.3 lacks -bscb; applies a "
                  "2-product dissociation to a bond inside a 5-molecule "
                  "ring that doesn't dissociate. bngsim (bscb on) + "
                  "RuleMonkey + ODE all agree; subprocess nf is the "
                  "outlier. Validated against the ODE oracle.",
    },
    "debug": {
        "ode_suffix": "A_ODE", "nf_suffix": "B_NFsim",
        "issue": "PyBNF-Private#54",
        "reason": "same -bscb root cause as ode_vs_nf_discrepancy "
                  "(MTOR/RPTOR pre-assembled complex). bngsim nf tracks "
                  "ODE; subprocess nf is the outlier.",
    },
    "debug_v3": {
        "ode_suffix": "A_ODE", "nf_suffix": "B_NFsim",
        "issue": "PyBNF-Private#54",
        "reason": "same -bscb root cause as ode_vs_nf_discrepancy "
                  "(simplified MTOR/RPTOR complex). bngsim nf tracks "
                  "ODE; subprocess nf is the outlier.",
    },
    "overlap_rules2": {
        "ode_suffix": "BNG", "nf_suffix": "NFS",
        "issue": "PyBNF-Private#54 (filed #55, corrected)",
        "reason": "same -bscb root cause: a 2-product dissociation on a "
                  "bond inside a size-2 ring. ODE keeps the rings (the "
                  "ring-opening reaction violates molecularity and is "
                  "dropped); bngsim nf agrees, subprocess nf wrongly "
                  "breaks them.",
    },
    "testrings_wsh": {
        "ode_suffix": "ode", "nf_suffix": "nf",
        "issue": "PyBNF-Private#54",
        "reason": "same -bscb root cause (ring complex). bngsim nf "
                  "tracks ODE; subprocess nf is the outlier.",
    },
}

# Relative tolerance for the ODE-oracle revalidation. Looser than the
# deterministic REL_TOL because an nf ensemble mean differs from the ODE
# (infinite-size) limit by a finite-size systematic offset on top of
# sampling scatter — observed up to ~25% on small-count observables for
# these models. The primary gate is still the per-cell sigma test against
# the nf mean's standard error; this rel term is a backstop for cells
# whose standard error has shrunk at high seed counts. The subprocess
# bscb divergence is 50-100x, so any value here far below 1.0 separates
# correct-bngsim from buggy-subprocess and still flags a real regression.
ODE_ORACLE_REL = 0.30

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


def _time_tol(times):
    """Per-file time-column tolerance, scale-relative.

    Two independent engines emit float-formatted output times whose last
    bits differ; exact equality is not a meaningful property of two
    simulations. We accept ``|t_sub - t_bng| <= max(TIME_TOL_FLOOR,
    TIME_TOL_REL * t_max)`` — at least 1 ns absolute, plus 1 ppb of the
    simulation length so the bar stays meaningful for both sub-second and
    Gyr-scale trajectories. Any real divergence (an engine bug producing
    wrong output times, i.e. a fraction of a sample step) is many decades
    larger than this floor.
    """
    if times is None or len(times) == 0:
        return TIME_TOL_FLOOR
    t_abs = np.abs(times)
    if not np.any(np.isfinite(t_abs)):
        return TIME_TOL_FLOOR
    t_max = float(np.nanmax(t_abs[np.isfinite(t_abs)]))
    return max(TIME_TOL_FLOOR, TIME_TOL_REL * t_max)


def _discontinuity_shift_mask(sub_data, bng_data, fail_mask, k=SHIFT_MASK_K):
    """Mark failing cells that are <=k-sample horizontal shifts.

    Two phenomena hit the same structural artifact: column values that are
    correct on both sides but sampled at slightly different times.

    (1) Step-function columns (``if(t<N, a, b)``) at a threshold on an output
        grid point — the two integrators' clock observables disagree by ~1e-13
        and land on opposite sides of ``<``, so the column transitions one
        sample early on one side. k=1 handles this.

    (2) Stiff relaxation oscillators with sharp switches — sub-step timing
        differs between integrators, so phase wanders a few samples over many
        cycles. Two periodic trajectories with matching period/amplitude/cycle
        count read as enormous per-cell diffs on the steep edges (rel ~2 on a
        sign flip). k>1 (default 3) handles this.

    (3) Multi-bucket staircase functions sampled at their discontinuities.

    A failing cell ``(r, c)`` is forgiven when there is some offset
    ``dr in [-k, +k] \\ {0}`` such that the value on the sub side equals the
    bng side's value at row ``r - dr``, and the bng side at row ``r`` equals
    the sub side at row ``r + dr`` — i.e. the column is consistent with bng
    leading sub by ``dr`` samples (or vice versa). Closeness is held to
    1ppm; a real divergence does not match a neighbour to 1ppm by accident
    in a smooth trajectory. Boundary rows fall back to a one-sided check
    against both adjacent rows on the same side.
    """
    R = sub_data.shape[0]
    forgiven = np.zeros_like(fail_mask)

    def close(x, y):
        d = abs(x - y)
        return d <= SHIFT_MATCH_TOL * max(abs(x), abs(y), 1e-12)

    rows, cols = np.nonzero(fail_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        s, b = sub_data[r, c], bng_data[r, c]
        ok = False
        for dr in range(1, k + 1):
            # bng leads sub by dr samples: bng[r-dr] == sub[r] AND
            # bng[r] == sub[r+dr].
            if r - dr >= 0 and r + dr < R:
                if close(s, bng_data[r - dr, c]) and close(b, sub_data[r + dr, c]):
                    ok = True
                    break
                # sub leads bng by dr samples.
                if close(s, bng_data[r + dr, c]) and close(b, sub_data[r - dr, c]):
                    ok = True
                    break
            # final-row boundary: one side stepped at the final sample, the
            # other has no room to follow. Require the stepped side to
            # agree with itself just before, and the other side to disagree.
            elif r == R - 1 and r - dr >= 0:
                sp = sub_data[r - dr, c]
                bp = bng_data[r - dr, c]
                if close(s, bp) and close(s, sp) and not close(b, bp):
                    ok = True  # bngsim stepped at the final sample
                    break
                if close(b, sp) and close(b, bp) and not close(s, sp):
                    ok = True  # subprocess stepped at the final sample
                    break
            # first-row boundary, symmetric.
            elif r == 0 and r + dr < R:
                sn = sub_data[r + dr, c]
                bn = bng_data[r + dr, c]
                if close(s, bn) and close(s, sn) and not close(b, bn):
                    ok = True  # bngsim stepped at the first sample
                    break
                if close(b, sn) and close(b, bn) and not close(s, sn):
                    ok = True  # subprocess stepped at the first sample
                    break
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
            t_tol = _time_tol(np.concatenate([sub[:, 0], bng[:, 0]]))
            per_file[f"{stem}{sp.suffix}<->{stem}{bp.suffix}"] = {
                "note": "cross-ext name match (likely empty observables NF)",
                "shape": list(sub.shape),
                "time_diff": time_diff,
                "time_tol": t_tol,
                "data_diff": data_diff,
            }
            if time_diff > t_tol:
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
        # Time column (col 0): scale-relative tolerance, see `_time_tol`.
        time_diff = float(np.max(np.abs(sub[:, 0] - bng[:, 0])))
        time_tol = _time_tol(np.concatenate([sub[:, 0], bng[:, 0]]))
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
        # solver error model |a-b| <= atol + rtol*|y|. Absolute term has
        # both a column-relative piece (forgives decay tails) and a
        # file-relative piece (forgives sub-scale columns); see the
        # ABS_TOL_COL / ABS_TOL_FILE constants.
        cell_tol = (ABS_TOL_FILE * scale
                    + ABS_TOL_COL * col_peak[np.newaxis, :]
                    + REL_TOL * colmag)
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
        # Option C — cell-fraction budget. A cell that exceeds either
        # hard ceiling is never forgiven (catches a real engine bug
        # concentrated in a small region). The rest of `effective_fail`
        # is "soft": cells past the per-cell tol but within the hard
        # ceilings, plausibly stiff-transient integrator noise. Forgive
        # the soft group iff their fraction is within FAIL_FRAC_BUDGET.
        hard_rel_fail = reld_clean > HARD_REL_CEILING
        hard_abs_fail = absd_clean > HARD_ABS_CEILING_FILE * scale
        hard_fail = effective_fail & (hard_rel_fail | hard_abs_fail)
        soft_fail = effective_fail & ~hard_fail
        n_hard_fail = int(np.sum(hard_fail))
        n_soft_fail = int(np.sum(soft_fail))
        total_cells = int(effective_fail.size)
        frac_soft_fail = (n_soft_fail / total_cells) if total_cells else 0.0
        budget_ok = frac_soft_fail <= FAIL_FRAC_BUDGET
        # Final fail set after budget: hard fails always; soft fails
        # only when they break the budget.
        remaining_fail = hard_fail if budget_ok else effective_fail
        n_remaining = int(np.sum(remaining_fail))
        n_budget_forgiven = n_soft_fail if budget_ok else 0
        # Reported figures: max over genuinely-failing cells (0 if the
        # file passes); raw figures over all cells kept for visibility.
        max_abs_raw = float(np.max(absd_clean)) if absd_clean.size else 0.0
        max_rel_raw = float(np.max(reld_clean)) if reld_clean.size else 0.0
        max_abs = (float(np.max(absd_clean[remaining_fail]))
                   if n_remaining else 0.0)
        max_rel = (float(np.max(reld_clean[remaining_fail]))
                   if n_remaining else 0.0)
        per_file[name] = {
            "shape": list(sub.shape),
            "time_diff": time_diff,
            "time_tol": time_tol,
            "max_abs": max_abs if np.isfinite(max_abs) else "inf",
            "max_rel": max_rel if np.isfinite(max_rel) else "inf",
        }
        if n_shift:
            per_file[name]["discontinuity_shifts"] = n_shift
        if n_near_zero:
            per_file[name]["near_zero_forgiven"] = n_near_zero
        if n_budget_forgiven:
            per_file[name]["budget_forgiven"] = n_budget_forgiven
            per_file[name]["frac_soft_fail"] = frac_soft_fail
        if n_shift or n_near_zero or n_budget_forgiven:
            per_file[name]["max_rel_raw"] = (
                max_rel_raw if np.isfinite(max_rel_raw) else "inf")
            per_file[name]["max_abs_raw"] = (
                max_abs_raw if np.isfinite(max_abs_raw) else "inf")
        overall_max_abs = max(overall_max_abs, max_abs if np.isfinite(max_abs) else float("inf"))
        overall_max_rel = max(overall_max_rel, max_rel if np.isfinite(max_rel) else float("inf"))
        if time_diff > time_tol:
            per_file[name]["fail"] = "time"
            overall_pass = False
        if n_remaining:
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
            sub_t_mean = np.mean(sub_stack[:, :, 0], axis=0)
            bng_t_mean = np.mean(bng_stack[:, :, 0], axis=0)
            time_diff = float(np.max(np.abs(sub_t_mean - bng_t_mean)))
            t_tol = _time_tol(np.concatenate([sub_t_mean, bng_t_mean]))
            per_file[stem] = {"note": "cross-ext name match (NF empty observables)",
                              "shape_seeds": [sub_stack.shape[0],
                                              *list(sub_stack.shape[1:])],
                              "time_diff": time_diff,
                              "time_tol": t_tol}
            if time_diff > t_tol:
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
        sub_time_mean = np.mean(sub_time, axis=0)
        bng_time_mean = np.mean(bng_time, axis=0)
        time_diff = float(np.max(np.abs(sub_time_mean - bng_time_mean)))
        time_tol = _time_tol(np.concatenate([sub_time_mean, bng_time_mean]))
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
            "time_tol": time_tol,
            "n_cells": n_total,
            "n_pass": n_pass,
            "frac_pass": frac_pass,
            "max_abs_mean_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        }
        if time_diff > time_tol:
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


def revalidate_nf_against_ode(sub_dirs, bng_dirs, model_stem, entry):
    """Validate a bngsim nf ensemble against the ODE network oracle.

    Used only for models in SUBPROCESS_NF_UNRELIABLE, where the subprocess
    nf reference is known-buggy (NFsim v1.14.3, no -bscb). The oracle is
    the subprocess run_network ODE result (canonical; bngsim reproduces it
    exactly, so this is non-circular). We compare the bngsim nf ensemble
    mean to that ODE trajectory with the same sigma test the ensemble path
    uses, plus a looser relative backstop (ODE_ORACLE_REL) for finite-size
    offsets. Returns ('pass'|'diff', details).
    """
    ode_suffix = entry["ode_suffix"]
    nf_suffix = entry["nf_suffix"]

    # ODE oracle: deterministic, identical across seeds — first one found.
    ode_arr = None
    for d in sub_dirs:
        p = Path(d) / f"{model_stem}_{ode_suffix}.gdat"
        if p.is_file():
            ode_arr, e = safe_load(p)
            if e is None and ode_arr is not None:
                break
            ode_arr = None
    if ode_arr is None:
        return "diff", {"oracle": "ode", "reason": f"ODE oracle "
                        f"{model_stem}_{ode_suffix}.gdat not found in subprocess"}

    # bngsim nf ensemble across seeds.
    nf_arrs = []
    for d in bng_dirs:
        p = Path(d) / f"{model_stem}_{nf_suffix}.gdat"
        if p.is_file():
            a, e = safe_load(p)
            if e is None and a is not None:
                nf_arrs.append(a)
    if not nf_arrs:
        return "diff", {"oracle": "ode", "reason": f"no bngsim nf outputs "
                        f"{model_stem}_{nf_suffix}.gdat"}
    if len({a.shape for a in nf_arrs}) != 1:
        return "diff", {"oracle": "ode",
                        "reason": "bngsim nf seed shapes inconsistent"}

    nf_stack = np.stack(nf_arrs)
    # Compare the leading common columns (time + observables + any shared
    # user functions). bngsim may append extra trailing function columns
    # (#53); the observable block aligns positionally with the ODE file.
    ncol = min(ode_arr.shape[1], nf_stack.shape[2])
    nrow = min(ode_arr.shape[0], nf_stack.shape[1])
    ode = ode_arr[:nrow, :ncol]
    nf_stack = nf_stack[:, :nrow, :ncol]

    nf_time = np.mean(nf_stack[:, :, 0], axis=0)
    time_diff = float(np.max(np.abs(nf_time - ode[:, 0])))
    time_tol = _time_tol(np.concatenate([nf_time, ode[:, 0]]))

    ode_obs = ode[:, 1:]
    nf_obs = nf_stack[:, :, 1:]
    N = nf_obs.shape[0]
    mu = np.mean(nf_obs, axis=0)
    sd = np.std(nf_obs, axis=0, ddof=1) if N > 1 else np.zeros_like(mu)
    se = sd / np.sqrt(N)
    scale = max(float(np.nanmax(np.abs(ode_obs))) if ode_obs.size else 0.0,
                float(np.nanmax(np.abs(mu))) if mu.size else 0.0, 1e-12)
    diff = np.abs(mu - ode_obs)
    floor = 1e-12 * scale
    threshold = ENSEMBLE_K * np.maximum(se, floor)
    rel_floor = np.maximum(np.maximum(np.abs(ode_obs), np.abs(mu)),
                           scale * NEAR_ZERO_FLOOR_REL)
    rel_ok = diff <= ODE_ORACLE_REL * rel_floor
    near_zero = np.maximum(np.abs(ode_obs), np.abs(mu)) < NEAR_ZERO_REL * scale
    cell_pass = (diff <= threshold) | rel_ok | near_zero
    n_total = int(cell_pass.size)
    n_pass = int(np.sum(cell_pass))
    frac_pass = n_pass / n_total if n_total else 1.0

    details = {
        "oracle": "ode",
        "ode_suffix": ode_suffix,
        "nf_suffix": nf_suffix,
        "n_seeds": N,
        "time_diff": time_diff,
        "n_cells": n_total,
        "n_pass": n_pass,
        "frac_pass": frac_pass,
        "max_abs_nf_vs_ode": float(np.nanmax(diff)) if diff.size else 0.0,
        "issue": entry.get("issue"),
        "reason": entry.get("reason"),
    }
    ok = (frac_pass >= ENSEMBLE_PASS_FRAC) and (time_diff <= time_tol)
    return ("pass" if ok else "diff"), details


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

    buckets = {"PASS": [], "PASS_REF_BUG": [], "DIFF": [], "KNOWN_ARTIFACT": [],
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
                # A model whose subprocess nf reference is known-buggy
                # (NFsim v1.14.3, no -bscb) is revalidated against the ODE
                # oracle: if bngsim's nf tracks ODE, bngsim is the correct
                # engine and the subprocess-comparison DIFF is reclassified
                # PASS_REF_BUG. Otherwise it stays DIFF.
                ref_bug = SUBPROCESS_NF_UNRELIABLE.get(Path(bngl).stem)
                ode_status, ode_details = (None, None)
                if ref_bug is not None:
                    ode_status, ode_details = revalidate_nf_against_ode(
                        sub_dirs, bng_dirs, Path(bngl).stem, ref_bug)
                if ref_bug is not None and ode_status == "pass":
                    buckets["PASS_REF_BUG"].append(bngl)
                    per_model[bngl] = {
                        "bucket": "PASS_REF_BUG", "regime": "stochastic",
                        "reason": ref_bug["reason"],
                        "issue": ref_bug.get("issue"),
                        "details": details,
                        "ode_oracle": ode_details,
                    }
                else:
                    per_model[bngl] = {"bucket": "DIFF", "regime": "stochastic",
                                       "details": details}
                    if ode_details is not None:
                        per_model[bngl]["ode_oracle"] = ode_details
                    buckets["DIFF"].append(bngl)

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
    lines.append(f"- tolerance: deterministic rel={REL_TOL}, "
                 f"time=max({TIME_TOL_FLOOR},{TIME_TOL_REL}*t_max), "
                 f"fail-frac budget<={FAIL_FRAC_BUDGET} (ceilings "
                 f"rel<={HARD_REL_CEILING}, abs<={HARD_ABS_CEILING_FILE}*file_scale); "
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
    for b in ("PASS", "PASS_REF_BUG", "DIFF", "KNOWN_ARTIFACT",
              "NOT_SUPPORTED", "ERROR"):
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
    for b in ("PASS_REF_BUG", "DIFF", "KNOWN_ARTIFACT", "NOT_SUPPORTED", "ERROR"):
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
            if "issue" in info and info["issue"]:
                lines.append(f"- issue: {info['issue']}")
            if "ode_oracle" in info and info["ode_oracle"]:
                oo = info["ode_oracle"]
                lines.append(f"- ODE-oracle revalidation: "
                             f"frac_pass={oo.get('frac_pass')}, "
                             f"n_pass={oo.get('n_pass')}/{oo.get('n_cells')}, "
                             f"max_abs_nf_vs_ode={oo.get('max_abs_nf_vs_ode')}, "
                             f"n_seeds={oo.get('n_seeds')}")
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
