"""Unit tests for the parity-sweep differ in `scripts/parity_diff.py`.

Covers the Option A patches added 2026-05-20:

  * scale-relative ``TIME_TOL`` (was a hard ``0.0`` reject)
  * file-scale absolute term ``ABS_TOL_FILE * file_scale``
  * k-sample shift mask (was a single-sample step mask)

The goal is to lock in that each forgiveness rule fires when it should
and stays out of the way of real divergences. Each test constructs a
synthetic pair of arrays and exercises the helper directly — no real
BNGL runs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# scripts/ is not on sys.path by default.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import parity_diff as pd  # noqa: E402


# ---------------------------------------------------------------------------
# _time_tol — scale-relative time-column tolerance
# ---------------------------------------------------------------------------

class TestTimeTol:
    def test_floor_for_subunit_trajectory(self):
        # TIME_TOL_REL * t_max < floor when t_max < FLOOR/REL = 1e-2, so the
        # floor wins for a truly sub-unit (millisecond-scale) trajectory.
        assert pd._time_tol(np.linspace(0, 0.005, 11)) == pd.TIME_TOL_FLOOR

    def test_scales_with_long_trajectory(self):
        # t_max=1e6: tolerance = TIME_TOL_REL * 1e6 = 1e-7 * 1e6 = 0.1.
        t = np.linspace(0, 1e6, 1001)
        assert pd._time_tol(t) == pytest.approx(0.1, rel=1e-9)

    def test_gdat_text_precision_forgiven(self):
        # BNG2.pl writes the .gdat time column as %.8e (9 sig figs), so a
        # non-terminating sample time is rounded with up to ~5e-9 relative
        # error while bngsim writes the true value. The HBF1998_brusselator
        # case: sample times 10 + k/30, max observed diff 3.3e-8 at t~10 on a
        # ~12-unit span. The bar must forgive it: 1e-7 * 12 = 1.2e-6 >> 3.3e-8.
        t = np.linspace(0, 12, 361)
        assert pd._time_tol(t) == pytest.approx(1.2e-6, rel=1e-9)
        assert pd._time_tol(t) > 3.3e-8
        # but a real off-by-one-sample misalignment (a whole step ~ 12/360 =
        # 3.3e-2) is still decades above the bar and must fail.
        assert pd._time_tol(t) < 3.3e-2

    def test_empty_returns_floor(self):
        assert pd._time_tol(np.array([])) == pd.TIME_TOL_FLOOR
        assert pd._time_tol(None) == pd.TIME_TOL_FLOOR

    def test_all_nan_returns_floor(self):
        assert pd._time_tol(np.array([np.nan, np.nan])) == pd.TIME_TOL_FLOOR

    def test_uses_max_not_mean(self):
        # mostly zeros plus one large t: scale uses the max.
        # TIME_TOL_REL * 1e9 = 1e-7 * 1e9 = 100.
        t = np.array([0.0, 0.0, 0.0, 1e9])
        assert pd._time_tol(t) == pytest.approx(100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# k-sample shift mask
# ---------------------------------------------------------------------------

def _all_fail(arr):
    return np.ones_like(arr, dtype=bool)


class TestShiftMaskKSample:
    def test_uniform_three_sample_shift_forgiven(self):
        """A column that is the same trajectory shifted by 3 samples in
        the interior is fully forgiven at k=3.
        """
        R, k = 30, 3
        # Distinct values per row so closeness matches uniquely.
        vals = np.linspace(1.0, 30.0, R).reshape(R, 1)
        sub = vals.copy()
        bng = vals.copy()
        # bng leads sub by 3 samples: bng[r-3] = sub[r] for r >= 3.
        bng[: R - k, 0] = sub[k:, 0]
        # Only interior cells (3..R-4) have both neighbours for the
        # two-sided closeness check; check those.
        fm = _all_fail(sub)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=k)
        # All cells with valid ±k neighbours should be forgiven.
        interior = np.zeros_like(fm)
        interior[k : R - k, :] = True
        assert forg[interior].all()

    def test_shift_just_over_k_not_forgiven(self):
        """A 4-sample shift is past k=3 and stays flagged."""
        R, shift, k = 30, 4, 3
        vals = np.linspace(1.0, 30.0, R).reshape(R, 1)
        sub = vals.copy()
        bng = vals.copy()
        bng[: R - shift, 0] = sub[shift:, 0]
        fm = _all_fail(sub)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=k)
        # In the interior, none should be forgiven (the neighbour
        # search radius doesn't reach the 4-sample offset).
        interior = np.zeros_like(fm)
        interior[k : R - k, :] = True
        assert not forg[interior].any()

    def test_single_sample_shift_still_forgiven(self):
        """The original single-sample step case still passes the
        generalized mask (regression check).
        """
        R, k = 30, 3
        a, b = 1.0, 5.0
        sub = np.full((R, 1), a)
        bng = np.full((R, 1), a)
        # sub steps to b at row 10; bng steps to b at row 11.
        sub[10:, 0] = b
        bng[11:, 0] = b
        fm = _all_fail(sub)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=k)
        # The disagreeing row is r=10 (sub=b, bng=a); a 1-sample shift.
        assert forg[10, 0]

    def test_random_divergence_not_forgiven(self):
        """A random divergence does not coincide with a neighbour to
        1ppm and stays flagged.
        """
        rng = np.random.default_rng(42)
        sub = rng.uniform(1.0, 10.0, (40, 3))
        bng = sub + rng.normal(0, 0.5, sub.shape)  # large random noise
        fm = _all_fail(sub)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=3)
        assert forg.sum() < 0.05 * forg.size  # < 5% coincidental matches

    def test_staircase_one_bucket_flip_forgiven(self):
        """A staircase function whose discontinuity lands one sample
        differently between sides is forgiven.
        """
        # Pre/mid/post values; sub jumps at row 10, bng jumps at row 11.
        R = 20
        v_pre, v_mid, v_post = 100.0, 124.5, 150.0
        sub = np.full((R, 1), v_pre)
        bng = np.full((R, 1), v_pre)
        sub[10, 0] = v_mid
        sub[11:, 0] = v_post
        bng[11, 0] = v_mid
        bng[12:, 0] = v_post
        fm = _all_fail(sub)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=3)
        # The two disagreeing rows are 10 (sub=v_mid, bng=v_pre) and
        # 11 (sub=v_post, bng=v_mid). Both must forgive.
        assert forg[10, 0]
        assert forg[11, 0]


# ---------------------------------------------------------------------------
# deterministic_compare end-to-end on synthetic .gdat pairs
# ---------------------------------------------------------------------------

def _write_gdat(path, arr, n_obs):
    """Write a minimal BNG-style .gdat: ``# time o0 o1 ...`` header + rows."""
    path = Path(path)
    header = "#         time " + "".join(
        f"   obs_{i:02d} " for i in range(n_obs)
    )
    with open(path, "w") as f:
        f.write(header.rstrip() + "\n")
        for row in arr:
            f.write(" ".join(f"{v: .12e}" for v in row) + "\n")


@pytest.fixture
def tmp_pair(tmp_path):
    sub_dir = tmp_path / "sub"
    bng_dir = tmp_path / "bng"
    sub_dir.mkdir()
    bng_dir.mkdir()
    return sub_dir, bng_dir


class TestDeterministicCompare:
    def test_bit_identical_passes(self, tmp_pair):
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        arr = np.column_stack([t, np.sin(t), np.cos(t)])
        _write_gdat(sub_dir / "m.gdat", arr, 2)
        _write_gdat(bng_dir / "m.gdat", arr, 2)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass"

    def test_time_diff_1e_minus_11_short_traj_passes(self, tmp_pair):
        """The motivating filamentation_blue_v1 case: bit-identical
        observables, time column off by 1e-11 on t_max ~ 1000.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 1000, 1001)
        obs = np.column_stack([np.sin(t), np.cos(t)])
        sub_arr = np.column_stack([t, obs])
        bng_arr = np.column_stack([t + 1e-11, obs])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details
        # Time tol on t_max=1000 = max(1e-9, 1e-12*1000) = 1e-9; 1e-11 fits.

    def test_time_diff_real_lag_still_fails(self, tmp_pair):
        """A 1 ms time lag on a short trajectory is real and must fail."""
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        obs = np.column_stack([np.sin(t), np.cos(t)])
        sub_arr = np.column_stack([t, obs])
        bng_arr = np.column_stack([t + 1e-3, obs])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_sub_scale_column_forgiven_by_file_scale_abs(self, tmp_pair):
        """A tiny column (peak 1e-8) with a 5e-10 diff is forgiven by
        the file-scale absolute term when the file peak is ~1.0.
        The motivating transport_v3 / fceri case.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        big = np.sin(t)  # peak ~1.0 -> dominates file_scale
        tiny_sub = 1e-8 * np.exp(-t / 5)  # peak 1e-8
        tiny_bng = tiny_sub.copy()
        tiny_bng[50] += 5e-10  # one cell off by 5e-10 (5% relative)
        sub_arr = np.column_stack([t, big, tiny_sub])
        bng_arr = np.column_stack([t, big, tiny_bng])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details

    def test_sub_scale_diff_in_sub_scale_file_still_fails(self, tmp_pair):
        """Same diff, but the file has no big column — file_scale stays
        small, so the file-relative term does not rescue it.
        A genuine divergence in a model whose whole signal is tiny must
        still be caught.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        tiny_sub = 1e-8 * np.exp(-t / 5)
        tiny_bng = tiny_sub.copy()
        tiny_bng[50] += 5e-10
        sub_arr = np.column_stack([t, tiny_sub])
        bng_arr = np.column_stack([t, tiny_bng])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 1)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 1)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_real_divergence_still_fails(self, tmp_pair):
        """A 10% sustained divergence on a column carrying the model's
        signal must be flagged. No forgiveness rule rescues it.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        sub_arr = np.column_stack([t, np.sin(t)])
        bng_arr = np.column_stack([t, 1.1 * np.sin(t)])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 1)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 1)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_isolated_transient_cell_forgiven_by_budget(self, tmp_pair):
        """A model with one cell failing the per-cell bar but below the
        hard ceilings and below FAIL_FRAC_BUDGET passes — the McMillen_2002
        / test2 case. Two stiff integrators disagree on one row.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 401)
        # Monotone non-zero signal, 4 obs columns at file_scale ~ 1200.
        signal = 1200 * np.exp(-t / 50)  # decays from 1200 to ~980
        cols = [signal, 0.5 * signal, 2.0 * signal, 0.1 * signal]
        sub_arr = np.column_stack([t] + cols)
        bng_arr = sub_arr.copy()
        # One cell off by ~0.1% (relative ~1e-3, above REL_TOL=1e-4 but
        # way below HARD_REL_CEILING=5e-2); 1/1604 = 0.062% << 0.5% budget.
        bng_arr[200, 2] += 0.001 * bng_arr[200, 2]
        _write_gdat(sub_dir / "m.gdat", sub_arr, 4)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 4)
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details

    def test_concentrated_divergence_caught_by_hard_rel_ceiling(self, tmp_pair):
        """A single cell with a large per-cell relative diff is caught
        even though only one cell fails (budget alone would forgive it).
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 401)
        # Constant non-zero signal so the relative bump is well-defined.
        signal = np.full_like(t, 10.0)
        sub_arr = np.column_stack([t, signal, signal])
        bng_arr = sub_arr.copy()
        # 50% relative divergence on one cell — way above 5% rel ceiling.
        bng_arr[200, 1] *= 1.5
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_near_zero_relative_blowup_forgiven(self, tmp_pair):
        """The bmp-signaling case: a transient in a sub-scale column
        settles to +5e-8 on one integrator and -5e-7 on the other (a sign
        flip near zero), so abs diff ~5.5e-7 sits a hair over the file-
        relative atol (1e-9 * file_scale 500 = 5e-7) and per-cell rel ~1.1.
        The value carries no real magnitude (5e-7 is 1e-9 of the file peak
        and 1e-4 of its own column peak), so the hard *relative* ceiling
        must not condemn it — it falls to the soft group and the
        fail-fraction budget forgives the handful of cells.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 801)
        big = np.full_like(t, 500.0)          # file_scale = 500
        tiny_sub = np.zeros_like(t)
        tiny_sub[100] = 3e-3                   # column peak 3e-3 (sub-scale)
        tiny_bng = tiny_sub.copy()
        # Near-zero sign-flip noise on 3 rows: sub ~ +5e-8, bng ~ -5e-7.
        for r in (312, 313, 314):
            tiny_sub[r] = 5e-8
            tiny_bng[r] = -5e-7
        sub_arr = np.column_stack([t, big, tiny_sub])
        bng_arr = np.column_stack([t, big, tiny_bng])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details

    def test_many_near_zero_cells_still_caught_by_budget(self, tmp_pair):
        """Guard for the near-zero rel-ceiling exemption: forgiving a
        *sprinkling* of near-zero cells must not forgive a wholesale
        sub-scale divergence. The same sign-flip noise on >0.5% of cells
        is soft (below the hard ceilings) but exceeds FAIL_FRAC_BUDGET, so
        the file is still DIFF.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 801)
        big = np.full_like(t, 500.0)
        tiny_sub = np.zeros_like(t)
        tiny_sub[100] = 3e-3
        tiny_bng = tiny_sub.copy()
        # ~2% of the 801 rows get near-zero sign-flip noise (> 0.5% budget).
        for r in range(0, 801, 50):           # 17 rows, 17/(801*2) ~ 1.06%
            tiny_sub[r] = 5e-8
            tiny_bng[r] = -5e-7
        sub_arr = np.column_stack([t, big, tiny_sub])
        bng_arr = np.column_stack([t, big, tiny_bng])
        _write_gdat(sub_dir / "m.gdat", sub_arr, 2)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 2)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_concentrated_divergence_caught_by_hard_abs_ceiling(self, tmp_pair):
        """A single cell with an absolute diff above 1% of file scale
        is caught even though it's only one cell — and the per-cell
        relative is below the rel ceiling.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 401)
        # Constant 1000 (file_scale=1000); bump one cell by 50 (5% of scale).
        # Per-cell rel = 50/1050 ~ 0.048, below the 5% ceiling, so only
        # the abs ceiling can catch it.
        signal = np.full_like(t, 1000.0)
        sub_arr = np.column_stack([t, signal])
        bng_arr = sub_arr.copy()
        bng_arr[200, 1] += 50.0
        _write_gdat(sub_dir / "m.gdat", sub_arr, 1)
        _write_gdat(bng_dir / "m.gdat", bng_arr, 1)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_many_soft_failing_cells_caught_by_budget(self, tmp_pair):
        """A model with 5% of cells failing the per-cell bar (each cell
        below hard ceilings) is still caught — above the 0.5% budget.
        The catalysis / Motivating_example_cBNGL_2 case.
        """
        sub_dir, bng_dir = tmp_pair
        rng = np.random.default_rng(7)
        t = np.linspace(0, 10, 401)
        signal = np.exp(-t / 5) * np.sin(2 * np.pi * t)
        sub_arr = np.column_stack([t, signal, signal, signal, signal])
        bng_arr = sub_arr.copy()
        # Perturb 5% of obs cells by ~3% relative (above per-cell tol,
        # below hard ceilings).
        n_obs = 4
        rows = sub_arr.shape[0]
        n_perturb = int(0.05 * rows * n_obs)
        flat = rng.choice(rows * n_obs, n_perturb, replace=False)
        for fi in flat:
            r, c = divmod(fi, n_obs)
            bng_arr[r, c + 1] *= 1.03  # 3% relative bump
        _write_gdat(sub_dir / "m.gdat", sub_arr, n_obs)
        _write_gdat(bng_dir / "m.gdat", bng_arr, n_obs)
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_phase_wander_oscillator_mostly_forgiven(self):
        """A periodic trajectory whose two integrations land 2 samples
        out of phase has nearly every interior cell forgiven by the
        k-sample shift mask. Boundary cells (within k of an end) cannot
        be forgiven point-wise — that's a true property of phase shift,
        and a model where only the boundary fails will need an
        additional "tail-trim" rule above the cell mask (not in Option A).
        """
        t = np.linspace(0, 10, 1001)
        sub = np.sin(2 * np.pi * t).reshape(-1, 1)
        bng = np.sin(2 * np.pi * (t + 2 * (t[1] - t[0]))).reshape(-1, 1)
        fm = np.ones_like(sub, dtype=bool)
        forg = pd._discontinuity_shift_mask(sub, bng, fm, k=3)
        # 99%+ of cells forgiven — only the first/last few rows can't shift.
        assert forg.mean() > 0.99
        # And the unforgiven cells must all be within k of a boundary.
        unforg_rows = np.where(~forg.flatten())[0]
        R = sub.shape[0]
        assert all(r < 3 or r >= R - 3 for r in unforg_rows)


# ---------------------------------------------------------------------------
# ODE-oracle revalidation (PASS_REF_BUG) — for models where subprocess
# NFsim is the buggy reference and bngsim is correct.
# ---------------------------------------------------------------------------

def _write_seg_gdat(seed_dir, stem, suffix, arr, n_obs):
    """Write <stem>_<suffix>.gdat into a seed dir."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_gdat(seed_dir / f"{stem}_{suffix}.gdat", arr, n_obs)


class TestOdeOracleRevalidation:
    ENTRY = {"ode_suffix": "A_ODE", "nf_suffix": "B_NFsim",
             "issue": "TEST#1", "reason": "test"}

    def _setup(self, tmp_path, ode_obs, nf_seed_obs):
        """ode_obs: (T,K) ODE oracle observable block.
        nf_seed_obs: list of (T,K) per-seed NF observable blocks."""
        t = np.linspace(0, 10, ode_obs.shape[0])
        sub_dir = tmp_path / "sub" / "seed1"
        _write_seg_gdat(sub_dir, "m", "A_ODE",
                        np.column_stack([t, ode_obs]), ode_obs.shape[1])
        bng_dirs = []
        for i, obs in enumerate(nf_seed_obs, 1):
            d = tmp_path / "bng" / f"seed{i}"
            _write_seg_gdat(d, "m", "B_NFsim",
                            np.column_stack([t, obs]), obs.shape[1])
            bng_dirs.append(str(d))
        return [str(sub_dir)], bng_dirs

    def test_nf_tracking_ode_passes(self, tmp_path):
        rng = np.random.default_rng(0)
        T, K, N = 50, 2, 10
        base = np.column_stack([100 * np.exp(-np.linspace(0, 3, T)),
                                50 + np.linspace(0, 10, T)])
        ode = base.copy()
        # NF ensemble = ODE + small per-seed noise -> mean tracks ODE.
        nf = [base + rng.normal(0, 1.0, base.shape) for _ in range(N)]
        sub_dirs, bng_dirs = self._setup(tmp_path, ode, nf)
        status, details = pd.revalidate_nf_against_ode(
            sub_dirs, bng_dirs, "m", self.ENTRY)
        assert status == "pass", details
        assert details["frac_pass"] >= pd.ENSEMBLE_PASS_FRAC

    def test_nf_50x_off_ode_fails(self, tmp_path):
        # The buggy-subprocess scenario: NF mean is ~50x the ODE value on
        # one observable. A divergence this large must never pass.
        T, K, N = 50, 2, 10
        ode = np.column_stack([np.full(T, 1.0), np.full(T, 5.0)])
        nf = [np.column_stack([np.full(T, 50.0), np.full(T, 5.0)])
              for _ in range(N)]
        sub_dirs, bng_dirs = self._setup(tmp_path, ode, nf)
        status, details = pd.revalidate_nf_against_ode(
            sub_dirs, bng_dirs, "m", self.ENTRY)
        assert status == "diff", details

    def test_missing_ode_oracle_fails(self, tmp_path):
        # bng has NF output but sub has no ODE oracle file.
        T = 50
        t = np.linspace(0, 10, T)
        bng_dir = tmp_path / "bng" / "seed1"
        _write_seg_gdat(bng_dir, "m", "B_NFsim",
                        np.column_stack([t, np.ones((T, 2))]), 2)
        empty_sub = tmp_path / "sub" / "seed1"
        empty_sub.mkdir(parents=True)
        status, details = pd.revalidate_nf_against_ode(
            [str(empty_sub)], [str(bng_dir)], "m", self.ENTRY)
        assert status == "diff"
        assert "not found" in details.get("reason", "")

    def test_small_count_finite_size_offset_passes(self, tmp_path):
        # ode_vs_nf_discrepancy-like: a small-count observable where the NF
        # ensemble mean (~0.5) sits below the ODE value (0.67) by a finite-
        # size systematic offset (~25%) — within ODE_ORACLE_REL and nowhere
        # near the 50x subprocess divergence. Low per-seed scatter (the
        # escalated 50-seed regime) so the offset, not sampling, is tested.
        rng = np.random.default_rng(3)
        T, N = 30, 50
        ode = np.column_stack([np.full(T, 0.67), np.full(T, 5.4)])
        nf = [np.column_stack([0.5 + rng.normal(0, 0.02, T),
                               5.4 + rng.normal(0, 0.05, T)]) for _ in range(N)]
        sub_dirs, bng_dirs = self._setup(tmp_path, ode, nf)
        status, details = pd.revalidate_nf_against_ode(
            sub_dirs, bng_dirs, "m", self.ENTRY)
        assert status == "pass", details
        # The 0.67-vs-0.5 offset (~0.17) is forgiven by ODE_ORACLE_REL,
        # not by the sigma test (scatter is tiny here).
        assert 0.17 <= pd.ODE_ORACLE_REL * 0.67


class TestSubprocessNfUnreliableRegistry:
    def test_entries_well_formed(self):
        assert pd.SUBPROCESS_NF_UNRELIABLE
        for stem, entry in pd.SUBPROCESS_NF_UNRELIABLE.items():
            for key in ("ode_suffix", "nf_suffix", "issue", "reason"):
                assert key in entry, f"{stem} missing {key}"
                assert isinstance(entry[key], str) and entry[key]


class TestRevalidateAgainstAnalytic:
    """The analytic-accept path for nf-only models whose subprocess reference
    is wrong and which have no ODE segment (SUBPROCESS_NF_ANALYTIC). bngsim is
    accepted iff its ensemble-mean final row matches the documented values."""

    ENTRY = {"suffix": "", "issue": "X", "reason": "y",
             "expect": {"A_tot": [100.0, 5.0], "B_tot": [500.0, 60.0]}}

    def _bng(self, tmp_path, final_A, final_B, n=10, jitter=0.0):
        rng = np.random.default_rng(0)
        T = 51
        t = np.linspace(0, 50, T)
        dirs = []
        for i in range(1, n + 1):
            d = tmp_path / "bng" / f"seed{i}"
            d.mkdir(parents=True)
            A = np.linspace(100, final_A, T) + rng.normal(0, jitter, T)
            B = np.linspace(0, final_B, T) + rng.normal(0, jitter, T)
            _write_gdat_named(d / "m.gdat", ["time", "A_tot", "B_tot"],
                              np.column_stack([t, A, B]))
            dirs.append(str(d))
        return dirs

    def test_matches_analytic_passes(self, tmp_path):
        # bngsim final A_tot=100, B_tot=496 -> within tol of 100/500.
        dirs = self._bng(tmp_path, 100.0, 496.0, jitter=0.3)
        status, det = pd.revalidate_against_analytic(dirs, "m", self.ENTRY)
        assert status == "pass", det
        assert det["checks"]["A_tot"]["pass"] and det["checks"]["B_tot"]["pass"]

    def test_ignored_clamp_fails(self, tmp_path):
        # the '$ ignored' failure: A_tot->0.8, B_tot->99 -> far outside tol.
        dirs = self._bng(tmp_path, 0.8, 99.0)
        status, det = pd.revalidate_against_analytic(dirs, "m", self.ENTRY)
        assert status == "diff", det
        assert not det["checks"]["A_tot"]["pass"]

    def test_missing_column_fails(self, tmp_path):
        d = tmp_path / "bng" / "seed1"
        d.mkdir(parents=True)
        _write_gdat_named(d / "m.gdat", ["time", "A_tot"],
                          np.column_stack([np.linspace(0, 50, 5),
                                           np.full(5, 100.0)]))
        status, det = pd.revalidate_against_analytic([str(d)], "m", self.ENTRY)
        assert status == "diff"
        assert det["checks"]["B_tot"]["reason"] == "column not found"

    def test_no_outputs_fails(self, tmp_path):
        empty = tmp_path / "bng" / "seed1"
        empty.mkdir(parents=True)
        status, det = pd.revalidate_against_analytic([str(empty)], "m", self.ENTRY)
        assert status == "diff"
        assert "no bngsim outputs" in det["reason"]

    def test_registry_well_formed(self):
        for stem, entry in pd.SUBPROCESS_NF_ANALYTIC.items():
            assert "suffix" in entry and isinstance(entry["suffix"], str)
            assert entry.get("reason") and entry.get("issue")
            assert entry["expect"]
            for col, spec in entry["expect"].items():
                assert len(spec) == 2 and spec[1] >= 0


# ---------------------------------------------------------------------------
# Cosmetic header/column normalization — bare-vs-() function headers and
# omitted _rateLaw<digits> columns (bngsim's method-independent schema).
# See parity_diff._normalize_columns / _align_columns and PyBNF-Private#58.
# ---------------------------------------------------------------------------

def _write_gdat_named(path, names, arr):
    """Write a BNG-style .gdat with an explicit header token list."""
    path = Path(path)
    with open(path, "w") as f:
        f.write("# " + "  ".join(names) + "\n")
        for row in arr:
            f.write(" ".join(f"{v: .12e}" for v in row) + "\n")


class TestColumnNormalizationUnits:
    def test_read_columns_strips_hash(self, tmp_path):
        p = tmp_path / "m.gdat"
        _write_gdat_named(p, ["time", "A", "kf()"], np.zeros((2, 3)))
        assert pd._read_columns(p) == ["time", "A", "kf()"]

    def test_canon_drops_trailing_parens_only(self):
        assert pd._canon("kf_BSA()") == "kf_BSA"
        assert pd._canon("kf_BSA") == "kf_BSA"
        # not a trailing (): leave intact
        assert pd._canon("f(x)") == "f(x)"

    def test_normalize_drops_ratelaw_and_canonicalizes(self):
        names = ["time", "A", "_rateLaw2", "kf()", "_rateLaw10"]
        data = np.arange(10).reshape(2, 5).astype(float)
        out, out_names = pd._normalize_columns(data, names)
        assert out_names == ["time", "A", "kf"]
        # columns 0,1,3 kept (the two _rateLaw* dropped)
        np.testing.assert_array_equal(out, data[:, [0, 1, 3]])

    def test_normalize_drops_ratelaw_with_paren_suffix(self):
        # BNG2.pl's NFsim path ()-suffixes every function column, including
        # synthetic intermediates: _rateLaw1() must still be dropped. (The
        # lone v8 straggler, "AD 3 State FREE Expanding nfs".)
        names = ["time", "A", "kf()", "_rateLaw1()"]
        data = np.arange(8).reshape(2, 4).astype(float)
        out, out_names = pd._normalize_columns(data, names)
        assert out_names == ["time", "A", "kf"]
        np.testing.assert_array_equal(out, data[:, [0, 1, 2]])

    def test_normalize_keeps_userfunc_not_matching_ratelaw(self):
        # _rateLaw without trailing digits, or a user obs, is NOT dropped.
        names = ["time", "_rateLaw", "rateLaw3", "myLaw2"]
        data = np.zeros((1, 4))
        _, out_names = pd._normalize_columns(data, names)
        assert out_names == ["time", "_rateLaw", "rateLaw3", "myLaw2"]

    def test_normalize_header_width_mismatch_falls_back(self):
        data = np.zeros((1, 3))
        out, out_names = pd._normalize_columns(data, ["time", "A"])
        assert out_names is None and out.shape == (1, 3)

    def test_align_reorders_when_sets_equal(self):
        sub = np.array([[1.0, 2.0, 3.0]])
        bng = np.array([[1.0, 3.0, 2.0]])  # B and A swapped
        s, b = pd._align_columns(sub, ["time", "A", "B"],
                                 bng, ["time", "B", "A"])
        np.testing.assert_array_equal(b, [[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(s, sub)

    def test_align_noop_on_real_schema_difference(self):
        sub = np.zeros((1, 3))
        bng = np.zeros((1, 3))
        # Different real column present -> left for shape/value check.
        s, b = pd._align_columns(sub, ["time", "A", "B"],
                                 bng, ["time", "A", "C"])
        np.testing.assert_array_equal(b, bng)


class TestDeterministicCosmeticNormalization:
    def test_bare_vs_parens_and_ratelaw_passes(self, tmp_pair):
        """BNG2.pl ODE side carries _rateLaw cols + ()-headers; bngsim is
        bare and omits _rateLaw. Identical values -> PASS, not a shape DIFF.
        """
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        a, b, kf = np.sin(t), np.cos(t), 0.5 * t
        # bngsim (sub): bare headers, no _rateLaw.
        _write_gdat_named(sub_dir / "m.gdat",
                          ["time", "A", "B", "kf"],
                          np.column_stack([t, a, b, kf]))
        # BNG2.pl (bng): () on function header + interspersed _rateLaw cols.
        _write_gdat_named(bng_dir / "m.gdat",
                          ["time", "A", "_rateLaw2", "B", "kf()", "_rateLaw3"],
                          np.column_stack([t, a, 99 * t, b, kf, 7 * t]))
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details

    def test_real_diff_still_caught_after_normalization(self, tmp_pair):
        """Normalization must not mask a genuine divergence in a kept col."""
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 101)
        _write_gdat_named(sub_dir / "m.gdat", ["time", "A", "kf"],
                          np.column_stack([t, np.sin(t), 0.5 * t]))
        _write_gdat_named(bng_dir / "m.gdat",
                          ["time", "A", "kf()", "_rateLaw2"],
                          np.column_stack([t, 1.1 * np.sin(t), 0.5 * t, t]))
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"

    def test_extra_real_column_still_fails(self, tmp_pair):
        """A real (non-_rateLaw) extra column is a schema divergence: DIFF."""
        sub_dir, bng_dir = tmp_pair
        t = np.linspace(0, 10, 51)
        _write_gdat_named(sub_dir / "m.gdat", ["time", "A"],
                          np.column_stack([t, np.sin(t)]))
        _write_gdat_named(bng_dir / "m.gdat", ["time", "A", "C"],
                          np.column_stack([t, np.sin(t), np.cos(t)]))
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"


class TestScanCosmeticNormalization:
    """`.scan` files flow through the same deterministic loop as `.gdat`
    (NUM_EXTENSIONS), so the ()/_rateLaw normalization must cover them too.
    A `.scan` is a 2D table: column 0 is the swept *parameter* (not time),
    remaining columns are each observable/function at t_end. Verifies the
    param column stays put under name-based alignment.
    """

    def test_scan_bare_vs_parens_and_ratelaw_passes(self, tmp_pair):
        sub_dir, bng_dir = tmp_pair
        kf = np.linspace(0.1, 1.0, 25)          # swept parameter (col 0)
        a, b, func = np.sqrt(kf), kf ** 2, 3 * kf
        # bngsim: bare headers, no _rateLaw, param col first.
        _write_gdat_named(sub_dir / "m.scan",
                          ["kf", "A", "B", "func"],
                          np.column_stack([kf, a, b, func]))
        # BNG2.pl: ()-function header + interspersed _rateLaw cols.
        _write_gdat_named(bng_dir / "m.scan",
                          ["kf", "A", "_rateLaw2", "B", "func()"],
                          np.column_stack([kf, a, 42 * kf, b, func]))
        status, details = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "pass", details

    def test_scan_real_param_divergence_still_fails(self, tmp_pair):
        # A genuine difference in the swept-parameter column 0 must flag,
        # not be absorbed by normalization.
        sub_dir, bng_dir = tmp_pair
        kf = np.linspace(0.1, 1.0, 25)
        _write_gdat_named(sub_dir / "m.scan", ["kf", "A", "func"],
                          np.column_stack([kf, np.sqrt(kf), 3 * kf]))
        _write_gdat_named(bng_dir / "m.scan",
                          ["kf", "A", "func()", "_rateLaw2"],
                          np.column_stack([1.5 * kf, np.sqrt(kf), 3 * kf, kf]))
        status, _ = pd.deterministic_compare(sub_dir, bng_dir)
        assert status == "diff"
