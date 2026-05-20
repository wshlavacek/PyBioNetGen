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
    def test_floor_for_short_trajectory(self):
        assert pd._time_tol(np.linspace(0, 10, 11)) == pd.TIME_TOL_FLOOR

    def test_scales_with_long_trajectory(self):
        # t_max=1e6: tolerance = 1e-12 * 1e6 = 1e-6, well above the floor.
        t = np.linspace(0, 1e6, 1001)
        assert pd._time_tol(t) == pytest.approx(1e-6, rel=1e-9)

    def test_empty_returns_floor(self):
        assert pd._time_tol(np.array([])) == pd.TIME_TOL_FLOOR
        assert pd._time_tol(None) == pd.TIME_TOL_FLOOR

    def test_all_nan_returns_floor(self):
        assert pd._time_tol(np.array([np.nan, np.nan])) == pd.TIME_TOL_FLOOR

    def test_uses_max_not_mean(self):
        # mostly zeros plus one large t: scale uses the max.
        t = np.array([0.0, 0.0, 0.0, 1e9])
        assert pd._time_tol(t) == pytest.approx(1e-3, rel=1e-9)


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
