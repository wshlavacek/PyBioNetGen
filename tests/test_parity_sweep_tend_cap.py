"""Unit tests for parity_sweep's runtime-trimming hooks used to keep the
candidate-corpus "glacial" tier (slow NFsim/SSA models) within the parity
budget — all applied identically to both stacks:

  * _cap_tend / TEND_OVERRIDES — CAP (not set) semantics: only t_end values
    EXCEEDING the cap are reduced, so a long equilibration phase shrinks while
    a short main run is left alone.
  * _set_nscanpts / NSCANPTS_OVERRIDES — reduce parameter_scan point count.
  * ACTION_INJECT — append a run action to a model that ships without one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "parity_sweep.py"
    spec = importlib.util.spec_from_file_location("parity_sweep", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ps = _load()


class TestCapTend:
    def test_reduces_value_above_cap(self):
        assert ps._cap_tend("t_end=>200", 30) == "t_end=>30"

    def test_leaves_value_at_or_below_cap_untouched(self):
        assert ps._cap_tend("t_end=>50", 60) == "t_end=>50"
        assert ps._cap_tend("t_end=>60", 60) == "t_end=>60"

    def test_evaluates_arithmetic_expression(self):
        # BNGL allows expressions; 10*60=600 > 60 -> capped
        assert ps._cap_tend("x t_end=>10*60,y", 60) == "x t_end=>60,y"

    def test_scientific_notation(self):
        assert ps._cap_tend("t_end=>5e3", 1800) == "t_end=>1800"
        assert ps._cap_tend("t_end=>1.5e3", 1800) == "t_end=>1.5e3"  # 1500 < 1800

    def test_none_cap_is_noop(self):
        assert ps._cap_tend("t_end=>200", None) == "t_end=>200"

    def test_two_phase_caps_each_occurrence_independently(self):
        # Long equilibration (25.2) gets capped; a 1.5 main run also exceeds a
        # cap of 1.0, so both land at the cap — but a phase below the cap would
        # be preserved (covered above). The key property is per-occurrence.
        line = (
            'simulate({suffix=>"equil",t_end=>25.2013,n_steps=>1});'
            'simulate({suffix=>"main",t_end=>1.5,n_steps=>12})'
        )
        out = ps._cap_tend(line, 5)
        assert 'suffix=>"equil",t_end=>5,n_steps=>1' in out  # 25.2 -> 5
        assert 'suffix=>"main",t_end=>1.5,n_steps=>12' in out  # 1.5 < 5, kept


class TestPatchBnglCap:
    def test_seed_and_cap_together(self):
        out = ps.patch_bngl('simulate({method=>"nf",t_end=>650})', seed=7, tend_override=100)
        assert "seed=>7," in out
        assert "t_end=>100" in out

    def test_comment_lines_untouched(self):
        out = ps.patch_bngl('#simulate({method=>"nf",t_end=>650})\n', seed=1, tend_override=100)
        assert out == '#simulate({method=>"nf",t_end=>650})\n'

    def test_tend_only_path_caps_without_seed(self):
        out = ps.patch_bngl_tend_only('simulate({method=>"ode",t_end=>300})', 50)
        assert "seed" not in out and "t_end=>50" in out


class TestNScanPts:
    def test_override_reduces_points(self):
        line = 'parameter_scan({method=>"nf",n_scan_pts=>18,t_end=>10})'
        out = ps._set_nscanpts(line, 6)
        assert "n_scan_pts=>6" in out

    def test_none_is_noop(self):
        line = "parameter_scan({n_scan_pts=>18})"
        assert ps._set_nscanpts(line, None) == line

    def test_combined_with_cap_in_patch(self):
        out = ps.patch_bngl_tend_only("parameter_scan({t_end=>10,n_scan_pts=>18})", 5, 6)
        assert "t_end=>5" in out and "n_scan_pts=>6" in out


class TestRegistries:
    def test_tend_overrides_values_numeric(self):
        for k, v in ps.TEND_OVERRIDES.items():
            assert isinstance(k, str) and k.endswith(".bngl")
            assert isinstance(v, (int, float)) and v > 0

    def test_action_inject_appends_runnable_action(self):
        for k, v in ps.ACTION_INJECT.items():
            assert k.endswith(".bngl")
            assert "simulate" in v
