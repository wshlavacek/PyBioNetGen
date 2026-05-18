"""Tests for the in-process BNGsim parameter_scan fast-path driver.

These cover the pure (non-BNGsim) machinery: scan-value spacing, action
detection / fallback triggers, and ``.scan`` file formatting. The
end-to-end in-process driver is exercised by the parity sweeps.
"""

import logging

import pytest

from bionetgen.core.tools.bngsim_bridge import BNGSIM_AVAILABLE
from bionetgen.core.tools.bngsim_parameter_scan import (
    ScanRequest,
    detect_inprocess_scan,
    scan_values,
    _write_scan_file,
)


class _Action:
    """Minimal stand-in for a parsed BNGL action."""

    def __init__(self, action_type, action_args=None):
        self.type = action_type
        self.name = action_type
        self.args = action_args or {}


def _scan_action(**overrides):
    args = {
        "parameter": '"k1"',
        "par_min": "0.1",
        "par_max": "10",
        "n_scan_pts": "5",
        "log_scale": "1",
        "method": '"ode"',
        "t_start": "0",
        "t_end": "100",
        "n_steps": "10",
    }
    args.update(overrides)
    return _Action("parameter_scan", args)


def _valid_sequence(**scan_overrides):
    return [
        _Action("generate_network", {"overwrite": "1"}),
        _scan_action(**scan_overrides),
    ]


# ─── scan_values spacing ───────────────────────────────────────────


def test_scan_values_linear_spacing_inclusive_endpoints():
    req = ScanRequest(
        parameter="k", par_min=0.0, par_max=100.0, n_scan_pts=5,
        log_scale=False, t_start=0.0, t_end=1.0, n_steps=1,
        suffix=None, prefix=None, reset_conc=True, atol=None, rtol=None,
        print_cdat=True,
    )
    vals = scan_values(req)
    assert vals == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])


def test_scan_values_log_spacing_is_geometric_and_inclusive():
    req = ScanRequest(
        parameter="k", par_min=1e-3, par_max=1e3, n_scan_pts=7,
        log_scale=True, t_start=0.0, t_end=1.0, n_steps=1,
        suffix=None, prefix=None, reset_conc=True, atol=None, rtol=None,
        print_cdat=True,
    )
    vals = scan_values(req)
    assert vals[0] == pytest.approx(1e-3)
    assert vals[-1] == pytest.approx(1e3)
    # geometric: a constant ratio between successive points
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
    assert ratios == pytest.approx([10.0] * 6)


def test_scan_values_single_point():
    req = ScanRequest(
        parameter="k", par_min=5.0, par_max=5.0, n_scan_pts=1,
        log_scale=False, t_start=0.0, t_end=1.0, n_steps=1,
        suffix=None, prefix=None, reset_conc=True, atol=None, rtol=None,
        print_cdat=True,
    )
    assert scan_values(req) == [5.0]


# ─── detect_inprocess_scan: accept ─────────────────────────────────


def test_detect_accepts_generate_network_plus_ode_scan():
    req = detect_inprocess_scan(_valid_sequence())
    assert req is not None
    assert req.parameter == "k1"
    assert req.par_min == 0.1 and req.par_max == 10.0
    assert req.n_scan_pts == 5
    assert req.log_scale is True
    assert req.t_start == 0.0 and req.t_end == 100.0
    assert req.n_steps == 10


def test_detect_accepts_setparameter_preamble():
    seq = [
        _Action("setParameter", {"name": '"k2"', "value": "3"}),
        _Action("generate_network", {"overwrite": "1"}),
        _scan_action(),
    ]
    assert detect_inprocess_scan(seq) is not None


def test_detect_accepts_cvode_method_and_absent_method():
    assert detect_inprocess_scan(_valid_sequence(method='"cvode"')) is not None
    seq = _valid_sequence()
    del seq[-1].args["method"]
    assert detect_inprocess_scan(seq) is not None


def test_detect_parses_optional_options():
    req = detect_inprocess_scan(_valid_sequence(
        suffix='"scn"', prefix='"pfx"', atol="1e-9", rtol="1e-7",
        reset_conc="1", print_CDAT="0",
    ))
    assert req.suffix == "scn" and req.prefix == "pfx"
    assert req.atol == 1e-9 and req.rtol == 1e-7
    assert req.print_cdat is False


# ─── detect_inprocess_scan: decline / fallback ─────────────────────


def test_detect_declines_empty_actions():
    assert detect_inprocess_scan(None) is None
    assert detect_inprocess_scan([]) is None


def test_detect_declines_non_ode_method():
    for method in ('"ssa"', '"nf"', '"pla"', '"psa"'):
        assert detect_inprocess_scan(_valid_sequence(method=method)) is None


def test_detect_declines_par_scan_vals():
    seq = _valid_sequence()
    seq[-1].args["par_scan_vals"] = "[1,2,3]"
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_sample_times():
    seq = _valid_sequence()
    seq[-1].args["sample_times"] = "[0,1,2]"
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_steady_state():
    assert detect_inprocess_scan(_valid_sequence(steady_state="1")) is None
    # steady_state=>0 is harmless and does not block the fast path
    assert detect_inprocess_scan(_valid_sequence(steady_state="0")) is not None


def test_detect_declines_reset_conc_zero():
    assert detect_inprocess_scan(_valid_sequence(reset_conc="0")) is None


def test_detect_declines_print_functions():
    assert detect_inprocess_scan(_valid_sequence(print_functions="1")) is None
    assert detect_inprocess_scan(_valid_sequence(print_functions="0")) is not None


def test_detect_declines_unknown_option():
    seq = _valid_sequence()
    seq[-1].args["mystery_option"] = "1"
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_extra_simulate_action():
    seq = [
        _Action("generate_network", {"overwrite": "1"}),
        _Action("simulate", {"method": '"ode"', "t_end": "10"}),
        _scan_action(),
    ]
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_scan_not_last():
    seq = [
        _Action("generate_network", {"overwrite": "1"}),
        _scan_action(),
        _Action("setParameter", {"name": '"k"', "value": "1"}),
    ]
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_missing_generate_network():
    assert detect_inprocess_scan([_scan_action()]) is None


def test_detect_declines_generate_network_after_scan():
    seq = [_scan_action(), _Action("generate_network", {"overwrite": "1"})]
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_multiple_scans():
    seq = [
        _Action("generate_network", {"overwrite": "1"}),
        _scan_action(),
        _scan_action(),
    ]
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_missing_required_arg():
    seq = _valid_sequence()
    del seq[-1].args["t_end"]
    assert detect_inprocess_scan(seq) is None


def test_detect_declines_log_scale_with_nonpositive_range():
    assert detect_inprocess_scan(
        _valid_sequence(par_min="0", log_scale="1")
    ) is None


def test_detect_ignores_harmless_parallel_options():
    req = detect_inprocess_scan(_valid_sequence(parallel="1", num_cores="4"))
    assert req is not None


def test_detect_declines_stray_backslash_in_action():
    # PyBioNetGen absorbs the stray "\" before log_scale; BNG2.pl treats
    # "\log_scale" as an unrecognized key. The fast path must defer.
    malformed = (
        'begin model\nend model\n\n'
        'generate_network({overwrite=>1})\n'
        'parameter_scan({parameter=>"RT",par_min=>1e3,par_max=>1e6,\\\n'
        'n_scan_pts=>101,\\log_scale=>1,method=>"ode",\\\n'
        't_start=>0,t_end=>300,n_steps=>31})\n'
    )
    assert detect_inprocess_scan(_valid_sequence(), bngl_text=malformed) is None


def test_detect_accepts_clean_line_continuations():
    # A backslash that is a genuine end-of-line continuation is fine.
    clean = (
        'begin model\nend model\n\n'
        'generate_network({overwrite=>1})\n'
        'parameter_scan({parameter=>"k1",par_min=>0.1,par_max=>10,\\\n'
        'n_scan_pts=>5,log_scale=>1,method=>"ode",\\\n'
        't_start=>0,t_end=>100,n_steps=>10})\n'
    )
    assert detect_inprocess_scan(_valid_sequence(), bngl_text=clean) is not None


# ─── .scan file formatting ─────────────────────────────────────────


def test_write_scan_file_matches_bng2_format(tmp_path):
    scan_path = tmp_path / "model_k1.scan"
    rows = [
        (1.0e-3, [1.234e3, 5.0e1]),
        (1.0e3, [9.876e8, 2.0e4]),
    ]
    _write_scan_file(str(scan_path), "k1", ["obs_a", "obs_b"], rows)
    lines = scan_path.read_text().splitlines()
    # header: "# " + param right-justified 14, then each obs right-just 16
    assert lines[0] == "# " + f"{'k1':>14}" + " " + f"{'obs_a':>16}" \
        + " " + f"{'obs_b':>16}"
    # data rows: %16.8e fields, single-space separated
    assert lines[1] == f"{1.0e-3:16.8e}" + " " + f"{1.234e3:16.8e}" \
        + " " + f"{5.0e1:16.8e}"
    assert len(lines) == 3


def test_write_scan_file_roundtrips_numerically(tmp_path):
    scan_path = tmp_path / "rt.scan"
    rows = [(float(i), [float(i) * 2, float(i) * 3]) for i in range(4)]
    _write_scan_file(str(scan_path), "p", ["x", "y"], rows)
    parsed = [
        [float(tok) for tok in ln.split()]
        for ln in scan_path.read_text().splitlines()
        if not ln.startswith("#")
    ]
    for (pv, obs), parsed_row in zip(rows, parsed):
        assert parsed_row[0] == pytest.approx(pv)
        assert parsed_row[1:] == pytest.approx(obs)


# ─── end-to-end (BNGsim + BNG2.pl required) ────────────────────────

# A tiny model whose scanned parameter (A0_scale) feeds an initial
# concentration through A0 = A0_scale*100 -- exercising the per-point
# set_concentration propagation the in-process driver must do itself.
_TINY_SCAN_MODEL = """\
begin model
begin parameters
  k1 0.5
  A0_scale 10
  A0 A0_scale*100
  kdeg 0.1
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() A0
  B() 0
end seed species
begin observables
  Molecules A_tot A()
  Molecules B_tot B()
end observables
begin reaction rules
  A() -> B() k1
  B() -> 0 kdeg
end reaction rules
end model

generate_network({overwrite=>1})
%s
"""

_FAST_SCAN_ACTION = (
    'parameter_scan({parameter=>"A0_scale",par_min=>1,par_max=>100,'
    'n_scan_pts=>5,log_scale=>1,method=>"ode",t_start=>0,t_end=>20,n_steps=>10})'
)
# par_scan_vals is out of Phase 1 scope -> the detector declines and the
# backend-hook route runs it instead.
_FALLBACK_SCAN_ACTION = (
    'parameter_scan({parameter=>"A0_scale",par_scan_vals=>[1,10,100],'
    'method=>"ode",t_start=>0,t_end=>20,n_steps=>10})'
)


def _load_scan(path):
    import numpy as np

    return np.array([
        [float(tok) for tok in ln.split()]
        for ln in open(path)
        if ln.strip() and not ln.startswith("#")
    ])


@pytest.mark.skipif(not BNGSIM_AVAILABLE, reason="BNGsim not installed")
def test_fast_path_runs_in_process_and_matches_subprocess(tmp_path, caplog):
    import bionetgen

    model = tmp_path / "tiny.bngl"
    model.write_text(_TINY_SCAN_MODEL % _FAST_SCAN_ACTION)

    fast_out = tmp_path / "fast"
    ref_out = tmp_path / "ref"
    with caplog.at_level(logging.INFO, logger="bionetgen.bngsim_bridge"):
        bionetgen.run(str(model), out=str(fast_out))
    bionetgen.run(str(model), out=str(ref_out), simulator="subprocess")

    # the in-process fast path was taken (not the backend-hook route)
    assert any("fast path" in rec.message for rec in caplog.records)

    scan_name = "tiny_A0_scale.scan"
    fast = _load_scan(fast_out / scan_name)
    ref = _load_scan(ref_out / scan_name)
    assert fast.shape == ref.shape == (5, 3)
    denom = (abs(fast) + abs(ref)).clip(min=1e-12)
    assert (abs(fast - ref) / denom).max() < 1e-4

    # per-point .gdat artifacts exist under the scan working directory
    work = fast_out / "tiny_A0_scale"
    gdats = sorted(p.name for p in work.iterdir() if p.suffix == ".gdat")
    assert len(gdats) == 5


@pytest.mark.skipif(not BNGSIM_AVAILABLE, reason="BNGsim not installed")
def test_unsupported_scan_option_falls_back_and_still_runs(tmp_path):
    import bionetgen

    model = tmp_path / "tiny_fb.bngl"
    model.write_text(_TINY_SCAN_MODEL % _FALLBACK_SCAN_ACTION)

    fb_out = tmp_path / "fb"
    ref_out = tmp_path / "fb_ref"
    bionetgen.run(str(model), out=str(fb_out))
    bionetgen.run(str(model), out=str(ref_out), simulator="subprocess")

    scan_name = "tiny_fb_A0_scale.scan"
    assert (fb_out / scan_name).is_file()
    fb = _load_scan(fb_out / scan_name)
    ref = _load_scan(ref_out / scan_name)
    assert fb.shape == ref.shape == (3, 3)
    denom = (abs(fb) + abs(ref)).clip(min=1e-12)
    assert (abs(fb - ref) / denom).max() < 1e-4
