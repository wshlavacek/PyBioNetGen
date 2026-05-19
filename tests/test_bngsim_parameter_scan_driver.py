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
    _write_bifurcation_file,
)


class _Action:
    """Minimal stand-in for a parsed BNGL action."""

    def __init__(self, action_type, action_args=None):
        self.type = action_type
        self.name = action_type
        self.args = action_args or {}


def _make_request(**overrides):
    """Build a ScanRequest with sensible defaults for the spacing tests."""
    fields = dict(
        action="parameter_scan", parameter="k", par_min=0.0, par_max=100.0,
        n_scan_pts=5, log_scale=False, method="ode", t_start=0.0, t_end=1.0,
        n_steps=1, suffix=None, prefix=None, reset_conc=True, seed=None,
        atol=None, rtol=None, print_cdat=True, print_functions=False,
    )
    fields.update(overrides)
    return ScanRequest(**fields)


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


def _bifurcate_action(**overrides):
    args = {
        "parameter": '"k1"',
        "par_min": "1.0",
        "par_max": "100.0",
        "n_scan_pts": "10",
        "log_scale": "1",
        "method": '"ode"',
        "t_start": "0",
        "t_end": "100",
        "n_steps": "10",
    }
    args.update(overrides)
    return _Action("bifurcate", args)


def _valid_bifurcate_sequence(**overrides):
    return [
        _Action("generate_network", {"overwrite": "1"}),
        _bifurcate_action(**overrides),
    ]


# ─── scan_values spacing ───────────────────────────────────────────


def test_scan_values_linear_spacing_inclusive_endpoints():
    vals = scan_values(_make_request(par_min=0.0, par_max=100.0, n_scan_pts=5))
    assert vals == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])


def test_scan_values_log_spacing_is_geometric_and_inclusive():
    vals = scan_values(_make_request(
        par_min=1e-3, par_max=1e3, n_scan_pts=7, log_scale=True,
    ))
    assert vals[0] == pytest.approx(1e-3)
    assert vals[-1] == pytest.approx(1e3)
    # geometric: a constant ratio between successive points
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
    assert ratios == pytest.approx([10.0] * 6)


def test_scan_values_single_point():
    assert scan_values(
        _make_request(par_min=5.0, par_max=5.0, n_scan_pts=1)
    ) == [5.0]


def test_scan_values_backward_pass_is_reversed_forward():
    # bifurcate's backward pass swaps par_min/par_max -> its value list is
    # exactly the forward list reversed (log or linear).
    fwd = scan_values(_make_request(par_min=1.0, par_max=1e2, n_scan_pts=10,
                                    log_scale=True))
    bwd = scan_values(_make_request(par_min=1e2, par_max=1.0, n_scan_pts=10,
                                    log_scale=True))
    assert bwd == pytest.approx(list(reversed(fwd)))


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
    assert req.action == "parameter_scan" and req.method == "ode"


def test_detect_accepts_ssa_method_and_seed():
    req = detect_inprocess_scan(_valid_sequence(method='"ssa"', seed="17"))
    assert req is not None
    assert req.method == "ssa"
    assert req.seed == 17
    # absent seed -> None (the driver then uses BNGsim's default)
    assert detect_inprocess_scan(_valid_sequence(method='"ssa"')).seed is None


def test_detect_accepts_bifurcate():
    req = detect_inprocess_scan(_valid_bifurcate_sequence())
    assert req is not None
    assert req.action == "bifurcate"
    # bifurcate always carries the prior point's end state
    assert req.reset_conc is False


def test_detect_bifurcate_ignores_reset_conc_arg():
    # BNG2.pl forces reset_conc=>0 for bifurcate regardless of any value
    # the user wrote.
    req = detect_inprocess_scan(_valid_bifurcate_sequence(reset_conc="1"))
    assert req is not None and req.reset_conc is False


def test_detect_accepts_ssa_bifurcate():
    req = detect_inprocess_scan(_valid_bifurcate_sequence(method='"ssa"'))
    assert req is not None and req.action == "bifurcate" and req.method == "ssa"


def test_detect_declines_bifurcate_with_steady_state():
    # ExampleModel4_v6's bifurcate carries steady_state=>1 (bngsim #47).
    assert detect_inprocess_scan(
        _valid_bifurcate_sequence(steady_state="1")
    ) is None


def test_detect_declines_scan_plus_bifurcate():
    # exactly one workflow action is allowed
    seq = [
        _Action("generate_network", {"overwrite": "1"}),
        _scan_action(),
        _bifurcate_action(),
    ]
    assert detect_inprocess_scan(seq) is None


# ─── detect_inprocess_scan: decline / fallback ─────────────────────


def test_detect_declines_empty_actions():
    assert detect_inprocess_scan(None) is None
    assert detect_inprocess_scan([]) is None


def test_detect_declines_unsupported_method():
    for method in ('"nf"', '"pla"', '"psa"'):
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


def test_detect_accepts_reset_conc_zero():
    # Phase 2: reset_conc=>0 (carry the prior point's end state) is now
    # supported for parameter_scan.
    req = detect_inprocess_scan(_valid_sequence(reset_conc="0"))
    assert req is not None and req.reset_conc is False
    # reset_conc=>1 stays the default-equivalent.
    assert detect_inprocess_scan(
        _valid_sequence(reset_conc="1")
    ).reset_conc is True


def test_detect_parses_print_functions():
    # print_functions is honored in-process — BNGL functions go into the
    # .gdat/.scan from Result.expressions, as the backend hook already does.
    assert detect_inprocess_scan(
        _valid_sequence(print_functions="1")
    ).print_functions is True
    assert detect_inprocess_scan(
        _valid_sequence(print_functions="0")
    ).print_functions is False
    # absent => default off, matching BNG2.pl.
    assert detect_inprocess_scan(_valid_sequence()).print_functions is False


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


def test_write_bifurcation_file_matches_bng2_format(tmp_path):
    bif_path = tmp_path / "m_bifurcation_A.scan"
    # forward column: ascending parameter axis; backward column the same
    # axis scanned in reverse (so backward[N-1-i] aligns with forward[i]).
    fwd_col = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
    bwd_col = [(3.0, 33.0), (2.0, 22.0), (1.0, 11.0)]
    _write_bifurcation_file(str(bif_path), "Kxy", "A", fwd_col, bwd_col)
    lines = bif_path.read_text().splitlines()
    assert lines[0] == "# " + f"{'Kxy':>14}" + " " + f"{'A_fwd':>16}" \
        + " " + f"{'A_bwd':>16}"
    # row i pairs forward[i] with backward[N-1-i]
    assert lines[1] == f"{1.0:16.8e} {10.0:16.8e} {11.0:16.8e}"
    assert lines[2] == f"{2.0:16.8e} {20.0:16.8e} {22.0:16.8e}"
    assert lines[3] == f"{3.0:16.8e} {30.0:16.8e} {33.0:16.8e}"
    assert len(lines) == 4


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


# A model with a functions block, scanned with print_functions=>1: the
# .gdat/.scan must carry the two BNGL function columns after the two
# observables.
_FUNC_SCAN_MODEL = """\
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
begin functions
  frac_B() B_tot/(A_tot+B_tot+1)
  total_AB() A_tot+B_tot
end functions
begin reaction rules
  A() -> B() k1
  B() -> 0 kdeg
end reaction rules
end model

generate_network({overwrite=>1})
parameter_scan({parameter=>"A0_scale",par_min=>1,par_max=>100,\
n_scan_pts=>5,log_scale=>1,method=>"ode",t_start=>0,t_end=>20,\
n_steps=>10,print_functions=>1})
"""


def _scan_header(path):
    for ln in open(path):
        if ln.startswith("#"):
            return ln.lstrip("#").split()
    return []


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
def test_print_functions_scan_includes_function_columns(tmp_path, caplog):
    import bionetgen

    model = tmp_path / "funcs.bngl"
    model.write_text(_FUNC_SCAN_MODEL)

    fast_out = tmp_path / "fast"
    ref_out = tmp_path / "ref"
    with caplog.at_level(logging.INFO, logger="bionetgen.bngsim_bridge"):
        bionetgen.run(str(model), out=str(fast_out))
    bionetgen.run(str(model), out=str(ref_out), simulator="subprocess")

    # print_functions no longer declines the fast path.
    assert any("fast path" in rec.message for rec in caplog.records)

    scan_name = "funcs_A0_scale.scan"
    # the .scan carries both observables and both BNGL functions, in the
    # same order and with the same column names as BNG2.pl.
    fast_header = _scan_header(fast_out / scan_name)
    assert fast_header == _scan_header(ref_out / scan_name)
    assert fast_header == [
        "A0_scale", "A_tot", "B_tot", "frac_B", "total_AB",
    ]

    fast = _load_scan(fast_out / scan_name)
    ref = _load_scan(ref_out / scan_name)
    assert fast.shape == ref.shape == (5, 5)
    denom = (abs(fast) + abs(ref)).clip(min=1e-12)
    assert (abs(fast - ref) / denom).max() < 1e-4

    # the per-point .gdat carries the function columns too
    # (time + 2 observables + 2 functions).
    work = fast_out / "funcs_A0_scale"
    gdat = sorted(p for p in work.iterdir() if p.suffix == ".gdat")[0]
    assert _scan_header(gdat) == [
        "time", "A_tot", "B_tot", "frac_B", "total_AB",
    ]


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


# ─── Phase 2: ssa parameter_scan ───────────────────────────────────

# An ssa scan of A0_scale (feeds the A() initial count via A0 =
# A0_scale*100): the log-spaced scan points yield fractional A0 values,
# which the driver must round to integers for SSA (bngsim issue #43).
_SSA_SCAN_ACTION = (
    'parameter_scan({parameter=>"A0_scale",par_min=>1,par_max=>100,'
    'n_scan_pts=>5,log_scale=>1,method=>"ssa",t_start=>0,t_end=>20,'
    'n_steps=>10,seed=>1234})'
)


@pytest.mark.skipif(not BNGSIM_AVAILABLE, reason="BNGsim not installed")
def test_ssa_scan_fast_path_is_taken_and_reproducible(tmp_path, caplog):
    import bionetgen

    model = tmp_path / "tinyssa.bngl"
    model.write_text(_TINY_SCAN_MODEL % _SSA_SCAN_ACTION)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    with caplog.at_level(logging.INFO, logger="bionetgen.bngsim_bridge"):
        bionetgen.run(str(model), out=str(out_a))
    bionetgen.run(str(model), out=str(out_b))

    # the in-process fast path ran the ssa scan
    assert any("fast path" in rec.message and "ssa" in rec.message
               for rec in caplog.records)

    scan_name = "tinyssa_A0_scale.scan"
    a = _load_scan(out_a / scan_name)
    b = _load_scan(out_b / scan_name)
    assert a.shape == b.shape == (5, 3)
    # same seed -> byte-reproducible across runs
    assert (a == b).all()
    # SSA observables are molecule counts: integer-valued (the parameter
    # column may be fractional from the log spacing).
    counts = a[:, 1:]
    assert (counts == counts.round()).all()

    # per-point .gdat artifacts exist
    work = out_a / "tinyssa_A0_scale"
    gdats = [p for p in work.iterdir() if p.suffix == ".gdat"]
    assert len(gdats) == 5


# ─── Phase 2: bifurcate ────────────────────────────────────────────

# A reversible A<->B model that does not fully equilibrate within
# t_end, so reset_conc=>0 carry-over genuinely matters (forward and
# backward passes differ). bifurcate scans the rate constant k1.
_BIFURCATE_MODEL = """\
begin model
begin parameters
  k1 1.0
  ktot 100
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() ktot
  B() 0
end seed species
begin observables
  Molecules A_tot A()
  Molecules B_tot B()
end observables
begin reaction rules
  A() <-> B() k1, k1
end reaction rules
end model

generate_network({overwrite=>1})
bifurcate({parameter=>"k1",par_min=>0.1,par_max=>10,n_scan_pts=>8,\
log_scale=>1,method=>"ode",t_start=>0,t_end=>5,n_steps=>5})
"""


@pytest.mark.skipif(not BNGSIM_AVAILABLE, reason="BNGsim not installed")
def test_bifurcate_fast_path_runs_and_matches_subprocess(tmp_path, caplog):
    import bionetgen

    model = tmp_path / "bif.bngl"
    model.write_text(_BIFURCATE_MODEL)

    fast_out = tmp_path / "fast"
    ref_out = tmp_path / "ref"
    with caplog.at_level(logging.INFO, logger="bionetgen.bngsim_bridge"):
        bionetgen.run(str(model), out=str(fast_out))
    bionetgen.run(str(model), out=str(ref_out), simulator="subprocess")

    assert any("bifurcate fast path" in rec.message for rec in caplog.records)

    # one _bifurcation_<obs>.scan per observable, matching subprocess
    for obs in ("A_tot", "B_tot"):
        name = f"bif_bifurcation_{obs}.scan"
        fast = _load_scan(fast_out / name)
        ref = _load_scan(ref_out / name)
        assert fast.shape == ref.shape == (8, 3)
        denom = (abs(fast) + abs(ref)).clip(min=1e-12)
        assert (abs(fast - ref) / denom).max() < 1e-4

    # per-point .gdat artifacts under the forward/backward workdirs
    for sub in ("bif_forward", "bif_backward"):
        gdats = [p for p in (fast_out / sub).iterdir() if p.suffix == ".gdat"]
        assert len(gdats) == 8

    # the intermediate forward/backward .scan files are not left behind
    assert not (fast_out / "bif_forward.scan").exists()
    assert not (fast_out / "bif_backward.scan").exists()
