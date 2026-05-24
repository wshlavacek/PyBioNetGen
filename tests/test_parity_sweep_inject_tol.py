"""Unit tests for parity_sweep.inject_tol — the per-model ODE tolerance
override injected into both stacks (see TOL_OVERRIDES). Covers clean
insertion, replacement of pre-existing atol/rtol (no duplicate keys),
comment lines, the no-op passthrough, and the registry contract.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


def _load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "parity_sweep.py"
    spec = importlib.util.spec_from_file_location("parity_sweep", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ps = _load()
TOL = {"atol": "1e-12", "rtol": "1e-12"}


def _params_of(line):
    """Return the dict of key=>value tokens inside the first action block."""
    blob = re.search(r"\{(.*)\}", line).group(1)
    out = {}
    for tok in blob.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, v = tok.split("=>")
        out[k.strip()] = v.strip()
    return out


class TestInjectTol:
    def test_clean_insert_when_absent(self):
        line = 'simulate({method=>"ode",t_end=>100,n_steps=>200})\n'
        p = _params_of(ps.inject_tol(line, TOL))
        assert p["atol"] == "1e-12" and p["rtol"] == "1e-12"
        # original params preserved
        assert p["method"] == '"ode"' and p["t_end"] == "100"

    def test_replaces_existing_without_duplicates(self):
        line = 'simulate({method=>"ode",atol=>1e-8,rtol=>1e-8,t_end=>100})\n'
        out = ps.inject_tol(line, TOL)
        # exactly one atol and one rtol token, at the override value
        assert out.count("atol=>") == 1 and out.count("rtol=>") == 1
        p = _params_of(out)
        assert p["atol"] == "1e-12" and p["rtol"] == "1e-12"

    def test_replaces_partial_existing(self):
        line = 'simulate({t_end=>50,method=>"ode",rtol=>1e-6})\n'
        out = ps.inject_tol(line, TOL)
        assert out.count("rtol=>") == 1
        p = _params_of(out)
        assert p["atol"] == "1e-12" and p["rtol"] == "1e-12"
        assert p["t_end"] == "50"

    def test_no_comma_artifacts(self):
        # No empty tokens / doubled commas / dangling comma before brace.
        for line in [
            'simulate({method=>"ode",atol=>1e-8,rtol=>1e-8})\n',
            'simulate({atol=>1e-8,rtol=>1e-8,method=>"ode"})\n',
        ]:
            out = ps.inject_tol(line, TOL)
            assert ",," not in out
            assert "{," not in out
            assert re.search(r",\s*\}", out) is None

    def test_comment_line_untouched(self):
        line = '#simulate({method=>"ode",t_end=>100})\n'
        assert ps.inject_tol(line, TOL) == line

    def test_none_override_is_passthrough(self):
        line = 'simulate({method=>"ode",t_end=>100})\n'
        assert ps.inject_tol(line, None) == line

    def test_parameter_scan_action_patched(self):
        line = ('parameter_scan({method=>"ode",t_end=>10,parameter=>"k",'
                'par_min=>1,par_max=>2,n_scan_pts=>3,log_scale=>0})\n')
        p = _params_of(ps.inject_tol(line, TOL))
        assert p["atol"] == "1e-12" and p["rtol"] == "1e-12"

    def test_registry_values_well_formed(self):
        assert ps.TOL_OVERRIDES
        for name, tol in ps.TOL_OVERRIDES.items():
            assert name.endswith(".bngl")
            assert set(tol) == {"atol", "rtol"}
            for v in tol.values():
                float(v)  # parseable numeric string
