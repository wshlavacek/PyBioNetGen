"""Tests for BNGResult, BNGNotebook, and bngsim_bridge pure-function helpers."""

import os
import math

import numpy as np
import pytest

from bionetgen.core.tools.result import BNGResult
from bionetgen.core.notebook import BNGNotebook
from bionetgen.core.tools.bngsim_bridge import (
    _strip_quotes,
    _safe_math_namespace,
    _eval_numeric,
    _parse_net_species_initializers,
    _resolve_scan_points,
    _write_scan_file,
    _write_bng_dat,
    _actions_need_network,
    _actions_need_xml,
    _extract_positional_args,
    _BNG2PL_ACTIONS,
    _SIMULATE_METHOD_MAP,
)
from bionetgen.modelapi.structs import Action


# ── Helpers ───────────────────────────────────────────────────────────


def _write_gdat(path, header_cols, rows):
    """Write a minimal gdat/cdat file for testing."""
    with open(path, "w") as f:
        f.write("# " + " ".join(header_cols) + "\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


# =====================================================================
# BNGResult tests
# =====================================================================


class TestBNGResultDirectPath:
    def test_loads_file_and_sets_attrs(self, tmp_path):
        gdat = tmp_path / "model.gdat"
        _write_gdat(gdat, ["time", "A", "B"], [[0, 1, 2], [1, 3, 4]])

        res = BNGResult(direct_path=str(gdat))

        assert res.file_name == "model"
        assert res.file_extension == ".gdat"
        assert "model" in res.gnames
        assert "model" in res.gdats
        arr = res.gdats["model"]
        assert arr.dtype.names == ("time", "A", "B")
        np.testing.assert_allclose(arr["A"], [1.0, 3.0])

    def test_stores_direct_path(self, tmp_path):
        gdat = tmp_path / "x.gdat"
        _write_gdat(gdat, ["time", "C"], [[0, 5]])
        res = BNGResult(direct_path=str(gdat))
        assert res.direct_path == str(gdat)


class TestBNGResultPath:
    def test_finds_and_loads_all_types(self, tmp_path):
        _write_gdat(tmp_path / "m1.gdat", ["time", "X"], [[0, 1]])
        _write_gdat(tmp_path / "m2.cdat", ["time", "Y"], [[0, 2]])
        _write_gdat(tmp_path / "m3.scan", ["p", "Z"], [[0.1, 3]])

        res = BNGResult(path=str(tmp_path))

        assert "m1" in res.gdats
        assert "m2" in res.cdats
        assert "m3" in res.scans
        np.testing.assert_allclose(res.gdats["m1"]["X"], [1.0])
        np.testing.assert_allclose(res.cdats["m2"]["Y"], [2.0])
        np.testing.assert_allclose(res.scans["m3"]["Z"], [3.0])


class TestBNGResultNeitherPath:
    def test_no_crash_empty_dicts(self):
        res = BNGResult()
        assert res.gdats == {}
        assert res.cdats == {}
        assert res.scans == {}


class TestBNGResultRepr:
    def test_repr_counts(self, tmp_path):
        _write_gdat(tmp_path / "a.gdat", ["time", "X"], [[0, 1]])
        _write_gdat(tmp_path / "b.gdat", ["time", "Y"], [[0, 2]])
        _write_gdat(tmp_path / "c.cdat", ["time", "Z"], [[0, 3]])
        _write_gdat(tmp_path / "d.scan", ["p", "W"], [[0, 4]])

        res = BNGResult(path=str(tmp_path))
        r = repr(res)

        assert "gdats from 2 models" in r
        assert "cdats from 1 models" in r
        assert "scans from 1 models" in r


class TestBNGResultGetitem:
    def test_int_key(self, tmp_path):
        _write_gdat(tmp_path / "m.gdat", ["time", "A"], [[0, 10]])
        res = BNGResult(path=str(tmp_path))
        arr = res[0]
        np.testing.assert_allclose(arr["A"], [10.0])

    def test_string_key(self, tmp_path):
        _write_gdat(tmp_path / "m.gdat", ["time", "A"], [[0, 10]])
        res = BNGResult(path=str(tmp_path))
        arr = res["m"]
        np.testing.assert_allclose(arr["A"], [10.0])


class TestBNGResultIter:
    def test_iterates_gdat_keys(self, tmp_path):
        _write_gdat(tmp_path / "alpha.gdat", ["time", "X"], [[0, 1]])
        _write_gdat(tmp_path / "beta.gdat", ["time", "Y"], [[0, 2]])
        res = BNGResult(path=str(tmp_path))
        keys = list(res)
        assert set(keys) == {"alpha", "beta"}


class TestBNGResultLoad:
    def test_gdat_dispatch(self, tmp_path):
        p = tmp_path / "f.gdat"
        _write_gdat(p, ["time", "V"], [[0, 7]])
        res = BNGResult()
        arr = res.load(str(p))
        assert arr.dtype.names == ("time", "V")

    def test_cdat_dispatch(self, tmp_path):
        p = tmp_path / "f.cdat"
        _write_gdat(p, ["time", "S"], [[0, 9]])
        res = BNGResult()
        arr = res.load(str(p))
        assert arr.dtype.names == ("time", "S")

    def test_scan_dispatch(self, tmp_path):
        p = tmp_path / "f.scan"
        _write_gdat(p, ["p", "O"], [[1, 2]])
        res = BNGResult()
        arr = res.load(str(p))
        assert arr is not None

    def test_unknown_ext_returns_none(self, tmp_path):
        p = tmp_path / "f.xyz"
        p.write_text("junk")
        res = BNGResult()
        assert res.load(str(p)) is None


class TestBNGResultLoadDat:
    def test_reads_recarray_correctly(self, tmp_path):
        p = tmp_path / "data.gdat"
        _write_gdat(p, ["time", "A", "B"], [[0, 1.5, 2.5], [1, 3.5, 4.5]])
        res = BNGResult()
        arr = res._load_dat(str(p))
        assert isinstance(arr, np.recarray)
        assert arr.dtype.names == ("time", "A", "B")
        np.testing.assert_allclose(arr["time"], [0.0, 1.0])
        np.testing.assert_allclose(arr["B"], [2.5, 4.5])


class TestBNGResultFindDatFiles:
    def test_populates_names(self, tmp_path):
        (tmp_path / "x.gdat").write_text("# time\n0\n")
        (tmp_path / "y.cdat").write_text("# time\n0\n")
        (tmp_path / "z.scan").write_text("# p\n0\n")
        (tmp_path / "w.txt").write_text("ignore")

        res = BNGResult()
        res.path = str(tmp_path)
        res.find_dat_files()

        assert "x" in res.gnames
        assert "y" in res.cnames
        assert "z" in res.snames
        assert len(res.gnames) == 1
        assert len(res.cnames) == 1
        assert len(res.snames) == 1


class TestBNGResultLoadResults:
    def test_loads_all_found(self, tmp_path):
        _write_gdat(tmp_path / "g.gdat", ["time", "A"], [[0, 1]])
        _write_gdat(tmp_path / "c.cdat", ["time", "B"], [[0, 2]])
        _write_gdat(tmp_path / "s.scan", ["p", "C"], [[0, 3]])

        res = BNGResult()
        res.path = str(tmp_path)
        res.find_dat_files()
        res.load_results()

        assert "g" in res.gdats
        assert "c" in res.cdats
        assert "s" in res.scans


# =====================================================================
# BNGNotebook tests
# =====================================================================


class TestBNGNotebook:
    def test_init_stores_template_and_kwargs(self):
        nb = BNGNotebook("tmpl.ipynb", FOO="bar", BAZ="qux")
        assert nb.template == "tmpl.ipynb"
        assert nb.odict == {"FOO": "bar", "BAZ": "qux"}

    def test_write_substitutes_keywords(self, tmp_path):
        tmpl = tmp_path / "template.txt"
        tmpl.write_text("Hello GREETING from PLACE\nBye GREETING\n")
        outfile = tmp_path / "output.txt"

        nb = BNGNotebook(str(tmpl), GREETING="world", PLACE="earth")
        nb.write(str(outfile))

        content = outfile.read_text()
        assert "Hello world from earth" in content
        assert "Bye world" in content
        assert "GREETING" not in content

    def test_write_no_kwargs_copies_file(self, tmp_path):
        tmpl = tmp_path / "t.txt"
        tmpl.write_text("unchanged line\n")
        out = tmp_path / "o.txt"

        nb = BNGNotebook(str(tmpl))
        nb.write(str(out))

        assert out.read_text() == "unchanged line\n"


# =====================================================================
# bngsim_bridge pure-function tests
# =====================================================================


class TestStripQuotes:
    def test_double_quotes(self):
        assert _strip_quotes('"hello"') == "hello"

    def test_single_quotes(self):
        assert _strip_quotes("'hello'") == "hello"

    def test_no_quotes(self):
        assert _strip_quotes("hello") == "hello"

    def test_empty_string(self):
        assert _strip_quotes("") == ""

    def test_single_char(self):
        assert _strip_quotes("x") == "x"

    def test_mismatched_quotes(self):
        assert _strip_quotes("\"hello'") == "\"hello'"


class TestSafeMathNamespace:
    def test_contains_math_functions(self):
        ns = _safe_math_namespace()
        assert "exp" in ns
        assert "log" in ns
        assert "sqrt" in ns
        assert "sin" in ns
        assert "pi" in ns
        assert callable(ns["exp"])
        assert ns["pi"] == math.pi

    def test_extra_params_added(self):
        ns = _safe_math_namespace(extra={"k": 42.0})
        assert ns["k"] == 42.0
        # math functions still present
        assert "exp" in ns

    def test_builtins_restricted(self):
        ns = _safe_math_namespace()
        assert ns["__builtins__"] == {}


class TestEvalNumeric:
    def test_plain_float(self):
        assert _eval_numeric("3.14") == pytest.approx(3.14)

    def test_arithmetic_expression(self):
        assert _eval_numeric("2 + 3 * 4") == pytest.approx(14.0)

    def test_math_expression(self):
        assert _eval_numeric("exp(0)") == pytest.approx(1.0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot evaluate"):
            _eval_numeric("undefined_var_xyz")

    def test_with_extra_ns(self):
        result = _eval_numeric("k * 2", extra_ns={"k": 5.0})
        assert result == pytest.approx(10.0)


class TestParseNetSpeciesInitializers:
    def test_parses_species_block(self, tmp_path):
        net = tmp_path / "model.net"
        net.write_text(
            "begin species\n"
            "  1 A(b) 5000\n"
            "  2 B(a) k_init*100\n"
            "end species\n"
        )
        result = _parse_net_species_initializers(str(net))
        assert len(result) == 2
        assert result[0] == ("A(b)", "5000")
        assert result[1] == ("B(a)", "k_init*100")

    def test_missing_file_returns_empty(self, tmp_path):
        result = _parse_net_species_initializers(str(tmp_path / "nonexistent.net"))
        assert result == []


class TestResolveScanPoints:
    def test_with_par_scan_vals(self):
        args = {"par_scan_vals": "[1.0, 2.0, 3.0]"}
        pts = _resolve_scan_points(args)
        np.testing.assert_allclose(pts, [1.0, 2.0, 3.0])

    def test_with_linspace(self):
        args = {"par_min": "0", "par_max": "10", "n_scan_pts": "5"}
        pts = _resolve_scan_points(args)
        np.testing.assert_allclose(pts, np.linspace(0, 10, 5))

    def test_with_log_scale(self):
        args = {"par_min": "1", "par_max": "1000", "n_scan_pts": "4", "log_scale": "1"}
        pts = _resolve_scan_points(args)
        np.testing.assert_allclose(pts, np.logspace(0, 3, 4))


class TestWriteScanFile:
    def test_writes_correct_format(self, tmp_path):
        out = tmp_path / "test.scan"
        rows = [
            [0.1, 1.0, 2.0],
            [0.2, 3.0, 4.0],
        ]
        _write_scan_file(str(out), "kf", ["obsA", "obsB"], rows)

        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert "kf" in lines[0]
        assert "obsA" in lines[0]
        assert "obsB" in lines[0]
        assert len(lines) == 3  # header + 2 data rows
        # Data rows should be parseable as floats
        vals = lines[1].split()
        assert len(vals) == 3
        assert float(vals[0]) == pytest.approx(0.1)


class TestWriteBngDat:
    def test_writes_correct_format(self, tmp_path):
        out = tmp_path / "test.gdat"
        time = np.array([0.0, 1.0, 2.0])
        data = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
        _write_bng_dat(str(out), time, data, ["A", "B"])

        lines = out.read_text().strip().split("\n")
        assert lines[0].startswith("#")
        assert "time" in lines[0]
        assert "A" in lines[0]
        assert "B" in lines[0]
        assert len(lines) == 4  # header + 3 data rows
        # Parse first data row
        vals = [float(v) for v in lines[1].split()]
        assert len(vals) == 3
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(10.0)
        assert vals[2] == pytest.approx(20.0)


class TestActionsNeedNetwork:
    def test_simulate_ode_needs_network(self):
        a = Action("simulate_ode", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_network([a]) is True

    def test_simulate_nf_still_returns_true(self):
        # _actions_need_network returns True by default (fallthrough)
        a = Action("simulate_nf", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_network([a]) is True


class TestActionsNeedXml:
    def test_simulate_nf_needs_xml(self):
        a = Action("simulate_nf", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_xml([a]) is True

    def test_writeXML_needs_xml(self):
        a = Action("writeXML", {})
        assert _actions_need_xml([a]) is True

    def test_simulate_ode_does_not_need_xml(self):
        a = Action("simulate_ode", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_xml([a]) is False


class TestExtractPositionalArgs:
    def test_extracts_name_and_value(self):
        a = Action("setParameter", {'"kf"': None, "1.0": None})
        name, value = _extract_positional_args(a)
        assert name == "kf"
        assert value == "1.0"

    def test_strips_quotes(self):
        a = Action("setParameter", {"'kf'": None, "'2.5'": None})
        name, value = _extract_positional_args(a)
        assert name == "kf"
        assert value == "2.5"


class TestBNG2PLActions:
    def test_is_frozenset(self):
        assert isinstance(_BNG2PL_ACTIONS, frozenset)

    def test_expected_members(self):
        assert "generate_network" in _BNG2PL_ACTIONS
        assert "writeXML" in _BNG2PL_ACTIONS
        assert "writeSBML" in _BNG2PL_ACTIONS
        assert "readFile" in _BNG2PL_ACTIONS
        assert "visualize" in _BNG2PL_ACTIONS
        assert "setModelName" in _BNG2PL_ACTIONS


class TestSimulateMethodMap:
    def test_maps_correctly(self):
        assert _SIMULATE_METHOD_MAP["simulate"] == "ode"
        assert _SIMULATE_METHOD_MAP["simulate_ode"] == "ode"
        assert _SIMULATE_METHOD_MAP["simulate_ssa"] == "ssa"
        assert _SIMULATE_METHOD_MAP["simulate_psa"] == "psa"
        assert _SIMULATE_METHOD_MAP["simulate_nf"] == "nf"
        assert _SIMULATE_METHOD_MAP["simulate_pla"] == "pla"
