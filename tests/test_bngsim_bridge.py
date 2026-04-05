"""Tests for the BNGsim bridge module.

Unit tests for format detection run without BNGsim installed.
Integration tests are skipped if BNGsim is not available.
"""

import os
import tempfile

import pytest

from bionetgen.core.exc import BNGFormatError, BNGSimError
from bionetgen.core.tools.bngsim_bridge import (
    BNGSIM_AVAILABLE,
    BNGSIM_HAS_NFSIM,
    BNGSIM_VERSION,
    FORMAT_ANTIMONY,
    FORMAT_BNG_XML,
    FORMAT_BNGL,
    FORMAT_NET,
    FORMAT_SBML,
    _normalize_method,
    _parse_protocol_block,
    _parse_simulate_params,
    _parse_table_functions,
    _resolve_sample_times,
    _sniff_xml_format,
    detect_input_format,
    run_with_bngsim,
)

tfold = os.path.dirname(__file__)


# ─── Format detection: extension-based ─────────────────────────────


class TestFormatDetectionByExtension:
    def test_bngl(self):
        assert detect_input_format("model.bngl") == FORMAT_BNGL

    def test_net(self):
        assert detect_input_format("model.net") == FORMAT_NET

    def test_antimony(self):
        assert detect_input_format("model.ant") == FORMAT_ANTIMONY

    def test_bngl_with_path(self):
        assert detect_input_format("/some/path/to/model.bngl") == FORMAT_BNGL

    def test_unknown_extension(self):
        with pytest.raises(BNGFormatError, match="Unrecognized file extension"):
            detect_input_format("model.txt")

    def test_no_extension(self):
        with pytest.raises(BNGFormatError, match="Unrecognized file extension"):
            detect_input_format("model")


# ─── Format detection: XML sniffing ────────────────────────────────


class TestXMLSniffing:
    def _write_xml(self, content):
        f = tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False
        )
        f.write(content)
        f.close()
        return f.name

    def test_sbml_detected(self):
        path = self._write_xml(
            '<?xml version="1.0"?>\n'
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core">\n'
            '  <model id="test"/>\n'
            "</sbml>"
        )
        try:
            assert _sniff_xml_format(path) == FORMAT_SBML
            assert detect_input_format(path) == FORMAT_SBML
        finally:
            os.unlink(path)

    def test_bng_xml_detected(self):
        path = self._write_xml(
            '<?xml version="1.0"?>\n'
            "<Model>\n"
            "  <ListOfMoleculeTypes>\n"
            "  </ListOfMoleculeTypes>\n"
            "</Model>"
        )
        try:
            assert _sniff_xml_format(path) == FORMAT_BNG_XML
            assert detect_input_format(path) == FORMAT_BNG_XML
        finally:
            os.unlink(path)

    def test_bng_xml_with_sbml_wrapper(self):
        """BNG XML that also has an <sbml> tag should be detected as BNG XML."""
        path = self._write_xml(
            '<?xml version="1.0"?>\n'
            "<sbml>\n"
            "  <Model>\n"
            "    <ListOfMoleculeTypes/>\n"
            "  </Model>\n"
            "</sbml>"
        )
        try:
            assert _sniff_xml_format(path) == FORMAT_BNG_XML
        finally:
            os.unlink(path)

    def test_ambiguous_xml_raises(self):
        path = self._write_xml(
            '<?xml version="1.0"?>\n<root><data/></root>'
        )
        try:
            with pytest.raises(BNGFormatError, match="Could not determine"):
                detect_input_format(path)
        finally:
            os.unlink(path)

    def test_nonexistent_xml_raises(self):
        with pytest.raises(BNGFormatError, match="Could not read file"):
            detect_input_format("/nonexistent/path/model.xml")

    def test_bng_xml_with_observables(self):
        path = self._write_xml(
            '<?xml version="1.0"?>\n'
            "<Model>\n"
            "  <ListOfObservables/>\n"
            "</Model>"
        )
        try:
            assert _sniff_xml_format(path) == FORMAT_BNG_XML
        finally:
            os.unlink(path)


# ─── Explicit format flag ──────────────────────────────────────────


class TestExplicitFormat:
    def test_explicit_bngl(self):
        assert detect_input_format("model.bngl", explicit_format="bngl") == FORMAT_BNGL

    def test_explicit_overrides_for_xml(self):
        """Explicit format for XML skips sniffing when file doesn't exist,
        but we test with a real SBML file."""
        f = tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False
        )
        f.write('<sbml xmlns="http://www.sbml.org/"><model/></sbml>')
        f.close()
        try:
            assert (
                detect_input_format(f.name, explicit_format="sbml") == FORMAT_SBML
            )
        finally:
            os.unlink(f.name)

    def test_explicit_conflicts_with_autodetect(self):
        """Saying --format=bng-xml on an SBML file should raise."""
        f = tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False
        )
        f.write(
            '<sbml xmlns="http://www.sbml.org/sbml/level3">'
            "<model/></sbml>"
        )
        f.close()
        try:
            with pytest.raises(BNGFormatError, match="Format conflict"):
                detect_input_format(f.name, explicit_format="bng-xml")
        finally:
            os.unlink(f.name)

    def test_explicit_conflicts_with_extension(self):
        with pytest.raises(BNGFormatError, match="Format conflict"):
            detect_input_format("model.bngl", explicit_format="sbml")

    def test_unknown_explicit_format(self):
        with pytest.raises(BNGFormatError, match="Unknown format"):
            detect_input_format("model.xml", explicit_format="foobar")

    def test_explicit_case_insensitive(self):
        assert detect_input_format("model.bngl", explicit_format="BNGL") == FORMAT_BNGL


# ─── Availability flags ────────────────────────────────────────────


class TestAvailabilityFlags:
    def test_bngsim_available_is_bool(self):
        assert isinstance(BNGSIM_AVAILABLE, bool)

    def test_bngsim_has_nfsim_is_bool(self):
        assert isinstance(BNGSIM_HAS_NFSIM, bool)

    def test_version_matches_availability(self):
        if BNGSIM_AVAILABLE:
            assert BNGSIM_VERSION is not None
        else:
            assert BNGSIM_VERSION is None


# ─── Public API exposure ───────────────────────────────────────────


class TestPublicAPI:
    def test_available_in_bionetgen_namespace(self):
        import bionetgen

        assert hasattr(bionetgen, "BNGSIM_AVAILABLE")
        assert hasattr(bionetgen, "BNGSIM_VERSION")

    def test_run_signature(self):
        import inspect
        import bionetgen

        sig = inspect.signature(bionetgen.run)
        params = list(sig.parameters.keys())
        assert "simulator" in params
        assert "format" in params
        assert "method" in params
        assert "t_span" in params
        assert "n_points" in params


# ─── Routing logic (no BNGsim needed) ─────────────────────────────


class TestRoutingWithoutBngsim:
    def test_sbml_without_bngsim_raises(self):
        """SBML format should raise if BNGsim is not available."""
        import unittest.mock as mock
        f = tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False
        )
        f.write('<sbml xmlns="http://www.sbml.org/"><model/></sbml>')
        f.close()
        try:
            with mock.patch("bionetgen.core.tools.bngsim_bridge.BNGSIM_AVAILABLE", False):
                with pytest.raises(BNGSimError, match="BNGsim is required"):
                    run_with_bngsim(f.name, "/tmp/out", fmt=FORMAT_SBML)
        finally:
            os.unlink(f.name)

    def test_antimony_without_bngsim_raises(self):
        """Antimony format should raise if BNGsim is not available."""
        import unittest.mock as mock
        with mock.patch("bionetgen.core.tools.bngsim_bridge.BNGSIM_AVAILABLE", False):
            with pytest.raises(BNGSimError, match="BNGsim is required"):
                run_with_bngsim("model.ant", "/tmp/out", fmt=FORMAT_ANTIMONY)


# ─── Integration tests (require BNGsim) ───────────────────────────


@pytest.mark.skipif(
    not BNGSIM_AVAILABLE,
    reason="BNGsim not installed",
)
class TestBngsimIntegration:
    def test_run_net_file(self):
        """Run a .net file through BNGsim and verify output files."""
        net_file = os.path.join(tfold, "test", "test.net")
        if not os.path.isfile(net_file):
            pytest.skip("test.net fixture not found")
        with tempfile.TemporaryDirectory() as out:
            result = run_with_bngsim(
                net_file,
                out,
                fmt=FORMAT_NET,
                method="ode",
                t_span=(0, 10),
                n_points=11,
            )
            assert result is not None
            assert result.process_return == 0
            # Check that output files were created
            files = os.listdir(out)
            assert any(f.endswith(".gdat") or f.endswith(".cdat") for f in files)

    def test_run_via_library_api(self):
        """Run a .net file via bionetgen.run() with simulator='bngsim'."""
        import bionetgen

        net_file = os.path.join(tfold, "test", "test.net")
        if not os.path.isfile(net_file):
            pytest.skip("test.net fixture not found")
        with tempfile.TemporaryDirectory() as out:
            result = bionetgen.run(
                net_file,
                out=out,
                simulator="bngsim",
                format="net",
                method="ode",
                t_span=(0, 10),
                n_points=11,
            )
            assert result is not None

    def test_bngsim_version_reported(self):
        """BNGsim version should be a non-empty string."""
        assert BNGSIM_VERSION is not None
        assert len(BNGSIM_VERSION) > 0


# ─── sample_times resolution ─────────────────────────────────────


class TestResolveSampleTimes:
    def test_basic_list_string(self):
        result = _resolve_sample_times({"sample_times": "[1,5,10,20,50]"})
        assert result == [1.0, 5.0, 10.0, 20.0, 50.0]

    def test_sorts_unordered(self):
        result = _resolve_sample_times({"sample_times": "[50,1,10,5,20]"})
        assert result == [1.0, 5.0, 10.0, 20.0, 50.0]

    def test_returns_none_when_absent(self):
        assert _resolve_sample_times({}) is None

    def test_returns_none_for_empty_string(self):
        assert _resolve_sample_times({"sample_times": "[]"}) is None

    def test_returns_none_for_too_few_points(self):
        assert _resolve_sample_times({"sample_times": "[1,2]"}) is None

    def test_n_steps_takes_precedence(self):
        """n_steps should suppress sample_times (BNG2.pl compat)."""
        result = _resolve_sample_times({
            "sample_times": "[1,5,10,20,50]",
            "n_steps": "100",
        })
        assert result is None

    def test_n_output_steps_takes_precedence(self):
        result = _resolve_sample_times({
            "sample_times": "[1,5,10,20,50]",
            "n_output_steps": "50",
        })
        assert result is None

    def test_t_end_appended_when_larger(self):
        result = _resolve_sample_times({
            "sample_times": "[1,5,10]",
            "t_end": "100",
        })
        assert result == [1.0, 5.0, 10.0, 100.0]

    def test_t_end_not_appended_when_smaller(self):
        result = _resolve_sample_times({
            "sample_times": "[1,5,100]",
            "t_end": "50",
        })
        assert result == [1.0, 5.0, 100.0]

    def test_list_input(self):
        """Also accept a Python list (not just a string)."""
        result = _resolve_sample_times({"sample_times": [1, 5, 10, 20, 50]})
        assert result == [1.0, 5.0, 10.0, 20.0, 50.0]

    def test_none_value(self):
        assert _resolve_sample_times({"sample_times": None}) is None


# ─── Protocol block parsing ──────────────────────────────────────


class TestParseProtocolBlock:
    def _write_bngl(self, content):
        f = tempfile.NamedTemporaryFile(
            suffix=".bngl", mode="w", delete=False
        )
        f.write(content)
        f.close()
        return f.name

    def test_extracts_protocol_lines(self):
        path = self._write_bngl(
            "begin model\nend model\n"
            "begin protocol\n"
            '  simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
            '  setConcentration("A",100)\n'
            '  simulate({method=>"ode",t_end=>20,n_steps=>10,continue=>1})\n'
            "end protocol\n"
            "parameter_scan({method=>\"protocol\"})\n"
        )
        try:
            lines = _parse_protocol_block(path)
            assert len(lines) == 3
            assert "simulate" in lines[0]
            assert "setConcentration" in lines[1]
            assert "continue" in lines[2]
        finally:
            os.unlink(path)

    def test_empty_when_no_protocol(self):
        path = self._write_bngl(
            "begin model\nend model\n"
            "simulate({method=>\"ode\",t_end=>10})\n"
        )
        try:
            lines = _parse_protocol_block(path)
            assert lines == []
        finally:
            os.unlink(path)

    def test_skips_comments_inside_protocol(self):
        path = self._write_bngl(
            "begin protocol\n"
            "# this is a comment\n"
            '  simulate({method=>"ode",t_end=>10,n_steps=>10})\n'
            "end protocol\n"
        )
        try:
            lines = _parse_protocol_block(path)
            # Comment lines are included in raw lines; _run_protocol skips them
            assert len(lines) == 2
            assert lines[0].strip().startswith("#")
            assert "simulate" in lines[1]
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        lines = _parse_protocol_block("/nonexistent/path/model.bngl")
        assert lines == []


# ─── Method normalization (SSA/PSA) ──────────────────────────────


class TestNormalizeMethod:
    def test_ode_unchanged(self):
        assert _normalize_method("ode") == ("ode", None)

    def test_ssa_unchanged_without_poplevel(self):
        assert _normalize_method("ssa") == ("ssa", None)

    def test_ssa_promoted_to_psa_with_poplevel(self):
        """BNG2.pl compat: ssa + poplevel → psa."""
        method, poplevel = _normalize_method("ssa", poplevel=200.0)
        assert method == "psa"
        assert poplevel == 200.0

    def test_psa_direct(self):
        method, poplevel = _normalize_method("psa", poplevel=500.0)
        assert method == "psa"
        assert poplevel == 500.0

    def test_psa_default_poplevel(self):
        """PSA without poplevel should default to 100."""
        method, poplevel = _normalize_method("psa")
        assert method == "psa"
        assert poplevel == 100.0

    def test_psa_low_poplevel_gets_default(self):
        """PSA with poplevel <= 1.0 should default to 100."""
        method, poplevel = _normalize_method("psa", poplevel=0.5)
        assert method == "psa"
        assert poplevel == 100.0

    def test_nf_unchanged(self):
        assert _normalize_method("nf") == ("nf", None)

    def test_case_insensitive(self):
        assert _normalize_method("SSA", poplevel=100.0) == ("psa", 100.0)
        assert _normalize_method("ODE") == ("ode", None)


# ─── Table function parsing ──────────────────────────────────────


class TestParseTableFunctions:
    def _write_bngl(self, content):
        f = tempfile.NamedTemporaryFile(
            suffix=".bngl", mode="w", delete=False
        )
        f.write(content)
        f.close()
        return f.name

    def test_file_based_tfun(self):
        path = self._write_bngl(
            "begin functions\n"
            "  f_drive() = tfun('drive_data.tfun', time)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["name"] == "f_drive"
            assert specs[0]["file"].endswith("drive_data.tfun")
            assert specs[0]["index"] == "time"
            assert specs[0]["method"] == "linear"
        finally:
            os.unlink(path)

    def test_inline_tfun(self):
        path = self._write_bngl(
            "begin functions\n"
            "  f_simple() = tfun([0,1,2], [1,2,4], time)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["name"] == "f_simple"
            assert specs[0]["times"] == [0.0, 1.0, 2.0]
            assert specs[0]["values"] == [1.0, 2.0, 4.0]
            assert specs[0]["index"] == "time"
        finally:
            os.unlink(path)

    def test_step_method(self):
        path = self._write_bngl(
            "begin functions\n"
            '  f_step() = tfun(\'data.tfun\', time, method=>"step")\n'
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["method"] == "step"
        finally:
            os.unlink(path)

    def test_custom_index_variable(self):
        path = self._write_bngl(
            "begin functions\n"
            "  f_dose() = tfun('dose.tfun', drug_conc)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["index"] == "drug_conc"
        finally:
            os.unlink(path)

    def test_no_functions_block(self):
        path = self._write_bngl(
            "begin model\nend model\n"
            "simulate({method=>\"ode\"})\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert specs == []
        finally:
            os.unlink(path)

    def test_non_tfun_functions_ignored(self):
        path = self._write_bngl(
            "begin functions\n"
            "  f_rate() = k1 * A_tot\n"
            "  f_drive() = tfun([0,1,2], [10,20,30], time)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["name"] == "f_drive"
        finally:
            os.unlink(path)

    def test_multiple_tfuns(self):
        path = self._write_bngl(
            "begin functions\n"
            "  f1() = tfun([0,1], [0,10], time)\n"
            "  f2() = tfun('other.tfun', time)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 2
            assert specs[0]["name"] == "f1"
            assert specs[1]["name"] == "f2"
        finally:
            os.unlink(path)

    def test_file_path_resolved_relative_to_bngl(self):
        """File paths in tfun should be resolved relative to the BNGL directory."""
        path = self._write_bngl(
            "begin functions\n"
            "  f() = tfun('subdir/data.tfun', time)\n"
            "end functions\n"
        )
        try:
            specs = _parse_table_functions(path)
            bngl_dir = os.path.dirname(os.path.abspath(path))
            expected = os.path.join(bngl_dir, "subdir/data.tfun")
            assert specs[0]["file"] == expected
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        specs = _parse_table_functions("/nonexistent/path/model.bngl")
        assert specs == []


# ─── _parse_simulate_params ──────────────────────────────────────


class _FakeAction:
    """Lightweight stand-in for bionetgen.modelapi.structs.Action."""

    def __init__(self, action_type, args):
        self.type = action_type
        self.name = action_type
        self.args = args


class TestParseSimulateParams:
    def test_simulate_ode_defaults(self):
        a = _FakeAction("simulate_ode", {})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "ode"
        assert sp["t_start"] == 0.0
        assert sp["t_end"] == 100.0
        assert sp["n_steps"] == 100
        assert sp["poplevel"] is None
        assert sp["gml"] is None
        assert sp["sample_times"] is None

    def test_simulate_ssa_no_poplevel(self):
        a = _FakeAction("simulate_ssa", {})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "ssa"
        assert sp["poplevel"] is None

    def test_simulate_ssa_with_poplevel_promotes_to_psa(self):
        """BNG2.pl compat: simulate_ssa + poplevel → psa."""
        a = _FakeAction("simulate_ssa", {"poplevel": "200"})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "psa"
        assert sp["poplevel"] == 200.0

    def test_simulate_with_method_psa(self):
        """simulate({method=>"psa"}) should work directly."""
        a = _FakeAction("simulate", {"method": "psa", "poplevel": "500"})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "psa"
        assert sp["poplevel"] == 500.0

    def test_simulate_psa_default_poplevel(self):
        a = _FakeAction("simulate", {"method": "psa"})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "psa"
        assert sp["poplevel"] == 100.0

    def test_simulate_nf(self):
        a = _FakeAction("simulate_nf", {"t_end": "50", "n_steps": "200"})
        sp = _parse_simulate_params(a)
        assert sp["method"] == "nf"
        assert sp["t_end"] == 50.0
        assert sp["n_steps"] == 200

    def test_gml_parsed(self):
        a = _FakeAction("simulate_nf", {"gml": "100000"})
        sp = _parse_simulate_params(a)
        assert sp["gml"] == 100000

    def test_sample_times_parsed(self):
        a = _FakeAction("simulate_ode", {"sample_times": "[0,5,10,50,100]"})
        sp = _parse_simulate_params(a)
        assert sp["sample_times"] == [0.0, 5.0, 10.0, 50.0, 100.0]

    def test_sample_times_suppressed_by_n_steps(self):
        a = _FakeAction("simulate_ode", {
            "sample_times": "[0,5,10,50,100]",
            "n_steps": "50",
        })
        sp = _parse_simulate_params(a)
        assert sp["sample_times"] is None
        assert sp["n_steps"] == 50

    def test_continue_flag(self):
        a = _FakeAction("simulate_ode", {"continue": "1"})
        sp = _parse_simulate_params(a)
        assert sp["continue_flag"] is True

    def test_tolerances_and_seed(self):
        a = _FakeAction("simulate_ode", {
            "atol": "1e-10", "rtol": "1e-8", "seed": "123",
        })
        sp = _parse_simulate_params(a)
        assert sp["atol"] == 1e-10
        assert sp["rtol"] == 1e-8
        assert sp["seed"] == 123

    def test_print_functions(self):
        a = _FakeAction("simulate_ode", {"print_functions": "1"})
        sp = _parse_simulate_params(a)
        assert sp["print_functions"] is True

    def test_stop_if(self):
        a = _FakeAction("simulate_ode", {"stop_if": '"A > 100"'})
        sp = _parse_simulate_params(a)
        assert sp["stop_if"] == "A > 100"

    def test_unrecognized_action_returns_none(self):
        a = _FakeAction("parameter_scan", {"method": "ode"})
        assert _parse_simulate_params(a) is None

    def test_suffix(self):
        a = _FakeAction("simulate_ode", {"suffix": '"my_run"'})
        sp = _parse_simulate_params(a)
        assert sp["suffix"] == "my_run"
