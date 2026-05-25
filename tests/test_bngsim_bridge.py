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
    _sniff_xml_format,
    detect_input_format,
    run_with_bngsim,
)

tfold = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def generated_net_file(tmp_path_factory, require_bng2):
    """Generate a .net fixture locally instead of depending on test order."""
    from bionetgen.core.defaults import BNGDefaults
    from bionetgen.core.tools.cli import BNGCLI

    out_dir = tmp_path_factory.mktemp("bngsim_net_fixture")
    cli = BNGCLI(
        os.path.join(tfold, "test.bngl"),
        str(out_dir),
        BNGDefaults().bng_path,
        suppress=True,
    )
    cli.run()
    assert cli.result.process_return == 0

    net_file = out_dir / "test.net"
    assert net_file.is_file()
    return str(net_file)


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
        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
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
        path = self._write_xml('<?xml version="1.0"?>\n<root><data/></root>')
        try:
            with pytest.raises(BNGFormatError, match="Could not determine"):
                detect_input_format(path)
        finally:
            os.unlink(path)

    def test_nonexistent_xml_raises(self):
        with pytest.raises(BNGFormatError, match="Could not read file"):
            detect_input_format("/nonexistent/path/model.xml")

    def test_bng_xml_with_observables(self):
        path = self._write_xml('<?xml version="1.0"?>\n<Model>\n  <ListOfObservables/>\n</Model>')
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
        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        f.write('<sbml xmlns="http://www.sbml.org/"><model/></sbml>')
        f.close()
        try:
            assert detect_input_format(f.name, explicit_format="sbml") == FORMAT_SBML
        finally:
            os.unlink(f.name)

    def test_explicit_conflicts_with_autodetect(self):
        """Saying --format=bng-xml on an SBML file should raise."""
        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        f.write('<sbml xmlns="http://www.sbml.org/sbml/level3"><model/></sbml>')
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

        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
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
    reason="requires bngsim package importable (e.g. editable install via PYBNG_DEV_BNGSIM_PATH)",
)
class TestBngsimIntegration:
    def test_run_net_file(self, generated_net_file):
        """Run a .net file through BNGsim and verify output files."""
        with tempfile.TemporaryDirectory() as out:
            result = run_with_bngsim(
                generated_net_file,
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
            assert any(f.endswith((".gdat", ".cdat")) for f in files)

    def test_run_via_library_api(self, generated_net_file):
        """Run a .net file via bionetgen.run() with simulator='bngsim'."""
        import bionetgen

        with tempfile.TemporaryDirectory() as out:
            result = bionetgen.run(
                generated_net_file,
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
