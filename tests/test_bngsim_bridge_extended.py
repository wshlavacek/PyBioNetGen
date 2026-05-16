"""Extended tests for bngsim_bridge to increase coverage.

These tests mock the bngsim library extensively so they run without
BNGsim installed.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bionetgen.core.exc import BNGSimError
# ─── Helpers ──────────────────────────────────────────────────────────

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


def _make_mock_result(obs_names=None, obs_data=None, species_names=None,
                      concentrations=None, n_times=10, time=None,
                      func_names=None, func_data=None):
    """Build a mock bngsim.Result-like object."""
    if obs_names is None:
        obs_names = ["obsA", "obsB"]
    if obs_data is None:
        obs_data = np.random.rand(n_times, len(obs_names))
    if species_names is None:
        species_names = ["S1", "S2"]
    if concentrations is None:
        concentrations = [1.0, 2.0]
    if time is None:
        time = np.linspace(0, 100, n_times)
    if func_names is None:
        func_names = []
    if func_data is None:
        func_data = np.empty((n_times, 0))

    core = MagicMock()
    core.expression_names = func_names
    core.expression_data = func_data

    result = MagicMock()
    result.observable_names = obs_names
    result.observables = obs_data
    result.n_observables = len(obs_names)
    result.n_times = n_times
    result.time = time
    result.expression_names = func_names
    result.expressions = func_data
    result.species_names = species_names
    result.concentrations = concentrations
    result._core = core
    result.to_cdat = MagicMock()
    return result


def _make_mock_bngsim_with_nfsim_session(result=None):
    """Build a mock bngsim module exposing the public NfsimSession API."""
    if result is None:
        result = _make_mock_result()

    session = MagicMock()
    session.simulate.return_value = result

    mock_bngsim = MagicMock()
    mock_bngsim.NfsimSession.return_value.__enter__.return_value = session
    return mock_bngsim, session


def _make_mock_model(param_names=None, params=None):
    """Build a mock bngsim.Model-like object."""
    if param_names is None:
        param_names = ["k1", "k2"]
    if params is None:
        params = {"k1": 0.1, "k2": 0.5}

    model = MagicMock()
    model.param_names = param_names
    model.get_param = MagicMock(side_effect=lambda n: params.get(n, 0.0))
    model.set_param = MagicMock()
    model.set_concentration = MagicMock()
    model.get_concentration = MagicMock(return_value=10.0)
    model.save_concentrations = MagicMock()
    model.reset = MagicMock()
    model.clone = MagicMock(return_value=MagicMock(
        param_names=param_names,
        get_param=MagicMock(side_effect=lambda n: params.get(n, 0.0)),
        set_param=MagicMock(),
        set_concentration=MagicMock(),
        save_concentrations=MagicMock(),
        reset=MagicMock(),
    ))
    model.add_table_function = MagicMock()
    return model


# ─── _write_bng_dat ──────────────────────────────────────────────────


class TestWriteBngDat:
    def test_writes_header_and_data(self):
        from bionetgen.core.tools.bngsim_bridge import _write_bng_dat

        with tempfile.NamedTemporaryFile(mode="w", suffix=".gdat", delete=False) as f:
            path = f.name

        try:
            time = np.array([0.0, 1.0, 2.0])
            data = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]])
            _write_bng_dat(path, time, data, ["obsA", "obsB"])

            with open(path) as f:
                lines = f.readlines()

            assert lines[0].startswith("# ")
            assert "time" in lines[0]
            assert "obsA" in lines[0]
            assert "obsB" in lines[0]
            assert len(lines) == 4  # header + 3 data rows
        finally:
            os.unlink(path)


# ─── _write_bngsim_results ───────────────────────────────────────────


class TestWriteBngsimResults:
    def test_writes_gdat_and_cdat(self):
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        result = _make_mock_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(result, tmpdir, "test_model")
            gdat = os.path.join(tmpdir, "test_model.gdat")
            cdat = os.path.join(tmpdir, "test_model.cdat")
            assert os.path.isfile(gdat)
            result.to_cdat.assert_called_once_with(cdat)

    def test_with_print_functions(self):
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        func_data = np.random.rand(10, 2)
        result = _make_mock_result(func_names=["f1", "f2"], func_data=func_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(result, tmpdir, "test_model", print_functions=True)
            gdat = os.path.join(tmpdir, "test_model.gdat")
            with open(gdat) as f:
                header = f.readline()
            assert "f1" in header
            assert "f2" in header

    def test_no_observables_no_funcs_skips_gdat(self):
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        result = _make_mock_result(obs_names=[], obs_data=np.empty((10, 0)))
        result.n_observables = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(result, tmpdir, "test_model")
            gdat = os.path.join(tmpdir, "test_model.gdat")
            assert not os.path.isfile(gdat)

    def test_print_functions_without_bngsim_expressions_does_not_eval_bngl(self):
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        obs_names = ["Y1", "Y2x2"]
        obs_data = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 0.0],
        ], dtype=float)
        result = _make_mock_result(
            obs_names=obs_names, obs_data=obs_data, n_times=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(
                result, tmpdir, "alt",
                print_functions=True,
            )
            gdat = os.path.join(tmpdir, "alt.gdat")
            with open(gdat) as f:
                lines = f.readlines()

        assert "Y1" in lines[0]
        assert "Y2x2" in lines[0]
        assert "Y2()" not in lines[0]
        assert "Sfree()" not in lines[0]
        assert "Lfree()" not in lines[0]

    def test_nfsim_fallback_skipped_when_bngsim_returns_expressions(self):
        # If BNGsim supplies a non-empty expression block, the writer
        # includes those direct result columns.
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        func_data = np.array([[7.0], [8.0], [9.0]])
        result = _make_mock_result(
            obs_names=["obsA"], obs_data=np.zeros((3, 1)), n_times=3,
            func_names=["actually_from_bngsim"], func_data=func_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(
                result, tmpdir, "ode",
                print_functions=True,
            )
            with open(os.path.join(tmpdir, "ode.gdat")) as f:
                header = f.readline()
        assert "actually_from_bngsim" in header


# ─── _make_bng_result ────────────────────────────────────────────────


class TestMakeBngResult:
    def test_returns_result(self):
        from bionetgen.core.tools.bngsim_bridge import _make_bng_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _make_bng_result(tmpdir, method="ode")
            assert result.process_return == 0
            assert "ode" in result.output[0]


# ─── run_nfsim ────────────────────────────────────────────────────────


class TestRunNfsim:
    def test_raises_when_bngsim_unavailable(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", False):
            with pytest.raises(BNGSimError, match="not installed"):
                run_nfsim("/dummy.xml", "/output")

    def test_raises_when_nfsim_unavailable(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", False):
            with pytest.raises(BNGSimError, match="not available"):
                run_nfsim("/dummy.xml", "/output")

    def test_happy_path(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            # Create a dummy xml file
            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            result = run_nfsim(xml_path, tmpdir)
            assert result.process_return == 0

            mock_bngsim.NfsimSession.assert_called_once_with(xml_path, molecule_limit=None)
            session.initialize.assert_called_once_with(42)
            session.simulate.assert_called_once_with(0.0, 100.0, 101)

    def test_param_overrides(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, param_overrides={"k1": 5.0})
            session.set_param.assert_called_with("k1", 5.0)

    def test_conc_overrides_set_exact_species_count(self):
        """conc_overrides should call set_species_count when available."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, conc_overrides={"A(b)": 200})
            session.set_species_count.assert_called_with("A(b)", 200)
            session.get_molecule_count.assert_not_called()
            session.add_molecules.assert_not_called()

    def test_conc_overrides_can_decrease_exact_species_count(self):
        """conc_overrides should allow decreases through set_species_count."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, conc_overrides={"A(b)": 50})
            session.set_species_count.assert_called_with("A(b)", 50)
            session.get_molecule_count.assert_not_called()
            session.add_molecules.assert_not_called()

    def test_conc_overrides_same_mol_type_stay_pattern_specific(self):
        """Patterned overrides should no longer collapse by molecule type."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(
                xml_path,
                tmpdir,
                conc_overrides={"A(b~0)": 50, "A(b~1)": 150},
            )
            session.set_species_count.assert_any_call("A(b~0)", 50)
            session.set_species_count.assert_any_call("A(b~1)", 150)
            assert session.set_species_count.call_count == 2
            session.get_molecule_count.assert_not_called()
            session.add_molecules.assert_not_called()

    def test_conc_overrides_and_deltas_remain_pattern_specific(self):
        """Absolute and relative NF concentration changes should replay by pattern."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(
                xml_path,
                tmpdir,
                conc_overrides={"A(b)": 100},
                conc_deltas={"A(c)": 25},
            )
            session.set_species_count.assert_called_once_with("A(b)", 100)
            session.add_species.assert_called_once_with("A(c)", 25)
            session.get_molecule_count.assert_not_called()
            session.add_molecules.assert_not_called()

    def test_conc_overrides_and_deltas_same_pattern_combine(self):
        """An override and delta for the same exact pattern should combine."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(
                xml_path,
                tmpdir,
                conc_overrides={"A(b)": 100},
                conc_deltas={"A(b)": 25},
            )
            session.set_species_count.assert_called_once_with("A(b)", 125)
            session.add_species.assert_not_called()

    def test_conc_deltas_can_decrease_exact_species_count(self):
        """Negative deltas should call remove_species when available."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, conc_deltas={"A(b)": -25})

            session.remove_species.assert_called_once_with("A(b)", 25)
            session.get_molecule_count.assert_not_called()
            session.add_molecules.assert_not_called()

    def test_legacy_patterned_conc_changes_are_molecule_type_granular(self):
        """Legacy fallback still collapses patterns by molecule type."""
        from bionetgen.core.tools.bngsim_bridge import (
            _collapse_nfsim_concentration_changes,
        )

        collapsed_overrides, collapsed_deltas = _collapse_nfsim_concentration_changes(
            conc_overrides={"A(b~0)": 50, "A(b~1)": 150},
            conc_deltas={"A(b~0)": -5, "A(b~1)": 20},
        )

        assert collapsed_overrides == {"A": 200}
        assert collapsed_deltas == {"A": 15}

    def test_conc_replay_falls_back_for_legacy_nfsim_session(self):
        """Older bngsim builds without species APIs keep the molecule-type path."""
        from bionetgen.core.tools.bngsim_bridge import (
            _apply_nfsim_concentration_changes,
        )

        class LegacyNfsimSession:
            def __init__(self):
                self.get_molecule_count = MagicMock(return_value=50)
                self.add_molecules = MagicMock()

        session = LegacyNfsimSession()
        _apply_nfsim_concentration_changes(
            session,
            conc_overrides={"A(b)": 200},
        )

        session.get_molecule_count.assert_called_once_with("A")
        session.add_molecules.assert_called_once_with("A", 150)

    def test_defaults(self):
        """Test default t_span, n_points, seed."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir)
            # Default: simulate(0.0, 100.0, 101)
            session.simulate.assert_called_once_with(0.0, 100.0, 101)
            session.initialize.assert_called_once_with(42)

    def test_simulation_failure_wraps_exception(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim = MagicMock()
        mock_bngsim.NfsimSession.side_effect = RuntimeError("boom")

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            with pytest.raises(BNGSimError, match="NFsim simulation failed"):
                run_nfsim(xml_path, tmpdir)

    def test_gml_is_set(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_bngsim, _ = _make_mock_bngsim_with_nfsim_session()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, gml=100000)
            mock_bngsim.NfsimSession.assert_called_once_with(xml_path, molecule_limit=100000)


# ─── run_with_bngsim ─────────────────────────────────────────────────


class TestRunWithBngsim:
    def test_raises_when_bngsim_unavailable(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", False):
            with pytest.raises(BNGSimError, match="not installed"):
                run_with_bngsim("/dummy.net", "/output", fmt="net")

    def test_bng_xml_routes_to_run_nfsim(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_run_nfsim = MagicMock()
        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.run_nfsim", mock_run_nfsim):
            run_with_bngsim("/model.xml", "/output", fmt="bng-xml", method="nf")
            mock_run_nfsim.assert_called_once()

    def test_bng_xml_defaults_to_nf(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_run_nfsim = MagicMock()
        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.run_nfsim", mock_run_nfsim):
            run_with_bngsim("/model.xml", "/output", fmt="bng-xml", method=None)
            mock_run_nfsim.assert_called_once()

    def test_bng_xml_bad_method_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True):
            with pytest.raises(BNGSimError, match="network-free simulation"):
                run_with_bngsim("/model.xml", "/output", fmt="bng-xml", method="ssa")

    def test_bng_xml_ode_method_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True):
            with pytest.raises(BNGSimError, match="network-free simulation"):
                run_with_bngsim("/model.xml", "/output", fmt="bng-xml", method="ode")

    def test_net_loads_from_net(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        mock_model = _make_mock_model()
        mock_bngsim.Model.from_net.return_value = mock_model
        mock_result = _make_mock_result()
        mock_sim = MagicMock()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            run_with_bngsim("/model.net", tmpdir, fmt="net", method="ode")
            mock_bngsim.Model.from_net.assert_called_once()

    def test_sbml_loads_from_sbml(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        mock_model = _make_mock_model()
        mock_bngsim.Model.from_sbml.return_value = mock_model
        mock_result = _make_mock_result()
        mock_sim = MagicMock()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            run_with_bngsim("/model.xml", tmpdir, fmt="sbml", method="ode")
            mock_bngsim.Model.from_sbml.assert_called_once()

    def test_antimony_loads_from_antimony(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        mock_bngsim.Model.from_antimony.return_value = _make_mock_model()
        mock_sim = MagicMock()
        mock_sim.run.return_value = _make_mock_result()
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            run_with_bngsim("/model.ant", tmpdir, fmt="antimony", method="ode")
            mock_bngsim.Model.from_antimony.assert_called_once()

    def test_nf_method_without_xml_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True):
            with pytest.raises(BNGSimError, match="requires a BioNetGen XML"):
                run_with_bngsim("/model.net", "/output", fmt="net", method="nf")

    def test_nf_with_xml_path_routes_to_nfsim(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_run_nfsim = MagicMock()
        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.run_nfsim", mock_run_nfsim):
            run_with_bngsim(
                "/model.net", "/output", fmt="net", method="nf",
                xml_path="/model.xml",
            )
            mock_run_nfsim.assert_called_once()

    def test_unsupported_format_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim):
            with pytest.raises(BNGSimError, match="Unsupported format"):
                run_with_bngsim("/model.bngl", "/output", fmt="bngl", method="ode")

    def test_simulation_exception_wrapped(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        mock_bngsim.Model.from_net.side_effect = RuntimeError("boom")

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim):
            with pytest.raises(BNGSimError, match="BNGsim simulation failed"):
                run_with_bngsim("/model.net", "/output", fmt="net")


# ─── _try_prepare_codegen ─────────────────────────────────────────────


class TestTryPrepareCodegen:
    def test_returns_empty_when_env_var_set(self):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        with patch.dict(os.environ, {"BIONETGEN_NO_CODEGEN": "1"}):
            assert _try_prepare_codegen("/dummy.net") == ""

    def test_returns_empty_when_codegen_unavailable(self):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        with patch.dict(os.environ, {}, clear=False):
            # Make sure BIONETGEN_NO_CODEGEN is not set
            os.environ.pop("BIONETGEN_NO_CODEGEN", None)
            # bngsim.prepare_codegen won't be importable
            assert _try_prepare_codegen("/dummy.net") == ""

    def test_returns_so_path_when_codegen_available(self):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        mock_bngsim = MagicMock()
        mock_bngsim.prepare_codegen.return_value = "/path/to/lib.so"

        with patch.dict(os.environ, {}, clear=False), \
             patch.dict("sys.modules", {"bngsim": mock_bngsim}):
            os.environ.pop("BIONETGEN_NO_CODEGEN", None)
            result = _try_prepare_codegen("/dummy.net")
            assert result == "/path/to/lib.so"


# ─── Regression: XML sniffer must not misread BNG-generated SBML ────


class TestSbmlWithBioNetGenComment:
    """BNG2.pl writeSBML emits a 'Created by BioNetGen' comment in the
    SBML output. The sniffer must not classify that as BNG XML."""

    def _write_xml(self, content):
        f = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_sbml_with_bionetgen_comment(self):
        from bionetgen.core.tools.bngsim_bridge import (
            FORMAT_SBML, _sniff_xml_format,
        )
        path = self._write_xml(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!-- Created by BioNetGen 2.9.3  -->\n"
            '<sbml xmlns="http://www.sbml.org/sbml/level2/version3" level="2" version="3">\n'
            '  <model id="test">\n'
            "    <listOfReactions/>\n"
            "  </model>\n"
            "</sbml>"
        )
        try:
            assert _sniff_xml_format(path) == FORMAT_SBML
        finally:
            os.unlink(path)
