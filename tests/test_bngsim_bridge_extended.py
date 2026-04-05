"""Extended tests for bngsim_bridge to increase coverage.

These tests mock the bngsim library extensively so they run without
BNGsim installed.
"""

import os
import tempfile
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from bionetgen.core.exc import BNGSimError
from bionetgen.modelapi.structs import Action


# ─── Helpers ──────────────────────────────────────────────────────────

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


def _make_action(action_type, action_args=None):
    """Create an Action object with the given type and args."""
    return Action(action_type=action_type, action_args=action_args or {})


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
    result.species_names = species_names
    result.concentrations = concentrations
    result._core = core
    result.to_cdat = MagicMock()
    return result


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

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result

        mock_result = _make_mock_result()
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.NfsimSimulator", mock_nfsim_cls, create=True), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            # Create a dummy xml file
            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            result = run_nfsim(xml_path, tmpdir)
            assert result.process_return == 0

            mock_nfsim_inst.initialize.assert_called_once_with(42)
            mock_nfsim_inst.simulate.assert_called_once()
            mock_nfsim_inst.destroy_session.assert_called_once()

    def test_param_overrides(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result

        mock_result = _make_mock_result()
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, param_overrides={"k1": 5.0})
            mock_nfsim_inst.set_param.assert_called_with("k1", 5.0)

    def test_conc_overrides_add_molecules(self):
        """conc_overrides should call get_molecule_count + add_molecules."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result
        # Current count is 50, target is 200 → add 150
        mock_nfsim_inst.get_molecule_count.return_value = 50

        mock_result = _make_mock_result()
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, conc_overrides={"A(b)": 200})
            mock_nfsim_inst.get_molecule_count.assert_called_with("A")
            mock_nfsim_inst.add_molecules.assert_called_with("A", 150)

    def test_conc_overrides_cannot_decrease(self):
        """conc_overrides should warn and skip when target < current."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result
        # Current count is 200, target is 50 → cannot decrease
        mock_nfsim_inst.get_molecule_count.return_value = 200

        mock_result = _make_mock_result()
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, conc_overrides={"A(b)": 50})
            mock_nfsim_inst.get_molecule_count.assert_called_with("A")
            # add_molecules should NOT have been called
            mock_nfsim_inst.add_molecules.assert_not_called()

    def test_defaults(self):
        """Test default t_span, n_points, seed."""
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result

        mock_result = _make_mock_result()
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir)
            # Default: simulate(0.0, 100.0, 101)
            mock_nfsim_inst.simulate.assert_called_once_with(0.0, 100.0, 101)
            mock_nfsim_inst.initialize.assert_called_once_with(42)

    def test_simulation_failure_wraps_exception(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_cls.side_effect = RuntimeError("boom")

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            with pytest.raises(BNGSimError, match="NFsim simulation failed"):
                run_nfsim(xml_path, tmpdir)

    def test_gml_is_set(self):
        from bionetgen.core.tools.bngsim_bridge import run_nfsim

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_nfsim_inst.simulate.return_value = MagicMock()

        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = _make_mock_result()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            run_nfsim(xml_path, tmpdir, gml=100000)
            mock_nfsim_inst.set_molecule_limit.assert_called_once_with(100000)


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

    def test_bng_xml_bad_method_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True):
            with pytest.raises(BNGSimError, match="network-free simulation"):
                run_with_bngsim("/model.xml", "/output", fmt="bng-xml", method="ssa")

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


# ─── _sync_species_concentrations ─────────────────────────────────────


class TestSyncSpeciesConcentrations:
    def test_syncs_concentrations(self):
        from bionetgen.core.tools.bngsim_bridge import _sync_species_concentrations

        model = _make_mock_model(
            param_names=["k1"],
            params={"k1": 10.0},
        )
        initializers = [("S1", "k1 * 2")]
        _sync_species_concentrations(model, initializers)
        model.set_concentration.assert_called_once_with("S1", 20.0)
        model.save_concentrations.assert_called_once()

    def test_empty_initializers(self):
        from bionetgen.core.tools.bngsim_bridge import _sync_species_concentrations

        model = _make_mock_model()
        _sync_species_concentrations(model, [])
        model.set_concentration.assert_not_called()

    def test_bad_expression_skipped(self):
        from bionetgen.core.tools.bngsim_bridge import _sync_species_concentrations

        model = _make_mock_model(param_names=["k1"], params={"k1": 1.0})
        initializers = [("S1", "undefined_var * 2")]
        # Should not raise — bad expressions are silently skipped
        _sync_species_concentrations(model, initializers)
        model.set_concentration.assert_not_called()
        model.save_concentrations.assert_called_once()


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
            # bngsim._codegen won't be importable
            assert _try_prepare_codegen("/dummy.net") == ""

    def test_returns_so_path_when_codegen_available(self):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        mock_codegen = MagicMock()
        mock_codegen.prepare_codegen.return_value = "/path/to/lib.so"

        with patch.dict(os.environ, {}, clear=False), \
             patch.dict("sys.modules", {"bngsim._codegen": mock_codegen}):
            os.environ.pop("BIONETGEN_NO_CODEGEN", None)
            result = _try_prepare_codegen("/dummy.net")
            assert result == "/path/to/lib.so"


# ─── _resolve_sample_times warning path ───────────────────────────────


class TestResolveSampleTimesWarnings:
    def test_unparseable_returns_none(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_sample_times

        assert _resolve_sample_times({"sample_times": "not_a_list"}) is None

    def test_less_than_3_points_returns_none(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_sample_times

        assert _resolve_sample_times({"sample_times": "[1,2]"}) is None

    def test_non_string_non_list_returns_none(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_sample_times

        assert _resolve_sample_times({"sample_times": 42}) is None

    def test_n_steps_takes_precedence(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_sample_times

        result = _resolve_sample_times({
            "sample_times": "[1,5,10,20]",
            "n_steps": "100",
        })
        assert result is None


# ─── _actions_need_network / _actions_need_xml ────────────────────────


class TestActionsNeedNetwork:
    def test_simulate_ode_needs_network(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_network([action]) is True

    def test_simulate_nf_still_defaults_true(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"})
        # _actions_need_network returns True by default even for NF
        assert _actions_need_network([action]) is True

    def test_parameter_scan_ode(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "5", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })
        assert _actions_need_network([action]) is True


class TestActionsNeedXml:
    def test_simulate_nf_needs_xml(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_xml

        action = _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_xml([action]) is True

    def test_simulate_ode_does_not_need_xml(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_xml

        action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_xml([action]) is False

    def test_writeXML_needs_xml(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_xml

        action = _make_action("writeXML", {})
        assert _actions_need_xml([action]) is True

    def test_parameter_scan_nf(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_xml

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "5", "method": "nf", "t_end": "100",
            "n_steps": "10",
        })
        assert _actions_need_xml([action]) is True


# ─── _scan_result_to_row ─────────────────────────────────────────────


class TestScanResultToRow:
    def test_basic(self):
        from bionetgen.core.tools.bngsim_bridge import _scan_result_to_row

        result = _make_mock_result(
            obs_names=["A", "B"],
            obs_data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        row, obs_names, func_names = _scan_result_to_row(result, 0.5)
        assert obs_names == ["A", "B"]
        assert func_names == []
        assert row[0] == 0.5  # scan value
        assert row[1] == 3.0  # final obs A
        assert row[2] == 4.0  # final obs B

    def test_with_print_functions(self):
        from bionetgen.core.tools.bngsim_bridge import _scan_result_to_row

        func_data = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = _make_mock_result(
            obs_names=["A"],
            obs_data=np.array([[1.0], [3.0]]),
            func_names=["f1", "f2"],
            func_data=func_data,
            n_times=2,
        )
        row, obs_names, func_names = _scan_result_to_row(result, 1.0, print_functions=True)
        assert func_names == ["f1", "f2"]
        assert row[0] == 1.0
        assert row[1] == 3.0
        assert row[2] == 30.0
        assert row[3] == 40.0

    def test_empty_observables(self):
        from bionetgen.core.tools.bngsim_bridge import _scan_result_to_row

        result = _make_mock_result(
            obs_names=[],
            obs_data=np.empty((0, 0)),
            n_times=0,
        )
        row, obs_names, func_names = _scan_result_to_row(result, 2.0)
        assert row[0] == 2.0
        assert len(obs_names) == 0


# ─── _execute_bngsim_actions ─────────────────────────────────────────


class TestExecuteBngsimActions:
    """Test the main action execution engine."""

    def _run(self, actions, model=None, **kwargs):
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        if model is None:
            model = _make_mock_model()

        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            result = _execute_bngsim_actions(
                actions, model, tmpdir, "test_model",
                **kwargs,
            )
            return result, model, mock_bngsim, mock_sim

    def test_simulate_ode(self):
        action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
        result, model, mock_bngsim, mock_sim = self._run([action])
        assert result.process_return == 0
        mock_sim.run.assert_called_once()

    def test_simulate_ssa(self):
        action = _make_action("simulate_ssa", {"t_end": "50", "n_steps": "20"})
        result, model, mock_bngsim, mock_sim = self._run([action])
        assert result.process_return == 0
        mock_bngsim.Simulator.assert_called()

    def test_set_parameter(self):
        action = _make_action("setParameter", {'"kf"': None, '1.5': None})
        result, model, mock_bngsim, mock_sim = self._run([action])
        model.set_param.assert_called_with("kf", 1.5)

    def test_set_concentration(self):
        action = _make_action("setConcentration", {'"S1"': None, '100': None})
        result, model, mock_bngsim, mock_sim = self._run([action])
        model.set_concentration.assert_called_with("S1", 100.0)

    def test_add_concentration(self):
        action = _make_action("addConcentration", {'"S1"': None, '50': None})
        model = _make_mock_model()
        model.get_concentration.return_value = 100.0
        result, model, mock_bngsim, mock_sim = self._run([action], model=model)
        model.set_concentration.assert_called_with("S1", 150.0)

    def test_set_concentration_propagates_to_nfsim(self):
        """setConcentration before simulate_nf should forward conc_overrides."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        set_conc = _make_action("setConcentration", {'"A(b)"': None, '200': None})
        sim_nf = _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"})

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}.run_nfsim") as mock_run_nfsim, \
             tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")
            _execute_bngsim_actions(
                [set_conc, sim_nf], model, tmpdir, "test_model",
                xml_path=xml_path,
            )
            mock_run_nfsim.assert_called_once()
            call_kwargs = mock_run_nfsim.call_args
            conc_ov = call_kwargs[1].get("conc_overrides") or call_kwargs.kwargs.get("conc_overrides")
            assert conc_ov == {"A(b)": 200}

    def test_reset_concentrations_clears_nf_conc_overrides(self):
        """resetConcentrations should clear nf_conc_overrides."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        set_conc = _make_action("setConcentration", {'"A(b)"': None, '200': None})
        reset_conc = _make_action("resetConcentrations", {})
        sim_nf = _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"})

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}.run_nfsim") as mock_run_nfsim, \
             tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")
            _execute_bngsim_actions(
                [set_conc, reset_conc, sim_nf], model, tmpdir, "test_model",
                xml_path=xml_path,
            )
            mock_run_nfsim.assert_called_once()
            call_kwargs = mock_run_nfsim.call_args
            conc_ov = call_kwargs[1].get("conc_overrides") or call_kwargs.kwargs.get("conc_overrides")
            # Should be None (empty dict is falsy, passed as None)
            assert not conc_ov

    def test_save_reset_concentrations(self):
        save_action = _make_action("saveConcentrations", {})
        reset_action = _make_action("resetConcentrations", {})
        result, model, mock_bngsim, mock_sim = self._run([save_action, reset_action])
        model.save_concentrations.assert_called()
        model.reset.assert_called()

    def test_save_reset_parameters(self):
        save_action = _make_action("saveParameters", {})
        reset_action = _make_action("resetParameters", {})
        result, model, mock_bngsim, mock_sim = self._run([save_action, reset_action])
        # saveParameters reads param values, resetParameters restores them
        assert result.process_return == 0

    def test_continue_flag_updates_t_start(self):
        """Test that continue=>1 uses model_time as t_start."""
        action1 = _make_action("simulate_ode", {"t_end": "50", "n_steps": "10"})
        action2 = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "10", "continue": "1",
        })

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action1, action2], model, tmpdir, "test_model",
            )
            # Second call should use t_start=50 (model_time from first sim)
            calls = mock_sim.run.call_args_list
            assert len(calls) == 2
            second_call_kwargs = calls[1]
            # The t_span should have t_start = 50.0
            t_span = second_call_kwargs[1].get("t_span", second_call_kwargs[0][0] if second_call_kwargs[0] else None)
            if t_span is None:
                t_span = calls[1].kwargs.get("t_span")
            assert t_span[0] == 50.0

    def test_skip_bng2pl_actions(self):
        action = _make_action("generate_network", {"overwrite": "1"})
        result, model, mock_bngsim, mock_sim = self._run([action])
        # Should be silently skipped
        mock_sim.run.assert_not_called()

    def test_parameter_scan_dispatches(self):
        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "3", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}._run_parameter_scan_bngsim") as mock_scan, \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action], model, tmpdir, "test_model",
            )
            mock_scan.assert_called_once()

    def test_bifurcate_dispatches(self):
        action = _make_action("bifurcate", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "3", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}._run_parameter_scan_bngsim") as mock_scan, \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action], model, tmpdir, "test_model",
            )
            mock_scan.assert_called_once()
            # Check is_bifurcate=True
            call_kwargs = mock_scan.call_args
            assert call_kwargs[1].get("is_bifurcate") is True or call_kwargs.kwargs.get("is_bifurcate") is True

    def test_simulate_with_suffix(self):
        action = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "10", "suffix": "test_suffix",
        })
        result, model, mock_bngsim, mock_sim = self._run([action])
        assert result.process_return == 0

    def test_simulate_pla_raises(self):
        action = _make_action("simulate_pla", {"t_end": "100", "n_steps": "10"})
        with pytest.raises(BNGSimError, match="pla"):
            self._run([action])

    def test_simulate_with_atol_rtol_seed(self):
        action = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "10",
            "atol": "1e-8", "rtol": "1e-6", "seed": "123",
        })
        result, model, mock_bngsim, mock_sim = self._run([action])
        run_kwargs = mock_sim.run.call_args[1]
        assert run_kwargs["atol"] == 1e-8
        assert run_kwargs["rtol"] == 1e-6
        assert run_kwargs["seed"] == 123

    def test_simulate_with_sample_times(self):
        action = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "10",
            "sample_times": "[0,10,50,100]",
        })

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action], model, tmpdir, "test_model",
            )
            # sample_times should be passed; n_steps takes precedence though
            # since both are set, sample_times returns None
            mock_sim.run.assert_called_once()

    def test_setParameter_invalidates_sim_cache(self):
        """After setParameter, the next simulate should rebuild the simulator."""
        set_action = _make_action("setParameter", {'"k1"': None, '5.0': None})
        sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [sim_action, set_action, sim_action], model, tmpdir, "test_model",
            )
            # Simulator should be created twice (invalidated after setParameter)
            assert mock_bngsim.Simulator.call_count == 2

    def test_nf_simulate_dispatches(self):
        action = _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"})

        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        model = _make_mock_model()
        mock_bngsim = MagicMock()

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}.run_nfsim") as mock_run_nfsim, \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            _execute_bngsim_actions(
                [action], model, tmpdir, "test_model",
                xml_path=xml_path,
            )
            mock_run_nfsim.assert_called_once()


# ─── _run_protocol ────────────────────────────────────────────────────


class TestRunProtocol:
    def test_simulate_action(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = ['simulate_ode({t_end=>100,n_steps=>10})']

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            result = _run_protocol(model, lines)
            assert result is mock_result

    def test_set_parameter_in_protocol(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        lines = ['setParameter("k1", 5.0)']

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            result = _run_protocol(model, lines)
            model.set_param.assert_called_with("k1", 5.0)
            assert result is None

    def test_set_concentration_in_protocol(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        lines = ['setConcentration("S1", 200.0)']

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            model.set_concentration.assert_called_with("S1", 200.0)

    def test_save_reset_concentrations_in_protocol(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'saveConcentrations()',
            'resetConcentrations()',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            model.save_concentrations.assert_called_once()
            model.reset.assert_called_once()

    def test_continue_updates_t_start(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'simulate_ode({t_end=>50,n_steps=>10})',
            'simulate_ode({t_end=>100,n_steps=>10,continue=>1})',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            calls = mock_sim.run.call_args_list
            assert len(calls) == 2
            # Second call t_start should be 50.0
            assert calls[1][1]["t_span"][0] == 50.0

    def test_comments_and_blank_lines_skipped(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            '',
            '# This is a comment',
            '   ',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            result = _run_protocol(model, lines)
            assert result is None

    def test_method_change_rebuilds_simulator(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'simulate_ode({t_end=>50,n_steps=>10})',
            'simulate_ssa({t_end=>100,n_steps=>10})',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            # Simulator created 3 times: initial ODE, then ODE sim, then SSA sim
            # Actually: initial + rebuild for SSA = at least 2
            assert mock_bngsim.Simulator.call_count >= 2

    def test_save_reset_parameters_in_protocol(self):
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        # get_param returns different values before/after setParameter
        param_vals = {"k1": 0.1, "k2": 0.5}
        model.get_param.side_effect = lambda n: param_vals.get(n, 0.0)

        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'saveParameters()',
            'setParameter("k1", 99.0)',
            'resetParameters()',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            # setParameter should be called with 99.0, then resetParameters
            # restores k1 to 0.1
            set_calls = model.set_param.call_args_list
            # First call: setParameter("k1", 99.0)
            assert set_calls[0] == (("k1", 99.0),)
            # resetParameters restores both k1=0.1 and k2=0.5
            restore_calls = {c[0][0]: c[0][1] for c in set_calls[1:]}
            assert restore_calls["k1"] == 0.1
            assert restore_calls["k2"] == 0.5


# ─── _run_parameter_scan_bngsim ──────────────────────────────────────


class TestRunParameterScanBngsim:
    def test_basic_time_course_scan(self):
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "1.0",
            "n_scan_pts": "3", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })

        model = _make_mock_model()
        clone = _make_mock_model()
        model.clone.return_value = clone

        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)

    def test_protocol_method_raises_without_protocol_lines(self):
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "1.0",
            "n_scan_pts": "3", "method": '"protocol"', "t_end": "100",
            "n_steps": "10",
        })

        model = _make_mock_model()
        mock_bngsim = MagicMock()

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            with pytest.raises(BNGSimError, match="protocol"):
                _run_parameter_scan_bngsim(model, action, "/tmp", "test_model")

    def test_nf_method_dispatches(self):
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "1.0",
            "n_scan_pts": "3", "method": "nf", "t_end": "100",
            "n_steps": "10",
        })

        model = _make_mock_model()

        with patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}._run_nfsim_scan") as mock_nfscan, \
             tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")
            _run_parameter_scan_bngsim(
                model, action, tmpdir, "test_model", xml_path=xml_path,
            )
            mock_nfscan.assert_called_once()

    def test_steady_state_converged(self):
        """When steady_state converges, use equilibrium concentrations."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "1.0",
            "n_scan_pts": "1", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()
        clone = _make_mock_model()
        model.clone.return_value = clone

        # steady_state result: converged
        ss_result = MagicMock()
        ss_result.converged = True
        ss_result.species_names = ["S1", "S2"]
        ss_result.concentrations = [5.0, 10.0]

        eval_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        ss_sim = MagicMock()
        ss_sim.steady_state.return_value = ss_result
        eval_sim = MagicMock()
        eval_sim.run.return_value = eval_result
        # First Simulator call is for _make_sim (ss_sim), second for eval_sim
        mock_bngsim.Simulator.side_effect = [ss_sim, eval_sim]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Verify steady_state was called
            ss_sim.steady_state.assert_called_once()
            # Verify concentrations were set on the clone
            clone.set_concentration.assert_any_call("S1", 5.0)
            clone.set_concentration.assert_any_call("S2", 10.0)

    def test_steady_state_not_converged_falls_back(self):
        """When steady_state doesn't converge, fall back to time-course."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "1.0",
            "n_scan_pts": "1", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()
        clone = _make_mock_model()
        fallback_clone = _make_mock_model()
        model.clone.side_effect = [clone, fallback_clone]

        # steady_state result: NOT converged
        ss_result = MagicMock()
        ss_result.converged = False
        ss_result.residual = 1.5e-2

        fallback_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        ss_sim = MagicMock()
        ss_sim.steady_state.return_value = ss_result
        fallback_sim = MagicMock()
        fallback_sim.run.return_value = fallback_result
        # First Simulator for ss_sim, second for fallback_sim
        mock_bngsim.Simulator.side_effect = [ss_sim, fallback_sim]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Verify fallback sim was used
            fallback_sim.run.assert_called_once()
            call_kwargs = fallback_sim.run.call_args
            assert call_kwargs[1]["t_span"] == (0, 100)

    def test_steady_state_exception_falls_back(self):
        """When steady_state raises, fall back to time-course."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "1.0",
            "n_scan_pts": "1", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()
        clone = _make_mock_model()
        fallback_clone = _make_mock_model()
        model.clone.side_effect = [clone, fallback_clone]

        fallback_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        ss_sim = MagicMock()
        ss_sim.steady_state.side_effect = RuntimeError("solver blew up")
        fallback_sim = MagicMock()
        fallback_sim.run.return_value = fallback_result
        # First Simulator for ss_sim, second for fallback_sim
        mock_bngsim.Simulator.side_effect = [ss_sim, fallback_sim]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Verify fallback sim was used
            fallback_sim.run.assert_called_once()

    def test_threaded_ss_scan_converged(self):
        """Threaded path used when >=4 points, no species_initializers."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()

        def make_clone():
            c = _make_mock_model()
            return c
        model.clone.side_effect = [make_clone() for _ in range(4)]

        # All 4 steady-state results converge
        ss_results = []
        for _ in range(4):
            sr = MagicMock()
            sr.converged = True
            sr.species_names = ["S1", "S2"]
            sr.concentrations = [5.0, 10.0]
            ss_results.append(sr)

        eval_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        # 4 _make_sim calls (for SS), then 4 eval Simulator calls
        ss_sims = []
        for sr in ss_results:
            s = MagicMock()
            s.steady_state.return_value = sr
            ss_sims.append(s)
        eval_sims = [MagicMock(run=MagicMock(return_value=eval_result)) for _ in range(4)]
        mock_bngsim.Simulator.side_effect = ss_sims + eval_sims

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # All 4 SS solvers should have been called
            for s in ss_sims:
                s.steady_state.assert_called_once()

    def test_threaded_ss_scan_with_fallback(self):
        """Threaded path falls back per-point when SS fails."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()
        model.clone.side_effect = [_make_mock_model() for _ in range(8)]

        eval_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )
        fallback_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[3.0], [4.0]]), n_times=2,
        )

        # Points 0,2 converge; point 1 fails; point 3 doesn't converge
        ss_results = []
        for i in range(4):
            sr = MagicMock()
            if i == 1:
                sr.steady_state = MagicMock(side_effect=RuntimeError("boom"))
            elif i == 3:
                sr.steady_state = MagicMock(return_value=MagicMock(converged=False, residual=0.1))
            else:
                res = MagicMock(converged=True, species_names=["S1"], concentrations=[5.0])
                sr.steady_state = MagicMock(return_value=res)
            ss_results.append(sr)

        # Simulator calls: 4 for _make_sim (SS), 2 eval (converged), 2 fallback _make_sim,
        # plus 2 fallback _prepare clones need _make_sim
        mock_bngsim = MagicMock()
        eval_sim = MagicMock(run=MagicMock(return_value=eval_result))
        fb_sim = MagicMock(run=MagicMock(return_value=fallback_result))
        mock_bngsim.Simulator.side_effect = (
            ss_results           # 4 _make_sim for initial SS
            + [eval_sim, eval_sim]  # 2 eval sims for converged points
            + [fb_sim, fb_sim]      # 2 fallback sims
        )

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Fallback sims should have been used for points 1 and 3
            assert fb_sim.run.call_count == 2

    def test_threaded_ss_not_used_with_species_initializers(self):
        """Sequential SS path when species_initializers present, even with >=4 points."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "n_steps": "10", "steady_state": "1",
        })

        model = _make_mock_model()
        model.clone.side_effect = [_make_mock_model() for _ in range(4)]

        ss_result = MagicMock(converged=True, species_names=["S1"], concentrations=[5.0])
        eval_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        ss_sim = MagicMock(steady_state=MagicMock(return_value=ss_result))
        eval_sim = MagicMock(run=MagicMock(return_value=eval_result))
        mock_bngsim.Simulator.side_effect = [ss_sim, eval_sim] * 4

        # Pass species_initializers — should force sequential path
        species_inits = [("S1", "k1*10")]
        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}._sync_species_concentrations"), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(
                model, action, tmpdir, "test_model",
                species_initializers=species_inits,
            )
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Sequential: each point gets its own SS + eval sim pair
            assert ss_sim.steady_state.call_count == 4

    def test_batch_time_course_scan(self):
        """Batch path used for time-course with >=4 points, no sample_times."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })

        model = _make_mock_model()

        batch_results = [
            _make_mock_result(obs_names=["A"], obs_data=np.array([[float(i)], [float(i)]]), n_times=2)
            for i in range(4)
        ]

        mock_bngsim = MagicMock()
        batch_sim = MagicMock()
        batch_sim.run_batch.return_value = batch_results
        mock_bngsim.Simulator.return_value = batch_sim
        # Ensure run_batch is detected via hasattr
        mock_bngsim.Simulator.run_batch = True

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            batch_sim.run_batch.assert_called_once()
            call_kwargs = batch_sim.run_batch.call_args[1]
            assert call_kwargs["t_span"] == (0, 100)
            assert call_kwargs["n_points"] == 2
            assert len(call_kwargs["params"]) == 4

    def test_batch_fallback_to_sequential(self):
        """Batch path falls back to sequential on run_batch exception."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })

        model = _make_mock_model()
        model.clone.side_effect = [_make_mock_model() for _ in range(4)]

        seq_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        # First Simulator call is for batch — run_batch fails
        batch_sim = MagicMock()
        batch_sim.run_batch.side_effect = RuntimeError("batch failed")
        # Subsequent calls are sequential sims
        seq_sim = MagicMock(run=MagicMock(return_value=seq_result))
        mock_bngsim.Simulator.side_effect = [batch_sim] + [seq_sim] * 4
        mock_bngsim.Simulator.run_batch = True

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # Batch was attempted then fell back
            batch_sim.run_batch.assert_called_once()
            assert seq_sim.run.call_count == 4

    def test_batch_not_used_with_sample_times(self):
        """Batch path not used when sample_times is specified."""
        from bionetgen.core.tools.bngsim_bridge import _run_parameter_scan_bngsim

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "1.0", "par_max": "4.0",
            "n_scan_pts": "4", "method": "ode", "t_end": "100",
            "sample_times": "[0, 25, 50, 75, 100]",
        })

        model = _make_mock_model()
        model.clone.side_effect = [_make_mock_model() for _ in range(4)]

        seq_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )

        mock_bngsim = MagicMock()
        seq_sim = MagicMock(run=MagicMock(return_value=seq_result))
        mock_bngsim.Simulator.side_effect = [seq_sim] * 4
        mock_bngsim.Simulator.run_batch = True

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:
            _run_parameter_scan_bngsim(model, action, tmpdir, "test_model")
            # Sequential: each point gets its own sim.run call
            assert seq_sim.run.call_count == 4


# ─── _parse_tfun_args ────────────────────────────────────────────────


class TestParseTfunArgs:
    def test_file_based_with_index(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "'data.tfun', time", "/models")
        assert result is not None
        assert result["name"] == "myfunc"
        assert result["file"] == "/models/data.tfun"
        assert result["index"] == "time"
        assert result["method"] == "linear"

    def test_file_based_no_index(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "'data.tfun'", "/models")
        assert result is not None
        assert result["index"] == "time"  # default

    def test_inline_array_form(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "[0,1,2], [10,20,30], time", "/models")
        assert result is not None
        assert result["times"] == [0.0, 1.0, 2.0]
        assert result["values"] == [10.0, 20.0, 30.0]
        assert result["index"] == "time"

    def test_inline_array_no_index(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "[0,1,2], [10,20,30]", "/models")
        assert result is not None
        assert result["index"] == "time"  # default

    def test_with_method_step(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args(
            "myfunc",
            '[0,1,2], [10,20,30], time, method=>"step"',
            "/models",
        )
        assert result is not None
        assert result["method"] == "step"

    def test_unparseable_returns_none(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "gibberish", "/models")
        assert result is None

    def test_file_based_absolute_path(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "'/abs/path/data.tfun', time", "/models")
        assert result is not None
        assert result["file"] == "/abs/path/data.tfun"

    def test_inline_bad_values_returns_none(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_tfun_args

        result = _parse_tfun_args("myfunc", "[a,b,c], [d,e,f]", "/models")
        assert result is None


# ─── _add_table_functions ────────────────────────────────────────────


class TestAddTableFunctions:
    def test_file_based(self):
        from bionetgen.core.tools.bngsim_bridge import _add_table_functions

        model = _make_mock_model()
        specs = [{"name": "f1", "file": "/path/data.tfun", "index": "time", "method": "linear"}]
        _add_table_functions(model, specs)
        model.add_table_function.assert_called_once_with(
            "f1", file="/path/data.tfun", index="time", method="linear",
        )

    def test_inline(self):
        from bionetgen.core.tools.bngsim_bridge import _add_table_functions

        model = _make_mock_model()
        specs = [{
            "name": "f1", "times": [0, 1, 2], "values": [10, 20, 30],
            "index": "time", "method": "step",
        }]
        _add_table_functions(model, specs)
        model.add_table_function.assert_called_once_with(
            "f1", times=[0, 1, 2], values=[10, 20, 30],
            index="time", method="step",
        )

    def test_failure_warning(self):
        from bionetgen.core.tools.bngsim_bridge import _add_table_functions

        model = _make_mock_model()
        model.add_table_function.side_effect = RuntimeError("fail")
        specs = [{"name": "f1", "file": "/path/data.tfun", "index": "time", "method": "linear"}]
        # Should not raise, just warn
        _add_table_functions(model, specs)


# ─── _parse_table_functions ──────────────────────────────────────────


class TestParseTableFunctions:
    def test_parses_tfun_from_bngl(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_table_functions

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            f.write(textwrap.dedent("""\
                begin functions
                    myfunc(time) = tfun('data.tfun', time)
                end functions
            """))
            path = f.name

        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["name"] == "myfunc"
        finally:
            os.unlink(path)

    def test_parses_inline_tfun(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_table_functions

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            f.write(textwrap.dedent("""\
                begin functions
                    myfunc(time) = tfun([0,1,2], [10,20,30], time)
                end functions
            """))
            path = f.name

        try:
            specs = _parse_table_functions(path)
            assert len(specs) == 1
            assert specs[0]["times"] == [0.0, 1.0, 2.0]
        finally:
            os.unlink(path)

    def test_no_functions_block(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_table_functions

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            f.write("begin model\nend model\n")
            path = f.name

        try:
            specs = _parse_table_functions(path)
            assert specs == []
        finally:
            os.unlink(path)


# ─── _parse_protocol_block ──────────────────────────────────────────


class TestParseProtocolBlock:
    def test_parses_protocol(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_protocol_block

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            f.write(textwrap.dedent("""\
                begin protocol
                    simulate_ode({t_end=>100,n_steps=>10})
                    setParameter("k1", 5.0)
                end protocol
            """))
            path = f.name

        try:
            lines = _parse_protocol_block(path)
            assert len(lines) == 2
            assert "simulate_ode" in lines[0]
            assert "setParameter" in lines[1]
        finally:
            os.unlink(path)

    def test_no_protocol_block(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_protocol_block

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            f.write("begin model\nend model\n")
            path = f.name

        try:
            lines = _parse_protocol_block(path)
            assert lines == []
        finally:
            os.unlink(path)


# ─── run_bngl_with_bngsim ───────────────────────────────────────────


class TestRunBnglWithBngsim:
    def test_raises_when_bngsim_unavailable(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", False):
            with pytest.raises(BNGSimError, match="not available"):
                run_bngl_with_bngsim("/model.bngl", "/output", "/bngpath")

    def test_basic_flow(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        # Create a minimal BNGL file
        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            # Mock bngmodel
            mock_model = MagicMock()
            mock_model.model_name = "test"
            sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            # Mock BNGCLI
            mock_cli = MagicMock()
            mock_cli.result = MagicMock()
            mock_cli.result.process_return = 0

            # Mock bngsim
            mock_bngsim = MagicMock()
            mock_bngsim_model = _make_mock_model()
            mock_bngsim.Model.from_net.return_value = mock_bngsim_model

            # Create the .net file that the code expects
            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_execute = MagicMock()
            mock_execute.return_value = MagicMock(process_return=0)

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", mock_execute):

                result = run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")
                mock_execute.assert_called_once()
                assert result.process_return == 0

    def test_no_sim_actions_returns_cli_result(self):
        """If no simulate actions and no CLI overrides, return BNG2.pl result."""
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            # Only generate_network, no simulate
            gen_action = _make_action("generate_network", {"overwrite": "1"})
            mock_model.actions.items = [gen_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock()
            mock_cli.result.process_return = 0

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", MagicMock()), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli):

                result = run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")
                assert result is mock_cli.result

    def test_no_sim_actions_with_method_creates_synthetic(self):
        """If no simulate actions but method is specified, create synthetic action."""
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            gen_action = _make_action("generate_network", {"overwrite": "1"})
            mock_model.actions.items = [gen_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock()

            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_bngsim = MagicMock()
            mock_bngsim.Model.from_net.return_value = _make_mock_model()

            mock_execute = MagicMock()
            mock_execute.return_value = MagicMock(process_return=0)

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", mock_execute):

                result = run_bngl_with_bngsim(
                    bngl_path, tmpdir, "/bngpath", method="ode",
                )
                mock_execute.assert_called_once()

    def test_cli_failure_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = None  # CLI failed

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", MagicMock()), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli):

                with pytest.raises(BNGSimError, match="BNG2.pl failed"):
                    run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")

    def test_net_not_generated_raises(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock()

            mock_bngsim = MagicMock()
            # No .net file exists in tmpdir

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli):

                with pytest.raises(BNGSimError, match="Expected .net file"):
                    run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")

    def test_table_functions_added(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock()

            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_bngsim = MagicMock()
            mock_bngsim_model = _make_mock_model()
            mock_bngsim.Model.from_net.return_value = mock_bngsim_model

            tfun_specs = [{"name": "f1", "file": "/data.tfun", "index": "time", "method": "linear"}]

            mock_execute = MagicMock()
            mock_execute.return_value = MagicMock(process_return=0)

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=tfun_specs), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._add_table_functions") as mock_add_tfun, \
                 patch(f"{BRIDGE}._execute_bngsim_actions", mock_execute):

                run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")
                mock_add_tfun.assert_called_once_with(mock_bngsim_model, tfun_specs)


# ─── _write_scan_file ────────────────────────────────────────────────


class TestWriteScanFile:
    def test_writes_scan_file(self):
        from bionetgen.core.tools.bngsim_bridge import _write_scan_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".scan", delete=False) as f:
            path = f.name

        try:
            rows = [
                np.array([0.1, 1.0, 2.0]),
                np.array([0.5, 3.0, 4.0]),
            ]
            _write_scan_file(path, "k1", ["obsA", "obsB"], rows)

            with open(path) as f:
                lines = f.readlines()
            assert lines[0].startswith("# ")
            assert "k1" in lines[0]
            assert "obsA" in lines[0]
            assert len(lines) == 3  # header + 2 data rows
        finally:
            os.unlink(path)


# ─── _resolve_scan_points ────────────────────────────────────────────


class TestResolveScanPoints:
    def test_linspace(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_scan_points

        points = _resolve_scan_points({
            "par_min": "0", "par_max": "1", "n_scan_pts": "3",
        })
        np.testing.assert_allclose(points, [0.0, 0.5, 1.0])

    def test_logspace(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_scan_points

        points = _resolve_scan_points({
            "par_min": "1", "par_max": "100", "n_scan_pts": "3",
            "log_scale": "1",
        })
        np.testing.assert_allclose(points, [1.0, 10.0, 100.0])

    def test_explicit_values(self):
        from bionetgen.core.tools.bngsim_bridge import _resolve_scan_points

        points = _resolve_scan_points({
            "par_scan_vals": "[0.1, 0.5, 1.0, 5.0]",
        })
        np.testing.assert_allclose(points, [0.1, 0.5, 1.0, 5.0])


# ─── _extract_positional_args ────────────────────────────────────────


class TestExtractPositionalArgs:
    def test_basic(self):
        from bionetgen.core.tools.bngsim_bridge import _extract_positional_args

        action = _make_action("setParameter", {'"kf"': None, '1.5': None})
        name, value = _extract_positional_args(action)
        assert name == "kf"
        assert value == "1.5"

    def test_empty_args(self):
        from bionetgen.core.tools.bngsim_bridge import _extract_positional_args

        action = _make_action("setParameter", {})
        name, value = _extract_positional_args(action)
        assert name == ""
        assert value == "0"


# ─── _parse_net_species_initializers ─────────────────────────────────


class TestParseNetSpeciesInitializers:
    def test_parses_species(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_net_species_initializers

        with tempfile.NamedTemporaryFile(mode="w", suffix=".net", delete=False) as f:
            f.write(textwrap.dedent("""\
                begin species
                    1 @b::X(p~0,y) 5000
                    2 @b::X(p~1,y) k_init*100
                end species
            """))
            path = f.name

        try:
            result = _parse_net_species_initializers(path)
            assert len(result) == 2
            assert result[0] == ("@b::X(p~0,y)", "5000")
            assert result[1] == ("@b::X(p~1,y)", "k_init*100")
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        from bionetgen.core.tools.bngsim_bridge import _parse_net_species_initializers

        result = _parse_net_species_initializers("/nonexistent.net")
        assert result == []


# ─── _run_nfsim_scan ─────────────────────────────────────────────────


class TestRunNfsimScan:
    def test_basic(self):
        from bionetgen.core.tools.bngsim_bridge import _run_nfsim_scan

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "1.0",
            "n_scan_pts": "2", "method": "nf", "t_end": "100",
            "n_steps": "10",
        })

        mock_nfsim_cls = MagicMock()
        mock_nfsim_inst = MagicMock()
        mock_nfsim_cls.return_value = mock_nfsim_inst
        mock_core_result = MagicMock()
        mock_nfsim_inst.simulate.return_value = mock_core_result

        mock_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )
        mock_bngsim = MagicMock()
        mock_bngsim.Result.return_value = mock_result

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch.dict("sys.modules", {"bngsim._bngsim_core": MagicMock(NfsimSimulator=mock_nfsim_cls)}), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            _run_nfsim_scan(xml_path, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # 2 scan points = 2 NfsimSimulator instances
            assert mock_nfsim_cls.call_count == 2
