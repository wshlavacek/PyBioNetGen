"""Extended tests for bngsim_bridge to increase coverage.

These tests mock the bngsim library extensively so they run without
BNGsim installed.
"""

import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

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

    def test_nfsim_fallback_evaluates_functions_per_timepoint(self):
        # NFsim Result.expressions comes back empty even when print_functions
        # is requested; the writer must fall back to evaluating BNGL function
        # bodies against (resolved-params + scan-overrides + obs) for every
        # row so the .gdat carries function columns matching BNG2.pl's
        # convention.
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

        bngmodel = MagicMock()
        bngmodel.functions.items = {
            # Y2 = Y2x2 / 2 (depends on observable only)
            "Y2": MagicMock(expr="Y2x2 / 2", args=[]),
            # Sfree = ST - Y1 - Y2x2 (depends on resolved param ST)
            "Sfree": MagicMock(expr="ST - Y1 - Y2x2", args=[]),
            # Lfree = LT_current - Y1 (depends on overridden param LT_current)
            "Lfree": MagicMock(expr="LT_current - Y1", args=[]),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(
                result, tmpdir, "alt",
                print_functions=True,
                bngmodel=bngmodel,
                bngmodel_params={"ST": 60.0, "LT_current": 331.0},
                param_overrides={"LT_current": 150.0},  # setParameter("LT_current","LT_low")
            )
            gdat = os.path.join(tmpdir, "alt.gdat")
            with open(gdat) as f:
                lines = f.readlines()

        # Header carries function columns rendered with BNG/NFsim parens style.
        assert "Y2()" in lines[0]
        assert "Sfree()" in lines[0]
        assert "Lfree()" in lines[0]

        # Per-row arithmetic. Columns are time, Y1, Y2x2, Y2(), Sfree(), Lfree().
        row1 = [float(v) for v in lines[2].split()[1:]]  # skip leading '#' guard? no, '#' only on header
        # Actually the data rows don't start with '#'; lines[1] is row 0.
        rows = [
            [float(v) for v in line.split()] for line in lines[1:]
        ]
        # row index 1 → t=1, Y1=1, Y2x2=2 → Y2=1, Sfree=60-1-2=57, Lfree=150-1=149
        assert rows[1][3] == pytest.approx(1.0)
        assert rows[1][4] == pytest.approx(57.0)
        assert rows[1][5] == pytest.approx(149.0)
        # row index 2 → t=2, Y1=3, Y2x2=0 → Y2=0, Sfree=57, Lfree=147
        assert rows[2][3] == pytest.approx(0.0)
        assert rows[2][4] == pytest.approx(57.0)
        assert rows[2][5] == pytest.approx(147.0)

    def test_nfsim_fallback_skipped_when_bngsim_returns_expressions(self):
        # If BNGsim already supplies a non-empty expression block (the
        # network/ODE path), the writer must not invoke the post-hoc
        # evaluator — even if a bngmodel is also threaded through.
        from bionetgen.core.tools.bngsim_bridge import _write_bngsim_results

        func_data = np.array([[7.0], [8.0], [9.0]])
        result = _make_mock_result(
            obs_names=["obsA"], obs_data=np.zeros((3, 1)), n_times=3,
            func_names=["actually_from_bngsim"], func_data=func_data,
        )

        # Provide a bngmodel that would evaluate to different values, just
        # to prove the fallback didn't run.
        bngmodel = MagicMock()
        bngmodel.functions.items = {
            "decoy": MagicMock(expr="999", args=[]),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_bngsim_results(
                result, tmpdir, "ode",
                print_functions=True,
                bngmodel=bngmodel,
                bngmodel_params={},
            )
            with open(os.path.join(tmpdir, "ode.gdat")) as f:
                header = f.readline()
        assert "actually_from_bngsim" in header
        assert "decoy" not in header


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
        # Must NOT save_concentrations: callers (e.g. parameter_scan) hold a
        # snapshot of post-time-course state that this overlay must not clobber.
        model.save_concentrations.assert_not_called()

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
        model.save_concentrations.assert_not_called()


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

    def test_simulate_nf_does_not_need_network(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"})
        assert _actions_need_network([action]) is False

    def test_parameter_scan_ode(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "5", "method": "ode", "t_end": "100",
            "n_steps": "10",
        })
        assert _actions_need_network([action]) is True

    def test_parameter_scan_nf_does_not_need_network(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        action = _make_action("parameter_scan", {
            "parameter": "k1", "par_min": "0.1", "par_max": "10",
            "n_scan_pts": "5", "method": "nf", "t_end": "100",
            "n_steps": "10",
        })
        assert _actions_need_network([action]) is False

    def test_nf_state_actions_do_not_force_network(self):
        from bionetgen.core.tools.bngsim_bridge import _actions_need_network

        actions = [
            _make_action("setParameter", {'"k1"': None, '5.0': None}),
            _make_action("addConcentration", {'"A(b)"': None, '50': None}),
            _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"}),
        ]
        assert _actions_need_network(actions) is False


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
             patch(f"{BRIDGE}._run_bifurcate_bngsim") as mock_bif, \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action], model, tmpdir, "test_model",
            )
            mock_bif.assert_called_once()

    def test_simulate_with_suffix(self):
        action = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "10", "suffix": "test_suffix",
        })
        result, model, mock_bngsim, mock_sim = self._run([action])
        assert result.process_return == 0

    def test_simulate_pla_skipped_for_bng2pl(self):
        # BNGsim has no PLA method, so simulate_pla is preserved in the BNGL
        # for BNG2.pl to run, and the bridge silently skips it during BNGsim
        # execution. The bridge must not call bngsim.Simulator for it.
        action = _make_action("simulate_pla", {"t_end": "100", "n_steps": "10"})
        result, model, mock_bngsim, mock_sim = self._run([action])
        assert result.process_return == 0
        mock_bngsim.Simulator.assert_not_called()
        mock_sim.run.assert_not_called()

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

    def test_pure_nf_actions_work_without_network_model(self):
        """Pure NF workflows should not require a generated .net model."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        actions = [
            _make_action("setParameter", {'"kf"': None, '2.0': None}),
            _make_action("addConcentration", {'"A(b)"': None, '50': None}),
            _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"}),
        ]

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             patch(f"{BRIDGE}.run_nfsim") as mock_run_nfsim, \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            _execute_bngsim_actions(
                actions, None, tmpdir, "test_model",
                xml_path=xml_path,
            )

        mock_run_nfsim.assert_called_once()
        assert mock_run_nfsim.call_args.kwargs["param_overrides"] == {"kf": 2.0}
        assert mock_run_nfsim.call_args.kwargs["conc_deltas"] == {"A(b)": 50}


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

    def test_set_parameter_invalidates_simulator_cache(self):
        # Pin the protocol path's defensive simulator-rebuild behavior on
        # setParameter so it stays consistent with _execute_bngsim_actions.
        # (BNGsim simulators read params fresh on each run(), so this is a
        # consistency guarantee rather than a correctness fix.)
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'simulate_ode({t_end=>50,n_steps=>10})',
            'setParameter("k1", 5.0)',
            'simulate_ode({t_end=>100,n_steps=>10})',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            # Initial ODE sim + rebuild after setParameter = at least 2.
            assert mock_bngsim.Simulator.call_count >= 2

    def test_set_concentration_invalidates_simulator_cache(self):
        # Pin the protocol path's defensive simulator-rebuild behavior on
        # setConcentration so it stays consistent with _execute_bngsim_actions.
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'simulate_ode({t_end=>50,n_steps=>10})',
            'setConcentration("S1", 200.0)',
            'simulate_ode({t_end=>100,n_steps=>10})',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            assert mock_bngsim.Simulator.call_count >= 2

    def test_reset_parameters_preserves_psa_poplevel(self):
        # Regression: resetParameters used to rebuild the simulator eagerly
        # with method=current_method and codegen_kw. For PSA this crashed
        # unconditionally (BNGsim raises ValueError because poplevel is
        # required); for SSA/PSA with codegen it crashed because BNGsim
        # rejects codegen=True for non-ODE methods. The fix defers rebuild
        # to the next simulate line, which branches correctly.
        from bionetgen.core.tools.bngsim_bridge import _run_protocol

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_result = _make_mock_result()
        mock_sim.run.return_value = mock_result
        mock_bngsim.Simulator.return_value = mock_sim

        lines = [
            'saveParameters()',
            'simulate({method=>"psa",poplevel=>50,t_end=>50,n_steps=>10})',
            'setParameter("k1", 5.0)',
            'resetParameters()',
            'simulate({method=>"psa",poplevel=>50,t_end=>100,n_steps=>10})',
        ]

        with patch(f"{BRIDGE}.bngsim", mock_bngsim):
            _run_protocol(model, lines)
            psa_calls = [
                c for c in mock_bngsim.Simulator.call_args_list
                if c.kwargs.get("method") == "psa"
            ]
            # Two psa simulate lines → at least two psa rebuilds, both with poplevel.
            assert len(psa_calls) >= 2
            for c in psa_calls:
                assert c.kwargs.get("poplevel") == 50.0


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

    def test_pure_nf_flow_skips_generate_network(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")
            xml_path = os.path.join(tmpdir, "test.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            sim_action = _make_action("simulate_nf", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock(process_return=0)

            mock_execute = MagicMock(return_value=MagicMock(process_return=0))

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", MagicMock()), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", mock_execute):

                result = run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")

            assert result.process_return == 0
            added_actions = [call.args[0] for call in mock_model.add_action.call_args_list]
            assert "generate_network" not in added_actions
            assert added_actions == ["writeXML"]
            assert mock_execute.call_args.args[1] is None

    def test_preserves_generate_network_max_stoich(self):
        # When the original BNGL specifies max_stoich on generate_network,
        # the bridge must not silently drop it — without the bound, BNG2.pl
        # tries to enumerate the full network and hangs/explodes for models
        # like blbr_dembo1978_with_rings.
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")

            mock_model = MagicMock()
            mock_model.model_name = "test"
            gen_action = _make_action(
                "generate_network",
                {"overwrite": "1", "max_stoich": "{R=>6}"},
            )
            sim_action = _make_action("simulate_ode", {"t_end": "100", "n_steps": "10"})
            mock_model.actions.items = [gen_action, sim_action]
            mock_model.actions.clear_actions = MagicMock()
            mock_model.add_action = MagicMock()
            mock_model.write_model = MagicMock()

            mock_cli = MagicMock()
            mock_cli.result = MagicMock(process_return=0)

            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_bngsim = MagicMock()
            mock_bngsim.Model.from_net.return_value = _make_mock_model()
            mock_execute = MagicMock(return_value=MagicMock(process_return=0))

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", mock_execute):

                run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath")

            gen_calls = [
                call for call in mock_model.add_action.call_args_list
                if call.args[0] == "generate_network"
            ]
            assert len(gen_calls) == 1
            args = gen_calls[0].args[1]
            assert args.get("max_stoich") == "{R=>6}"
            assert args.get("overwrite") == 1

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

                _result = run_bngl_with_bngsim(
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

                with pytest.raises(BNGSimError, match=r"BNG2\.pl failed"):
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

                with pytest.raises(BNGSimError, match=r"Expected \.net file"):
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
            # Constant initializers (numeric literals) are filtered out — they
            # don't need re-evaluation when scan parameters change, and including
            # them would force the slow sequential path and clobber any saved
            # concentration snapshot. Only parameter expressions remain.
            result = _parse_net_species_initializers(path)
            assert result == [("@b::X(p~1,y)", "k_init*100")]
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

        mock_result = _make_mock_result(
            obs_names=["A"], obs_data=np.array([[1.0], [2.0]]), n_times=2,
        )
        mock_bngsim, session = _make_mock_bngsim_with_nfsim_session(mock_result)

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             tempfile.TemporaryDirectory() as tmpdir:

            xml_path = os.path.join(tmpdir, "model.xml")
            with open(xml_path, "w") as f:
                f.write("<model/>")

            _run_nfsim_scan(xml_path, action, tmpdir, "test_model")
            scan_file = os.path.join(tmpdir, "test_model_scan.scan")
            assert os.path.isfile(scan_file)
            # 2 scan points = 2 NfsimSession contexts
            assert mock_bngsim.NfsimSession.call_count == 2
            assert session.initialize.call_count == 2


# ─── Regression: parameter-name resolution in setParameter / setConcentration ──


class TestParamNameResolution:
    """setParameter / setConcentration / addConcentration values can reference
    model parameter names, not just literal numbers. The bridge must resolve
    those names against the loaded BNGsim model."""

    def _run_actions(self, actions, model):
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_sim.run.return_value = _make_mock_result()
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(actions, model, tmpdir, "test_model")

    def test_set_parameter_resolves_param_name(self):
        """setParameter("kf", k_init) must look up k_init in the model."""
        model = _make_mock_model(
            param_names=["kf", "k_init"],
            params={"kf": 0.0, "k_init": 0.42},
        )
        action = _make_action("setParameter", {'"kf"': None, "k_init": None})
        self._run_actions([action], model)
        model.set_param.assert_called_with("kf", 0.42)

    def test_set_concentration_resolves_expression(self):
        """setConcentration("S0", I0 * kfactor) must resolve both names."""
        model = _make_mock_model(
            param_names=["I0", "kfactor"],
            params={"I0": 100.0, "kfactor": 0.5},
        )
        action = _make_action(
            "setConcentration", {'"S0"': None, "I0*kfactor": None},
        )
        self._run_actions([action], model)
        model.set_concentration.assert_called_with("S0", 50.0)

    def test_add_concentration_resolves_param_name(self):
        """addConcentration("S0", delta) must resolve `delta` from the model."""
        model = _make_mock_model(
            param_names=["delta"],
            params={"delta": 25.0},
        )
        model.get_concentration.return_value = 100.0
        action = _make_action(
            "addConcentration", {'"S0"': None, "delta": None},
        )
        self._run_actions([action], model)
        # 100 (current) + 25 (delta) = 125 on the network model side
        model.set_concentration.assert_called_with("S0", 125.0)

    def test_unresolved_name_still_raises(self):
        """A value that names an unknown variable must still fail loudly."""
        from bionetgen.core.tools.bngsim_bridge import _eval_numeric

        with pytest.raises(ValueError, match="Cannot evaluate"):
            _eval_numeric("not_a_param", extra_ns={"k_known": 1.0})

    def test_resolve_bngmodel_params_iterative(self):
        """_resolve_bngmodel_params handles parameters that reference each other,
        regardless of declaration order."""
        from bionetgen.core.tools.bngsim_bridge import _resolve_bngmodel_params

        bngmodel = MagicMock()
        bngmodel.parameters.items = {
            "RT": MagicMock(value="30"),
            "LT_low": MagicMock(value="150"),
            "LT_peak": MagicMock(value="2*RT + LT_low"),
            "koff": MagicMock(value="0.01"),
            "jm": MagicMock(value="koff"),
        }
        resolved = _resolve_bngmodel_params(bngmodel)
        assert resolved["RT"] == 30.0
        assert resolved["LT_low"] == 150.0
        assert resolved["LT_peak"] == 210.0
        assert resolved["koff"] == 0.01
        assert resolved["jm"] == 0.01

    def test_resolve_bngmodel_params_unresolvable(self):
        """Unresolvable parameters are silently dropped, not raised."""
        from bionetgen.core.tools.bngsim_bridge import _resolve_bngmodel_params

        bngmodel = MagicMock()
        bngmodel.parameters.items = {
            "good": MagicMock(value="42"),
            "bad": MagicMock(value="some_undefined_name"),
        }
        resolved = _resolve_bngmodel_params(bngmodel)
        assert resolved == {"good": 42.0}

    def test_evaluate_bngmodel_functions_basic(self):
        """Functions resolve against parameters + observables in declaration
        order; the helper is what backfills NFsim scan output where
        BNGsim's NFsim binding leaves Result.expressions empty."""
        from bionetgen.core.tools.bngsim_bridge import _evaluate_bngmodel_functions

        bngmodel = MagicMock()
        bngmodel.functions.items = {
            "w1": MagicMock(expr="Obs_NCL_R1 / RT_1", args=[]),
            "w2": MagicMock(expr="Obs_NCL_R2 / RT_2", args=[]),
            "x_poly": MagicMock(
                expr="1 - (Obs_NCL_R1 + Obs_NCL_R2) / RT", args=[],
            ),
        }
        names, vals = _evaluate_bngmodel_functions(
            bngmodel,
            base_params={"RT": 50.0, "RT_1": 25.0, "RT_2": 25.0},
            obs_dict={"Obs_NCL_R1": 11.0, "Obs_NCL_R2": 5.0},
        )
        assert names == ["w1", "w2", "x_poly"]
        assert vals == pytest.approx([11.0 / 25.0, 5.0 / 25.0, 1 - 16.0 / 50.0])

    def test_evaluate_bngmodel_functions_chained(self):
        """One function may reference another via name() — the iterative
        resolver handles that without forcing the user to pre-order."""
        from bionetgen.core.tools.bngsim_bridge import _evaluate_bngmodel_functions

        bngmodel = MagicMock()
        # b() depends on a(), declared after; a() resolves first then b().
        bngmodel.functions.items = {
            "b": MagicMock(expr="2 * a()", args=[]),
            "a": MagicMock(expr="x + 1", args=[]),
        }
        names, vals = _evaluate_bngmodel_functions(
            bngmodel, base_params={"x": 3.0}, obs_dict={},
        )
        assert set(names) == {"a", "b"}
        d = dict(zip(names, vals))
        assert d["a"] == pytest.approx(4.0)
        assert d["b"] == pytest.approx(8.0)

    def test_evaluate_bngmodel_functions_skips_parameterized(self):
        """Functions taking arguments aren't scan columns — skip them."""
        from bionetgen.core.tools.bngsim_bridge import _evaluate_bngmodel_functions

        bngmodel = MagicMock()
        bngmodel.functions.items = {
            "f": MagicMock(expr="x*2", args=["x"]),
            "g": MagicMock(expr="42", args=[]),
        }
        names, vals = _evaluate_bngmodel_functions(
            bngmodel, base_params={}, obs_dict={},
        )
        assert names == ["g"]
        assert vals == [42.0]

    def test_evaluate_bngmodel_functions_drops_unresolvable(self):
        """Functions referencing unknown names are dropped, not raised."""
        from bionetgen.core.tools.bngsim_bridge import _evaluate_bngmodel_functions

        bngmodel = MagicMock()
        bngmodel.functions.items = {
            "good": MagicMock(expr="x + 1", args=[]),
            "bad": MagicMock(expr="some_unknown_name * 2", args=[]),
        }
        names, vals = _evaluate_bngmodel_functions(
            bngmodel, base_params={"x": 5.0}, obs_dict={},
        )
        assert names == ["good"]
        assert vals == [6.0]

    def test_evaluate_bngmodel_functions_empty(self):
        """No functions block, or empty items — return ([], []) cleanly."""
        from bionetgen.core.tools.bngsim_bridge import _evaluate_bngmodel_functions

        assert _evaluate_bngmodel_functions(None, {}, {}) == ([], [])
        empty = MagicMock()
        empty.functions = None
        assert _evaluate_bngmodel_functions(empty, {}, {}) == ([], [])
        empty.functions = MagicMock()
        empty.functions.items = {}
        assert _evaluate_bngmodel_functions(empty, {}, {}) == ([], [])

    def test_pure_nf_uses_bngmodel_params_fallback(self):
        """Pure-NF runs (bngsim_model is None) must resolve parameter
        names from the parsed bngmodel parameter block."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        # Pure-NF action sequence: setParameter then setConcentration
        # using the parameter that was just set.
        set_p = _make_action("setParameter", {'"LT_current"': None, "LT_low": None})
        set_c = _make_action(
            "setConcentration", {'"L(r,r)"': None, "LT_current": None},
        )
        sim_nf = _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"})

        mock_bngsim = MagicMock()
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
                [set_p, set_c, sim_nf], None, tmpdir, "test_model",
                xml_path=xml_path,
                bngmodel_params={"LT_low": 150.0, "LT_current": 331.0},
            )

            kwargs = mock_run_nfsim.call_args.kwargs
            # setParameter should have stored 150.0 (the resolved value of LT_low)
            assert kwargs["param_overrides"] == {"LT_current": 150.0}
            # setConcentration("L(r,r)", LT_current) should see LT_current=150
            # (the live value after setParameter), not 331 (the BNGL block default).
            assert kwargs["conc_overrides"] == {"L(r,r)": 150}


# ─── Regression: continue=>1 must append, not overwrite ─────────────


class TestContinueAppendsOutput:
    """A second simulate() with continue=>1 and no suffix change must
    append rows to the prior segment's .gdat / .cdat instead of clobbering
    them. Matches BNG2.pl ``run_network -x`` semantics."""

    def test_two_segment_continue_appends(self):
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        # First segment: t = 0..50 (5 rows including endpoints)
        seg1_n = 6
        seg1_time = np.linspace(0.0, 50.0, seg1_n)
        seg1_obs = np.column_stack([
            np.linspace(100.0, 50.0, seg1_n),
            np.linspace(0.0, 50.0, seg1_n),
        ])
        seg1 = _make_mock_result(
            obs_names=["A", "B"], obs_data=seg1_obs, n_times=seg1_n, time=seg1_time,
        )
        seg1.species = np.zeros((seg1_n, 1))

        # Second segment: t = 50..100 (6 rows; first row duplicates seg1's tail)
        seg2_n = 6
        seg2_time = np.linspace(50.0, 100.0, seg2_n)
        seg2_obs = np.column_stack([
            np.linspace(50.0, 25.0, seg2_n),
            np.linspace(50.0, 75.0, seg2_n),
        ])
        seg2 = _make_mock_result(
            obs_names=["A", "B"], obs_data=seg2_obs, n_times=seg2_n, time=seg2_time,
        )
        seg2.species = np.zeros((seg2_n, 1))

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_sim.run.side_effect = [seg1, seg2]
        mock_bngsim.Simulator.return_value = mock_sim

        # Make to_cdat write a real file so the append branch sees it exist.
        def _fake_to_cdat(self_result, path):
            with open(path, "w") as fh:
                fh.write("# time S1\n")
                for t in self_result.time:
                    fh.write(f"  {t:22.12e}  0.000000000000e+00\n")

        seg1.to_cdat.side_effect = lambda path: _fake_to_cdat(seg1, path)
        seg2.to_cdat.side_effect = lambda path: _fake_to_cdat(seg2, path)

        action1 = _make_action("simulate_ode", {"t_end": "50", "n_steps": "5"})
        action2 = _make_action("simulate_ode", {
            "t_end": "100", "n_steps": "5", "continue": "1",
        })

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action1, action2], model, tmpdir, "test_model",
            )

            gdat = os.path.join(tmpdir, "test_model.gdat")
            cdat = os.path.join(tmpdir, "test_model.cdat")
            assert os.path.isfile(gdat)
            assert os.path.isfile(cdat)

            with open(gdat) as fh:
                gdat_lines = fh.readlines()
            # 1 header + 6 (seg1) + 5 (seg2 minus duplicate t=50) = 12 lines
            assert gdat_lines[0].startswith("# ")
            assert len(gdat_lines) == 1 + seg1_n + (seg2_n - 1)

            # The duplicate t=50 row from seg2 must be dropped on append
            time_col_values = [float(line.split()[0]) for line in gdat_lines[1:]]
            assert time_col_values[seg1_n - 1] == pytest.approx(50.0)
            assert time_col_values[seg1_n] == pytest.approx(60.0)
            assert time_col_values[-1] == pytest.approx(100.0)

            # cdat: 1 header + same 11 data rows
            with open(cdat) as fh:
                cdat_lines = fh.readlines()
            assert cdat_lines[0].startswith("# ")
            assert len(cdat_lines) == 1 + seg1_n + (seg2_n - 1)

    def test_continue_with_different_suffix_does_not_append(self):
        """If continue=>1 but suffix changes, the second segment is a
        separate output stream — write fresh, don't append to a foreign file."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        seg1 = _make_mock_result(n_times=4, time=np.linspace(0, 10, 4))
        seg2 = _make_mock_result(n_times=4, time=np.linspace(10, 20, 4))

        def _fake_to_cdat(self_result, path):
            with open(path, "w") as fh:
                fh.write("# time S1\n")
                for t in self_result.time:
                    fh.write(f"  {t:22.12e}  0.000000000000e+00\n")

        seg1.to_cdat.side_effect = lambda path: _fake_to_cdat(seg1, path)
        seg2.to_cdat.side_effect = lambda path: _fake_to_cdat(seg2, path)

        model = _make_mock_model()
        mock_bngsim = MagicMock()
        mock_sim = MagicMock()
        mock_sim.run.side_effect = [seg1, seg2]
        mock_bngsim.Simulator.return_value = mock_sim

        action1 = _make_action("simulate_ode", {
            "t_end": "10", "n_steps": "3", "suffix": "first",
        })
        action2 = _make_action("simulate_ode", {
            "t_end": "20", "n_steps": "3", "continue": "1", "suffix": "second",
        })

        with patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}._try_prepare_codegen", return_value=""), \
             patch(f"{BRIDGE}._parse_net_species_initializers", return_value=[]), \
             tempfile.TemporaryDirectory() as tmpdir:
            _execute_bngsim_actions(
                [action1, action2], model, tmpdir, "test_model",
            )

            first_gdat = os.path.join(tmpdir, "test_model_first.gdat")
            second_gdat = os.path.join(tmpdir, "test_model_second.gdat")
            assert os.path.isfile(first_gdat)
            assert os.path.isfile(second_gdat)
            with open(second_gdat) as fh:
                lines = fh.readlines()
            # Different out_name → fresh write with header + all rows
            assert lines[0].startswith("# ")
            assert len(lines) == 1 + 4


# ─── Regression: addConcentration must propagate to NFsim as a delta ────


class TestAddConcentrationNfsimDelta:
    """addConcentration must NOT use the network model's count as the
    NFsim absolute target — NFsim's live count diverges from the network
    model. Track an additive delta instead."""

    def test_addconcentration_alone_becomes_delta(self):
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        add = _make_action("addConcentration", {'"A(b)"': None, '50': None})
        sim_nf = _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"})

        model = _make_mock_model()
        # Network-model concentration is 100 — older code would have set
        # nf_conc_overrides["A(b)"] = 150. The fix tracks the delta only.
        model.get_concentration.return_value = 100.0

        mock_bngsim = MagicMock()
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
                [add, sim_nf], model, tmpdir, "test_model",
                xml_path=xml_path,
            )

            mock_run_nfsim.assert_called_once()
            kwargs = mock_run_nfsim.call_args.kwargs
            # No prior setConcentration → no overrides, just a delta
            assert kwargs.get("conc_overrides") in (None, {})
            assert kwargs.get("conc_deltas") == {"A(b)": 50}

    def test_addconcentration_after_setconcentration_bumps_override(self):
        """If setConcentration set an absolute target, a subsequent
        addConcentration should bump that override (not start a new delta)."""
        from bionetgen.core.tools.bngsim_bridge import _execute_bngsim_actions

        set_c = _make_action("setConcentration", {'"A(b)"': None, '200': None})
        add = _make_action("addConcentration", {'"A(b)"': None, '50': None})
        sim_nf = _make_action("simulate_nf", {"t_end": "10", "n_steps": "10"})

        model = _make_mock_model()
        model.get_concentration.return_value = 200.0

        mock_bngsim = MagicMock()
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
                [set_c, add, sim_nf], model, tmpdir, "test_model",
                xml_path=xml_path,
            )

            mock_run_nfsim.assert_called_once()
            kwargs = mock_run_nfsim.call_args.kwargs
            assert kwargs.get("conc_overrides") == {"A(b)": 250}
            # Bumped via override, not a separate delta
            assert kwargs.get("conc_deltas") in (None, {})


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


# ─── Regression: BIONETGEN_SS_WORKERS env var ───────────────────────


class TestSsWorkersEnv:
    def test_default_when_unset(self, monkeypatch):
        from bionetgen.core.tools.bngsim_bridge import (
            _DEFAULT_SS_WORKERS, _resolve_ss_workers,
        )
        monkeypatch.delenv("BIONETGEN_SS_WORKERS", raising=False)
        assert _resolve_ss_workers() == _DEFAULT_SS_WORKERS

    def test_overrides_default(self, monkeypatch):
        from bionetgen.core.tools.bngsim_bridge import _resolve_ss_workers
        monkeypatch.setenv("BIONETGEN_SS_WORKERS", "8")
        assert _resolve_ss_workers() == 8

    def test_invalid_falls_back_to_default(self, monkeypatch):
        from bionetgen.core.tools.bngsim_bridge import _resolve_ss_workers
        monkeypatch.setenv("BIONETGEN_SS_WORKERS", "not-a-number")
        assert _resolve_ss_workers(default=3) == 3

    def test_zero_or_negative_falls_back(self, monkeypatch):
        from bionetgen.core.tools.bngsim_bridge import _resolve_ss_workers
        monkeypatch.setenv("BIONETGEN_SS_WORKERS", "0")
        assert _resolve_ss_workers(default=2) == 2
        monkeypatch.setenv("BIONETGEN_SS_WORKERS", "-1")
        assert _resolve_ss_workers(default=2) == 2


# ─── Regression: backslash-continued tfun() in functions block ──────


class TestParseTableFunctionsMultilineTfun:
    """The inline-array ``tfun()`` form often spans multiple physical
    lines via BNGL's ``\\`` continuation. ``_parse_table_functions``
    must join those before scanning, otherwise the table function never
    gets registered with BNGsim and the codegen path segfaults."""

    def test_multiline_tfun_is_recognized(self, tmp_path):
        from bionetgen.core.tools.bngsim_bridge import _parse_table_functions

        bngl = tmp_path / "model.bngl"
        bngl.write_text(
            "begin model\n"
            "begin parameters\n  IPTG 0\nend parameters\n"
            "begin functions\n"
            "  exp_gfp() = tfun(\\\n"
            "    [0, 1e-5, 1e-2],\\\n"
            "    [0.03, 0.5, 0.99],\\\n"
            "    IPTG)\n"
            "end functions\n"
            "end model\n"
        )
        specs = _parse_table_functions(str(bngl))
        assert len(specs) == 1
        spec = specs[0]
        assert spec["name"] == "exp_gfp"
        assert spec["index"] == "IPTG"
        assert spec["times"] == [0.0, 1e-5, 1e-2]
        assert spec["values"] == [0.03, 0.5, 0.99]
        assert spec["method"] == "linear"

    def test_singleline_tfun_still_recognized(self, tmp_path):
        from bionetgen.core.tools.bngsim_bridge import _parse_table_functions

        bngl = tmp_path / "model.bngl"
        bngl.write_text(
            "begin model\n"
            "begin parameters\n  x 0\nend parameters\n"
            "begin functions\n"
            "  f() = tfun([0,1,2], [10,20,30], x)\n"
            "end functions\n"
            "end model\n"
        )
        specs = _parse_table_functions(str(bngl))
        assert len(specs) == 1
        assert specs[0]["times"] == [0.0, 1.0, 2.0]


# ─── Regression: codegen+tfun guard ────────────────────────────────


class TestCodegenTfunGuard:
    """BNGsim's codegen .so calls a ``tfun_eval`` function pointer that is
    set up by NetworkModel post-load. The current code path is fragile
    (segfaults if the stars don't align), so the bridge skips codegen
    when the .net file uses ``tfun()``. Interpreted RHS handles tfun fine."""

    def test_net_with_tfun_skips_codegen(self, tmp_path):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        net = tmp_path / "model.net"
        net.write_text(
            "# Created by BioNetGen\n"
            "begin parameters\n  1 IPTG 0\nend parameters\n"
            "begin functions\n"
            "  1 f() tfun([0,1],[10,20],IPTG)\n"
            "end functions\n"
        )
        # _try_prepare_codegen short-circuits to "" when net has tfun
        assert _try_prepare_codegen(str(net)) == ""

    def test_net_without_tfun_attempts_codegen(self, tmp_path, monkeypatch):
        """Sanity: net files without tfun() still go through prepare_codegen."""
        from bionetgen.core.tools import bngsim_bridge as bb

        net = tmp_path / "model.net"
        net.write_text(
            "# Created by BioNetGen\n"
            "begin parameters\n  1 k 0.1\nend parameters\n"
            "begin functions\n"
            "  1 g() k*2\n"
            "end functions\n"
        )
        called = {"yes": False}

        def fake_prepare(_):
            called["yes"] = True
            return "/tmp/fake.so"

        # Patch the lazy import inside _try_prepare_codegen via the bngsim
        # module attribute it pulls from.
        import bngsim
        monkeypatch.setattr(bngsim, "prepare_codegen", fake_prepare, raising=False)

        out = bb._try_prepare_codegen(str(net))
        assert called["yes"]
        assert out == "/tmp/fake.so"

    def test_no_codegen_env_short_circuits(self, tmp_path, monkeypatch):
        from bionetgen.core.tools.bngsim_bridge import _try_prepare_codegen

        monkeypatch.setenv("BIONETGEN_NO_CODEGEN", "1")
        net = tmp_path / "model.net"
        net.write_text("begin parameters\nend parameters\n")
        assert _try_prepare_codegen(str(net)) == ""
