import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bionetgen.modelapi.structs import Action

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


def _make_action(action_type, action_args=None):
    return Action(action_type=action_type, action_args=action_args or {})


def _make_mock_bngmodel(action):
    model = MagicMock()
    model.model_name = "test"
    model.actions.items = [action]
    model.actions.clear_actions = MagicMock()
    model.add_action = MagicMock()
    model.write_model = MagicMock()
    return model


class TestRunBnglWithBngsimMethodOverrides:
    def test_preserves_declared_method_without_override(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")
            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_model = _make_mock_bngmodel(
                _make_action("simulate_nf", {"t_end": "10", "n_steps": "5"})
            )
            mock_cli = MagicMock()
            mock_cli.result = MagicMock(process_return=0)

            mock_bngsim = MagicMock()
            mock_bngsim.Model.from_net.return_value = MagicMock()

            captured = {}

            def fake_execute(actions, *args, **kwargs):
                captured["types"] = [action.type for action in actions]
                return MagicMock(process_return=0)

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", side_effect=fake_execute):

                run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath", method=None)

            assert captured["types"] == ["simulate_nf"]

    def test_explicit_override_rewrites_declared_method(self):
        from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

        with tempfile.TemporaryDirectory() as tmpdir:
            bngl_path = os.path.join(tmpdir, "test.bngl")
            net_path = os.path.join(tmpdir, "test.net")
            with open(net_path, "w") as f:
                f.write("# empty net\n")

            mock_model = _make_mock_bngmodel(
                _make_action("simulate_nf", {"t_end": "10", "n_steps": "5"})
            )
            mock_cli = MagicMock()
            mock_cli.result = MagicMock(process_return=0)

            mock_bngsim = MagicMock()
            mock_bngsim.Model.from_net.return_value = MagicMock()

            captured = {}

            def fake_execute(actions, *args, **kwargs):
                captured["types"] = [action.type for action in actions]
                return MagicMock(process_return=0)

            with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
                 patch(f"{BRIDGE}.bngsim", mock_bngsim), \
                 patch("bionetgen.modelapi.model.bngmodel", return_value=mock_model), \
                 patch(f"{BRIDGE}._parse_protocol_block", return_value=[]), \
                 patch(f"{BRIDGE}._parse_table_functions", return_value=[]), \
                 patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli), \
                 patch(f"{BRIDGE}._execute_bngsim_actions", side_effect=fake_execute):

                run_bngl_with_bngsim(bngl_path, tmpdir, "/bngpath", method="ode")

            assert captured["types"] == ["simulate_ode"]


class TestLibraryMethodDefaults:
    def test_bngl_run_passes_no_method_override_by_default(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
            BngsimRouteDecision,
        )
        from bionetgen.modelapi.runner import run

        sentinel = object()

        with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
             patch(
                 f"{BRIDGE}.classify_bngsim_route",
                 return_value=BngsimRouteDecision(ROUTE_BNGL_BNGSIM, "atomic BNGL"),
             ), \
             patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=sentinel) as mock_run:

            result = run("model.bngl", out="/tmp/out")

        assert result is sentinel
        assert mock_run.call_args.kwargs["method"] is None

    def test_bngl_run_passes_explicit_method_override(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
            BngsimRouteDecision,
        )
        from bionetgen.modelapi.runner import run

        sentinel = object()

        with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
             patch(
                 f"{BRIDGE}.classify_bngsim_route",
                 return_value=BngsimRouteDecision(ROUTE_BNGL_BNGSIM, "atomic BNGL"),
             ), \
             patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=sentinel) as mock_run:

            result = run("model.bngl", out="/tmp/out", method="ode")

        assert result is sentinel
        assert mock_run.call_args.kwargs["method"] == "ode"


class TestCliMethodDefaults:
    def test_cli_run_passes_no_method_override_by_default(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
            BngsimRouteDecision,
        )
        from bionetgen.main import BioNetGenTest

        with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
             patch(
                 f"{BRIDGE}.classify_bngsim_route",
                 return_value=BngsimRouteDecision(ROUTE_BNGL_BNGSIM, "atomic BNGL"),
             ), \
             patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=MagicMock()) as mock_run, \
             patch("bionetgen.main.test_perl"):

            with BioNetGenTest(argv=["run", "-i", "model.bngl", "-o", "/tmp/out"]) as app:
                app.run()

        assert mock_run.call_args.kwargs["method"] is None
        assert mock_run.call_args.kwargs["timeout"] is None

    def test_cli_run_passes_explicit_method_override(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
            BngsimRouteDecision,
        )
        from bionetgen.main import BioNetGenTest

        with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
             patch(
                 f"{BRIDGE}.classify_bngsim_route",
                 return_value=BngsimRouteDecision(ROUTE_BNGL_BNGSIM, "atomic BNGL"),
             ), \
             patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=MagicMock()) as mock_run, \
             patch("bionetgen.main.test_perl"):

            with BioNetGenTest(
                argv=["run", "-i", "model.bngl", "-o", "/tmp/out", "--method", "ode"]
            ) as app:
                app.run()

        assert mock_run.call_args.kwargs["method"] == "ode"

    def test_cli_run_passes_timeout_override(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
            BngsimRouteDecision,
        )
        from bionetgen.main import BioNetGenTest

        with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
             patch(
                 f"{BRIDGE}.classify_bngsim_route",
                 return_value=BngsimRouteDecision(ROUTE_BNGL_BNGSIM, "atomic BNGL"),
             ), \
             patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=MagicMock()) as mock_run, \
             patch("bionetgen.main.test_perl"):

            with BioNetGenTest(
                argv=["run", "-i", "model.bngl", "-o", "/tmp/out", "--timeout", "17"]
            ) as app:
                app.run()

        assert mock_run.call_args.kwargs["timeout"] == 17


class TestDirectInputMethodDefaults:
    def test_direct_net_input_defaults_to_ode(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        mock_bngsim = MagicMock()
        mock_model = MagicMock()
        mock_bngsim.Model.from_net.return_value = mock_model
        mock_sim = MagicMock()
        mock_sim.run.return_value = MagicMock(
            to_cdat=MagicMock(),
            observable_names=[],
            n_observables=0,
            n_times=2,
        )
        mock_bngsim.Simulator.return_value = mock_sim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()), \
             tempfile.TemporaryDirectory() as tmpdir:

            run_with_bngsim("/model.net", tmpdir, fmt="net", method=None)

        mock_bngsim.Simulator.assert_called_once_with(mock_model, method="ode")

    def test_direct_bng_xml_input_defaults_to_nf(self):
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        sentinel = object()

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.run_nfsim", return_value=sentinel) as mock_run:

            result = run_with_bngsim("/model.xml", "/tmp/out", fmt="bng-xml", method=None)

        assert result is sentinel
        mock_run.assert_called_once()

    def test_direct_bng_xml_input_rejects_ode_override(self):
        from bionetgen.core.exc import BNGSimError
        from bionetgen.core.tools.bngsim_bridge import run_with_bngsim

        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True):
            with pytest.raises(BNGSimError, match="network-free simulation"):
                run_with_bngsim("/model.xml", "/tmp/out", fmt="bng-xml", method="ode")
