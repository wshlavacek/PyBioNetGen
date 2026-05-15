import textwrap
from unittest.mock import MagicMock, patch

import pytest

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


class _Action:
    def __init__(self, action_type, action_args=None):
        self.type = action_type
        self.name = action_type
        self.args = action_args or {}


def _action(action_type, action_args=None):
    return _Action(action_type, action_args)


def _classify(fmt, actions=None, **kwargs):
    from bionetgen.core.tools.bngsim_bridge import classify_bngsim_route

    return classify_bngsim_route(
        "model.bngl" if fmt == "bngl" else f"model.{fmt}",
        fmt,
        bngl_actions=actions,
        has_protocol=False,
        **kwargs,
    )


def _write_minimal_bngl(tmp_path, action_text):
    path = tmp_path / "stage3_complex.bngl"
    path.write_text(
        textwrap.dedent(
            f"""\
            begin model
            begin parameters
              k 1
            end parameters
            begin molecule types
              A()
            end molecule types
            begin seed species
              A() 1
            end seed species
            begin observables
              Molecules A A()
            end observables
            begin reaction rules
            end reaction rules
            {action_text}
            end model
            """
        ),
        encoding="utf-8",
    )
    return path


COMPLEX_BNGL_ACTION_CASES = [
    pytest.param(
        'setParameter("k", 2)\nsimulate_ode({t_end=>1,n_steps=>1})',
        id="setParameter",
    ),
    pytest.param(
        'setConcentration("A()", 10)\nsimulate_ode({t_end=>1,n_steps=>1})',
        id="setConcentration",
    ),
    pytest.param(
        "saveConcentrations()\nresetConcentrations()\nsimulate_ode({t_end=>1,n_steps=>1})",
        id="save-reset-concentrations",
    ),
    pytest.param(
        "saveParameters()\nresetParameters()\nsimulate_ode({t_end=>1,n_steps=>1})",
        id="save-reset-parameters",
    ),
    pytest.param(
        'parameter_scan({method=>"ode",parameter=>"k",par_min=>0,par_max=>1,n_scan_pts=>2})',
        id="parameter-scan",
    ),
    pytest.param(
        'bifurcate({method=>"ode",parameter=>"k",par_min=>0,par_max=>1,n_scan_pts=>2})',
        id="bifurcate",
    ),
    pytest.param(
        'parameter_scan({method=>"protocol",parameter=>"k",par_min=>0,par_max=>1,n_scan_pts=>2})',
        id="protocol-parameter-scan",
    ),
    pytest.param(
        "writeSBML()\nsimulate_ode({t_end=>1,n_steps=>1})",
        id="write-action",
    ),
]


class TestBngsimRouteClassifier:
    def test_no_bngsim_for_bngl_uses_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="subprocess",
            bngsim_available=True,
            actions=[_action("simulate_ode")],
        )

        assert decision.route == ROUTE_SUBPROCESS

    def test_bngsim_unavailable_for_bngl_uses_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=False,
            actions=[_action("simulate_ode")],
        )

        assert decision.route == ROUTE_SUBPROCESS

    @pytest.mark.parametrize("fmt", ["net", "sbml", "antimony"])
    def test_direct_formats_use_bngsim_when_available(self, fmt):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_DIRECT_BNGSIM

        decision = _classify(fmt, simulator="auto", bngsim_available=True)

        assert decision.route == ROUTE_DIRECT_BNGSIM

    def test_direct_net_pla_uses_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "net",
            simulator="auto",
            bngsim_available=True,
            method="pla",
        )

        assert decision.route == ROUTE_SUBPROCESS
        assert decision.method == "pla"

    @pytest.mark.parametrize("fmt", ["sbml", "antimony"])
    def test_direct_required_formats_reject_pla(self, fmt):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_ERROR

        decision = _classify(
            fmt,
            simulator="auto",
            bngsim_available=True,
            method="pla",
        )

        assert decision.route == ROUTE_ERROR
        assert decision.method == "pla"

    def test_bng_xml_defaults_to_direct_nf(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_DIRECT_BNGSIM

        decision = _classify(
            "bng-xml",
            simulator="auto",
            bngsim_available=True,
            bngsim_has_nfsim=True,
        )

        assert decision.route == ROUTE_DIRECT_BNGSIM
        assert decision.method == "nf"

    def test_bng_xml_without_nfsim_support_uses_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bng-xml",
            simulator="auto",
            bngsim_available=True,
            bngsim_has_nfsim=False,
        )

        assert decision.route == ROUTE_SUBPROCESS
        assert decision.method == "nf"

    @pytest.mark.parametrize("fmt", ["sbml", "antimony"])
    def test_required_formats_error_without_bngsim(self, fmt):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_ERROR

        decision = _classify(fmt, simulator="auto", bngsim_available=False)

        assert decision.route == ROUTE_ERROR
        assert "requires BNGsim" in decision.reason

    @pytest.mark.parametrize(
        ("action", "expected_method"),
        [
            (_action("simulate_ode"), "ode"),
            (_action("simulate_ssa"), "ssa"),
            (_action("simulate_psa"), "psa"),
            (_action("simulate", {"method": "psa"}), "psa"),
            (_action("simulate_ssa", {"poplevel": "100"}), "psa"),
            (_action("simulate", {"method": "rm"}), "rm"),
        ],
    )
    def test_atomic_supported_bngl_methods_use_bngsim(self, action, expected_method):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_BNGL_BNGSIM

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            actions=[action],
        )

        assert decision.route == ROUTE_BNGL_BNGSIM
        assert decision.method == expected_method

    def test_bngl_method_override_preserves_legacy_psa_classification(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_BNGL_BNGSIM

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            method="ssa",
            actions=[_action("simulate_ssa", {"poplevel": "100"})],
        )

        assert decision.route == ROUTE_BNGL_BNGSIM
        assert decision.method == "psa"

    @pytest.mark.parametrize(
        "action",
        [
            _action("simulate_pla"),
            _action("simulate", {"method": "pla"}),
        ],
    )
    def test_bngl_pla_uses_subprocess(self, action):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            actions=[action],
        )

        assert decision.route == ROUTE_SUBPROCESS
        assert decision.method == "pla"

    def test_bngl_method_override_does_not_pull_pla_into_bngsim(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            method="ode",
            actions=[_action("simulate_pla")],
        )

        assert decision.route == ROUTE_SUBPROCESS
        assert decision.method == "pla"

    def test_bngl_without_simulation_actions_uses_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            actions=[],
        )

        assert decision.route == ROUTE_SUBPROCESS

    @pytest.mark.parametrize(
        "actions",
        [
            [_action("setParameter", {'"k"': None, "2": None}), _action("simulate_ode")],
            [_action("setConcentration", {'"A()"': None, "10": None}), _action("simulate_ode")],
            [_action("saveConcentrations"), _action("simulate_ode")],
            [_action("resetConcentrations"), _action("simulate_ode")],
            [_action("saveParameters"), _action("simulate_ode")],
            [_action("resetParameters"), _action("simulate_ode")],
            [_action("parameter_scan", {"method": "ode"})],
            [_action("parameter_scan", {"method": "protocol"})],
            [_action("bifurcate", {"method": "ode"})],
            [_action("writeSBML"), _action("simulate_ode")],
            [_action("simulate_ode", {"prefix": "equil"})],
            [_action("simulate_ode", {"suffix": "prod"})],
            [_action("simulate_ode", {"continue": "1"})],
            [_action("simulate_ode"), _action("simulate_ssa")],
        ],
    )
    def test_complex_bngl_workflows_use_subprocess(self, actions):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            actions=actions,
        )

        assert decision.route == ROUTE_SUBPROCESS

    def test_bngl_method_override_does_not_hide_multi_sim_workflow(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_SUBPROCESS

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            method="ode",
            actions=[_action("simulate_ode"), _action("simulate_ssa")],
        )

        assert decision.route == ROUTE_SUBPROCESS

    def test_protocol_blocks_use_subprocess(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_SUBPROCESS,
            classify_bngsim_route,
        )

        decision = classify_bngsim_route(
            "model.bngl",
            "bngl",
            simulator="auto",
            bngsim_available=True,
            bngl_actions=[_action("simulate_ode")],
            has_protocol=True,
        )

        assert decision.route == ROUTE_SUBPROCESS


@pytest.mark.parametrize("action_text", COMPLEX_BNGL_ACTION_CASES)
def test_parser_backed_complex_bngl_actions_use_subprocess(tmp_path, action_text):
    from bionetgen.core.tools.bngsim_bridge import (
        ROUTE_SUBPROCESS,
        classify_bngsim_route,
    )

    bngl_path = _write_minimal_bngl(tmp_path, action_text)

    decision = classify_bngsim_route(
        str(bngl_path),
        "bngl",
        simulator="auto",
        bngsim_available=True,
        bngsim_has_nfsim=True,
    )

    assert decision.route == ROUTE_SUBPROCESS
    assert "BNG2.pl" in decision.reason


@pytest.mark.parametrize("action_text", COMPLEX_BNGL_ACTION_CASES)
def test_library_complex_bngl_uses_bngcli_not_python_executor(tmp_path, action_text):
    from bionetgen.modelapi.runner import run

    bngl_path = _write_minimal_bngl(tmp_path, action_text)
    out_dir = tmp_path / "out"
    sentinel = object()
    mock_cli = MagicMock()
    mock_cli.result = sentinel

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
         patch(f"{BRIDGE}.run_bngl_with_bngsim") as mock_bngsim_run, \
         patch(
             f"{BRIDGE}._execute_bngsim_actions",
             side_effect=AssertionError("complex BNGL must not use Python action replay"),
         ), \
         patch("bionetgen.modelapi.runner.get_conf", return_value={"bngpath": "/fake/bng"}), \
         patch("bionetgen.modelapi.runner.BNGCLI", return_value=mock_cli) as mock_bngcli:
        result = run(str(bngl_path), out=str(out_dir))

    assert result is sentinel
    mock_bngsim_run.assert_not_called()
    mock_bngcli.assert_called_once()
    assert mock_bngcli.call_args.args[:3] == (
        str(bngl_path),
        str(out_dir),
        "/fake/bng",
    )


def test_run_bngl_with_bngsim_complex_action_falls_back_without_executor(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

    bngl_path = _write_minimal_bngl(
        tmp_path,
        'setParameter("k", 2)\nsimulate_ode({t_end=>1,n_steps=>1})',
    )
    sentinel = object()
    mock_cli = MagicMock()
    mock_cli.result = sentinel

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
         patch(
             f"{BRIDGE}._execute_bngsim_actions",
             side_effect=AssertionError("complex BNGL must not use Python action replay"),
         ), \
         patch("bionetgen.core.tools.cli.BNGCLI", return_value=mock_cli) as mock_bngcli:
        result = run_bngl_with_bngsim(str(bngl_path), str(tmp_path / "out"), "/fake/bng")

    assert result is sentinel
    mock_bngcli.assert_called_once()
    assert mock_bngcli.call_args.args[:3] == (
        str(bngl_path),
        str(tmp_path / "out"),
        "/fake/bng",
    )


def test_library_subprocess_route_uses_bngcli(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import (
        ROUTE_SUBPROCESS,
        BngsimRouteDecision,
    )
    from bionetgen.modelapi.runner import run

    sentinel = MagicMock()
    mock_cli = MagicMock()
    mock_cli.result = sentinel

    with patch(f"{BRIDGE}.detect_input_format", return_value="bngl"), \
         patch(
             f"{BRIDGE}.classify_bngsim_route",
             return_value=BngsimRouteDecision(ROUTE_SUBPROCESS, "complex BNGL"),
         ), \
         patch("bionetgen.modelapi.runner.get_conf", return_value={"bngpath": "/fake/bng"}), \
         patch("bionetgen.modelapi.runner.BNGCLI", return_value=mock_cli):
        result = run("model.bngl", out=str(tmp_path))

    assert result is sentinel
    mock_cli.run.assert_called_once()
