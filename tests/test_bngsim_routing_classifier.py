import os
import textwrap
import time
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
    def test_supported_complex_bngl_workflows_use_backend_hook_route(self, actions):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_BNGL_BNGSIM

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            actions=actions,
        )

        assert decision.route == ROUTE_BNGL_BNGSIM

    def test_bngl_method_override_keeps_multi_sim_workflow_on_backend_hook_route(self):
        from bionetgen.core.tools.bngsim_bridge import ROUTE_BNGL_BNGSIM

        decision = _classify(
            "bngl",
            simulator="auto",
            bngsim_available=True,
            method="ode",
            actions=[_action("simulate_ode"), _action("simulate_ssa")],
        )

        assert decision.route == ROUTE_BNGL_BNGSIM

    def test_protocol_blocks_use_backend_hook_route(self):
        from bionetgen.core.tools.bngsim_bridge import (
            ROUTE_BNGL_BNGSIM,
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

        assert decision.route == ROUTE_BNGL_BNGSIM


@pytest.mark.parametrize("action_text", COMPLEX_BNGL_ACTION_CASES)
def test_parser_backed_supported_complex_bngl_actions_use_backend_hook_route(tmp_path, action_text):
    from bionetgen.core.tools.bngsim_bridge import (
        ROUTE_BNGL_BNGSIM,
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

    assert decision.route == ROUTE_BNGL_BNGSIM
    assert "BNG2.pl" in decision.reason


@pytest.mark.parametrize("action_text", COMPLEX_BNGL_ACTION_CASES)
def test_library_complex_bngl_uses_bngsim_route_not_subprocess_classifier(tmp_path, action_text):
    from bionetgen.modelapi.runner import run

    bngl_path = _write_minimal_bngl(tmp_path, action_text)
    out_dir = tmp_path / "out"
    sentinel = object()

    with (
        patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True),
        patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True),
        patch(f"{BRIDGE}.run_bngl_with_bngsim", return_value=sentinel) as mock_bngsim_run,
        patch("bionetgen.modelapi.runner.get_conf", return_value={"bngpath": "/fake/bng"}),
        patch("bionetgen.modelapi.runner.BNGCLI") as mock_bngcli,
    ):
        result = run(str(bngl_path), out=str(out_dir))

    assert result is sentinel
    mock_bngsim_run.assert_called_once()
    assert mock_bngsim_run.call_args.args[:3] == (
        str(bngl_path),
        str(out_dir),
        "/fake/bng",
    )
    mock_bngcli.assert_not_called()


def test_run_bngl_with_bngsim_complex_action_uses_backend_hook_without_executor(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

    bngl_path = _write_minimal_bngl(
        tmp_path,
        'setParameter("k", 2)\nsimulate_ode({t_end=>1,n_steps=>1})',
    )
    sentinel = object()

    with (
        patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True),
        patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True),
        patch(
            f"{BRIDGE}.run_bngl_with_bngsim_backend_hook",
            return_value=sentinel,
        ) as mock_hook,
    ):
        result = run_bngl_with_bngsim(str(bngl_path), str(tmp_path / "out"), "/fake/bng")

    assert result is sentinel
    mock_hook.assert_called_once()
    assert mock_hook.call_args.args[:3] == (
        str(bngl_path),
        str(tmp_path / "out"),
        "/fake/bng",
    )


def test_stage6_removed_python_bngl_interpreter_symbols():
    import bionetgen.core.tools.bngsim_bridge as bridge

    removed_symbols = [
        "_execute_bngsim_actions",
        "_parse_simulate_params",
        "_resolve_sample_times",
        "_resolve_scan_points",
        "_run_parameter_scan_bngsim",
        "_run_bifurcate_bngsim",
        "_run_protocol",
        "_parse_protocol_block",
        "_safe_math_namespace",
        "_safe_eval_expr",
        "_eval_numeric",
        "_normalize_bngl_expr",
        "_aliased_keyword_namespace",
        "_resolve_bngmodel_params",
        "_evaluate_bngmodel_functions",
        "_evaluate_functions_per_timepoint",
        "_strip_zero_arg_calls",
        "_parse_net_species_initializers",
        "_sync_species_concentrations",
        "_parse_bngmodel_seed_species_initializers",
        "_parse_xml_parameter_table",
        "_resolve_xml_params",
        "_apply_nfsim_derived_params",
        "_apply_nfsim_seed_species_initializers",
        "_write_scan_file",
        "_read_scan_file",
        "_scan_result_to_row",
        "_parse_table_functions",
        "_parse_tfun_args",
        "_add_table_functions",
    ]

    for name in removed_symbols:
        assert not hasattr(bridge, name)


def test_run_bngl_with_bngsim_protocol_uses_backend_hook_without_python_parser(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

    bngl_path = _write_minimal_bngl(
        tmp_path,
        "simulate_ode({t_end=>1,n_steps=>1})",
    )
    bngl_path.write_text(
        bngl_path.read_text(encoding="utf-8")
        + "\nbegin protocol\nsimulate_ode({t_end=>1,n_steps=>1})\nend protocol\n",
        encoding="utf-8",
    )
    sentinel = object()

    with (
        patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True),
        patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True),
        patch(
            f"{BRIDGE}.run_bngl_with_bngsim_backend_hook",
            return_value=sentinel,
        ) as mock_hook,
    ):
        result = run_bngl_with_bngsim(str(bngl_path), str(tmp_path / "out"), "/fake/bng")

    assert result is sentinel
    mock_hook.assert_called_once()


def test_run_bngl_with_bngsim_scan_uses_backend_hook_without_python_scan_outputs(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import run_bngl_with_bngsim

    bngl_path = _write_minimal_bngl(
        tmp_path,
        'parameter_scan({method=>"ode",parameter=>"k",par_min=>0,par_max=>1,n_scan_pts=>2})',
    )
    sentinel = object()

    with (
        patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True),
        patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True),
        patch(
            f"{BRIDGE}.run_bngl_with_bngsim_backend_hook",
            return_value=sentinel,
        ) as mock_hook,
    ):
        result = run_bngl_with_bngsim(str(bngl_path), str(tmp_path / "out"), "/fake/bng")

    assert result is sentinel
    mock_hook.assert_called_once()
    assert not list((tmp_path / "out").glob("*.scan"))


def test_library_subprocess_route_uses_bngcli(tmp_path):
    from bionetgen.core.tools.bngsim_bridge import (
        ROUTE_SUBPROCESS,
        BngsimRouteDecision,
    )
    from bionetgen.modelapi.runner import run

    sentinel = MagicMock()
    mock_cli = MagicMock()
    mock_cli.result = sentinel

    with (
        patch(f"{BRIDGE}.detect_input_format", return_value="bngl"),
        patch(
            f"{BRIDGE}.classify_bngsim_route",
            return_value=BngsimRouteDecision(ROUTE_SUBPROCESS, "complex BNGL"),
        ),
        patch("bionetgen.modelapi.runner.get_conf", return_value={"bngpath": "/fake/bng"}),
        patch("bionetgen.modelapi.runner.BNGCLI", return_value=mock_cli),
    ):
        result = run("model.bngl", out=str(tmp_path))

    assert result is sentinel
    mock_cli.run.assert_called_once()


class TestRoutingActionCache:
    """The route-classification action parse is memoized per file identity.

    Routing re-asks for a BNGL's action list ~4 times per ``bionetgen.run``
    (the classifier from ``runner.run`` and again inside
    ``run_bngl_with_bngsim``, the in-process-scan detector, the
    network-free-method probe). Each uncached parse builds a ``bngmodel``
    that shells out to BNG2.pl — serial timing measured ~1.9 s of redundant
    pre-flight per run before this cache existed, which made the BNGsim
    route slower than plain subprocess on every model.
    """

    def test_repeated_routing_queries_parse_the_file_once(self, tmp_path):
        from bionetgen.core.tools import bngsim_bridge as bridge

        bridge._clear_routing_actions_cache()
        bngl = tmp_path / "memo.bngl"
        bngl.write_text("generate_network({overwrite=>1})\n", encoding="utf-8")
        fake_model = MagicMock()
        fake_model.actions.items = [_action("generate_network")]

        with patch("bionetgen.modelapi.model.bngmodel", return_value=fake_model) as bngmodel:
            first = bridge._load_bngl_actions_for_routing(str(bngl))
            second = bridge._load_bngl_actions_for_routing(str(bngl))
            third = bridge._load_bngl_actions_for_routing(str(bngl))

        assert bngmodel.call_count == 1
        assert first is second is third
        assert [a.type for a in first] == ["generate_network"]

    def test_cache_reparses_after_the_file_changes(self, tmp_path):
        from bionetgen.core.tools import bngsim_bridge as bridge

        bridge._clear_routing_actions_cache()
        bngl = tmp_path / "memo.bngl"
        bngl.write_text("generate_network({overwrite=>1})\n", encoding="utf-8")
        fake_model = MagicMock()
        fake_model.actions.items = [_action("generate_network")]

        with patch("bionetgen.modelapi.model.bngmodel", return_value=fake_model) as bngmodel:
            bridge._load_bngl_actions_for_routing(str(bngl))
            # Edit the file: different size, and a strictly later mtime so
            # the change is caught even on coarse-resolution clocks.
            bngl.write_text(
                'generate_network({overwrite=>1})\nsimulate({method=>"ode"})\n',
                encoding="utf-8",
            )
            future = time.time() + 10
            os.utime(bngl, (future, future))
            bridge._load_bngl_actions_for_routing(str(bngl))

        assert bngmodel.call_count == 2

    def test_parse_failure_is_cached_not_retried(self, tmp_path):
        from bionetgen.core.tools import bngsim_bridge as bridge

        bridge._clear_routing_actions_cache()
        bngl = tmp_path / "broken.bngl"
        bngl.write_text("not valid bngl\n", encoding="utf-8")

        with patch(
            "bionetgen.modelapi.model.bngmodel", side_effect=RuntimeError("boom")
        ) as bngmodel:
            first = bridge._load_bngl_actions_for_routing(str(bngl))
            second = bridge._load_bngl_actions_for_routing(str(bngl))

        assert first is None and second is None
        assert bngmodel.call_count == 1

    def test_unstattable_path_parses_without_caching(self):
        from bionetgen.core.tools import bngsim_bridge as bridge

        bridge._clear_routing_actions_cache()
        with patch(f"{BRIDGE}._parse_bngl_actions_for_routing", return_value=None) as parse:
            bridge._load_bngl_actions_for_routing("/no/such/file.bngl")
            bridge._load_bngl_actions_for_routing("/no/such/file.bngl")

        assert parse.call_count == 2
