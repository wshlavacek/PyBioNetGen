import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bionetgen.core.exc import BNGSimError
from bionetgen.core.tools.bngsim_bridge import (
    BngsimDirectJob,
    FORMAT_ANTIMONY,
    FORMAT_BNG_XML,
    FORMAT_NET,
    FORMAT_SBML,
    execute_bngsim_direct_job,
)

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


def _mock_result():
    def write_cdat(path):
        with open(path, "w") as f:
            f.write("cdat\n")

    result = MagicMock()
    result.time = np.array([0.0, 1.0])
    result.observable_names = ["obsA"]
    result.observables = np.array([[0.0], [2.0]])
    result.n_observables = 1
    result.n_times = 2
    result.expression_names = []
    result.expressions = np.empty((2, 0))
    result.species = np.array([[10.0], [9.0]])
    result.to_cdat = MagicMock(side_effect=write_cdat)
    return result


def _network_job(fmt=FORMAT_NET, method="ode", options=None, output_dir="/tmp/out"):
    return BngsimDirectJob(
        input_path=f"/model.{fmt}",
        input_format=fmt,
        method=method,
        t_span=(0.0, 10.0),
        n_points=11,
        output_dir=output_dir,
        output_root="model",
        bngsim_options=options,
    )


@pytest.mark.parametrize("method", ["ode", "ssa", "psa", "rm"])
def test_net_job_instantiates_simulator_with_expected_method(method):
    mock_bngsim = MagicMock()
    model = MagicMock()
    mock_bngsim.Model.from_net.return_value = model
    mock_bngsim.Simulator.return_value.run.return_value = _mock_result()

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim), \
         patch(f"{BRIDGE}._write_bngsim_results"), \
         patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
        execute_bngsim_direct_job(_network_job(method=method))

    mock_bngsim.Simulator.assert_called_once_with(model, method=method)


def test_psa_job_passes_poplevel_to_simulator():
    mock_bngsim = MagicMock()
    model = MagicMock()
    mock_bngsim.Model.from_net.return_value = model
    mock_bngsim.Simulator.return_value.run.return_value = _mock_result()

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim), \
         patch(f"{BRIDGE}._write_bngsim_results"), \
         patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
        execute_bngsim_direct_job(
            _network_job(method="psa", options={"poplevel": 250.0})
        )

    mock_bngsim.Simulator.assert_called_once_with(
        model, method="psa", poplevel=250.0
    )


@pytest.mark.parametrize(
    ("fmt", "loader"),
    [
        (FORMAT_SBML, "from_sbml"),
        (FORMAT_ANTIMONY, "from_antimony"),
    ],
)
def test_sbml_and_antimony_jobs_load_mocked_bngsim_models(fmt, loader):
    mock_bngsim = MagicMock()
    model = MagicMock()
    getattr(mock_bngsim.Model, loader).return_value = model
    mock_bngsim.Simulator.return_value.run.return_value = _mock_result()

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim), \
         patch(f"{BRIDGE}._write_bngsim_results"), \
         patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
        execute_bngsim_direct_job(_network_job(fmt=fmt, method="ode"))

    getattr(mock_bngsim.Model, loader).assert_called_once()
    mock_bngsim.Simulator.assert_called_once_with(model, method="ode")


def test_bng_xml_job_uses_nfsim_session_for_nf_method_only():
    mock_bngsim = MagicMock()
    session = MagicMock()
    session.simulate.return_value = _mock_result()
    mock_bngsim.NfsimSession.return_value.__enter__.return_value = session

    job = BngsimDirectJob(
        input_path="/model.xml",
        input_format=FORMAT_BNG_XML,
        method="nf",
        t_span=(0.0, 10.0),
        n_points=11,
        output_dir="/tmp/out",
        output_root="model",
        bngsim_options={"seed": 7, "gml": 1000},
    )

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim), \
         patch(f"{BRIDGE}._write_bngsim_results"), \
         patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
        execute_bngsim_direct_job(job)

    mock_bngsim.NfsimSession.assert_called_once_with(
        "/model.xml", molecule_limit=1000
    )
    session.initialize.assert_called_once_with(7)
    session.simulate.assert_called_once_with(0.0, 10.0, 11)
    mock_bngsim.Simulator.assert_not_called()

    bad_job = BngsimDirectJob(
        input_path="/model.xml",
        input_format=FORMAT_BNG_XML,
        method="ode",
        t_span=(0.0, 10.0),
        n_points=11,
        output_dir="/tmp/out",
        output_root="model",
    )
    mock_bngsim.NfsimSession.reset_mock()

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.BNGSIM_HAS_NFSIM", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim):
        with pytest.raises(BNGSimError, match="network-free simulation"):
            execute_bngsim_direct_job(bad_job)

    mock_bngsim.NfsimSession.assert_not_called()


def test_bng_xml_rm_job_writes_final_state_and_replays_conc():
    """rm multi-segment continuation regression.

    ``method=>"rm"`` with ``get_final_state=>1`` must call the
    RuleMonkeySession's ``save_species`` so BNG2.pl's ``readNFspecies``
    can continue across segments, and setConcentration/addConcentration
    must replay via ``set_species_count`` / ``add_species`` — both rely on
    the bngsim >= 0.9.2 RuleMonkeySession surface and were previously
    dropped by ``_run_rulemonkey_job``.
    """
    mock_bngsim = MagicMock()
    session = MagicMock()
    session.simulate.return_value = _mock_result()
    mock_bngsim.RuleMonkeySession.return_value.__enter__.return_value = session

    job = BngsimDirectJob(
        input_path="/model.xml",
        input_format=FORMAT_BNG_XML,
        method="rm",
        t_span=(0.0, 10.0),
        n_points=11,
        output_dir="/tmp/out",
        output_root="model",
        bngsim_options={
            "seed": 7,
            "gml": 1000,
            "conc_overrides": {"A()": 5},
            "conc_deltas": {"B()": 3},
        },
        get_final_state=True,
    )

    with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
         patch(f"{BRIDGE}.BNGSIM_HAS_RULEMONKEY", True), \
         patch(f"{BRIDGE}.bngsim", mock_bngsim), \
         patch(f"{BRIDGE}._write_bngsim_results"), \
         patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
        execute_bngsim_direct_job(job)

    mock_bngsim.RuleMonkeySession.assert_called_once_with(
        "/model.xml", molecule_limit=1000
    )
    session.initialize.assert_called_once_with(7)
    session.simulate.assert_called_once_with(0.0, 10.0, 11)
    # get_final_state writeback -> <root>.species (was missing for rm)
    assert session.save_species.call_count == 1
    assert session.save_species.call_args[0][0].endswith("model.species")
    # concentration replay (was dropped for rm)
    session.set_species_count.assert_any_call("A()", 5)
    session.add_species.assert_any_call("B()", 3)
    mock_bngsim.NfsimSession.assert_not_called()


def test_direct_job_writer_creates_gdat_and_cdat_files():
    mock_bngsim = MagicMock()
    mock_bngsim.Model.from_net.return_value = MagicMock()
    mock_bngsim.Simulator.return_value.run.return_value = _mock_result()

    with tempfile.TemporaryDirectory() as tmpdir:
        job = _network_job(method="ode", output_dir=tmpdir)
        with patch(f"{BRIDGE}.BNGSIM_AVAILABLE", True), \
             patch(f"{BRIDGE}.bngsim", mock_bngsim), \
             patch(f"{BRIDGE}._make_bng_result", return_value=MagicMock()):
            execute_bngsim_direct_job(job)

        assert os.path.isfile(os.path.join(tmpdir, "model.gdat"))
        assert os.path.isfile(os.path.join(tmpdir, "model.cdat"))
