import os
import tempfile
import textwrap
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from bionetgen.modelapi.blocks import (
    ActionBlock,
    CompartmentBlock,
    EnergyPatternBlock,
    FunctionBlock,
    MoleculeTypeBlock,
    ObservableBlock,
    ParameterBlock,
    PopulationMapBlock,
    RuleBlock,
    SpeciesBlock,
)

SAMPLE_BNGL = textwrap.dedent("""\
    begin model
    begin parameters
      k1 0.1
    end parameters
    begin molecule types
      A()
    end molecule types
    begin species
      A() 100
    end species
    begin reaction rules
      A() -> 0 k1
    end reaction rules
    begin observables
      Molecules Atot A()
    end observables
    end model
    simulate({method=>"ode",t_end=>10,n_steps=>100})
""")


def _make_model_bypass_init():
    from bionetgen.modelapi.model import bngmodel

    obj = object.__new__(bngmodel)
    obj.active_blocks = []
    obj._block_order = [
        "parameters",
        "compartments",
        "molecule_types",
        "species",
        "observables",
        "functions",
        "energy_patterns",
        "population_maps",
        "rules",
        "actions",
    ]
    obj.model_name = "test_model"
    obj.model_path = "/fake/test.bngl"
    obj.parameters = ParameterBlock()
    obj.compartments = CompartmentBlock()
    obj.molecule_types = MoleculeTypeBlock()
    obj.species = SpeciesBlock()
    obj.observables = ObservableBlock()
    obj.functions = FunctionBlock()
    obj.energy_patterns = EnergyPatternBlock()
    obj.population_maps = PopulationMapBlock()
    obj.rules = RuleBlock()
    obj.actions = ActionBlock()
    return obj


@patch(
    "bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl")
)
@patch("bionetgen.modelapi.bngfile.run_command", return_value=(1, "error"))
def test_generate_xml_failure_raises_bngfile_error(mock_run, mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "model.bngl")
        with open(src, "w", encoding="UTF-8") as handle:
            handle.write(SAMPLE_BNGL)

        xml_file = StringIO()
        with patch.object(bf.logger, "error") as mock_error:
            with pytest.raises(BNGFileError, match="BNG-XML generation failed"):
                bf.generate_xml(xml_file, model_file=src)
        mock_error.assert_called_once()


@patch(
    "bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl")
)
@patch("bionetgen.modelapi.bngfile.run_command", return_value=(0, ""))
def test_generate_xml_missing_output_raises_bngfile_error(mock_run, mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "model.bngl")
        with open(src, "w", encoding="UTF-8") as handle:
            handle.write(SAMPLE_BNGL)

        with pytest.raises(BNGFileError, match="did not produce an XML file"):
            bf.generate_xml(StringIO(), model_file=src)


@patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", None))
def test_generate_xml_no_bngexec_uses_minimal_xml(mock_find):
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "model.bngl")
        with open(src, "w", encoding="UTF-8") as handle:
            handle.write(SAMPLE_BNGL)

        xml_file = StringIO()
        assert bf.generate_xml(xml_file, model_file=src) is True
        xml_file.seek(0)
        content = xml_file.read()
        assert "<sbml>" in content
        assert '<model id="model">' in content


@patch(
    "bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl")
)
@patch("bionetgen.modelapi.bngfile.run_command", return_value=(1, "error"))
def test_write_xml_bngxml_failure_raises_bngfile_error(mock_run, mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")
    with patch.object(bf.logger, "error") as mock_error:
        with pytest.raises(BNGFileError, match="BNG-XML generation failed"):
            bf.write_xml(
                StringIO(), xml_type="bngxml", bngl_str="begin model\nend model\n"
            )
    mock_error.assert_called_once()


@patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", None))
def test_write_xml_bngxml_no_bngexec_raises_bngfile_error(mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")
    with pytest.raises(BNGFileError, match="BNG-XML generation requires BNG2.pl"):
        bf.write_xml(StringIO(), xml_type="bngxml", bngl_str="begin model\nend model\n")


@patch(
    "bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl")
)
def test_write_xml_unknown_type_raises_bngfile_error(mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")
    with pytest.raises(BNGFileError, match="XML type unknown not recognized"):
        bf.write_xml(
            StringIO(), xml_type="unknown", bngl_str="begin model\nend model\n"
        )


@patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", None))
def test_write_xml_sbml_no_bngexec_raises_bngfile_error(mock_find):
    from bionetgen.core.exc import BNGFileError
    from bionetgen.modelapi.bngfile import BNGFile

    bf = BNGFile("/some/model.bngl")
    with pytest.raises(BNGFileError, match="SBML generation requires BNG2.pl"):
        bf.write_xml(StringIO(), xml_type="sbml", bngl_str="begin model\nend model\n")


@patch("bionetgen.modelapi.bngparser.BNGFile")
def test_parse_model_xml_generation_failure_wraps_bngfile_error(mock_bngfile_cls):
    from bionetgen.core.exc import BNGFileError, BNGModelError
    from bionetgen.modelapi.bngparser import BNGParser

    mock_bngfile = MagicMock()
    mock_bngfile.path = "/some/model.bngl"
    mock_bngfile.parsed_actions = []
    mock_bngfile.generate_xml.side_effect = BNGFileError(
        "/some/model.bngl", message="BNG-XML generation failed"
    )
    mock_bngfile_cls.return_value = mock_bngfile

    parser = BNGParser("/some/model.bngl")
    with pytest.raises(BNGModelError, match="XML file couldn't be generated"):
        parser.parse_model(MagicMock())


def test_setup_simulator_write_xml_failure_raises_bngmodel_error_and_restores_actions(
    tmp_path, monkeypatch
):
    from bionetgen.core.exc import BNGFileError, BNGModelError

    monkeypatch.chdir(tmp_path)
    model = _make_model_bypass_init()
    model.add_action("simulate", {"method": '"ode"'})
    model.bngparser = MagicMock()
    model.bngparser.bngfile.write_xml.side_effect = BNGFileError(
        model.model_path, message="SBML generation failed for /fake/test.bngl"
    )

    with pytest.raises(BNGModelError, match="SBML couldn't be generated"):
        model.setup_simulator(sim_type="libRR")

    assert len(model.actions.items) == 1
    assert model.actions.items[0].type == "simulate"
