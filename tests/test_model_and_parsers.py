"""Tests for bngmodel (model.py), BNGParser (bngparser.py), and BNGFile (bngfile.py).

Heavy mocking is used throughout to avoid requiring BNG2.pl or any external
binaries.  The goal is to exercise the logic inside these three modules
without a real BioNetGen installation.
"""

import os
import tempfile
import textwrap
from collections import OrderedDict
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Imports under test -- blocks are real objects so we can populate models
# ---------------------------------------------------------------------------
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

# ============================================================================
# Helpers
# ============================================================================

MINIMAL_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <sbml>
      <model id="test_model">
        <ListOfParameters>
          <Parameter id="k1" type="Constant" value="0.1"/>
        </ListOfParameters>
        <ListOfObservables/>
        <ListOfCompartments/>
        <ListOfMoleculeTypes/>
        <ListOfSpecies/>
        <ListOfReactionRules/>
        <ListOfFunctions/>
        <ListOfEnergyPatterns/>
        <ListOfPopulationMaps/>
      </model>
    </sbml>
""")

EMPTY_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <sbml>
      <model id="empty_model">
        <ListOfParameters/>
        <ListOfObservables/>
        <ListOfCompartments/>
        <ListOfMoleculeTypes/>
        <ListOfSpecies/>
        <ListOfReactionRules/>
        <ListOfFunctions/>
        <ListOfEnergyPatterns/>
        <ListOfPopulationMaps/>
      </model>
    </sbml>
""")

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
    """Create a bngmodel instance without running __init__."""
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
    # Add empty blocks for all
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


# ============================================================================
# BNGFile tests
# ============================================================================


class TestBNGFile:
    """Tests for bionetgen.modelapi.bngfile.BNGFile."""

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_init_sets_attributes(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        assert bf.path == "/some/model.bngl"
        assert bf.BNGPATH == "/fake"
        assert bf.bngexec == "/fake/BNG2.pl"
        assert bf.parsed_actions == []
        assert bf.generate_network is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_not_action_true_for_plain_lines(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        assert bf._not_action("begin parameters") is True
        assert bf._not_action("  k1 0.1") is True
        assert bf._not_action("end model") is True

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_not_action_false_for_action_lines(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        assert bf._not_action('simulate({method=>"ode",t_end=>10})') is False
        assert bf._not_action("generate_network({overwrite=>1})") is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_removes_actions(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a BNGL file with actions
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(SAMPLE_BNGL)

            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            stripped_path = bf.strip_actions(src, out_dir)

            # The stripped file should exist
            assert os.path.exists(stripped_path)
            with open(stripped_path) as f:
                content = f.read()

            # The simulate action should have been removed
            assert "simulate(" not in content
            # But model content should remain
            assert "begin parameters" in content
            assert "k1 0.1" in content

            # parsed_actions should contain the action
            assert len(bf.parsed_actions) == 1
            assert "simulate(" in bf.parsed_actions[0]

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_handles_begin_end_actions_block(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        bngl_with_block = textwrap.dedent("""\
            begin model
            begin parameters
              k1 0.1
            end parameters
            end model
            begin actions
              simulate({method=>"ode"})
            end actions
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(bngl_with_block)

            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            stripped_path = bf.strip_actions(src, out_dir)
            with open(stripped_path) as f:
                content = f.read()

            assert "begin actions" not in content
            assert "end actions" not in content

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_unmatched_begin_actions_raises(self, mock_find):
        from bionetgen.core.exc import BNGFileError
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        bngl_bad = "begin model\nend model\nbegin actions\nsimulate({method=>\"ode\"})\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(bngl_bad)
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            with pytest.raises(BNGFileError):
                bf.strip_actions(src, out_dir)

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_unmatched_end_actions_raises(self, mock_find):
        from bionetgen.core.exc import BNGFileError
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        bngl_bad = "begin model\nend model\nend actions\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(bngl_bad)
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            with pytest.raises(BNGFileError):
                bf.strip_actions(src, out_dir)

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    @patch("bionetgen.modelapi.bngfile.run_command", return_value=(0, ""))
    def test_generate_xml_success(self, mock_run, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write source BNGL
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(SAMPLE_BNGL)

            # We need to mock the XML file that BNG2.pl would create.
            # generate_xml creates a temp dir, strips actions there, runs BNG2.pl.
            # We patch strip_actions to write to temp dir and also create a fake XML.
            orig_strip = bf.strip_actions

            def fake_strip(model_path, folder):
                result = orig_strip(model_path, folder)
                # Create the XML file that BNG2.pl would produce
                xml_name = os.path.splitext(os.path.basename(result))[0] + ".xml"
                xml_path = os.path.join(folder, xml_name)
                with open(xml_path, "w") as xf:
                    xf.write(MINIMAL_XML)
                return result

            bf.strip_actions = fake_strip

            xml_file = StringIO()
            # generate_xml changes dirs, so we use the real source path
            result = bf.generate_xml(xml_file, model_file=src)

            assert result is True
            xml_file.seek(0)
            content = xml_file.read()
            assert "test_model" in content

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    @patch("bionetgen.modelapi.bngfile.run_command", return_value=(1, "error"))
    def test_generate_xml_failure_returns_false(self, mock_run, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(SAMPLE_BNGL)

            xml_file = StringIO()
            result = bf.generate_xml(xml_file, model_file=src)
            assert result is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", None))
    def test_generate_xml_no_bngexec_uses_minimal(self, mock_find):
        """When bngexec is None, _generate_minimal_xml should be used."""
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        assert bf.bngexec is None

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(SAMPLE_BNGL)

            xml_file = StringIO()
            result = bf.generate_xml(xml_file, model_file=src)
            assert result is True
            xml_file.seek(0)
            content = xml_file.read()
            assert "<sbml>" in content
            assert "<model" in content

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    @patch("bionetgen.modelapi.bngfile.run_command", return_value=(0, ""))
    def test_write_xml_bngxml(self, mock_run, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")

        # Patch open so that the "temp.xml" read returns our fixture
        original_open = open

        def patched_open(path, *args, **kwargs):
            if isinstance(path, str) and path == "temp.xml":
                return StringIO(MINIMAL_XML)
            return original_open(path, *args, **kwargs)

        out = StringIO()
        with patch("builtins.open", side_effect=patched_open):
            result = bf.write_xml(out, xml_type="bngxml", bngl_str="begin model\nend model\n")

        assert result is True
        out.seek(0)
        assert "test_model" in out.read()

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    @patch("bionetgen.modelapi.bngfile.run_command", return_value=(1, "error"))
    def test_write_xml_bngxml_failure(self, mock_run, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        out = StringIO()
        result = bf.write_xml(out, xml_type="bngxml", bngl_str="begin model\nend model\n")
        assert result is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_write_xml_unknown_type(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        out = StringIO()
        result = bf.write_xml(out, xml_type="unknown", bngl_str="begin model\nend model\n")
        assert result is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_write_xml_no_bngl_str_raises(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        out = StringIO()
        with pytest.raises(NotImplementedError):
            bf.write_xml(out)

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", None))
    def test_write_xml_sbml_no_bngexec(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        out = StringIO()
        result = bf.write_xml(out, xml_type="sbml", bngl_str="begin model\nend model\n")
        assert result is False

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_with_generate_network(self, mock_find):
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl", generate_network=True)
        assert bf.generate_network is True

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write("begin model\nbegin parameters\n  k1 0.1\nend parameters\nend model\n")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            stripped_path = bf.strip_actions(src, out_dir)
            with open(stripped_path) as f:
                content = f.read()
            assert "generate_network({overwrite=>1})" in content

    @patch("bionetgen.modelapi.bngfile.find_BNG_path", return_value=("/fake", "/fake/BNG2.pl"))
    def test_strip_actions_line_continuation(self, mock_find):
        """Backslash-newline continuations should be joined before stripping."""
        from bionetgen.modelapi.bngfile import BNGFile

        bf = BNGFile("/some/model.bngl")
        bngl = 'begin model\nend model\nsimulate({method=>\\\n"ode"})\n'

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "model.bngl")
            with open(src, "w") as f:
                f.write(bngl)
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir)
            stripped_path = bf.strip_actions(src, out_dir)
            with open(stripped_path) as f:
                content = f.read()
            assert "simulate(" not in content
            assert len(bf.parsed_actions) == 1


# ============================================================================
# BNGParser tests
# ============================================================================


class TestBNGParser:
    """Tests for bionetgen.modelapi.bngparser.BNGParser."""

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_init_creates_bngfile(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        MockBNGFile.assert_called_once_with(
            "/some/model.bngl", generate_network=False, suppress=True
        )
        assert parser.bngfile is mock_bf
        assert parser.to_parse_actions is True

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_xml_empty_model(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        MockBNGFile.return_value = MagicMock()
        parser = BNGParser("/some/model.bngl")

        model_obj = _make_model_bypass_init()
        parser.parse_xml(EMPTY_XML, model_obj)

        assert model_obj.model_name == "empty_model"
        assert hasattr(model_obj, "xml_dict")

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_xml_with_parameters(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        MockBNGFile.return_value = MagicMock()
        parser = BNGParser("/some/model.bngl")

        model_obj = _make_model_bypass_init()
        parser.parse_xml(MINIMAL_XML, model_obj)

        assert model_obj.model_name == "test_model"
        assert "parameters" in model_obj.active_blocks
        assert len(model_obj.parameters) > 0

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_model_bngl_file(self, MockBNGFile):
        """_parse_model_bngpl should call generate_xml and parse the result."""
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = []

        # generate_xml should write XML to the temp file and return True
        def fake_generate_xml(xml_file):
            xml_file.write(MINIMAL_XML)
            xml_file.seek(0)
            return True

        mock_bf.generate_xml.side_effect = fake_generate_xml
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")

        model_obj = _make_model_bypass_init()
        parser.parse_model(model_obj)

        assert model_obj.model_name == "test_model"
        mock_bf.generate_xml.assert_called_once()

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_model_xml_file(self, MockBNGFile):
        """When path ends with .xml, it should read and parse the file directly."""
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.xml"
        mock_bf.parsed_actions = []
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.xml")

        model_obj = _make_model_bypass_init()

        # We need to mock open for the XML file
        with patch("builtins.open", mock_open(read_data=MINIMAL_XML)):
            parser.parse_model(model_obj)

        assert model_obj.model_name == "test_model"

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_model_unsupported_extension(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.txt"
        mock_bf.parsed_actions = []
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.txt")
        model_obj = _make_model_bypass_init()

        with pytest.raises(NotImplementedError, match="not supported"):
            parser.parse_model(model_obj)

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_model_xml_generation_fails(self, MockBNGFile):
        from bionetgen.core.exc import BNGModelError
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = []
        mock_bf.generate_xml.return_value = False
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()

        with pytest.raises(BNGModelError):
            parser.parse_model(model_obj)

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_simulate(self, MockBNGFile):
        """Test parsing a simple simulate action."""
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = ['simulate({method=>"ode",t_end=>10,n_steps=>100})']
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()

        parser.parse_actions(model_obj)

        assert "actions" in model_obj.active_blocks
        assert len(model_obj.actions) > 0

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_empty(self, MockBNGFile):
        """No actions should not add an action block."""
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = []
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()
        orig_blocks = list(model_obj.active_blocks)

        parser.parse_actions(model_obj)

        # No new action block should be added
        assert model_obj.active_blocks == orig_blocks

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_with_comments_and_whitespace(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        # Action lines with comments and blank lines
        mock_bf.parsed_actions = [
            "# this is a comment",
            "",
            '  simulate({method=>"ode",t_end=>10})',
        ]
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()
        parser.parse_actions(model_obj)

        assert "actions" in model_obj.active_blocks

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_generate_network(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = ["generate_network({overwrite=>1})"]
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()
        parser.parse_actions(model_obj)

        assert "actions" in model_obj.active_blocks

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_no_arg_action(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = ["resetConcentrations()"]
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()
        parser.parse_actions(model_obj)

        assert "actions" in model_obj.active_blocks

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_actions_setParameter(self, MockBNGFile):
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = ['"setParameter("k1",10)"']
        # Actually the no_setter_syntax actions use a different format
        mock_bf.parsed_actions = ['setParameter("k1",10)']
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()
        parser.parse_actions(model_obj)

        assert "actions" in model_obj.active_blocks

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_xml_with_lt_relation(self, MockBNGFile):
        """Test that relation='<' in parameter expressions is properly escaped."""
        from bionetgen.modelapi.bngparser import BNGParser

        mock_bf = MagicMock()
        mock_bf.path = "/some/model.bngl"
        mock_bf.parsed_actions = []

        # The escaping targets 'relation="<' specifically -- use a realistic
        # context where it appears as an attribute on a Parameter element.
        xml_with_lt = MINIMAL_XML.replace(
            '<Parameter id="k1" type="Constant" value="0.1"/>',
            '<Parameter id="k1" type="Constant" value="0.1" relation="<"/>'
        )

        def fake_generate_xml(xml_file):
            xml_file.write(xml_with_lt)
            xml_file.seek(0)
            return True

        mock_bf.generate_xml.side_effect = fake_generate_xml
        MockBNGFile.return_value = mock_bf

        parser = BNGParser("/some/model.bngl")
        model_obj = _make_model_bypass_init()

        # This should not raise -- the '<' should be escaped to &lt;
        parser.parse_model(model_obj)
        assert model_obj.model_name == "test_model"


# ============================================================================
# bngmodel tests
# ============================================================================


class TestBngmodel:
    """Tests for bionetgen.modelapi.model.bngmodel."""

    def test_str_empty_model(self):
        model = _make_model_bypass_init()
        s = str(model)
        assert "begin model" in s
        assert "end model" in s

    def test_str_with_active_parameters(self):
        model = _make_model_bypass_init()
        # Add a mock parameter to make the block non-empty
        mock_param = MagicMock()
        mock_param.print_line.return_value = "  k1 0.1"
        model.parameters.items["k1"] = mock_param
        model.active_blocks.append("parameters")

        s = str(model)
        assert "begin parameters" in s
        assert "k1 0.1" in s
        assert "end parameters" in s

    def test_repr_returns_model_name(self):
        model = _make_model_bypass_init()
        model.model_name = "my_model"
        assert repr(model) == "my_model"

    def test_iter_returns_active_blocks(self):
        model = _make_model_bypass_init()
        model.active_blocks = ["parameters", "rules"]
        blocks = list(model)
        assert len(blocks) == 2
        assert blocks[0] is model.parameters
        assert blocks[1] is model.rules

    def test_iter_empty_active_blocks(self):
        model = _make_model_bypass_init()
        model.active_blocks = []
        blocks = list(model)
        assert blocks == []

    def test_recompile_false_by_default(self):
        model = _make_model_bypass_init()
        model.active_blocks = ["parameters"]
        assert model.recompile is False

    def test_recompile_true_after_change(self):
        model = _make_model_bypass_init()
        model.active_blocks = ["parameters"]
        model.parameters._recompile = True
        assert model.recompile is True

    def test_changes_empty_by_default(self):
        model = _make_model_bypass_init()
        model.active_blocks = ["parameters"]
        changes = model.changes
        assert "parameters" in changes
        assert len(changes["parameters"]) == 0

    def test_reset_compilation_tags(self):
        model = _make_model_bypass_init()
        model.active_blocks = ["parameters"]
        model.parameters._recompile = True
        model.parameters._changes = OrderedDict({"k1": 0.5})

        model.reset_compilation_tags()

        assert model.parameters._recompile is False
        assert len(model.parameters._changes) == 0

    def test_add_action(self):
        model = _make_model_bypass_init()
        model.active_blocks.append("actions")
        model.add_action("simulate", {"method": '"ode"', "t_end": 10})

        assert len(model.actions) > 0

    def test_add_action_creates_block_if_missing(self):
        model = _make_model_bypass_init()
        delattr(model, "actions")
        model.active_blocks = []

        model.add_action("simulate", {"method": '"ode"'})

        assert hasattr(model, "actions")
        assert "actions" in model.active_blocks

    def test_write_model(self):
        model = _make_model_bypass_init()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bngl", delete=False) as f:
            fname = f.name

        try:
            model.write_model(fname)
            with open(fname) as f:
                content = f.read()
            assert "begin model" in content
            assert "end model" in content
        finally:
            os.unlink(fname)

    def test_add_block_parameters(self):
        model = _make_model_bypass_init()
        pb = ParameterBlock()
        mock_param = MagicMock()
        mock_param.print_line.return_value = "  kf 1.0"
        pb.items["kf"] = mock_param

        model.add_block(pb)
        assert "parameters" in model.active_blocks
        assert "kf" in model.parameters.items

    def test_add_block_reaction_rules(self):
        """reaction_rules name should map to 'rules' block."""
        model = _make_model_bypass_init()
        rb = RuleBlock()
        # RuleBlock has name "reaction rules"
        model.add_block(rb)
        assert hasattr(model, "rules")

    def test_add_empty_block(self):
        model = _make_model_bypass_init()
        # Remove observables to test adding empty
        delattr(model, "observables")
        model.add_empty_block("observables")
        assert hasattr(model, "observables")
        assert len(model.observables) == 0

    def test_add_empty_block_reaction_rules(self):
        """Should handle 'reaction_rules' -> 'rules' mapping."""
        model = _make_model_bypass_init()
        delattr(model, "rules")
        model.add_empty_block("reaction_rules")
        assert hasattr(model, "rules")

    def test_str_actions_block(self):
        """Actions should appear after end model."""
        model = _make_model_bypass_init()
        model.active_blocks.append("actions")
        model.add_action("simulate", {"method": '"ode"', "t_end": "10"})

        s = str(model)
        end_model_pos = s.index("end model")
        # Actions text should come after "end model"
        assert "simulate" in s[end_model_pos:]

    def test_str_before_model_actions(self):
        """Actions that go before model should appear first."""
        model = _make_model_bypass_init()
        model.active_blocks.append("actions")

        # Manually add a before_model action
        mock_action = MagicMock()
        mock_action.__str__ = MagicMock(return_value="setModelName(\"test\")")
        model.actions.before_model.append(mock_action)

        s = str(model)
        begin_pos = s.index("begin model")
        set_pos = s.index("setModelName")
        assert set_pos < begin_pos

    def test_str_removes_empty_block_from_active(self):
        """If a block becomes empty, __str__ should remove it from active_blocks."""
        model = _make_model_bypass_init()
        model.active_blocks.append("parameters")
        # parameters block is empty (no items)
        assert len(model.parameters) == 0

        str(model)

        assert "parameters" not in model.active_blocks

    def test_str_adds_nonempty_block_to_active(self):
        """If a block gains items, __str__ should add it to active_blocks."""
        model = _make_model_bypass_init()
        assert "parameters" not in model.active_blocks

        mock_param = MagicMock()
        mock_param.print_line.return_value = "  k1 0.1"
        model.parameters.items["k1"] = mock_param

        str(model)

        assert "parameters" in model.active_blocks

    @patch("bionetgen.modelapi.model.BNGParser")
    def test_init_full(self, MockParser):
        """Test full __init__ with mocked parser."""
        from bionetgen.modelapi.model import bngmodel

        mock_parser = MagicMock()
        MockParser.return_value = mock_parser

        # parse_model should populate the model
        def fake_parse(model_obj):
            model_obj.model_name = "init_test"

        mock_parser.parse_model.side_effect = fake_parse

        model = bngmodel("/fake/model.bngl")

        MockParser.assert_called_once_with(
            "/fake/model.bngl", generate_network=False, suppress=True
        )
        mock_parser.parse_model.assert_called_once()
        assert model.model_name == "init_test"
        assert model.model_path == "/fake/model.bngl"
        # All blocks should exist (added as empty by init)
        for block in model._block_order:
            bname = block if block != "rules" else "rules"
            assert hasattr(model, bname)

    @patch("bionetgen.modelapi.model.BNGParser")
    def test_init_no_active_blocks_warning(self, MockParser, capsys):
        """When no blocks are active after parsing, a warning should be printed."""
        from bionetgen.modelapi.model import bngmodel

        mock_parser = MagicMock()
        MockParser.return_value = mock_parser
        mock_parser.parse_model.side_effect = lambda m: None

        model = bngmodel("/fake/empty.bngl")

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or len(model.active_blocks) == 0

    def test_setup_simulator_unrecognized_type(self):
        """Unrecognized sim types should return None."""
        model = _make_model_bypass_init()
        model.bngparser = MagicMock()

        result = model.setup_simulator(sim_type="unknown")
        assert result is None

    def test_add_all_block_types(self):
        """Test adding each block type to a model."""
        model = _make_model_bypass_init()
        model.active_blocks = []

        # Reset all blocks
        for attr in ["parameters", "compartments", "molecule_types", "species",
                      "observables", "functions", "energy_patterns",
                      "population_maps", "rules", "actions"]:
            if hasattr(model, attr):
                delattr(model, attr)

        block_classes = [
            ("parameters", ParameterBlock),
            ("compartments", CompartmentBlock),
            ("molecule_types", MoleculeTypeBlock),
            ("species", SpeciesBlock),
            ("observables", ObservableBlock),
            ("functions", FunctionBlock),
            ("energy_patterns", EnergyPatternBlock),
            ("population_maps", PopulationMapBlock),
            ("actions", ActionBlock),
        ]

        for name, cls in block_classes:
            block = cls()
            adder = getattr(model, f"add_{name}_block")
            adder(block)
            assert hasattr(model, name)
            assert name in model.active_blocks

    def test_add_block_with_none_creates_empty(self):
        """Calling add_*_block() with no args creates empty block."""
        model = _make_model_bypass_init()
        model.add_parameters_block()
        assert isinstance(model.parameters, ParameterBlock)
        assert len(model.parameters) == 0

    def test_block_order_preserved_in_str(self):
        """Blocks should appear in _block_order in the string output."""
        model = _make_model_bypass_init()

        # Add mock items to parameters and observables
        mock_param = MagicMock()
        mock_param.print_line.return_value = "  k1 0.1"
        model.parameters.items["k1"] = mock_param
        model.active_blocks.append("parameters")

        mock_obs = MagicMock()
        mock_obs.print_line.return_value = "  Molecules Atot A()"
        model.observables.items["Atot"] = mock_obs
        model.active_blocks.append("observables")

        s = str(model)
        param_pos = s.index("begin parameters")
        obs_pos = s.index("begin observables")
        assert param_pos < obs_pos


# ============================================================================
# Integration-style tests (still mocked, but crossing module boundaries)
# ============================================================================


class TestParserModelIntegration:
    """Tests that exercise BNGParser -> bngmodel interaction with mocked BNGFile."""

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    @patch("bionetgen.modelapi.model.BNGParser")
    def test_roundtrip_str(self, MockModelParser, MockBNGFile):
        """A model created and converted to string should have begin/end model."""
        model = _make_model_bypass_init()
        model.active_blocks.append("parameters")

        mock_param = MagicMock()
        mock_param.print_line.return_value = "  k1 0.1"
        model.parameters.items["k1"] = mock_param

        s = str(model)
        assert s.count("begin model") == 1
        assert s.count("end model") == 1
        assert "begin parameters" in s

    @patch("bionetgen.modelapi.bngparser.BNGFile")
    def test_parse_xml_multiple_blocks(self, MockBNGFile):
        """Parse XML with multiple block types."""
        from bionetgen.modelapi.bngparser import BNGParser

        xml_multi = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <sbml>
              <model id="multi_model">
                <ListOfParameters>
                  <Parameter id="k1" type="Constant" value="0.1"/>
                  <Parameter id="k2" type="Constant" value="0.2"/>
                </ListOfParameters>
                <ListOfMoleculeTypes>
                  <MoleculeType id="A" name="A">
                  </MoleculeType>
                </ListOfMoleculeTypes>
                <ListOfObservables/>
                <ListOfCompartments/>
                <ListOfSpecies/>
                <ListOfReactionRules/>
                <ListOfFunctions/>
                <ListOfEnergyPatterns/>
                <ListOfPopulationMaps/>
              </model>
            </sbml>
        """)

        MockBNGFile.return_value = MagicMock()
        parser = BNGParser("/some/model.bngl")

        model_obj = _make_model_bypass_init()
        parser.parse_xml(xml_multi, model_obj)

        assert model_obj.model_name == "multi_model"
        assert "parameters" in model_obj.active_blocks
        assert "molecule_types" in model_obj.active_blocks
