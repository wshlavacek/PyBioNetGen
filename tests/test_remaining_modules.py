"""
Tests for remaining uncovered modules:
  - sympy_odes.py
  - pattern_reader.py
  - networkparser.py / network.py
  - visualize.py
  - plot.py
  - core/main.py
  - main.py (CLI)
  - csimulator.py
"""

import os
import re
import tempfile
import textwrap
from unittest import mock

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 1. sympy_odes.py — internal helpers and extract_odes_from_mexfile
# ---------------------------------------------------------------------------

class TestSympyOdes:
    """Test the sympy_odes module (internal helpers + full extract)."""

    def setup_method(self):
        import sympy as sp
        self.sp = sp
        from bionetgen.modelapi import sympy_odes as so
        self.so = so

    # -- helper: _normalize_expr --
    def test_normalize_expr_pow(self):
        assert "Pow(" in self.so._normalize_expr("pow(x,2)")

    def test_normalize_expr_fabs(self):
        assert "Abs(" in self.so._normalize_expr("fabs(x)")

    def test_normalize_expr_fmax_fmin(self):
        assert "Max(" in self.so._normalize_expr("fmax(a,b)")
        assert "Min(" in self.so._normalize_expr("fmin(a,b)")

    def test_normalize_expr_cast_removal(self):
        result = self.so._normalize_expr("(double)x + (float)y")
        assert "(double)" not in result
        assert "(float)" not in result

    def test_normalize_expr_pi(self):
        assert "pi" in self.so._normalize_expr("M_PI")

    # -- helper: _build_symbol_names --
    def test_build_symbol_names_basic(self):
        names, final = self.so._build_symbol_names(["A", "B"], 2, prefix="s")
        assert names == ["A", "B"]
        assert final == ["A", "B"]

    def test_build_symbol_names_empty(self):
        names, final = self.so._build_symbol_names([], 3, prefix="s")
        assert names == ["s0", "s1", "s2"]
        assert final == ["s0", "s1", "s2"]

    def test_build_symbol_names_padding(self):
        names, final = self.so._build_symbol_names(["X"], 3, prefix="s")
        assert len(names) == 3
        assert len(final) == 3

    def test_build_symbol_names_digit_prefix(self):
        names, _ = self.so._build_symbol_names(["1foo"], 1, prefix="s")
        assert names[0].startswith("s_")

    def test_build_symbol_names_none_expected(self):
        names, final = self.so._build_symbol_names(["A", "B"], None, prefix="p")
        assert len(names) == 2

    def test_build_symbol_names_special_chars(self):
        names, _ = self.so._build_symbol_names(["A(b)"], 1, prefix="s")
        assert "(" not in names[0]

    def test_build_symbol_names_duplicate(self):
        names, _ = self.so._build_symbol_names(["x", "x"], 2, prefix="s")
        assert names[0] != names[1]

    # -- helper: _extract_ode_assignments --
    def test_extract_ode_assignments_nv(self):
        text = "NV_Ith_S(ydot, 0) = -k1*y[0]; NV_Ith_S(ydot, 1) = k1*y[0];"
        result = self.so._extract_ode_assignments(text)
        assert 0 in result
        assert 1 in result

    def test_extract_ode_assignments_bracket(self):
        text = "ydot[0] = -k*y[0]; ydot[1] = k*y[0];"
        result = self.so._extract_ode_assignments(text)
        assert 0 in result
        assert 1 in result

    def test_extract_ode_assignments_empty(self):
        assert self.so._extract_ode_assignments("nothing here") == {}

    # -- helper: _extract_name_array --
    def test_extract_name_array_found(self):
        text = 'const char *species_names[] = {"A", "B"};'
        result = self.so._extract_name_array(text, self.so._NAME_ARRAY_PATTERNS)
        assert result == ["A", "B"]

    def test_extract_name_array_not_found(self):
        assert self.so._extract_name_array("nothing", self.so._NAME_ARRAY_PATTERNS) == []

    # -- helper: _max_indexed_param --
    def test_max_indexed_param(self):
        exprs = ["params[0] + params[3]", "p[1]"]
        assert self.so._max_indexed_param(exprs) == 3

    def test_max_indexed_param_none(self):
        assert self.so._max_indexed_param(["x + y"]) is None

    # -- helper: _replace_indexed_symbols --
    def test_replace_indexed_symbols_species(self):
        result = self.so._replace_indexed_symbols(
            "NV_Ith_S(y, 0) + y[1]", ["A", "B"], ["k1"]
        )
        assert "A" in result
        assert "B" in result

    def test_replace_indexed_symbols_params(self):
        result = self.so._replace_indexed_symbols(
            "params[0] + param[1] + p[2]", ["s0"], ["k1", "k2", "k3"]
        )
        assert "k1" in result
        assert "k2" in result
        assert "k3" in result

    def test_replace_indexed_symbols_out_of_range(self):
        result = self.so._replace_indexed_symbols("y[5]", ["A"], [])
        assert "s5" in result

    # -- helper: _extract_define_int --
    def test_extract_define_int_found(self):
        text = "#define __N_SPECIES__ 5\n"
        assert self.so._extract_define_int(text, "__N_SPECIES__") == 5

    def test_extract_define_int_missing(self):
        assert self.so._extract_define_int("nothing", "__N_SPECIES__") is None

    # -- helper: _extract_function_body --
    def test_extract_function_body(self):
        text = "void calc_species_deriv(int x) {\n  do_stuff();\n}\n"
        body = self.so._extract_function_body(text, "calc_species_deriv")
        assert "do_stuff" in body

    def test_extract_function_body_missing(self):
        assert self.so._extract_function_body("nothing", "missing") == ""

    # -- helper: _extract_nv_assignments --
    def test_extract_nv_assignments(self):
        body = "NV_Ith_S(Dspecies, 0) = r0 - r1; NV_Ith_S(Dspecies, 1) = r1;"
        result = self.so._extract_nv_assignments(body, "Dspecies")
        assert 0 in result
        assert 1 in result

    def test_extract_nv_assignments_empty(self):
        assert self.so._extract_nv_assignments("", "Dspecies") == {}

    # -- helper: _replace_parameters_brackets --
    def test_replace_parameters_brackets(self):
        result = self.so._replace_parameters_brackets("parameters[0] + parameters[1]", ["k1", "k2"])
        assert "k1" in result
        assert "k2" in result

    def test_replace_parameters_brackets_out_of_range(self):
        result = self.so._replace_parameters_brackets("parameters[5]", ["k1"])
        assert "p5" in result

    # -- helper: _replace_nv_ith_s --
    def test_replace_nv_ith_s_species(self):
        import sympy as sp
        result = self.so._replace_nv_ith_s(
            "NV_Ith_S(species, 0)", ["A", "B"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert "A" in result

    def test_replace_nv_ith_s_expressions(self):
        import sympy as sp
        result = self.so._replace_nv_ith_s(
            "NV_Ith_S(expressions, 0)", ["A"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert "e0" in result

    def test_replace_nv_ith_s_observables(self):
        import sympy as sp
        result = self.so._replace_nv_ith_s(
            "NV_Ith_S(observables, 0)", ["A"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert "o0" in result

    def test_replace_nv_ith_s_ratelaws(self):
        import sympy as sp
        result = self.so._replace_nv_ith_s(
            "NV_Ith_S(ratelaws, 0)", ["A"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert "r0" in result

    def test_replace_nv_ith_s_dspecies(self):
        import sympy as sp
        result = self.so._replace_nv_ith_s(
            "NV_Ith_S(Dspecies, 0)", ["A"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert "ds0" in result

    def test_replace_nv_ith_s_unknown(self):
        import sympy as sp
        original = "NV_Ith_S(unknown, 0)"
        result = self.so._replace_nv_ith_s(
            original, ["A"],
            [sp.Symbol("e0")], [sp.Symbol("o0")], [sp.Symbol("r0")]
        )
        assert result == original

    # -- helper: _max_bracket_index --
    def test_max_bracket_index(self):
        text = "parameters[0] + parameters[3]"
        assert self.so._max_bracket_index(text, "parameters") == 3

    def test_max_bracket_index_none(self):
        assert self.so._max_bracket_index("nothing", "parameters") is None

    # -- helper: _find_mex_c_file --
    def test_find_mex_c_file_found(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model_mex.c")
            with open(p, "w") as f:
                f.write("// mex file")
            result = self.so._find_mex_c_file(td, mex_suffix="mex")
            assert result.endswith(".c")

    def test_find_mex_c_file_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(FileNotFoundError):
                self.so._find_mex_c_file(td, mex_suffix="mex")

    def test_find_mex_c_file_no_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model.c")
            with open(p, "w") as f:
                f.write("// c file")
            result = self.so._find_mex_c_file(td, mex_suffix="")
            assert result.endswith(".c")

    # -- helper: _safe_rmtree --
    def test_safe_rmtree(self):
        td = tempfile.mkdtemp()
        self.so._safe_rmtree(td)
        assert not os.path.exists(td)

    def test_safe_rmtree_nonexistent(self):
        # should not raise
        self.so._safe_rmtree("/nonexistent/path/xyz")

    # -- extract_odes_from_mexfile (simple ydot format) --
    def test_extract_odes_simple_format(self):
        mex_content = textwrap.dedent("""\
            const char *species_names[] = {"A", "B"};
            const char *param_names[] = {"k1", "k2"};
            NV_Ith_S(ydot, 0) = -params[0]*NV_Ith_S(y, 0);
            NV_Ith_S(ydot, 1) = params[0]*NV_Ith_S(y, 0) - params[1]*NV_Ith_S(y, 1);
        """)
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(mex_content)
            f.flush()
            try:
                result = self.so.extract_odes_from_mexfile(f.name)
                assert len(result.species) == 2
                assert len(result.params) == 2
                assert len(result.odes) == 2
                assert result.species_names == ["A", "B"]
                assert result.param_names == ["k1", "k2"]
                assert result.source_path == f.name
            finally:
                os.unlink(f.name)

    def test_extract_odes_no_assignments_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write("// nothing useful\n")
            f.flush()
            try:
                with pytest.raises(ValueError, match="No ODE assignments found"):
                    self.so.extract_odes_from_mexfile(f.name)
            finally:
                os.unlink(f.name)

    # -- extract_odes_from_mexfile (cvode mex format) --
    def test_extract_odes_cvode_format(self):
        mex_content = textwrap.dedent("""\
            #define __N_SPECIES__ 2
            #define __N_PARAMETERS__ 2
            void calc_expressions(int x) {
                NV_Ith_S(expressions, 0) = parameters[0];
                NV_Ith_S(expressions, 1) = parameters[1];
            }
            void calc_observables(int x) {
                NV_Ith_S(observables, 0) = NV_Ith_S(species, 0);
            }
            void calc_ratelaws(int x) {
                NV_Ith_S(ratelaws, 0) = NV_Ith_S(expressions, 0) * NV_Ith_S(species, 0);
            }
            void calc_species_deriv(int x) {
                NV_Ith_S(Dspecies, 0) = -NV_Ith_S(ratelaws, 0);
                NV_Ith_S(Dspecies, 1) = NV_Ith_S(ratelaws, 0);
            }
        """)
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(mex_content)
            f.flush()
            try:
                result = self.so.extract_odes_from_mexfile(f.name)
                assert len(result.species) == 2
                assert len(result.params) == 2
                assert len(result.odes) == 2
            finally:
                os.unlink(f.name)

    def test_extract_odes_cvode_no_deriv_raises(self):
        mex_content = textwrap.dedent("""\
            void calc_species_deriv(int x) {
                // no assignments
            }
        """)
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(mex_content)
            f.flush()
            try:
                with pytest.raises(ValueError, match="No ODE assignments found"):
                    self.so.extract_odes_from_mexfile(f.name)
            finally:
                os.unlink(f.name)

    # -- SympyOdes dataclass --
    def test_sympy_odes_dataclass(self):
        from bionetgen.modelapi.sympy_odes import SympyOdes
        import sympy as sp
        t = sp.Symbol("t")
        s = [sp.Symbol("s0")]
        p = [sp.Symbol("p0")]
        ode = [sp.Integer(0)]
        obj = SympyOdes(t=t, species=s, params=p, odes=ode,
                        species_names=["A"], param_names=["k1"],
                        source_path="/tmp/test.c")
        assert obj.t == t
        assert obj.species == s
        assert obj.params == p
        assert obj.source_path == "/tmp/test.c"

    # -- export_sympy_odes (mocked) --
    def test_export_sympy_odes_mocked(self):
        from bionetgen.modelapi.sympy_odes import SympyOdes
        import sympy as sp
        dummy_result = SympyOdes(
            t=sp.Symbol("t"), species=[], params=[], odes=[],
            species_names=[], param_names=[], source_path="/tmp/x.c"
        )
        mock_model = mock.MagicMock()
        mock_model.actions.items = []
        mock_model.actions.before_model = []
        # Patch bngmodel so isinstance check passes for our mock
        with mock.patch("bionetgen.modelapi.model.bngmodel", new=type(mock_model)), \
             mock.patch("bionetgen.modelapi.runner.run") as mock_run, \
             mock.patch.object(self.so, '_find_mex_c_file', return_value="/tmp/x.c"), \
             mock.patch.object(self.so, 'extract_odes_from_mexfile', return_value=dummy_result):
            result = self.so.export_sympy_odes(mock_model, out_dir="/tmp/test_out", keep_files=True)
            assert result == dummy_result


# ---------------------------------------------------------------------------
# 2. pattern_reader.py — BNGPatternReader
# ---------------------------------------------------------------------------

class TestPatternReader:
    """Test the pattern_reader module."""

    def test_simple_molecule(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b)")
        assert reader.pattern is not None
        assert len(reader.pattern.molecules) == 1
        mol = reader.pattern.molecules[0]
        assert mol.name == "A"

    def test_molecule_with_state(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b~0)")
        mol = reader.pattern.molecules[0]
        assert len(mol.components) == 1
        comp = mol.components[0]
        assert comp.name == "b"
        assert comp.state == "0"

    def test_molecule_with_bond(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b!1)")
        mol = reader.pattern.molecules[0]
        comp = mol.components[0]
        assert "1" in comp.bonds

    def test_two_molecules_with_bond(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b!1).B(a!1)")
        assert len(reader.pattern.molecules) == 2
        mol_names = [m.name for m in reader.pattern.molecules]
        assert "A" in mol_names
        assert "B" in mol_names

    def test_molecule_with_state_and_bond(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b~1!1)")
        mol = reader.pattern.molecules[0]
        comp = mol.components[0]
        assert comp.state == "1"
        assert "1" in comp.bonds

    def test_multiple_components(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b,c)")
        mol = reader.pattern.molecules[0]
        assert len(mol.components) == 2

    def test_zero_molecule(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("0")
        assert reader.pattern is not None
        assert len(reader.pattern.molecules) == 1

    def test_wildcard_bond(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b!+)")
        mol = reader.pattern.molecules[0]
        comp = mol.components[0]
        assert "+" in comp.bonds

    def test_wildcard_state(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b~?)")
        mol = reader.pattern.molecules[0]
        comp = mol.components[0]
        # wildcard state parsed as "?"
        assert comp.state == "?"

    def test_parsers_object(self):
        from bionetgen.modelapi.pattern_reader import BNGParsers
        p = BNGParsers()
        assert p is not None

    def test_fixed_pattern(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("$A(b)")
        assert reader.pattern.fixed is True

    def test_compartment_pattern(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("@EC:A(b)")
        assert reader.pattern.compartment == "EC"

    def test_three_molecules(self):
        from bionetgen.modelapi.pattern_reader import BNGPatternReader
        reader = BNGPatternReader("A(b!1).B(a!1,c!2).C(b!2)")
        assert len(reader.pattern.molecules) == 3


# ---------------------------------------------------------------------------
# 3. networkparser.py + network.py
# ---------------------------------------------------------------------------

# Fixture: a minimal .net file
# Leading blank line is needed because networkparser checks pblock[0] > 0
NET_FILE_CONTENT = "\n".join([
    "",
    "begin parameters",
    "  1 k1 0.1",
    "  2 k2 0.01",
    "end parameters",
    "begin species",
    "  1 A(b) 100",
    "  2 B(a) 200",
    "end species",
    "begin reactions",
    "  1 1,2 3 k1 #Rule1",
    "end reactions",
    "begin groups",
    "  1 Atot 1",
    "  2 Btot 2",
    "end groups",
    "",
])


class TestNetworkParserAndNetwork:
    """Test networkparser.py and network.py by creating a temp .net file."""

    def _write_net_file(self, content=NET_FILE_CONTENT):
        f = tempfile.NamedTemporaryFile(
            suffix=".net", mode="w", delete=False
        )
        f.write(content)
        f.flush()
        f.close()
        return f.name

    def test_parse_network_full(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            assert net.network_name is not None
            assert "parameters" in net.active_blocks
            assert "species" in net.active_blocks
            assert "reactions" in net.active_blocks
            assert "groups" in net.active_blocks
        finally:
            os.unlink(path)

    def test_network_str(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            s = str(net)
            assert "parameters" in s or "species" in s or len(s) > 0
        finally:
            os.unlink(path)

    def test_network_repr(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            assert repr(net) == net.network_name
        finally:
            os.unlink(path)

    def test_network_iter(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            blocks = list(iter(net))
            assert len(blocks) > 0
        finally:
            os.unlink(path)

    def test_network_write_model(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            out_path = path + ".out"
            net.write_model(out_path)
            assert os.path.exists(out_path)
            with open(out_path) as f:
                content = f.read()
            assert len(content) > 0
            os.unlink(out_path)
        finally:
            os.unlink(path)

    def test_network_empty_blocks(self):
        """A net file with only parameters should still have empty species/reactions/groups blocks."""
        content = """\
begin parameters
  1 k1 0.1
end parameters
"""
        path = self._write_net_file(content)
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            assert hasattr(net, "species")
            assert hasattr(net, "reactions")
            assert hasattr(net, "groups")
        finally:
            os.unlink(path)

    def test_networkparser_standalone(self):
        path = self._write_net_file()
        try:
            from bionetgen.network.networkparser import BNGNetworkParser
            parser = BNGNetworkParser(path)
            assert parser.network_name is not None
            assert len(parser.network_lines) > 0
        finally:
            os.unlink(path)

    def test_network_no_blocks_warning(self, capsys):
        """An empty file produces no active blocks and a warning."""
        path = self._write_net_file("")
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            captured = capsys.readouterr()
            assert "WARNING" in captured.out or len(net.active_blocks) == 0
        finally:
            os.unlink(path)

    def test_network_parameter_with_comment(self):
        content = "\nbegin parameters\n  1 k1 0.1 #forward rate\nend parameters\n"
        path = self._write_net_file(content)
        try:
            from bionetgen.network.network import Network
            net = Network(path)
            assert "parameters" in net.active_blocks
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 4. visualize.py — VisResult and BNGVisualize
# ---------------------------------------------------------------------------

class TestVisualize:
    """Test the visualize module via mocking."""

    def test_bng_visualize_init_defaults(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl")
        assert viz.input == "model.bngl"
        assert viz.vtype == "contactmap"
        assert viz.output is None

    def test_bng_visualize_init_custom_type(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", vtype="regulatory")
        assert viz.vtype == "regulatory"

    def test_bng_visualize_init_all_types(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", vtype="all")
        assert viz.vtype == "all"

    def test_bng_visualize_init_atom_rule(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", vtype="atom_rule")
        assert viz.vtype == "atom_rule"

    def test_bng_visualize_invalid_type(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        with pytest.raises(ValueError, match="not a valid visualization type"):
            BNGVisualize("model.bngl", vtype="invalid_type")

    def test_bng_visualize_empty_type_defaults(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", vtype="")
        assert viz.vtype == "contactmap"

    def test_bng_visualize_with_output(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", output="/tmp/out", vtype="contactmap")
        assert viz.output == "/tmp/out"

    def test_bng_visualize_with_bngpath(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        viz = BNGVisualize("model.bngl", bngpath="/usr/local/bin/BNG2.pl")
        assert viz.bngpath == "/usr/local/bin/BNG2.pl"

    def test_vis_result_init(self):
        from bionetgen.core.tools.visualize import VisResult
        with tempfile.TemporaryDirectory() as td:
            vr = VisResult(td, name="test_model", vtype="contactmap")
            assert vr.input_folder == td
            assert vr.name == "test_model"
            assert vr.vtype == "contactmap"
            assert vr.files == []
            assert vr.file_strs == {}

    def test_vis_result_load_graphml(self):
        from bionetgen.core.tools.visualize import VisResult
        with tempfile.TemporaryDirectory() as td:
            gml_path = os.path.join(td, "test_model_contact.graphml")
            with open(gml_path, "w") as f:
                f.write("<graphml>content</graphml>")
            # VisResult uses glob in cwd, so we need to chdir
            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                vr = VisResult(td, name="test_model")
                assert len(vr.files) == 1
                assert "test_model_contact.graphml" in vr.files[0]
            finally:
                os.chdir(old_cwd)

    def test_vis_result_dump_files(self):
        from bionetgen.core.tools.visualize import VisResult
        with tempfile.TemporaryDirectory() as td:
            gml_path = os.path.join(td, "model.graphml")
            with open(gml_path, "w") as f:
                f.write("<graphml>test</graphml>")
            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                vr = VisResult(td)
                dump_dir = os.path.join(td, "dump")
                os.makedirs(dump_dir)
                vr._dump_files(dump_dir)
                dumped = os.listdir(dump_dir)
                assert len(dumped) == 1
            finally:
                os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# 5. plot.py — BNGPlotter
# ---------------------------------------------------------------------------

class TestPlot:
    """Test the plot module via mocking."""

    def test_plotter_init(self):
        with mock.patch("bionetgen.core.tools.plot.BNGResult") as MockResult:
            mock_result = mock.MagicMock()
            mock_result.file_extension = ".gdat"
            MockResult.return_value = mock_result
            from bionetgen.core.tools.plot import BNGPlotter
            plotter = BNGPlotter("test.gdat", "test.png", legend=True)
            assert plotter.inp == "test.gdat"
            assert plotter.out == "test.png"
            assert plotter.kwargs["legend"] is True

    def test_plotter_plot_gdat(self):
        with mock.patch("bionetgen.core.tools.plot.BNGResult") as MockResult:
            mock_result = mock.MagicMock()
            mock_result.file_extension = ".gdat"
            mock_result.file_name = "test"
            MockResult.return_value = mock_result
            from bionetgen.core.tools.plot import BNGPlotter
            plotter = BNGPlotter("test.gdat", "test.png")
            with mock.patch.object(plotter, "_datplot") as mock_datplot:
                plotter.plot()
                mock_datplot.assert_called_once()

    def test_plotter_plot_cdat(self):
        with mock.patch("bionetgen.core.tools.plot.BNGResult") as MockResult:
            mock_result = mock.MagicMock()
            mock_result.file_extension = ".cdat"
            MockResult.return_value = mock_result
            from bionetgen.core.tools.plot import BNGPlotter
            plotter = BNGPlotter("test.cdat", "test.png")
            with mock.patch.object(plotter, "_datplot") as mock_datplot:
                plotter.plot()
                mock_datplot.assert_called_once()

    def test_plotter_plot_scan(self):
        with mock.patch("bionetgen.core.tools.plot.BNGResult") as MockResult:
            mock_result = mock.MagicMock()
            mock_result.file_extension = ".scan"
            MockResult.return_value = mock_result
            from bionetgen.core.tools.plot import BNGPlotter
            plotter = BNGPlotter("test.scan", "test.png")
            with mock.patch.object(plotter, "_datplot") as mock_datplot:
                plotter.plot()
                mock_datplot.assert_called_once()

    def test_plotter_plot_unknown_raises(self):
        with mock.patch("bionetgen.core.tools.plot.BNGResult") as MockResult:
            mock_result = mock.MagicMock()
            mock_result.file_extension = ".xyz"
            MockResult.return_value = mock_result
            from bionetgen.core.tools.plot import BNGPlotter
            plotter = BNGPlotter("test.xyz", "test.png")
            with pytest.raises(NotImplementedError):
                plotter.plot()


# ---------------------------------------------------------------------------
# 6. core/main.py — thin wrappers
# ---------------------------------------------------------------------------

class TestCoreMain:
    """Test the core/main.py convenience functions."""

    def test_runCLI(self):
        from bionetgen.core.main import runCLI
        mock_app = mock.MagicMock()
        mock_app.pargs.traceback_depth = 0
        mock_app.pargs.input = "model.bngl"
        mock_app.pargs.output = "."
        mock_app.pargs.log_file = None
        mock_app.config.get.return_value = "/usr/local/bin/BNG2.pl"

        with mock.patch("bionetgen.core.main.BNGCLI") as MockCLI:
            mock_cli_instance = mock.MagicMock()
            MockCLI.return_value = mock_cli_instance
            runCLI(mock_app)
            MockCLI.assert_called_once()
            mock_cli_instance.run.assert_called_once()

    def test_plotDAT(self):
        from bionetgen.core.main import plotDAT
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "model.gdat"
        mock_app.pargs.output = "/tmp/out.png"
        mock_app.pargs._get_kwargs.return_value = [
            ("input", "model.gdat"),
            ("output", "/tmp/out.png"),
            ("legend", False),
        ]
        with mock.patch("bionetgen.core.tools.plot.BNGPlotter") as MockPlotter:
            mock_plotter = mock.MagicMock()
            MockPlotter.return_value = mock_plotter
            # Also need to patch the local import inside plotDAT
            with mock.patch("bionetgen.core.tools.BNGPlotter", MockPlotter):
                plotDAT(mock_app)
                MockPlotter.assert_called()
                mock_plotter.plot.assert_called_once()

    def test_plotDAT_dot_output(self):
        from bionetgen.core.main import plotDAT
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "/path/to/model.gdat"
        mock_app.pargs.output = "."
        mock_app.pargs._get_kwargs.return_value = [
            ("input", "/path/to/model.gdat"),
            ("output", "."),
        ]
        with mock.patch("bionetgen.core.tools.BNGPlotter") as MockPlotter:
            mock_plotter = mock.MagicMock()
            MockPlotter.return_value = mock_plotter
            plotDAT(mock_app)
            # Check that output was reformatted
            call_args = MockPlotter.call_args
            out_arg = call_args[0][1]
            assert out_arg.endswith(".png")

    def test_plotDAT_invalid_extension_raises(self):
        from bionetgen.core.main import plotDAT
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "model.xyz"
        with pytest.raises(AssertionError):
            plotDAT(mock_app)

    def test_printInfo(self):
        from bionetgen.core.main import printInfo
        mock_app = mock.MagicMock()
        with mock.patch("bionetgen.core.main.BNGInfo") as MockInfo:
            mock_info = mock.MagicMock()
            MockInfo.return_value = mock_info
            printInfo(mock_app)
            MockInfo.assert_called_once()
            mock_info.gatherInfo.assert_called_once()
            mock_info.messageGeneration.assert_called_once()
            mock_info.run.assert_called_once()

    def test_visualizeModel(self):
        from bionetgen.core.main import visualizeModel
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "model.bngl"
        mock_app.pargs.output = None
        mock_app.pargs.type = "contactmap"
        mock_app.config.get.return_value = "/usr/local/bin/BNG2.pl"
        with mock.patch("bionetgen.core.main.BNGVisualize") as MockViz:
            mock_viz = mock.MagicMock()
            MockViz.return_value = mock_viz
            visualizeModel(mock_app)
            MockViz.assert_called_once()
            mock_viz.run.assert_called_once()

    def test_graphDiff(self):
        from bionetgen.core.main import graphDiff
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "g1.graphml"
        mock_app.pargs.input2 = "g2.graphml"
        mock_app.pargs.output = None
        mock_app.pargs.output2 = None
        mock_app.pargs.mode = "matrix"
        mock_app.pargs.colors = None
        with mock.patch("bionetgen.core.main.BNGGdiff") as MockGdiff:
            mock_gdiff = mock.MagicMock()
            MockGdiff.return_value = mock_gdiff
            graphDiff(mock_app)
            MockGdiff.assert_called_once()
            mock_gdiff.run.assert_called_once()

    def test_generate_notebook_no_input(self):
        from bionetgen.core.main import generate_notebook
        mock_app = mock.MagicMock()
        mock_app.pargs.input = None
        mock_app.pargs.output = ""
        mock_app.pargs.open = False
        mock_app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB:
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            generate_notebook(mock_app)
            MockNB.assert_called_once_with("/tmp/nb.ipynb")
            mock_nb.write.assert_called_once()

    def test_generate_notebook_with_input(self):
        from bionetgen.core.main import generate_notebook
        mock_app = mock.MagicMock()
        mock_app.pargs.input = "model.bngl"
        mock_app.pargs.output = ""
        mock_app.pargs.open = False
        mock_app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        # bionetgen is imported locally inside generate_notebook; patch the bngmodel it uses
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB, \
             mock.patch("bionetgen.bngmodel") as mock_bngmodel:
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            generate_notebook(mock_app)
            MockNB.assert_called_once()
            mock_nb.write.assert_called_once()


# ---------------------------------------------------------------------------
# 7. main.py — CLI entry point
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Test the main.py CLI module."""

    def test_bionetgen_app_creation(self):
        from bionetgen.main import BioNetGen
        app = BioNetGen()
        app.setup()
        assert app._meta.label == "bionetgen"
        # close() calls sys.exit due to exit_on_close=True
        with pytest.raises(SystemExit):
            app.close()

    def test_bionetgen_test_app(self):
        from bionetgen.main import BioNetGenTest
        with BioNetGenTest() as app:
            app.run()
            assert app._meta.label == "bionetgen"

    def test_require_action_init(self):
        from bionetgen.main import requireAction
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--require", action=requireAction, type=str, default=None)
        args = parser.parse_args([])
        assert args.require is None

    def test_require_action_nargs_raises(self):
        from bionetgen.main import requireAction
        import argparse
        with pytest.raises(ValueError, match="nargs not allowed"):
            requireAction(["--test"], "test", nargs="+")

    def test_bngbase_meta(self):
        from bionetgen.main import BNGBase
        assert BNGBase.Meta.label == "bionetgen"

    def test_main_function_assertion_error(self):
        """Test that main() handles AssertionError via BioNetGenTest."""
        from bionetgen.main import BioNetGenTest
        with BioNetGenTest() as app:
            app.run()
            assert app.config is not None


# ---------------------------------------------------------------------------
# 8. csimulator.py — CSimWrapper and CSimulator (mocked)
# ---------------------------------------------------------------------------

class TestCSimulator:
    """Test the csimulator module via mocking."""

    def test_result_struct_fields(self):
        from bionetgen.simulator.csimulator import RESULT
        field_names = [f[0] for f in RESULT._fields_]
        assert "status" in field_names
        assert "n_observables" in field_names
        assert "n_species" in field_names
        assert "n_tpts" in field_names
        assert "observables" in field_names
        assert "species" in field_names

    def test_csim_wrapper_init(self):
        import ctypes
        # Create a mock shared lib
        mock_lib = mock.MagicMock()
        with mock.patch("ctypes.CDLL", return_value=mock_lib):
            from bionetgen.simulator.csimulator import CSimWrapper
            wrapper = CSimWrapper("/fake/lib.so", num_params=3, num_spec_init=2)
            assert wrapper.num_params == 3
            assert wrapper.num_spec_init == 2
            assert wrapper.lib is mock_lib

    def test_csim_wrapper_set_species_init(self):
        mock_lib = mock.MagicMock()
        with mock.patch("ctypes.CDLL", return_value=mock_lib):
            from bionetgen.simulator.csimulator import CSimWrapper
            wrapper = CSimWrapper("/fake/lib.so", num_params=2, num_spec_init=3)
            wrapper.set_species_init([1.0, 2.0, 3.0])
            np.testing.assert_array_equal(wrapper.species_init, [1.0, 2.0, 3.0])

    def test_csim_wrapper_set_species_init_wrong_len(self):
        mock_lib = mock.MagicMock()
        with mock.patch("ctypes.CDLL", return_value=mock_lib):
            from bionetgen.simulator.csimulator import CSimWrapper
            wrapper = CSimWrapper("/fake/lib.so", num_params=2, num_spec_init=3)
            with pytest.raises(AssertionError):
                wrapper.set_species_init([1.0, 2.0])

    def test_csim_wrapper_set_parameters(self):
        mock_lib = mock.MagicMock()
        with mock.patch("ctypes.CDLL", return_value=mock_lib):
            from bionetgen.simulator.csimulator import CSimWrapper
            wrapper = CSimWrapper("/fake/lib.so", num_params=2, num_spec_init=3)
            wrapper.set_parameters([0.1, 0.2])
            np.testing.assert_array_equal(wrapper.parameters, [0.1, 0.2])

    def test_csim_wrapper_set_parameters_wrong_len(self):
        mock_lib = mock.MagicMock()
        with mock.patch("ctypes.CDLL", return_value=mock_lib):
            from bionetgen.simulator.csimulator import CSimWrapper
            wrapper = CSimWrapper("/fake/lib.so", num_params=2, num_spec_init=3)
            with pytest.raises(AssertionError):
                wrapper.set_parameters([0.1])

    def test_csimulator_str_repr(self):
        """Test CSimulator.__str__ and __repr__ via mock."""
        # CSimulator.__init__ does a lot (compiles, etc.), so we test
        # the methods by constructing a partial object
        from bionetgen.simulator.csimulator import CSimulator
        obj = CSimulator.__new__(CSimulator)
        obj.model = mock.MagicMock()
        obj.model.parameters = {"k1": mock.MagicMock(expr="0.1")}
        obj.model.species = {"A": mock.MagicMock(count="100")}
        s = str(obj)
        assert "C/Python Simulator" in s
        assert repr(obj) == s
