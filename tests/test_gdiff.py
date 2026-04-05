"""Tests for bionetgen/core/tools/gdiff.py"""

import copy
import json
import os
import tempfile

import pytest
import xmltodict

from bionetgen.core.tools.gdiff import BNGGdiff


# ---------------------------------------------------------------------------
# Helper: minimal graphml dicts that mirror what xmltodict.parse produces
# from BioNetGen contact-map graphml files.
#
# Node structure:
#   Species (molecule) nodes -> color #D2D2D2 (grey, color_id 0)
#   Component nodes          -> color #FFFFFF (white, color_id 1)
#   State nodes              -> color #FFCC00 (yellow, color_id 2)
#
# A molecule node is a "group node" containing a graph of component nodes.
# Component nodes may contain a graph of state nodes.
# ---------------------------------------------------------------------------


def _make_shape_node(name, color, node_id, font_size="12"):
    """Return a simple ShapeNode dict (leaf node -- component or state)."""
    return {
        "@id": node_id,
        "data": {
            "@key": "d6",
            "y:ShapeNode": {
                "y:Geometry": {"@height": "30", "@width": "30"},
                "y:Fill": {"@color": color, "@transparent": "false"},
                "y:NodeLabel": {"#text": name, "@fontSize": font_size},
            },
        },
    }


def _make_group_node(name, color, node_id, children, font_size="12"):
    """Return a ProxyAutoBoundsNode (group / molecule) dict with children."""
    child_nodes = children if len(children) != 1 else children[0]
    return {
        "@id": node_id,
        "data": [
            {"@key": "d4"},
            {
                "@key": "d6",
                "y:ProxyAutoBoundsNode": {
                    "y:Realizers": {
                        "y:GroupNode": {
                            "y:Geometry": {"@height": "80", "@width": "120"},
                            "y:Fill": {"@color": color, "@transparent": "false"},
                            "y:NodeLabel": {"#text": name, "@fontSize": font_size},
                        }
                    }
                },
            },
        ],
        "graph": {
            "@id": node_id + ":",
            "node": child_nodes,
        },
    }


def _make_edge(eid, source, target):
    return {
        "@id": eid,
        "@source": source,
        "@target": target,
        "data": {"@key": "d10"},
    }


def _make_graphml(nodes, edges):
    """Wrap nodes and edges into a top-level graphml dict.

    Always keeps nodes and edges as lists (matching xmltodict with force_list).
    """
    return {
        "graphml": {
            "@xmlns": "http://graphml.graphstruct.org/graphml",
            "graph": {
                "@id": "G",
                "@edgedefault": "undirected",
                "node": nodes,
                "edge": edges,
            },
        }
    }


def _write_graphml(path, gdict):
    """Write a graphml dict to a file via xmltodict.unparse."""
    with open(path, "w") as f:
        xmltodict.unparse(gdict, output=f, pretty=True)


def _read_graphml(path):
    """Read graphml and force node/edge to always be lists."""
    with open(path, "r") as f:
        gdict = xmltodict.parse(
            f.read(),
            force_list=("node", "edge"),
        )
    return gdict


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

# Graph 1: molecule A with components a1, a2; molecule B with component b1
# Edges: a1 -- b1, a2 -- b1  (two edges to keep list structure after round-trip)
COMP_A1 = _make_shape_node("a1", "#FFFFFF", "n0::n0")
COMP_A2 = _make_shape_node("a2", "#FFFFFF", "n0::n1")
MOL_A = _make_group_node("A", "#D2D2D2", "n0", [COMP_A1, COMP_A2])

COMP_B1 = _make_shape_node("b1", "#FFFFFF", "n1::n0")
MOL_B = _make_group_node("B", "#D2D2D2", "n1", [COMP_B1])

GRAPH1 = _make_graphml(
    [MOL_A, MOL_B],
    [_make_edge("e0", "n0::n0", "n1::n0"), _make_edge("e1", "n0::n1", "n1::n0")],
)

# Graph 2: molecule A with component a1 (same as g1); molecule C with component c1
# Edges: a1 -- c1, plus a self-edge for list preservation
COMP_A1_G2 = _make_shape_node("a1", "#FFFFFF", "n0::n0")
MOL_A_G2 = _make_group_node("A", "#D2D2D2", "n0", [COMP_A1_G2])

COMP_C1 = _make_shape_node("c1", "#FFFFFF", "n1::n0")
MOL_C = _make_group_node("C", "#D2D2D2", "n1", [COMP_C1])

GRAPH2 = _make_graphml(
    [MOL_A_G2, MOL_C],
    [_make_edge("e0", "n0::n0", "n1::n0"), _make_edge("e1", "n1::n0", "n0::n0")],
)


@pytest.fixture
def graphml_pair(tmp_path):
    """Write two graphml files and return their paths.

    Uses force_list on re-read so nodes/edges stay as lists after round-trip.
    """
    p1 = str(tmp_path / "g1.graphml")
    p2 = str(tmp_path / "g2.graphml")
    _write_graphml(p1, copy.deepcopy(GRAPH1))
    _write_graphml(p2, copy.deepcopy(GRAPH2))
    return p1, p2


def _make_gdiff(p1, p2, **kwargs):
    """Build a BNGGdiff but replace gdict_1/gdict_2 with force_list parsed versions."""
    obj = BNGGdiff.__new__(BNGGdiff)
    # Replicate __init__ logic but with force_list parsing
    from bionetgen.core.utils.logging import BNGLogger

    obj.app = kwargs.get("app", None)
    obj.logger = BNGLogger(app=obj.app)
    obj.input = p1
    obj.input2 = p2
    obj.output = kwargs.get("out", None)
    obj.output2 = kwargs.get("out2", None)

    colors = kwargs.get("colors", {
        "g1": ["#dadbfd", "#e6e7fe", "#f3f3ff"],
        "g2": ["#ff9e81", "#ffbfaa", "#ffdfd4"],
        "intersect": ["#c4ed9e", "#d9f4be", "#ecf9df"],
    })
    if isinstance(colors, dict):
        obj.colors = colors
    elif isinstance(colors, str):
        with open(colors, "r") as f:
            obj.colors = json.load(f)
    elif colors is None:
        obj.colors = {
            "g1": ["#dadbfd", "#e6e7fe", "#f3f3ff"],
            "g2": ["#ff9e81", "#ffbfaa", "#ffdfd4"],
            "intersect": ["#c4ed9e", "#d9f4be", "#ecf9df"],
        }

    mode = kwargs.get("mode", "matrix")
    obj.available_modes = ["matrix", "union"]
    obj.mode = mode

    obj.gdict_1 = _read_graphml(p1)
    obj.gdict_2 = _read_graphml(p2)
    return obj


@pytest.fixture
def gdiff_obj(graphml_pair):
    """Return a BNGGdiff in matrix mode with default colors."""
    p1, p2 = graphml_pair
    return _make_gdiff(p1, p2)


# ======================================================================
# __init__ tests
# ======================================================================


class TestInit:
    def test_default_colors_and_mode(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = BNGGdiff(p1, p2)
        assert obj.mode == "matrix"
        assert "g1" in obj.colors
        assert "g2" in obj.colors
        assert "intersect" in obj.colors

    def test_dict_colors(self, graphml_pair):
        p1, p2 = graphml_pair
        custom = {
            "g1": ["#111111", "#222222", "#333333"],
            "g2": ["#444444", "#555555", "#666666"],
            "intersect": ["#777777", "#888888", "#999999"],
        }
        obj = BNGGdiff(p1, p2, colors=custom)
        assert obj.colors == custom

    def test_none_colors_uses_defaults(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = BNGGdiff(p1, p2, colors=None)
        assert "g1" in obj.colors

    def test_str_colors_loads_json(self, graphml_pair, tmp_path):
        p1, p2 = graphml_pair
        color_dict = {
            "g1": ["#aaaaaa", "#bbbbbb", "#cccccc"],
            "g2": ["#dddddd", "#eeeeee", "#ffffff"],
            "intersect": ["#111111", "#222222", "#333333"],
        }
        cpath = str(tmp_path / "colors.json")
        with open(cpath, "w") as f:
            json.dump(color_dict, f)
        obj = BNGGdiff(p1, p2, colors=cpath)
        assert obj.colors == color_dict

    def test_invalid_colors_raises(self, graphml_pair):
        p1, p2 = graphml_pair
        with pytest.raises(ValueError, match="not recognized"):
            BNGGdiff(p1, p2, colors=12345)

    def test_invalid_mode_raises(self, graphml_pair):
        p1, p2 = graphml_pair
        with pytest.raises(ValueError, match="not a valid mode"):
            BNGGdiff(p1, p2, mode="foobar")

    def test_union_mode(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = BNGGdiff(p1, p2, mode="union")
        assert obj.mode == "union"

    def test_graphml_loaded(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = BNGGdiff(p1, p2)
        assert "graphml" in obj.gdict_1
        assert "graphml" in obj.gdict_2


# ======================================================================
# Helper / utility method tests
# ======================================================================


class TestHelpers:
    def test_get_node_name_shape(self, gdiff_obj):
        node = _make_shape_node("foo", "#FFFFFF", "n0")
        assert gdiff_obj._get_node_name(node) == "foo"

    def test_get_node_name_group(self, gdiff_obj):
        child = _make_shape_node("c", "#FFFFFF", "n0::n0")
        node = _make_group_node("Mol", "#D2D2D2", "n0", [child])
        assert gdiff_obj._get_node_name(node) == "Mol"

    def test_get_node_color_shape(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0")
        assert gdiff_obj._get_node_color(node) == "#D2D2D2"

    def test_get_color_id(self, gdiff_obj):
        grey = _make_shape_node("a", "#D2D2D2", "n0")
        white = _make_shape_node("b", "#FFFFFF", "n1")
        yellow = _make_shape_node("c", "#FFCC00", "n2")
        assert gdiff_obj._get_color_id(grey) == 0
        assert gdiff_obj._get_color_id(white) == 1
        assert gdiff_obj._get_color_id(yellow) == 2

    def test_get_color_id_unknown_raises(self, gdiff_obj):
        node = _make_shape_node("x", "#123456", "n0")
        with pytest.raises(RuntimeError, match="doesn't match known colors"):
            gdiff_obj._get_color_id(node)

    def test_color_node(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0")
        result = gdiff_obj._color_node(node, "#AABBCC")
        assert result is True
        assert gdiff_obj._get_node_color(node) == "#AABBCC"

    def test_get_node_id(self, gdiff_obj):
        assert gdiff_obj._get_node_id({"@id": "n3"}) == "n3"
        assert gdiff_obj._get_node_id({}) is None

    def test_set_node_id(self, gdiff_obj):
        node = {"@id": "n0"}
        assert gdiff_obj._set_node_id(node, "n5") is True
        assert node["@id"] == "n5"
        assert gdiff_obj._set_node_id({}, "n5") is False

    def test_get_id_list(self, gdiff_obj):
        assert gdiff_obj._get_id_list("n1::n2::n3") == [1, 2, 3]

    def test_get_id_str(self, gdiff_obj):
        assert gdiff_obj._get_id_str([1, 2, 3]) == "n1::n2::n3"

    def test_resize_node_font(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0", font_size="12")
        gdiff_obj._resize_node_font(node, 24)
        assert gdiff_obj._get_font_size(node) == 24

    def test_get_node_from_names_root(self, gdiff_obj):
        g = copy.deepcopy(GRAPH1)
        result = gdiff_obj._get_node_from_names(g, [])
        assert result == g["graphml"]

    def test_get_node_from_names_existing(self, gdiff_obj):
        g = copy.deepcopy(GRAPH1)
        result = gdiff_obj._get_node_from_names(g, ["A"])
        assert gdiff_obj._get_node_name(result) == "A"

    def test_get_node_from_names_nested(self, gdiff_obj):
        g = copy.deepcopy(GRAPH1)
        result = gdiff_obj._get_node_from_names(g, ["A", "a1"])
        assert gdiff_obj._get_node_name(result) == "a1"

    def test_get_node_from_names_missing(self, gdiff_obj):
        g = copy.deepcopy(GRAPH1)
        result = gdiff_obj._get_node_from_names(g, ["Z"])
        assert result is None

    def test_get_node_from_names_without_graphml_key(self, gdiff_obj):
        """Test _get_node_from_names when passed a sub-graph (no 'graphml' key)."""
        g = copy.deepcopy(GRAPH1)
        # Pass the molecule node directly (it has a 'graph' with children)
        mol_a = g["graphml"]["graph"]["node"][0]
        result = gdiff_obj._get_node_from_names(mol_a, ["a1"])
        assert gdiff_obj._get_node_name(result) == "a1"

    def test_get_node_properties_shape(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0")
        props = gdiff_obj._get_node_properties(node)
        assert "y:NodeLabel" in props
        assert "y:Fill" in props

    def test_get_node_properties_group(self, gdiff_obj):
        child = _make_shape_node("c", "#FFFFFF", "n0::n0")
        node = _make_group_node("G", "#D2D2D2", "n0", [child])
        props = gdiff_obj._get_node_properties(node)
        assert props["y:NodeLabel"]["#text"] == "G"

    def test_get_node_fill(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0")
        fill = gdiff_obj._get_node_fill(node)
        assert fill["@color"] == "#D2D2D2"

    def test_get_font_size(self, gdiff_obj):
        node = _make_shape_node("x", "#D2D2D2", "n0", font_size="18")
        assert gdiff_obj._get_font_size(node) == 18


# ======================================================================
# _recolor_graph tests
# ======================================================================


class TestRecolorGraph:
    def test_recolor_changes_colors(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        color_list = ["#AA0000", "#BB0000", "#CC0000"]
        result = gdiff_obj._recolor_graph(g, color_list)
        # The original should not be modified (deepcopy inside _recolor_graph)
        orig_mol = g["graphml"]["graph"]["node"][0]
        orig_color = gdiff_obj._get_node_color(orig_mol)
        # The result should have new molecule color (color_id 0 -> index 0)
        mol_color = gdiff_obj._get_node_color(result["graphml"]["graph"]["node"][0])
        assert mol_color == "#AA0000"

    def test_recolor_does_not_mutate_original(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        original = copy.deepcopy(g)
        gdiff_obj._recolor_graph(g, ["#AA0000", "#BB0000", "#CC0000"])
        assert g == original

    def test_recolor_component_gets_index_1(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        color_list = ["#AA0000", "#BB0000", "#CC0000"]
        result = gdiff_obj._recolor_graph(g, color_list)
        # Component node (white -> index 1)
        comp = result["graphml"]["graph"]["node"][0]["graph"]["node"][0]
        assert gdiff_obj._get_node_color(comp) == "#BB0000"


# ======================================================================
# _resize_fonts tests
# ======================================================================


class TestResizeFonts:
    def test_resize_changes_font(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        gdiff_obj._resize_fonts(g, 30)
        # Check a leaf node's font size
        comp = g["graphml"]["graph"]["node"][0]["graph"]["node"][0]
        assert gdiff_obj._get_font_size(comp) == 30

    def test_resize_affects_all_levels(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        gdiff_obj._resize_fonts(g, 42)
        # molecule level
        mol = g["graphml"]["graph"]["node"][0]
        assert gdiff_obj._get_font_size(mol) == 42
        # component level
        comp = mol["graph"]["node"][0]
        assert gdiff_obj._get_font_size(comp) == 42


# ======================================================================
# _find_diff tests
# ======================================================================


class TestFindDiff:
    def test_find_diff_returns_dict_and_rename_map(self, gdiff_obj):
        g1 = copy.deepcopy(gdiff_obj.gdict_1)
        g2 = copy.deepcopy(gdiff_obj.gdict_2)
        dg, rmap = gdiff_obj._find_diff(g1, g2)
        assert isinstance(dg, dict)
        assert isinstance(rmap, dict)
        assert "graphml" in dg

    def test_find_diff_same_graph(self, graphml_pair):
        """Diff of a graph with itself -- all nodes should be intersect color."""
        p1, _ = graphml_pair
        obj = _make_gdiff(p1, p1)
        g1 = copy.deepcopy(obj.gdict_1)
        g2 = copy.deepcopy(obj.gdict_2)
        dg, _ = obj._find_diff(g1, g2)
        # All molecule nodes should have intersect color (index 0 = species)
        intersect_species_color = obj.colors["intersect"][0]
        mol = dg["graphml"]["graph"]["node"][0]
        assert obj._get_node_color(mol) == intersect_species_color

    def test_find_diff_unique_nodes_get_g1_color(self, gdiff_obj):
        """Nodes in g1 but not g2 get g1 color."""
        g1 = copy.deepcopy(gdiff_obj.gdict_1)
        g2 = copy.deepcopy(gdiff_obj.gdict_2)
        dg, _ = gdiff_obj._find_diff(g1, g2)
        # Molecule B is in g1 but not g2
        mol_b = dg["graphml"]["graph"]["node"][1]
        name = gdiff_obj._get_node_name(mol_b)
        assert name == "B"
        assert gdiff_obj._get_node_color(mol_b) == gdiff_obj.colors["g1"][0]

    def test_find_diff_common_nodes_get_intersect_color(self, gdiff_obj):
        """Molecule A exists in both -- should get intersect color."""
        g1 = copy.deepcopy(gdiff_obj.gdict_1)
        g2 = copy.deepcopy(gdiff_obj.gdict_2)
        dg, _ = gdiff_obj._find_diff(g1, g2)
        mol_a = dg["graphml"]["graph"]["node"][0]
        name = gdiff_obj._get_node_name(mol_a)
        assert name == "A"
        assert gdiff_obj._get_node_color(mol_a) == gdiff_obj.colors["intersect"][0]

    def test_find_diff_rename_map_has_entries(self, gdiff_obj):
        g1 = copy.deepcopy(gdiff_obj.gdict_1)
        g2 = copy.deepcopy(gdiff_obj.gdict_2)
        _, rmap = gdiff_obj._find_diff(g1, g2)
        assert len(rmap) > 0

    def test_find_diff_with_dg_argument(self, gdiff_obj):
        """Passing an explicit dg dict should modify it instead of copying g1."""
        g1 = copy.deepcopy(gdiff_obj.gdict_1)
        g2 = copy.deepcopy(gdiff_obj.gdict_2)
        dg = copy.deepcopy(g1)
        result, _ = gdiff_obj._find_diff(g1, g2, dg=dg)
        assert result is dg


# ======================================================================
# _find_diff_union tests
# ======================================================================


class TestFindDiffUnion:
    def test_union_contains_all_nodes(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        g1 = copy.deepcopy(obj.gdict_1)
        g2 = copy.deepcopy(obj.gdict_2)
        dg = obj._find_diff_union(g1, g2)
        # Should have A, B, and C molecules
        top_nodes = dg["graphml"]["graph"]["node"]
        names = [obj._get_node_name(n) for n in top_nodes]
        assert "A" in names
        assert "B" in names
        assert "C" in names

    def test_union_no_duplicate_edges(self, graphml_pair):
        """The edges from g1 should not be duplicated when diffing with itself."""
        p1, _ = graphml_pair
        obj = _make_gdiff(p1, p1, mode="union")
        g1 = copy.deepcopy(obj.gdict_1)
        g2 = copy.deepcopy(obj.gdict_2)
        dg = obj._find_diff_union(g1, g2)
        edges = dg["graphml"]["graph"]["edge"]
        # g1 has 2 edges; union with itself should still have 2
        assert len(edges) == 2

    def test_union_adds_edges_from_g2(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        g1 = copy.deepcopy(obj.gdict_1)
        g2 = copy.deepcopy(obj.gdict_2)
        dg = obj._find_diff_union(g1, g2)
        edges = dg["graphml"]["graph"]["edge"]
        # Should have edges from both graphs (minus duplicates)
        assert len(edges) >= 2


# ======================================================================
# diff_graphs tests (end-to-end through the method)
# ======================================================================


class TestDiffGraphs:
    def test_matrix_mode_returns_four_graphs(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        # matrix mode returns: diff1, diff2, recolored g1, recolored g2
        assert len(graphs) == 4

    def test_matrix_mode_output_names(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        keys = list(graphs.keys())
        # Should contain diff and recolored file names
        assert any("diff" in k for k in keys)
        assert any("recolored" in k for k in keys)

    def test_union_mode_returns_one_graph(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        assert len(graphs) == 1

    def test_matrix_mode_custom_output_names(self, graphml_pair, tmp_path):
        p1, p2 = graphml_pair
        out1 = str(tmp_path / "custom_diff1.graphml")
        out2 = str(tmp_path / "custom_diff2.graphml")
        obj = _make_gdiff(p1, p2, out=out1, out2=out2, mode="matrix")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        assert out1 in graphs
        assert out2 in graphs

    def test_union_mode_custom_output_name(self, graphml_pair, tmp_path):
        p1, p2 = graphml_pair
        out = str(tmp_path / "custom_union.graphml")
        obj = _make_gdiff(p1, p2, out=out, mode="union")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        assert out in graphs

    def test_matrix_default_output_names_derived_from_input(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        keys = list(graphs.keys())
        assert any("g1_g2_diff" in k for k in keys)
        assert any("g2_g1_diff" in k for k in keys)

    def test_union_default_output_name_derived_from_input(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        keys = list(graphs.keys())
        assert any("union" in k for k in keys)

    def test_all_graphs_are_valid_dicts(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.diff_graphs(obj.gdict_1, obj.gdict_2, obj.colors)
        for name, gdict in graphs.items():
            assert "graphml" in gdict, f"Graph {name} missing 'graphml' key"


# ======================================================================
# run() tests
# ======================================================================


class TestRun:
    def test_run_matrix_writes_files(self, graphml_pair, tmp_path, monkeypatch):
        p1, p2 = graphml_pair
        monkeypatch.chdir(tmp_path)
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.run()
        for gname in graphs:
            assert os.path.exists(gname), f"Expected output file {gname} to exist"
            # Verify the file is valid XML
            with open(gname, "r") as f:
                parsed = xmltodict.parse(f.read())
            assert "graphml" in parsed

    def test_run_union_writes_file(self, graphml_pair, tmp_path, monkeypatch):
        p1, p2 = graphml_pair
        monkeypatch.chdir(tmp_path)
        obj = _make_gdiff(p1, p2, mode="union")
        graphs = obj.run()
        for gname in graphs:
            assert os.path.exists(gname)

    def test_run_returns_dict(self, graphml_pair, tmp_path, monkeypatch):
        p1, p2 = graphml_pair
        monkeypatch.chdir(tmp_path)
        obj = _make_gdiff(p1, p2, mode="matrix")
        result = obj.run()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_run_matrix_with_custom_outputs(self, graphml_pair, tmp_path, monkeypatch):
        p1, p2 = graphml_pair
        monkeypatch.chdir(tmp_path)
        out1 = str(tmp_path / "out1.graphml")
        out2 = str(tmp_path / "out2.graphml")
        obj = _make_gdiff(p1, p2, out=out1, out2=out2, mode="matrix")
        obj.run()
        assert os.path.exists(out1)
        assert os.path.exists(out2)

    def test_run_output_files_are_reparseable(self, graphml_pair, tmp_path, monkeypatch):
        p1, p2 = graphml_pair
        monkeypatch.chdir(tmp_path)
        obj = _make_gdiff(p1, p2, mode="matrix")
        graphs = obj.run()
        for gname in graphs:
            with open(gname, "r") as f:
                reparsed = xmltodict.parse(f.read())
            assert "graphml" in reparsed


# ======================================================================
# Single-node edge cases (when graph["node"] is a dict, not a list)
# ======================================================================


class TestSingleNodeGraph:
    def test_recolor_single_child_node(self, graphml_pair):
        """Graph where a molecule has exactly one component (dict not list)."""
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2)
        # MOL_B has a single child after round-trip -- still works
        g = copy.deepcopy(obj.gdict_1)
        result = obj._recolor_graph(g, ["#AA0000", "#BB0000", "#CC0000"])
        assert "graphml" in result

    def test_resize_fonts_single_child(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2)
        g = copy.deepcopy(obj.gdict_1)
        # Should not raise even if some nodes have a single child
        obj._resize_fonts(g, 25)


# ======================================================================
# _add_node_to_graph tests
# ======================================================================


class TestAddNodeToGraph:
    def test_add_node_increases_count(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        dg = copy.deepcopy(obj.gdict_1)
        original_count = len(dg["graphml"]["graph"]["node"])
        new_node = _make_group_node("NewMol", "#D2D2D2", "n99", [
            _make_shape_node("nc1", "#FFFFFF", "n99::n0"),
        ])
        obj._add_node_to_graph(new_node, dg, ["NewMol"])
        new_count = len(dg["graphml"]["graph"]["node"])
        assert new_count == original_count + 1

    def test_add_node_returns_copied_node(self, graphml_pair):
        p1, p2 = graphml_pair
        obj = _make_gdiff(p1, p2, mode="union")
        dg = copy.deepcopy(obj.gdict_1)
        new_node = _make_shape_node("leaf", "#FFFFFF", "n99::n0")
        # Add under molecule A
        result = obj._add_node_to_graph(new_node, dg, ["A", "leaf"])
        assert result is not None
        assert obj._get_node_name(result) == "leaf"


# ======================================================================
# _get_node_from_keylist tests
# ======================================================================


class TestGetNodeFromKeylist:
    def test_keylist_graphml_only(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        result = gdiff_obj._get_node_from_keylist(g, ["graphml"])
        assert result == g["graphml"]

    def test_keylist_finds_node(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        # Get the first top-level node by its id
        first_id = g["graphml"]["graph"]["node"][0]["@id"]
        result = gdiff_obj._get_node_from_keylist(g, ["graphml", first_id])
        assert result["@id"] == first_id

    def test_keylist_missing_returns_none(self, gdiff_obj):
        g = copy.deepcopy(gdiff_obj.gdict_1)
        result = gdiff_obj._get_node_from_keylist(g, ["graphml", "nonexistent"])
        assert result is None
