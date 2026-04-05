"""Tests for bionetgen.network.structs and bionetgen.network.blocks."""

import pytest

from bionetgen.network.structs import (
    NetworkObj,
    NetworkParameter,
    NetworkCompartment,
    NetworkGroup,
    NetworkSpecies,
    NetworkFunction,
    NetworkReaction,
    NetworkEnergyPattern,
    NetworkPopulationMap,
)
from bionetgen.network.blocks import (
    NetworkBlock,
    NetworkParameterBlock,
    NetworkCompartmentBlock,
    NetworkGroupBlock,
    NetworkSpeciesBlock,
    NetworkFunctionBlock,
    NetworkReactionBlock,
    NetworkEnergyPatternBlock,
    NetworkPopulationMapBlock,
)


# ===== NetworkObj base class =====


class TestNetworkObj:
    def test_comment_strips_hash(self):
        obj = NetworkObj()
        obj.comment = "#this is a comment"
        assert obj.comment == "this is a comment"

    def test_comment_no_hash(self):
        obj = NetworkObj()
        obj.comment = "this is a comment"
        assert obj.comment == "this is a comment"

    def test_comment_none(self):
        obj = NetworkObj()
        obj.comment = None
        assert obj.comment is None

    def test_comment_empty_string(self):
        obj = NetworkObj()
        obj.comment = ""
        assert obj.comment is None

    def test_default_comment_is_empty_string(self):
        obj = NetworkObj()
        assert obj.comment == ""

    def test_line_label_int(self):
        obj = NetworkObj()
        obj.line_label = 5
        assert obj.line_label == "5 "

    def test_line_label_str_int(self):
        obj = NetworkObj()
        obj.line_label = "10"
        assert obj.line_label == "10 "

    def test_line_label_str_non_int(self):
        obj = NetworkObj()
        obj.line_label = "myLabel"
        assert obj.line_label == "myLabel: "

    def test_print_line_no_label_no_comment(self):
        obj = NetworkObj()
        obj.comment = None
        # gen_string not overridden, so we need a subclass; test with NetworkParameter
        p = NetworkParameter(1, "k1", "0.5")
        p._line_label = None
        p.comment = None
        line = p.print_line()
        assert line == "  k1 0.5"

    def test_print_line_with_label_and_comment(self):
        p = NetworkParameter(1, "k1", "0.5", comment="rate constant")
        line = p.print_line()
        assert line == "  1 k1 0.5 #rate constant"

    def test_str_and_repr(self):
        p = NetworkParameter(1, "k1", "0.5")
        assert str(p) == "k1 0.5"
        assert repr(p) == "k1 0.5"

    def test_contains(self):
        p = NetworkParameter(1, "k1", "0.5")
        assert "name" in p
        assert "nonexistent" not in p

    def test_getitem(self):
        p = NetworkParameter(1, "k1", "0.5")
        assert p["name"] == "k1"
        assert p["value"] == "0.5"

    def test_setitem(self):
        p = NetworkParameter(1, "k1", "0.5")
        p["value"] = "1.0"
        assert p.value == "1.0"

    def test_delitem(self):
        p = NetworkParameter(1, "k1", "0.5")
        p["value"] = "0.5"
        del p["value"]
        assert not hasattr(p, "value")


# ===== NetworkParameter =====


class TestNetworkParameter:
    def test_gen_string(self):
        p = NetworkParameter(1, "kf", "0.01")
        assert p.gen_string() == "kf 0.01"

    def test_line_label_from_pid(self):
        p = NetworkParameter(3, "kf", "0.01")
        assert p.line_label == "3 "

    def test_comment_set(self):
        p = NetworkParameter(1, "kf", "0.01", comment="forward rate")
        assert p.comment == "forward rate"

    def test_comment_default_empty(self):
        p = NetworkParameter(1, "kf", "0.01")
        # default comment="" -> setter sets to None
        assert p.comment is None


# ===== NetworkCompartment =====


class TestNetworkCompartment:
    def test_gen_string_without_outside(self):
        c = NetworkCompartment("cytoplasm", 3, "1.0")
        assert c.gen_string() == "cytoplasm 3 1.0"

    def test_gen_string_with_outside(self):
        c = NetworkCompartment("membrane", 2, "0.5", outside="cytoplasm")
        assert c.gen_string() == "membrane 2 0.5 cytoplasm"

    def test_str(self):
        c = NetworkCompartment("EC", 3, "1e-6")
        assert str(c) == "EC 3 1e-6"


# ===== NetworkGroup =====


class TestNetworkGroup:
    def test_gen_string(self):
        g = NetworkGroup(1, "Atot", members=["1", "2", "3"])
        assert g.gen_string() == "Atot 1,2,3 "

    def test_gen_string_single_member(self):
        g = NetworkGroup(1, "Afree", members=["1"])
        assert g.gen_string() == "Afree 1 "

    def test_line_label_from_gid(self):
        g = NetworkGroup(5, "obs", members=["1"])
        assert g.line_label == "5 "

    def test_comment(self):
        g = NetworkGroup(1, "obs", members=["1"], comment="total")
        assert g.comment == "total"


# ===== NetworkSpecies =====


class TestNetworkSpecies:
    def test_gen_string(self):
        s = NetworkSpecies(1, "A(b)", count=100)
        assert s.gen_string() == "A(b) 100"

    def test_gen_string_default_count(self):
        s = NetworkSpecies(1, "A(b)")
        assert s.gen_string() == "A(b) 0"

    def test_line_label(self):
        s = NetworkSpecies(7, "A(b)", count=50)
        assert s.line_label == "7 "

    def test_comment(self):
        s = NetworkSpecies(1, "A(b)", count=100, comment="initial")
        assert s.comment == "initial"


# ===== NetworkFunction =====


class TestNetworkFunction:
    def test_gen_string_no_args(self):
        f = NetworkFunction("rate", "k1*A")
        assert f.gen_string() == "rate = k1*A"

    def test_gen_string_with_args(self):
        f = NetworkFunction("rate", "k1*x", args=["x"])
        assert f.gen_string() == "rate(x) = k1*x"

    def test_gen_string_with_multiple_args(self):
        f = NetworkFunction("rate", "k1*x+y", args=["x", "y"])
        assert f.gen_string() == "rate(x,y) = k1*x+y"

    def test_str(self):
        f = NetworkFunction("f1", "2*x", args=["x"])
        assert str(f) == "f1(x) = 2*x"


# ===== NetworkReaction =====


class TestNetworkReaction:
    def test_gen_string(self):
        r = NetworkReaction(1, reactants=["1", "2"], products=["3"], rate_constant="k1")
        assert r.gen_string() == "1,2 3 k1"

    def test_line_label(self):
        r = NetworkReaction(4, reactants=["1"], products=["2"], rate_constant="k1")
        assert r.line_label == "4 "

    def test_comment(self):
        r = NetworkReaction(
            1,
            reactants=["1"],
            products=["2"],
            rate_constant="k1",
            comment="binding",
        )
        assert r.comment == "binding"

    def test_comment_none(self):
        r = NetworkReaction(1, reactants=["1"], products=["2"], rate_constant="k1")
        assert r.comment is None


# ===== NetworkEnergyPattern =====


class TestNetworkEnergyPattern:
    def test_gen_string(self):
        ep = NetworkEnergyPattern("ep1", "A(b!1).B(a!1)", "epsilon")
        assert ep.gen_string() == "A(b!1).B(a!1) epsilon"

    def test_str(self):
        ep = NetworkEnergyPattern("ep1", "A(b)", "e1")
        assert str(ep) == "A(b) e1"


# ===== NetworkPopulationMap =====


class TestNetworkPopulationMap:
    def test_gen_string(self):
        pm = NetworkPopulationMap("pm1", "A(b~0)", "A_pop", "lump1")
        assert pm.gen_string() == "A(b~0) -> A_pop lump1"

    def test_attributes(self):
        pm = NetworkPopulationMap("pm1", "A(b~0)", "A_pop", "lump1")
        assert pm.name == "pm1"
        assert pm.species == "A(b~0)"
        assert pm.population == "A_pop"
        assert pm.rate == "lump1"


# ===== NetworkBlock base class =====


class TestNetworkBlock:
    def test_init(self):
        b = NetworkBlock()
        assert b.name == "NetworkBlock"
        assert b.comment == (None, None)
        assert len(b.items) == 0

    def test_len(self):
        b = NetworkBlock()
        assert len(b) == 0

    def test_add_item(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        assert len(b) == 1
        assert "k1" in b

    def test_add_item_none_name(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item((None, p))
        assert 0 in b

    def test_add_items(self):
        b = NetworkBlock()
        p1 = NetworkParameter(1, "k1", "0.5")
        p2 = NetworkParameter(2, "k2", "1.0")
        b.add_items([("k1", p1), ("k2", p2)])
        assert len(b) == 2

    def test_getitem_int(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        # int access returns the key name
        assert b[0] == "k1"

    def test_getitem_str(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        assert b["k1"] is p

    def test_setitem(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b["k1"] = p
        assert "k1" in b

    def test_delitem_existing(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        del b["k1"]
        assert "k1" not in b

    def test_delitem_nonexistent(self, capsys):
        b = NetworkBlock()
        del b["missing"]
        captured = capsys.readouterr()
        assert "Item missing not found" in captured.out

    def test_iter(self):
        b = NetworkBlock()
        p1 = NetworkParameter(1, "k1", "0.5")
        p2 = NetworkParameter(2, "k2", "1.0")
        b.add_items([("k1", p1), ("k2", p2)])
        keys = list(b)
        assert keys == ["k1", "k2"]

    def test_contains(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        assert "k1" in b
        assert "k2" not in b

    def test_repr(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        r = repr(b)
        assert "NetworkBlock" in r
        assert "1 item(s)" in r
        assert "k1" in r

    def test_gen_string_no_comments(self):
        b = NetworkBlock()
        p = NetworkParameter(1, "k1", "0.5")
        p.comment = None
        b.add_item(("k1", p))
        s = b.gen_string()
        assert s.startswith("\nbegin NetworkBlock")
        assert "end NetworkBlock" in s
        assert "k1 0.5" in s

    def test_gen_string_with_comments(self):
        b = NetworkBlock()
        b.__dict__["comment"] = ("start comment", "end comment")
        s = b.gen_string()
        assert "#start comment" in s
        assert "#end comment" in s

    def test_str(self):
        b = NetworkBlock()
        assert str(b) == b.gen_string()

    def test_setattr_updates_items(self):
        b = NetworkBlock()
        # _changes must be set via __dict__ because the base NetworkBlock
        # __setattr__ silently drops attributes not already in items when
        # items exists (no else branch for the items-key check).
        b.__dict__["_changes"] = {}
        p = NetworkParameter(1, "k1", "0.5")
        b.add_item(("k1", p))
        # setting a numeric value through setattr should update items
        b.k1 = 2.0
        assert b.items["k1"] == 2.0


# ===== NetworkParameterBlock =====


class TestNetworkParameterBlock:
    def test_name(self):
        pb = NetworkParameterBlock()
        assert pb.name == "parameters"

    def test_add_parameter(self):
        pb = NetworkParameterBlock()
        pb.add_parameter(1, "k1", "0.5")
        assert "k1" in pb
        assert isinstance(pb["k1"], NetworkParameter)
        assert pb["k1"].gen_string() == "k1 0.5"

    def test_setattr_with_float(self):
        pb = NetworkParameterBlock()
        pb._changes = {}
        pb.add_parameter(1, "k1", "0.5")
        pb.k1 = 2.0
        # Should update value in the item
        assert pb.items["k1"]["value"] == 2.0

    def test_setattr_with_string_expr(self):
        pb = NetworkParameterBlock()
        pb._changes = {}
        pb.add_parameter(1, "k1", "0.5")
        pb.k1 = "kf*2"
        assert pb.items["k1"]["value"] == "kf*2"

    def test_setattr_with_parameter_obj(self):
        pb = NetworkParameterBlock()
        pb._changes = {}
        pb.add_parameter(1, "k1", "0.5")
        new_p = NetworkParameter(2, "k1", "1.0")
        pb.k1 = new_p
        assert pb.items["k1"] is new_p


# ===== NetworkCompartmentBlock =====


class TestNetworkCompartmentBlock:
    def test_name(self):
        cb = NetworkCompartmentBlock()
        assert cb.name == "compartments"

    def test_add_compartment(self):
        cb = NetworkCompartmentBlock()
        cb.add_compartment("cytoplasm", 3, "1.0")
        assert "cytoplasm" in cb
        assert isinstance(cb["cytoplasm"], NetworkCompartment)

    def test_add_compartment_with_outside(self):
        cb = NetworkCompartmentBlock()
        cb.add_compartment("membrane", 2, "0.5", outside="cytoplasm")
        assert cb["membrane"].outside == "cytoplasm"


# ===== NetworkGroupBlock =====


class TestNetworkGroupBlock:
    def test_name(self):
        gb = NetworkGroupBlock()
        assert gb.name == "groups"

    def test_add_group(self):
        gb = NetworkGroupBlock()
        gb.add_group(1, "Atot", members=["1", "2"])
        assert "Atot" in gb
        assert isinstance(gb["Atot"], NetworkGroup)
        assert gb["Atot"].members == ["1", "2"]


# ===== NetworkSpeciesBlock =====


class TestNetworkSpeciesBlock:
    def test_name(self):
        sb = NetworkSpeciesBlock()
        assert sb.name == "species"

    def test_add_species(self):
        sb = NetworkSpeciesBlock()
        sb.add_species(1, "A(b)", count=100)
        # species are keyed by integer counter
        assert 0 in sb
        assert isinstance(sb[0], NetworkSpecies)

    def test_add_multiple_species(self):
        sb = NetworkSpeciesBlock()
        sb.add_species(1, "A(b)", count=100)
        sb.add_species(2, "B(a)", count=200)
        assert len(sb) == 2
        assert isinstance(sb[1], NetworkSpecies)

    def test_getitem_returns_item_directly(self):
        """NetworkSpeciesBlock.__getitem__ returns self.items[key] directly,
        unlike base class which returns key name for int keys."""
        sb = NetworkSpeciesBlock()
        sb.add_species(1, "A(b)", count=100)
        result = sb[0]
        assert isinstance(result, NetworkSpecies)

    def test_setitem(self):
        sb = NetworkSpeciesBlock()
        s = NetworkSpecies(1, "A(b)", count=50)
        sb[0] = s
        assert sb[0] is s


# ===== NetworkFunctionBlock =====


class TestNetworkFunctionBlock:
    def test_name(self):
        fb = NetworkFunctionBlock()
        assert fb.name == "functions"

    def test_add_function(self):
        fb = NetworkFunctionBlock()
        fb.add_function("rate", "k1*A")
        assert "rate" in fb
        assert isinstance(fb["rate"], NetworkFunction)

    def test_add_function_with_args(self):
        fb = NetworkFunctionBlock()
        fb.add_function("rate", "k1*x", args=["x"])
        assert fb["rate"].args == ["x"]


# ===== NetworkReactionBlock =====


class TestNetworkReactionBlock:
    def test_name(self):
        rb = NetworkReactionBlock()
        assert rb.name == "reactions"

    def test_add_reaction(self):
        rb = NetworkReactionBlock()
        rb.add_reaction(1, reactants=["1", "2"], products=["3"], rate_constant="k1")
        # reaction name is the int rid, so key is 1
        assert 1 in rb
        # __getitem__ with int returns key by position, so use items directly
        rxn = rb.items[1]
        assert isinstance(rxn, NetworkReaction)
        assert rxn.gen_string() == "1,2 3 k1"


# ===== NetworkEnergyPatternBlock =====


class TestNetworkEnergyPatternBlock:
    def test_name(self):
        epb = NetworkEnergyPatternBlock()
        assert epb.name == "energy patterns"

    def test_add_energy_pattern(self):
        epb = NetworkEnergyPatternBlock()
        epb.add_energy_pattern("ep1", "A(b!1).B(a!1)", "epsilon")
        assert "ep1" in epb
        assert isinstance(epb["ep1"], NetworkEnergyPattern)
        assert epb["ep1"].gen_string() == "A(b!1).B(a!1) epsilon"


# ===== NetworkPopulationMapBlock =====


class TestNetworkPopulationMapBlock:
    def test_name(self):
        pmb = NetworkPopulationMapBlock()
        assert pmb.name == "population maps"

    def test_add_population_map(self):
        pmb = NetworkPopulationMapBlock()
        pmb.add_population_map("pm1", "A(b~0)", "A_pop", "lump1")
        assert "pm1" in pmb
        assert isinstance(pmb["pm1"], NetworkPopulationMap)
        assert pmb["pm1"].gen_string() == "A(b~0) -> A_pop lump1"
