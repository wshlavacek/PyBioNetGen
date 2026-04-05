"""Tests for bionetgen/modelapi/blocks.py block classes."""

import pytest
from collections import OrderedDict

from bionetgen.modelapi.blocks import (
    ModelBlock,
    ParameterBlock,
    CompartmentBlock,
    ObservableBlock,
    SpeciesBlock,
    MoleculeTypeBlock,
    FunctionBlock,
    RuleBlock,
    ActionBlock,
    EnergyPatternBlock,
    PopulationMapBlock,
)
from bionetgen.modelapi.structs import (
    Parameter,
    Compartment,
    Observable,
    Species,
    Function,
    Action,
    Rule,
    EnergyPattern,
    PopulationMap,
    MoleculeType,
)


class FakePattern:
    """Minimal mock for Pattern objects used by Species, Observable, etc."""

    def __init__(self, name="A()"):
        self.name = name
        self.MatchOnce = False

    def __str__(self):
        return self.name


# ---- ModelBlock base class tests ----


class TestModelBlock:
    def test_init_defaults(self):
        b = ModelBlock()
        assert b.name == "ModelBlock"
        assert isinstance(b.items, OrderedDict)
        assert len(b.items) == 0
        assert b._recompile is False
        assert b.comment == (None, None)

    def test_len_empty(self):
        b = ModelBlock()
        assert len(b) == 0

    def test_len_with_items(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        assert len(b) == 1

    def test_repr(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        r = repr(b)
        assert "ModelBlock" in r
        assert "1 item(s)" in r
        assert "k1" in r

    def test_getitem_int_key(self):
        b = ModelBlock()
        p1 = Parameter("k1", 1.0)
        p2 = Parameter("k2", 2.0)
        b.items["k1"] = p1
        b.items["k2"] = p2
        # int key returns the key name at that position
        assert b[0] == "k1"
        assert b[1] == "k2"

    def test_getitem_string_key(self):
        b = ModelBlock()
        p = Parameter("k1", 1.0)
        b.items["k1"] = p
        assert b["k1"] is p

    def test_setitem(self):
        b = ModelBlock()
        p = Parameter("k1", 1.0)
        b["k1"] = p
        assert b.items["k1"] is p

    def test_delitem_existing(self):
        b = ModelBlock()
        p = Parameter("k1", 1.0)
        b.items["k1"] = p
        del b["k1"]
        assert "k1" not in b.items

    def test_delitem_missing_prints(self, capsys):
        b = ModelBlock()
        del b["nonexistent"]
        captured = capsys.readouterr()
        assert "nonexistent" in captured.out

    def test_iter(self):
        b = ModelBlock()
        b.items["a"] = Parameter("a", 1)
        b.items["b"] = Parameter("b", 2)
        keys = list(b)
        assert keys == ["a", "b"]

    def test_contains(self):
        b = ModelBlock()
        b.items["k1"] = Parameter("k1", 1)
        assert "k1" in b
        assert "k2" not in b

    def test_str_calls_gen_string(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        s = str(b)
        assert "begin ModelBlock" in s
        assert "end ModelBlock" in s
        assert "k1" in s

    def test_gen_string_with_comments(self):
        b = ModelBlock()
        # ModelBlock.__setattr__ silently drops assignments to non-item attrs
        # after items is initialized, so set comment via __dict__ directly
        b.__dict__["comment"] = ("start comment", "end comment")
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        s = b.gen_string()
        assert "#start comment" in s
        assert "#end comment" in s

    def test_gen_string_no_comments(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        s = b.gen_string()
        assert "#" not in s.split("\n")[0] or "begin ModelBlock" in s.split("\n")[0]

    def test_reset_compilation_tags(self):
        # NOTE: ModelBlock.__setattr__ silently drops assignments to non-item
        # attributes once self.items exists. This means reset_compilation_tags
        # cannot actually reset _changes/_recompile on the base ModelBlock.
        # The subclasses (ParameterBlock, etc.) have an else branch that does
        # set __dict__, so reset works there. We test a ParameterBlock instead.
        from bionetgen.modelapi.blocks import ParameterBlock

        pb = ParameterBlock()
        pb._recompile = True
        pb._changes["foo"] = "bar"
        pb.reset_compilation_tags()
        assert pb._recompile is False
        assert len(pb._changes) == 0

    def test_add_item(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.add_item(("k1", p))
        assert "k1" in b.items
        # NOTE: _recompile = True in add_item is silently dropped by
        # ModelBlock.__setattr__ (no else branch for non-item attrs).
        # Subclasses handle this correctly; see test_add_item_subclass.
        assert b._recompile is False  # actual behavior of base class

    def test_add_item_subclass(self):
        """Subclasses have the else branch so _recompile gets set."""
        pb = ParameterBlock()
        p = Parameter("k1", 0.5)
        pb.add_item(("k1", p))
        assert "k1" in pb.items
        assert pb._recompile is True

    def test_add_item_none_name(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.add_item((None, p))
        # uses index 0 as key
        assert 0 in b.items

    def test_add_item_sets_attribute(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.add_item(("k1", p))
        # attribute should be set (via __setattr__ which updates items)
        assert "k1" in b.items

    def test_add_items(self):
        b = ModelBlock()
        p1 = Parameter("k1", 1.0)
        p2 = Parameter("k2", 2.0)
        b.add_items([("k1", p1), ("k2", p2)])
        assert len(b) == 2
        assert "k1" in b.items
        assert "k2" in b.items

    def test_setattr_updates_existing_item(self):
        b = ModelBlock()
        p = Parameter("k1", 0.5)
        b.items["k1"] = p
        # Setting attribute with a float should update items and track change
        b.k1 = 2.0
        assert b.items["k1"] == 2.0
        assert "k1" in b._changes
        assert b._changes["k1"] == 2.0

    def test_setattr_new_attribute_base_class(self):
        # ModelBlock.__setattr__ silently drops non-item attrs after init
        b = ModelBlock()
        b.foo = "bar"
        assert "foo" not in b.__dict__  # actual base class behavior

    def test_setattr_new_attribute_subclass(self):
        # Subclasses have the else branch so non-item attrs work
        pb = ParameterBlock()
        pb.foo = "bar"
        assert pb.__dict__["foo"] == "bar"


# ---- ParameterBlock tests ----


class TestParameterBlock:
    def test_init(self):
        pb = ParameterBlock()
        assert pb.name == "parameters"
        assert len(pb) == 0

    def test_add_parameter(self):
        pb = ParameterBlock()
        pb.add_parameter("k1", 0.5)
        assert "k1" in pb.items
        assert isinstance(pb.items["k1"], Parameter)

    def test_setattr_with_parameter_object(self):
        pb = ParameterBlock()
        p = Parameter("k1", 0.5)
        pb.add_item(("k1", p))
        new_p = Parameter("k1", 1.5)
        pb.k1 = new_p
        assert pb.items["k1"] is new_p
        assert "k1" in pb._changes

    def test_setattr_with_string_expression(self):
        pb = ParameterBlock()
        p = Parameter("k1", 0.5)
        pb.add_item(("k1", p))
        pb.k1 = "2*k2"
        assert pb.items["k1"]["value"] == "2*k2"
        assert pb.items["k1"].write_expr is True
        assert "k1" in pb._changes

    def test_setattr_with_float(self):
        pb = ParameterBlock()
        p = Parameter("k1", 0.5)
        pb.add_item(("k1", p))
        pb.k1 = 3.14
        assert pb.items["k1"]["value"] == 3.14
        assert pb.items["k1"].write_expr is False
        assert "k1" in pb._changes

    def test_setattr_no_change_string(self):
        pb = ParameterBlock()
        p = Parameter("k1", "0.5")
        pb.add_item(("k1", p))
        pb._changes.clear()
        # Set to same value - no change recorded
        pb.k1 = "0.5"
        assert "k1" not in pb._changes

    def test_setattr_non_item_attribute(self):
        pb = ParameterBlock()
        pb.foo = "bar"
        assert pb.__dict__["foo"] == "bar"


# ---- CompartmentBlock tests ----


class TestCompartmentBlock:
    def test_init(self):
        cb = CompartmentBlock()
        assert cb.name == "compartments"

    def test_add_compartment(self):
        cb = CompartmentBlock()
        cb.add_compartment("EC", 3, 1.0)
        assert "EC" in cb.items
        assert isinstance(cb.items["EC"], Compartment)

    def test_setattr_with_compartment_object(self):
        cb = CompartmentBlock()
        c = Compartment("EC", 3, 1.0)
        cb.add_item(("EC", c))
        new_c = Compartment("EC", 3, 2.0)
        cb.EC = new_c
        assert cb.items["EC"] is new_c
        assert "EC" in cb._changes

    def test_setattr_with_float_updates_size(self):
        cb = CompartmentBlock()
        c = Compartment("EC", 3, 1.0)
        cb.add_item(("EC", c))
        cb.EC = 5.0
        assert cb.items["EC"]["size"] == 5.0
        assert "EC" in cb._changes

    def test_setattr_with_string_updates_name(self):
        cb = CompartmentBlock()
        c = Compartment("EC", 3, 1.0)
        cb.add_item(("EC", c))
        cb.EC = "newname"
        assert cb.items["EC"]["name"] == "newname"


# ---- ObservableBlock tests ----


class TestObservableBlock:
    def test_init(self):
        ob = ObservableBlock()
        assert ob.name == "observables"

    def test_add_observable(self):
        ob = ObservableBlock()
        fp = FakePattern("A()")
        ob.add_observable("obsA", "Molecules", [fp])
        assert "obsA" in ob.items
        assert isinstance(ob.items["obsA"], Observable)

    def test_setattr_with_observable_object(self):
        ob = ObservableBlock()
        fp = FakePattern("A()")
        o = Observable("obsA", "Molecules", [fp])
        ob.add_item(("obsA", o))
        new_o = Observable("obsA", "Species", [fp])
        ob.obsA = new_o
        assert ob.items["obsA"] is new_o
        assert "obsA" in ob._changes

    def test_setattr_with_string_updates_name(self):
        ob = ObservableBlock()
        fp = FakePattern("A()")
        o = Observable("obsA", "Molecules", [fp])
        ob.add_item(("obsA", o))
        ob.obsA = "obsB"
        assert ob.items["obsA"]["name"] == "obsB"

    def test_setattr_invalid_type_prints(self, capsys):
        ob = ObservableBlock()
        fp = FakePattern("A()")
        o = Observable("obsA", "Molecules", [fp])
        ob.add_item(("obsA", o))
        ob.obsA = 42
        captured = capsys.readouterr()
        assert "can't set observable" in captured.out


# ---- SpeciesBlock tests ----


class TestSpeciesBlock:
    def test_init(self):
        sb = SpeciesBlock()
        assert sb.name == "species"

    def test_add_species(self):
        sb = SpeciesBlock()
        fp = FakePattern("A()")
        sb.add_species(pattern=fp, count=100)
        assert len(sb) == 1
        # species are keyed by integer counter
        assert 0 in sb.items

    def test_getitem_uses_items_directly(self):
        sb = SpeciesBlock()
        fp = FakePattern("A()")
        s = Species(pattern=fp, count=100)
        sb.items[0] = s
        # SpeciesBlock.__getitem__ returns items[key] directly (no int->key-name logic)
        assert sb[0] is s

    def test_setitem(self):
        sb = SpeciesBlock()
        fp = FakePattern("A()")
        s = Species(pattern=fp, count=100)
        sb[0] = s
        assert sb.items[0] is s

    def test_setattr_with_species_object(self):
        sb = SpeciesBlock()
        fp = FakePattern("A()")
        s = Species(pattern=fp, count=100)
        sb.items["A()"] = s
        new_s = Species(pattern=fp, count=200)
        # Use attribute to update
        setattr(sb, "A()", new_s)
        assert sb.items["A()"] is new_s


# ---- MoleculeTypeBlock tests ----


class TestMoleculeTypeBlock:
    def test_init(self):
        mtb = MoleculeTypeBlock()
        assert mtb.name == "molecule types"

    def test_add_molecule_type(self):
        mtb = MoleculeTypeBlock()
        mtb.add_molecule_type("A", [])
        assert "A" in mtb.items
        assert isinstance(mtb.items["A"], MoleculeType)

    def test_setattr_with_molecule_type_object(self):
        mtb = MoleculeTypeBlock()
        mt = MoleculeType("A", [])
        mtb.add_item(("A", mt))
        new_mt = MoleculeType("A", ["b"])
        mtb.A = new_mt
        assert mtb.items["A"] is new_mt
        assert "A" in mtb._changes

    def test_setattr_with_string_updates_name(self):
        mtb = MoleculeTypeBlock()
        mt = MoleculeType("A", [])
        mtb.add_item(("A", mt))
        mtb.A = "B"
        assert mtb.items["A"]["name"] == "B"

    def test_setattr_invalid_type_prints(self, capsys):
        mtb = MoleculeTypeBlock()
        mt = MoleculeType("A", [])
        mtb.add_item(("A", mt))
        mtb.A = 42
        captured = capsys.readouterr()
        assert "can't set molecule type" in captured.out


# ---- FunctionBlock tests ----


class TestFunctionBlock:
    def test_init(self):
        fb = FunctionBlock()
        assert fb.name == "functions"

    def test_add_function(self):
        fb = FunctionBlock()
        fb.add_function("f1", "k1*A")
        assert "f1" in fb.items
        assert isinstance(fb.items["f1"], Function)

    def test_setattr_with_function_object(self):
        fb = FunctionBlock()
        f = Function("f1", "k1*A")
        fb.add_item(("f1", f))
        new_f = Function("f1", "k2*B")
        fb.f1 = new_f
        assert fb.items["f1"] is new_f
        assert "f1" in fb._changes

    def test_setattr_with_string_updates_expr(self):
        fb = FunctionBlock()
        f = Function("f1", "k1*A")
        fb.add_item(("f1", f))
        fb.f1 = "k2*B"
        assert fb.items["f1"]["expr"] == "k2*B"

    def test_setattr_invalid_type_prints(self, capsys):
        fb = FunctionBlock()
        f = Function("f1", "k1*A")
        fb.add_item(("f1", f))
        fb.f1 = 42
        captured = capsys.readouterr()
        assert "can't set function" in captured.out


# ---- RuleBlock tests ----


class TestRuleBlock:
    def test_init(self):
        rb = RuleBlock()
        assert rb.name == "reaction rules"

    def test_add_rule(self):
        rb = RuleBlock()
        fp_r = FakePattern("A()")
        fp_p = FakePattern("B()")
        rb.add_rule("r1", reactants=[fp_r], products=[fp_p], rate_constants=("k1",))
        assert "r1" in rb.items
        assert isinstance(rb.items["r1"], Rule)

    def test_setattr_with_rule_object(self):
        rb = RuleBlock()
        fp = FakePattern("A()")
        r = Rule("r1", reactants=[fp], products=[fp], rate_constants=("k1",))
        rb.add_item(("r1", r))
        new_r = Rule("r1", reactants=[fp], products=[fp], rate_constants=("k2",))
        rb.r1 = new_r
        assert rb.items["r1"] is new_r
        assert "r1" in rb._changes

    def test_setattr_with_string_updates_name(self):
        rb = RuleBlock()
        fp = FakePattern("A()")
        r = Rule("r1", reactants=[fp], products=[fp], rate_constants=("k1",))
        rb.add_item(("r1", r))
        rb.r1 = "r2"
        assert rb.items["r1"]["name"] == "r2"

    def test_consolidate_rules(self):
        rb = RuleBlock()
        fp_a = FakePattern("A()")
        fp_b = FakePattern("B()")
        # Forward rule
        r_fwd = Rule("bind", reactants=[fp_a], products=[fp_b], rate_constants=("kf",))
        rb.items["bind"] = r_fwd
        # Reverse rule (generated by XML loading)
        r_rev = Rule(
            "_reverse_bind",
            reactants=[fp_b],
            products=[fp_a],
            rate_constants=("kr",),
        )
        rb.items["_reverse_bind"] = r_rev
        rb.consolidate_rules()
        # Reverse should be removed
        assert "_reverse_bind" not in rb.items
        # Forward should now be bidirectional
        assert rb.items["bind"].bidirectional is True
        assert rb.items["bind"].rate_constants == ["kf", "kr"]

    def test_consolidate_rules_no_reverse(self):
        rb = RuleBlock()
        fp = FakePattern("A()")
        r = Rule("r1", reactants=[fp], products=[fp], rate_constants=("k1",))
        rb.items["r1"] = r
        # Should not error when no reverse rules exist
        rb.consolidate_rules()
        assert "r1" in rb.items
        assert rb.items["r1"].bidirectional is False

    def test_consolidate_rules_arrhenius(self):
        rb = RuleBlock()
        fp_a = FakePattern("A()")
        fp_b = FakePattern("B()")
        r_fwd = Rule("bind", reactants=[fp_a], products=[fp_b], rate_constants=("Arrhenius(1,2)",))
        rb.items["bind"] = r_fwd
        r_rev = Rule(
            "_reverse_bind",
            reactants=[fp_b],
            products=[fp_a],
            rate_constants=("kr",),
        )
        rb.items["_reverse_bind"] = r_rev
        rb.consolidate_rules()
        # Arrhenius rules keep only forward rate
        assert rb.items["bind"].rate_constants == ["Arrhenius(1,2)"]
        assert rb.items["bind"].bidirectional is False


# ---- ActionBlock tests ----


class TestActionBlock:
    def test_init(self):
        ab = ActionBlock()
        assert ab.name == "actions"
        assert isinstance(ab.items, list)
        assert len(ab.items) == 0

    def test_add_action(self):
        ab = ActionBlock()
        ab.add_action("simulate", {"method": '"ode"', "t_end": "100", "n_steps": "10"})
        assert len(ab.items) == 1

    def test_add_action_invalid_type(self, capsys):
        ab = ActionBlock()
        ab.add_action("not_a_real_action", {})
        captured = capsys.readouterr()
        assert "not recognized" in captured.out
        assert len(ab.items) == 0

    def test_clear_actions(self):
        ab = ActionBlock()
        ab.add_action("simulate", {"method": '"ode"', "t_end": "100", "n_steps": "10"})
        assert len(ab.items) == 1
        ab.clear_actions()
        assert len(ab.items) == 0

    def test_gen_string_no_begin_end(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        s = ab.gen_string()
        assert "begin" not in s
        assert "end" not in s
        assert "generate_network" in s

    def test_repr(self):
        ab = ActionBlock()
        r = repr(ab)
        assert "actions" in r
        assert "0 item(s)" in r

    def test_getitem(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        item = ab[0]
        assert item.type == "generate_network"

    def test_setitem(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        a = Action(action_type="generate_network", action_args={"overwrite": "0"})
        ab[0] = a
        assert ab.items[0] is a

    def test_delitem(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        del ab[0]
        assert len(ab.items) == 0

    def test_delitem_invalid(self, capsys):
        ab = ActionBlock()
        del ab[99]
        captured = capsys.readouterr()
        assert "99" in captured.out

    def test_iter(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        ab.add_action("simulate", {"method": '"ode"', "t_end": "100", "n_steps": "10"})
        indices = list(ab)
        assert indices == [0, 1]

    def test_contains_by_object(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        item = ab.items[0]
        assert item in ab

    def test_contains_by_name(self):
        ab = ActionBlock()
        ab.add_action("generate_network", {"overwrite": "1"})
        assert "generate_network" in ab

    def test_contains_missing(self):
        ab = ActionBlock()
        assert "simulate" not in ab

    def test_setattr_goes_to_dict(self):
        ab = ActionBlock()
        ab.foo = "bar"
        assert ab.__dict__["foo"] == "bar"


# ---- EnergyPatternBlock tests ----


class TestEnergyPatternBlock:
    def test_init(self):
        epb = EnergyPatternBlock()
        assert epb.name == "energy patterns"

    def test_add_energy_pattern(self):
        epb = EnergyPatternBlock()
        fp = FakePattern("A(b!1).B(a!1)")
        epb.add_energy_pattern("ep0", fp, "Eab")
        assert "ep0" in epb.items
        assert isinstance(epb.items["ep0"], EnergyPattern)

    def test_setattr_with_energy_pattern_object(self):
        epb = EnergyPatternBlock()
        fp = FakePattern("A(b!1).B(a!1)")
        ep = EnergyPattern("ep0", fp, "Eab")
        epb.add_item(("ep0", ep))
        new_ep = EnergyPattern("ep0", fp, "Ecd")
        epb.ep0 = new_ep
        assert epb.items["ep0"] is new_ep
        assert "ep0" in epb._changes

    def test_setattr_with_string_updates_name(self):
        epb = EnergyPatternBlock()
        fp = FakePattern("A(b!1).B(a!1)")
        ep = EnergyPattern("ep0", fp, "Eab")
        epb.add_item(("ep0", ep))
        epb.ep0 = "ep1"
        assert epb.items["ep0"]["name"] == "ep1"

    def test_setattr_invalid_type_prints(self, capsys):
        epb = EnergyPatternBlock()
        fp = FakePattern("A(b!1).B(a!1)")
        ep = EnergyPattern("ep0", fp, "Eab")
        epb.add_item(("ep0", ep))
        epb.ep0 = 42
        captured = capsys.readouterr()
        assert "can't set energy pattern" in captured.out


# ---- PopulationMapBlock tests ----


class TestPopulationMapBlock:
    def test_init(self):
        pmb = PopulationMapBlock()
        assert pmb.name == "population maps"

    def test_add_population_map(self):
        pmb = PopulationMapBlock()
        fp1 = FakePattern("A(b~0)")
        fp2 = FakePattern("Apop")
        pmb.add_population_map("pm0", fp1, fp2, "lump_rate")
        assert "pm0" in pmb.items
        assert isinstance(pmb.items["pm0"], PopulationMap)

    def test_setattr_with_population_map_object(self):
        pmb = PopulationMapBlock()
        fp1 = FakePattern("A(b~0)")
        fp2 = FakePattern("Apop")
        pm = PopulationMap("pm0", fp1, fp2, "lump_rate")
        pmb.add_item(("pm0", pm))
        new_pm = PopulationMap("pm0", fp1, fp2, "new_rate")
        pmb.pm0 = new_pm
        assert pmb.items["pm0"] is new_pm
        assert "pm0" in pmb._changes

    def test_setattr_with_string_updates_name(self):
        pmb = PopulationMapBlock()
        fp1 = FakePattern("A(b~0)")
        fp2 = FakePattern("Apop")
        pm = PopulationMap("pm0", fp1, fp2, "lump_rate")
        pmb.add_item(("pm0", pm))
        pmb.pm0 = "pm1"
        assert pmb.items["pm0"]["name"] == "pm1"

    def test_setattr_invalid_type_prints(self, capsys):
        pmb = PopulationMapBlock()
        fp1 = FakePattern("A(b~0)")
        fp2 = FakePattern("Apop")
        pm = PopulationMap("pm0", fp1, fp2, "lump_rate")
        pmb.add_item(("pm0", pm))
        pmb.pm0 = 42
        captured = capsys.readouterr()
        assert "can't set population map" in captured.out
