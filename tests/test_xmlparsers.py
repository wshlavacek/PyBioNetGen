"""Tests for bionetgen/modelapi/xmlparsers.py"""

from collections import OrderedDict

import pytest

from bionetgen.modelapi.blocks import (
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
from bionetgen.modelapi.pattern import Molecule, Pattern
from bionetgen.modelapi.xmlparsers import (
    BondsXML,
    CompartmentBlockXML,
    EnergyPatternBlockXML,
    FunctionBlockXML,
    MoleculeTypeBlockXML,
    ObservableBlockXML,
    ParameterBlockXML,
    PatternListXML,
    PatternXML,
    PopulationMapBlockXML,
    RuleBlockXML,
    SpeciesBlockXML,
    XMLObj,
)

# ---- Helpers to build XML-like OrderedDicts ----

def _simple_molecule_xml(name, comp_list=None, compartment=None, label=None):
    """Build a Molecule OrderedDict with optional components."""
    mol = OrderedDict([("@id", "M1"), ("@name", name)])
    if compartment:
        mol["@compartment"] = compartment
    if label:
        mol["@label"] = label
    if comp_list is not None:
        mol["ListOfComponents"] = OrderedDict([("Component", comp_list)])
    return mol


def _simple_component_xml(cid, name, num_bonds="0", state=None, label=None):
    c = OrderedDict([("@id", cid), ("@name", name), ("@numberOfBonds", num_bonds)])
    if state is not None:
        c["@state"] = state
    if label is not None:
        c["@label"] = label
    return c


def _simple_pattern_xml(molecules, bonds=None, compartment=None, label=None,
                         fixed=None, match_once=None, relation=None, quantity=None):
    """Build a pattern-level OrderedDict."""
    pat = OrderedDict()
    if compartment:
        pat["@compartment"] = compartment
    if label:
        pat["@label"] = label
    if fixed:
        pat["@Fixed"] = "1"
    if match_once:
        pat["@matchOnce"] = "1"
    if relation and quantity:
        pat["@relation"] = relation
        pat["@quantity"] = quantity
    if bonds is not None:
        pat["ListOfBonds"] = OrderedDict([("Bond", bonds)])
    mol_data = molecules
    pat["ListOfMolecules"] = OrderedDict([("Molecule", mol_data)])
    return pat


# ---- XMLObj base class ----

class TestXMLObj:
    def test_parse_xml_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            XMLObj(None)


# ---- BondsXML ----

class TestBondsXML:
    def test_no_bonds(self):
        b = BondsXML()
        assert b.bonds_dict == {}

    def test_single_bond(self):
        bond = OrderedDict([
            ("@id", "B1"),
            ("@site1", "O1_P1_M1_C1"),
            ("@site2", "O1_P1_M2_C1"),
        ])
        b = BondsXML(bond)
        assert ("O1", "P1", "M1", "C1") in b.bonds_dict
        assert ("O1", "P1", "M2", "C1") in b.bonds_dict
        assert b.bonds_dict[("O1", "P1", "M1", "C1")] == [1]
        assert b.bonds_dict[("O1", "P1", "M2", "C1")] == [1]

    def test_list_of_bonds(self):
        bonds = [
            OrderedDict([("@id", "B1"), ("@site1", "O1_P1_M1_C1"), ("@site2", "O1_P1_M2_C1")]),
            OrderedDict([("@id", "B2"), ("@site1", "O1_P1_M2_C2"), ("@site2", "O1_P1_M3_C1")]),
        ]
        b = BondsXML(bonds)
        assert b.bonds_dict[("O1", "P1", "M1", "C1")] == [1]
        assert b.bonds_dict[("O1", "P1", "M2", "C1")] == [1]
        assert b.bonds_dict[("O1", "P1", "M2", "C2")] == [2]
        assert b.bonds_dict[("O1", "P1", "M3", "C1")] == [2]

    def test_set_xml(self):
        b = BondsXML()
        assert b.bonds_dict == {}
        bond = OrderedDict([
            ("@id", "B1"),
            ("@site1", "O1_P1_M1_C1"),
            ("@site2", "O1_P1_M2_C1"),
        ])
        b.set_xml(bond)
        assert len(b.bonds_dict) == 2

    def test_get_tpl_from_id(self):
        b = BondsXML()
        tpl = b.get_tpl_from_id("O1_P1_M1_C2")
        assert tpl == ("O1", "P1", "M1", "C2")

    def test_tpls_from_bond(self):
        b = BondsXML()
        bond = OrderedDict([
            ("@id", "B1"),
            ("@site1", "O1_P1_M1_C1"),
            ("@site2", "O1_P1_M2_C1"),
        ])
        t1, t2 = b.tpls_from_bond(bond)
        assert t1 == ("O1", "P1", "M1", "C1")
        assert t2 == ("O1", "P1", "M2", "C1")

    def test_get_bond_id_numeric(self):
        bond = OrderedDict([
            ("@id", "B1"),
            ("@site1", "O1_P1_M1_C1"),
            ("@site2", "O1_P1_M2_C1"),
        ])
        b = BondsXML(bond)
        comp = OrderedDict([("@id", "O1_P1_M1_C1"), ("@numberOfBonds", "1")])
        result = b.get_bond_id(comp)
        assert result == [1]

    def test_get_bond_id_wildcard(self):
        b = BondsXML()
        comp = OrderedDict([("@id", "O1_P1_M1_C1"), ("@numberOfBonds", "+")])
        result = b.get_bond_id(comp)
        assert result == "+"

    def test_get_bond_id_question(self):
        b = BondsXML()
        comp = OrderedDict([("@id", "O1_P1_M1_C1"), ("@numberOfBonds", "?")])
        result = b.get_bond_id(comp)
        assert result == "?"


# ---- PatternXML ----

class TestPatternXML:
    def test_single_molecule_no_components(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        assert isinstance(px.parsed_obj, Pattern)
        assert len(px.parsed_obj.molecules) == 1
        assert px.parsed_obj.molecules[0].name == "A"

    def test_multiple_molecules(self):
        m1 = _simple_molecule_xml("A")
        m1["@id"] = "M1"
        m2 = _simple_molecule_xml("B")
        m2["@id"] = "M2"
        pat_xml = _simple_pattern_xml([m1, m2])
        px = PatternXML(pat_xml)
        assert len(px.parsed_obj.molecules) == 2
        names = [m.name for m in px.parsed_obj.molecules]
        assert "A" in names
        assert "B" in names

    def test_pattern_with_compartment(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol, compartment="cytoplasm")
        px = PatternXML(pat_xml)
        assert px.parsed_obj.compartment == "cytoplasm"

    def test_pattern_with_label(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol, label="p1")
        px = PatternXML(pat_xml)
        assert px.parsed_obj.label == "p1"

    def test_pattern_fixed(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol, fixed=True)
        px = PatternXML(pat_xml)
        assert px.parsed_obj.fixed is True

    def test_pattern_match_once(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol, match_once=True)
        px = PatternXML(pat_xml)
        assert px.parsed_obj.MatchOnce is True

    def test_pattern_relation_quantity(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol, relation="==", quantity="5")
        px = PatternXML(pat_xml)
        assert px.parsed_obj.relation == "=="
        assert px.parsed_obj.quantity == "5"

    def test_molecule_with_components(self):
        comp = _simple_component_xml("O1_P1_M1_C1", "x", state="active")
        mol = _simple_molecule_xml("A", comp_list=comp)
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        m = px.parsed_obj.molecules[0]
        assert len(m.components) == 1
        assert m.components[0].name == "x"
        assert m.components[0].state == "active"

    def test_molecule_with_multiple_components(self):
        comps = [
            _simple_component_xml("O1_P1_M1_C1", "x", state="u"),
            _simple_component_xml("O1_P1_M1_C2", "y"),
        ]
        mol = _simple_molecule_xml("A", comp_list=comps)
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        m = px.parsed_obj.molecules[0]
        assert len(m.components) == 2
        assert m.components[0].name == "x"
        assert m.components[0].state == "u"
        assert m.components[1].name == "y"

    def test_component_with_label(self):
        comp = _simple_component_xml("O1_P1_M1_C1", "x", label="c1")
        mol = _simple_molecule_xml("A", comp_list=comp)
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        assert px.parsed_obj.molecules[0].components[0].label == "c1"

    def test_molecule_compartment_and_label(self):
        mol = _simple_molecule_xml("A", compartment="EC", label="m1")
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        m = px.parsed_obj.molecules[0]
        assert m.compartment == "EC"
        assert m.label == "m1"

    def test_pattern_with_bonds(self):
        bond = OrderedDict([
            ("@id", "B1"),
            ("@site1", "O1_P1_M1_C1"),
            ("@site2", "O1_P1_M2_C1"),
        ])
        comp_a = _simple_component_xml("O1_P1_M1_C1", "x", num_bonds="1")
        comp_b = _simple_component_xml("O1_P1_M2_C1", "y", num_bonds="1")
        m1 = _simple_molecule_xml("A", comp_list=comp_a)
        m1["@id"] = "M1"
        m2 = _simple_molecule_xml("B", comp_list=comp_b)
        m2["@id"] = "M2"
        pat_xml = _simple_pattern_xml([m1, m2], bonds=bond)
        px = PatternXML(pat_xml)
        assert len(px.parsed_obj.molecules) == 2
        # Both components should have bond 1
        assert 1 in px.parsed_obj.molecules[0].components[0].bonds
        assert 1 in px.parsed_obj.molecules[1].components[0].bonds

    def test_str_and_repr(self):
        mol = _simple_molecule_xml("A")
        pat_xml = _simple_pattern_xml(mol)
        px = PatternXML(pat_xml)
        s = str(px)
        assert "A" in s
        assert repr(px) == str(px)


# ---- PatternListXML ----

class TestPatternListXML:
    def test_single_pattern(self):
        mol = _simple_molecule_xml("A")
        pat = _simple_pattern_xml(mol)
        xml = OrderedDict([("Pattern", pat)])
        pl = PatternListXML(xml)
        assert len(pl.patterns) == 1
        assert pl.patterns[0].molecules[0].name == "A"

    def test_multiple_patterns(self):
        mol_a = _simple_molecule_xml("A")
        mol_b = _simple_molecule_xml("B")
        pat_a = _simple_pattern_xml(mol_a)
        pat_b = _simple_pattern_xml(mol_b)
        xml = OrderedDict([("Pattern", [pat_a, pat_b])])
        pl = PatternListXML(xml)
        assert len(pl.patterns) == 2
        names = [p.molecules[0].name for p in pl.patterns]
        assert "A" in names
        assert "B" in names


# ---- ParameterBlockXML ----

class TestParameterBlockXML:
    def test_list_of_parameters(self):
        xml = [
            OrderedDict([("@id", "k1"), ("@type", "Constant"), ("@value", "0.1")]),
            OrderedDict([("@id", "k2"), ("@type", "Constant"), ("@value", "0.01")]),
        ]
        pb = ParameterBlockXML(xml)
        assert isinstance(pb.parsed_obj, ParameterBlock)
        assert "k1" in pb.parsed_obj.items
        assert "k2" in pb.parsed_obj.items

    def test_single_parameter(self):
        xml = OrderedDict([("@id", "kf"), ("@type", "Constant"), ("@value", "1.5")])
        pb = ParameterBlockXML(xml)
        assert "kf" in pb.parsed_obj.items

    def test_parameter_with_expr(self):
        xml = OrderedDict([
            ("@id", "kf"),
            ("@type", "Constant"),
            ("@value", "0.5"),
            ("@expr", "k1*k2"),
        ])
        pb = ParameterBlockXML(xml)
        # expr should override value
        param_str = str(pb.parsed_obj.items["kf"])
        assert "k1*k2" in param_str

    def test_str_repr(self):
        xml = [OrderedDict([("@id", "k1"), ("@type", "Constant"), ("@value", "0.1")])]
        pb = ParameterBlockXML(xml)
        s = str(pb)
        assert "k1" in s


# ---- CompartmentBlockXML ----

class TestCompartmentBlockXML:
    def test_single_compartment(self):
        xml = OrderedDict([
            ("@id", "EC"),
            ("@spatialDimensions", "3"),
            ("@size", "1e-6"),
        ])
        cb = CompartmentBlockXML(xml)
        assert isinstance(cb.parsed_obj, CompartmentBlock)
        assert "EC" in cb.parsed_obj.items

    def test_list_of_compartments(self):
        xml = [
            OrderedDict([("@id", "EC"), ("@spatialDimensions", "3"), ("@size", "1e-6")]),
            OrderedDict([("@id", "PM"), ("@spatialDimensions", "2"), ("@size", "1e-8"), ("@outside", "EC")]),
        ]
        cb = CompartmentBlockXML(xml)
        assert "EC" in cb.parsed_obj.items
        assert "PM" in cb.parsed_obj.items

    def test_compartment_with_outside(self):
        xml = OrderedDict([
            ("@id", "CP"),
            ("@spatialDimensions", "3"),
            ("@size", "1e-12"),
            ("@outside", "PM"),
        ])
        cb = CompartmentBlockXML(xml)
        assert "CP" in cb.parsed_obj.items


# ---- ObservableBlockXML ----

class TestObservableBlockXML:
    def _obs_xml(self, name, otype, mol_name):
        mol = _simple_molecule_xml(mol_name)
        pat = _simple_pattern_xml(mol)
        return OrderedDict([
            ("@name", name),
            ("@type", otype),
            ("ListOfPatterns", OrderedDict([("Pattern", pat)])),
        ])

    def test_single_observable(self):
        xml = self._obs_xml("Atot", "Molecules", "A")
        ob = ObservableBlockXML(xml)
        assert isinstance(ob.parsed_obj, ObservableBlock)
        assert "Atot" in ob.parsed_obj.items

    def test_list_of_observables(self):
        xml = [
            self._obs_xml("Atot", "Molecules", "A"),
            self._obs_xml("Btot", "Species", "B"),
        ]
        ob = ObservableBlockXML(xml)
        assert "Atot" in ob.parsed_obj.items
        assert "Btot" in ob.parsed_obj.items

    def test_observable_with_multiple_patterns(self):
        mol_a = _simple_molecule_xml("A")
        mol_b = _simple_molecule_xml("B")
        pat_a = _simple_pattern_xml(mol_a)
        pat_b = _simple_pattern_xml(mol_b)
        xml = OrderedDict([
            ("@name", "ABtot"),
            ("@type", "Molecules"),
            ("ListOfPatterns", OrderedDict([("Pattern", [pat_a, pat_b])])),
        ])
        ob = ObservableBlockXML(xml)
        assert "ABtot" in ob.parsed_obj.items


# ---- SpeciesBlockXML ----

class TestSpeciesBlockXML:
    def test_single_species(self):
        mol = _simple_molecule_xml("A")
        xml = _simple_pattern_xml(mol)
        xml["@concentration"] = "100"
        sb = SpeciesBlockXML(xml)
        assert isinstance(sb.parsed_obj, SpeciesBlock)
        # SpeciesBlock uses an integer counter as key
        assert len(sb.parsed_obj.items) == 1

    def test_list_of_species(self):
        mol_a = _simple_molecule_xml("A")
        xml_a = _simple_pattern_xml(mol_a)
        xml_a["@concentration"] = "100"
        mol_b = _simple_molecule_xml("B")
        xml_b = _simple_pattern_xml(mol_b)
        xml_b["@concentration"] = "200"
        sb = SpeciesBlockXML([xml_a, xml_b])
        assert len(sb.parsed_obj.items) == 2


# ---- MoleculeTypeBlockXML ----

class TestMoleculeTypeBlockXML:
    def test_simple_molecule_type(self):
        xml = OrderedDict([("@id", "A")])
        mt = MoleculeTypeBlockXML(xml)
        assert isinstance(mt.parsed_obj, MoleculeTypeBlock)
        assert "A" in mt.parsed_obj.items

    def test_molecule_type_with_single_component(self):
        xml = OrderedDict([
            ("@id", "A"),
            ("ListOfComponentTypes", OrderedDict([
                ("ComponentType", OrderedDict([("@id", "x")])),
            ])),
        ])
        mt = MoleculeTypeBlockXML(xml)
        assert "A" in mt.parsed_obj.items

    def test_molecule_type_with_component_states(self):
        xml = OrderedDict([
            ("@id", "A"),
            ("ListOfComponentTypes", OrderedDict([
                ("ComponentType", OrderedDict([
                    ("@id", "x"),
                    ("ListOfAllowedStates", OrderedDict([
                        ("AllowedState", [
                            OrderedDict([("@id", "u")]),
                            OrderedDict([("@id", "p")]),
                        ]),
                    ])),
                ])),
            ])),
        ])
        mt = MoleculeTypeBlockXML(xml)
        assert "A" in mt.parsed_obj.items

    def test_molecule_type_single_state(self):
        xml = OrderedDict([
            ("@id", "A"),
            ("ListOfComponentTypes", OrderedDict([
                ("ComponentType", OrderedDict([
                    ("@id", "x"),
                    ("ListOfAllowedStates", OrderedDict([
                        ("AllowedState", OrderedDict([("@id", "active")])),
                    ])),
                ])),
            ])),
        ])
        mt = MoleculeTypeBlockXML(xml)
        assert "A" in mt.parsed_obj.items

    def test_molecule_type_multiple_components(self):
        xml = OrderedDict([
            ("@id", "R"),
            ("ListOfComponentTypes", OrderedDict([
                ("ComponentType", [
                    OrderedDict([("@id", "x")]),
                    OrderedDict([("@id", "y")]),
                ]),
            ])),
        ])
        mt = MoleculeTypeBlockXML(xml)
        assert "R" in mt.parsed_obj.items

    def test_list_of_molecule_types(self):
        xml = [
            OrderedDict([("@id", "A")]),
            OrderedDict([("@id", "B")]),
        ]
        mt = MoleculeTypeBlockXML(xml)
        assert "A" in mt.parsed_obj.items
        assert "B" in mt.parsed_obj.items

    def test_multiple_components_with_states(self):
        xml = OrderedDict([
            ("@id", "R"),
            ("ListOfComponentTypes", OrderedDict([
                ("ComponentType", [
                    OrderedDict([
                        ("@id", "x"),
                        ("ListOfAllowedStates", OrderedDict([
                            ("AllowedState", OrderedDict([("@id", "on")])),
                        ])),
                    ]),
                    OrderedDict([("@id", "y")]),
                ]),
            ])),
        ])
        mt = MoleculeTypeBlockXML(xml)
        assert "R" in mt.parsed_obj.items


# ---- FunctionBlockXML ----

class TestFunctionBlockXML:
    def test_single_function_no_args(self):
        xml = OrderedDict([("@id", "f1"), ("Expression", "k1*A")])
        fb = FunctionBlockXML(xml)
        assert isinstance(fb.parsed_obj, FunctionBlock)
        assert "f1" in fb.parsed_obj.items

    def test_list_of_functions(self):
        xml = [
            OrderedDict([("@id", "f1"), ("Expression", "k1*A")]),
            OrderedDict([("@id", "f2"), ("Expression", "k2*B")]),
        ]
        fb = FunctionBlockXML(xml)
        assert "f1" in fb.parsed_obj.items
        assert "f2" in fb.parsed_obj.items

    def test_function_with_single_argument(self):
        xml = OrderedDict([
            ("@id", "f1"),
            ("Expression", "k1*x"),
            ("ListOfArguments", OrderedDict([
                ("Argument", OrderedDict([("@id", "x")])),
            ])),
        ])
        fb = FunctionBlockXML(xml)
        assert "f1" in fb.parsed_obj.items

    def test_function_with_multiple_arguments(self):
        xml = OrderedDict([
            ("@id", "f1"),
            ("Expression", "k1*x + k2*y"),
            ("ListOfArguments", OrderedDict([
                ("Argument", [
                    OrderedDict([("@id", "x")]),
                    OrderedDict([("@id", "y")]),
                ]),
            ])),
        ])
        fb = FunctionBlockXML(xml)
        assert "f1" in fb.parsed_obj.items

    def test_get_arguments_single(self):
        fb_xml = OrderedDict([("@id", "f1"), ("Expression", "x")])
        fb = FunctionBlockXML(fb_xml)
        args = fb.get_arguments(OrderedDict([("@id", "x")]))
        assert args == ["x"]

    def test_get_arguments_multiple(self):
        fb_xml = OrderedDict([("@id", "f1"), ("Expression", "x")])
        fb = FunctionBlockXML(fb_xml)
        args = fb.get_arguments([
            OrderedDict([("@id", "x")]),
            OrderedDict([("@id", "y")]),
        ])
        assert args == ["x", "y"]


# ---- RuleBlockXML ----

def _make_rule_xml(name, reactant_mol_name, product_mol_name, rate_value,
                    rate_type="Ele", operations=None):
    """Build a rule XML dict.

    Note: ListOfOperations must be a non-empty OrderedDict containing at least
    one recognized operation key (e.g. "StateChange") for the list-of-rules
    code path to work, because the source does not initialize ``operations``
    before the conditional block that populates it.  For the single-rule path,
    ``ListOfOperations`` must not be ``None`` because ``get_operations`` is
    called unconditionally.  Provide a dict with a recognized key to exercise
    both paths safely.
    """
    react_mol = _simple_molecule_xml(reactant_mol_name)
    react_pat = _simple_pattern_xml(react_mol)
    prod_mol = _simple_molecule_xml(product_mol_name)
    prod_pat = _simple_pattern_xml(prod_mol)

    rate_law = OrderedDict([
        ("@id", "RL1"),
        ("@type", rate_type),
    ])
    if rate_type == "Ele":
        rate_law["ListOfRateConstants"] = OrderedDict([
            ("RateConstant", OrderedDict([("@value", rate_value)])),
        ])
    elif rate_type == "Function":
        rate_law["@name"] = rate_value
        rate_law["@totalrate"] = 0

    if operations is None:
        # Provide a minimal valid operations dict so both code paths work.
        operations = OrderedDict([
            ("StateChange", OrderedDict([
                ("@id", "O1"),
                ("@site", "O1_P1_M1_C1"),
                ("@finalState", "active"),
            ])),
        ])

    rule = OrderedDict([
        ("@id", "R1"),
        ("@name", name),
        ("ListOfReactantPatterns", OrderedDict([("ReactantPattern", react_pat)])),
        ("ListOfProductPatterns", OrderedDict([("ProductPattern", prod_pat)])),
        ("RateLaw", rate_law),
        ("ListOfOperations", operations),
    ])
    return rule


class TestRuleBlockXML:
    def test_single_rule_ele(self):
        xml = _make_rule_xml("r1", "A", "B", "0.1")
        rb = RuleBlockXML(xml)
        assert isinstance(rb.parsed_obj, RuleBlock)
        assert "r1" in rb.parsed_obj.items

    def test_list_of_rules(self):
        r1 = _make_rule_xml("r1", "A", "B", "0.1")
        r2 = _make_rule_xml("r2", "B", "C", "0.2")
        rb = RuleBlockXML([r1, r2])
        assert "r1" in rb.parsed_obj.items
        assert "r2" in rb.parsed_obj.items

    def test_rule_function_rate(self):
        xml = _make_rule_xml("r1", "A", "B", "myFunc", rate_type="Function")
        rb = RuleBlockXML(xml)
        assert "r1" in rb.parsed_obj.items
        assert rb.parsed_obj.items["r1"].rate_constants[0] == "myFunc"

    def test_resolve_ratelaw_ele(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "Ele"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", OrderedDict([("@value", "0.5")])),
            ])),
        ])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "0.5"

    def test_resolve_ratelaw_function(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([("@type", "Function"), ("@name", "myFunc")])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "myFunc"

    def test_resolve_ratelaw_mm(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "MM"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", [
                    OrderedDict([("@value", "kcat")]),
                    OrderedDict([("@value", "Km")]),
                ]),
            ])),
        ])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "MM(kcat,Km)"

    def test_resolve_ratelaw_mm_single(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "Sat"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", OrderedDict([("@value", "k1")])),
            ])),
        ])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "Sat(k1)"

    def test_resolve_rxn_side_none(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        result = rb.resolve_rxn_side(None)
        assert len(result) == 1
        assert isinstance(result[0], Molecule)

    def test_resolve_rxn_side_reactants(self):
        mol = _simple_molecule_xml("A")
        pat = _simple_pattern_xml(mol)
        xml_side = OrderedDict([("ReactantPattern", pat)])
        dummy = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(dummy)
        result = rb.resolve_rxn_side(xml_side)
        assert len(result) == 1
        assert result[0].molecules[0].name == "A"

    def test_resolve_rxn_side_products(self):
        mol = _simple_molecule_xml("B")
        pat = _simple_pattern_xml(mol)
        xml_side = OrderedDict([("ProductPattern", pat)])
        dummy = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(dummy)
        result = rb.resolve_rxn_side(xml_side)
        assert len(result) == 1
        assert result[0].molecules[0].name == "B"

    def test_resolve_rxn_side_multiple_reactants(self):
        mol_a = _simple_molecule_xml("A")
        mol_b = _simple_molecule_xml("B")
        pat_a = _simple_pattern_xml(mol_a)
        pat_b = _simple_pattern_xml(mol_b)
        xml_side = OrderedDict([("ReactantPattern", [pat_a, pat_b])])
        dummy = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(dummy)
        result = rb.resolve_rxn_side(xml_side)
        assert len(result) == 2

    def test_consolidate_reverse_rules(self):
        r1 = _make_rule_xml("bind", "A", "B", "0.1")
        r2 = _make_rule_xml("_reverse_bind", "B", "A", "0.05")
        rb = RuleBlockXML([r1, r2])
        # After consolidation, the reverse rule should be merged
        assert "bind" in rb.parsed_obj.items
        assert "_reverse_bind" not in rb.parsed_obj.items
        assert rb.parsed_obj.items["bind"].bidirectional is True

    def test_get_rule_mod_none_ops(self):
        """get_rule_mod returns None when ListOfOperations is None."""
        xml = _make_rule_xml("r1", "A", "B", "0.1")
        rb = RuleBlockXML(xml)
        rule_xml = OrderedDict([
            ("@name", "r1"),
            ("ListOfOperations", None),
        ])
        result = rb.get_rule_mod(rule_xml)
        assert result is None

    def test_get_rule_mod_delete_molecules(self):
        xml = _make_rule_xml("r1", "A", "B", "0.1")
        rb = RuleBlockXML(xml)
        rule_xml = OrderedDict([
            ("@name", "r1"),
            ("ListOfOperations", OrderedDict([
                ("Delete", OrderedDict([
                    ("@id", "O1"),
                    ("@DeleteMolecules", 1),
                ])),
            ])),
            ("ListOfReactantPatterns", None),
            ("ListOfProductPatterns", None),
        ])
        result = rb.get_rule_mod(rule_xml)
        assert result.type == "DeleteMolecules"

    def test_get_operations_empty(self):
        xml = _make_rule_xml("r1", "A", "B", "0.1")
        rb = RuleBlockXML(xml)
        ops = rb.get_operations(OrderedDict())
        assert ops == []

    def test_resolve_ratelaw_hill(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "Hill"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", [
                    OrderedDict([("@value", "Vmax")]),
                    OrderedDict([("@value", "Kd")]),
                    OrderedDict([("@value", "n")]),
                ]),
            ])),
        ])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "Hill(Vmax,Kd,n)"

    def test_resolve_ratelaw_arrhenius(self):
        xml = _make_rule_xml("r1", "A", "B", "0.5")
        rb = RuleBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "Arrhenius"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", [
                    OrderedDict([("@value", "phi")]),
                    OrderedDict([("@value", "Ea")]),
                ]),
            ])),
        ])
        result = rb.resolve_ratelaw(rate_xml)
        assert result == "Arrhenius(phi,Ea)"


# ---- EnergyPatternBlockXML ----

class TestEnergyPatternBlockXML:
    def _ep_xml(self, epid, mol_name, expr):
        mol = _simple_molecule_xml(mol_name)
        pat = _simple_pattern_xml(mol)
        return OrderedDict([
            ("@id", epid),
            ("@expression", expr),
            ("Pattern", pat),
        ])

    def test_single_energy_pattern(self):
        xml = self._ep_xml("ep1", "A", "Gf_A")
        ep = EnergyPatternBlockXML(xml)
        assert isinstance(ep.parsed_obj, EnergyPatternBlock)
        assert "ep1" in ep.parsed_obj.items

    def test_list_of_energy_patterns(self):
        xml = [
            self._ep_xml("ep1", "A", "Gf_A"),
            self._ep_xml("ep2", "B", "Gf_B"),
        ]
        ep = EnergyPatternBlockXML(xml)
        assert "ep1" in ep.parsed_obj.items
        assert "ep2" in ep.parsed_obj.items


# ---- PopulationMapBlockXML ----

class TestPopulationMapBlockXML:
    def _pm_xml(self, pmid, struct_name, pop_name, rate_value):
        struct_mol = _simple_molecule_xml(struct_name)
        struct_pat = _simple_pattern_xml(struct_mol)
        pop_mol = _simple_molecule_xml(pop_name)
        pop_pat = _simple_pattern_xml(pop_mol)
        return OrderedDict([
            ("@id", pmid),
            ("StructuredSpecies", OrderedDict([("Species", struct_pat)])),
            ("PopulationSpecies", OrderedDict([("Species", pop_pat)])),
            ("RateLaw", OrderedDict([
                ("@id", "RL1"),
                ("@type", "Ele"),
                ("ListOfRateConstants", OrderedDict([
                    ("RateConstant", OrderedDict([("@value", rate_value)])),
                ])),
            ])),
        ])

    def test_single_population_map(self):
        xml = self._pm_xml("pm1", "A", "Apop", "0.5")
        pm = PopulationMapBlockXML(xml)
        assert isinstance(pm.parsed_obj, PopulationMapBlock)
        assert "pm1" in pm.parsed_obj.items

    def test_list_of_population_maps(self):
        xml = [
            self._pm_xml("pm1", "A", "Apop", "0.5"),
            self._pm_xml("pm2", "B", "Bpop", "0.3"),
        ]
        pm = PopulationMapBlockXML(xml)
        assert "pm1" in pm.parsed_obj.items
        assert "pm2" in pm.parsed_obj.items

    def test_population_map_function_rate(self):
        xml = self._pm_xml("pm1", "A", "Apop", "0.5")
        # override rate law to Function type
        xml["RateLaw"] = OrderedDict([
            ("@id", "RL1"),
            ("@type", "Function"),
            ("@name", "lumpFunc"),
        ])
        pm = PopulationMapBlockXML(xml)
        assert "pm1" in pm.parsed_obj.items

    def test_resolve_ratelaw_ele(self):
        xml = self._pm_xml("pm1", "A", "Apop", "0.5")
        pm = PopulationMapBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "Ele"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", OrderedDict([("@value", "1.0")])),
            ])),
        ])
        result = pm.resolve_ratelaw(rate_xml)
        assert result == "1.0"

    def test_resolve_ratelaw_function(self):
        xml = self._pm_xml("pm1", "A", "Apop", "0.5")
        pm = PopulationMapBlockXML(xml)
        rate_xml = OrderedDict([("@type", "Function"), ("@name", "myFunc")])
        result = pm.resolve_ratelaw(rate_xml)
        assert result == "myFunc"

    def test_resolve_ratelaw_mm(self):
        xml = self._pm_xml("pm1", "A", "Apop", "0.5")
        pm = PopulationMapBlockXML(xml)
        rate_xml = OrderedDict([
            ("@type", "MM"),
            ("ListOfRateConstants", OrderedDict([
                ("RateConstant", [
                    OrderedDict([("@value", "kcat")]),
                    OrderedDict([("@value", "Km")]),
                ]),
            ])),
        ])
        result = pm.resolve_ratelaw(rate_xml)
        assert result == "MM(kcat,Km)"
