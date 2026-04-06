"""Tests for bionetgen/modelapi/pattern.py"""

import sys
import types

import pytest

from bionetgen.modelapi.pattern import Component, Molecule, Pattern

# ── Helper factories ──────────────────────────────────────────────


def make_component(name, state=None, states=None, bonds=None, label=None):
    """Create a Component with the given attributes."""
    c = Component()
    c.name = name
    if state is not None:
        c.state = state
    if states is not None:
        c.states = states
    if bonds is not None:
        c.bonds = bonds
    if label is not None:
        c.label = label
    return c


def make_molecule(name, components=None, compartment=None, label=None):
    """Create a Molecule with the given attributes."""
    comps = components if components is not None else []
    return Molecule(name=name, components=comps, compartment=compartment, label=label)


def make_pattern(molecules=None, compartment=None, label=None):
    """Create a Pattern with the given attributes."""
    mols = molecules if molecules is not None else []
    return Pattern(molecules=mols, compartment=compartment, label=label)


# ══════════════════════════════════════════════════════════════════
# Component tests
# ══════════════════════════════════════════════════════════════════


class TestComponent:
    def test_str_name_only(self):
        c = make_component("b")
        assert str(c) == "b"

    def test_str_with_state(self):
        c = make_component("b", state="0")
        assert str(c) == "b~0"

    def test_str_with_multiple_states(self):
        """Molecule-type component listing all possible states."""
        c = make_component("b", states=["0", "1"])
        assert str(c) == "b~0~1"

    def test_str_with_states_and_current_state(self):
        """Both states list and current state are printed."""
        c = make_component("b", states=["0", "1"], state="0")
        assert str(c) == "b~0~1~0"

    def test_str_with_single_bond(self):
        c = make_component("b", bonds=["1"])
        assert str(c) == "b!1"

    def test_str_with_multiple_bonds(self):
        c = make_component("b", bonds=["1", "2"])
        assert str(c) == "b!1!2"

    def test_str_wildcard_bond_plus(self):
        c = make_component("b", bonds=["+"])
        assert str(c) == "b!+"

    def test_str_wildcard_bond_question(self):
        c = make_component("b", bonds=["?"])
        assert str(c) == "b!?"

    def test_str_with_label(self):
        c = make_component("b", label="x")
        assert str(c) == "b%x"

    def test_str_state_label_bond(self):
        c = make_component("b", state="0", label="x", bonds=["1"])
        assert str(c) == "b~0%x!1"

    def test_repr_matches_str(self):
        c = make_component("b", state="0")
        assert repr(c) == str(c)

    # ── Properties ────────────────────────────────────────────

    def test_name_property(self):
        c = Component()
        c.name = "site"
        assert c.name == "site"

    def test_label_property(self):
        c = Component()
        assert c.label is None
        c.label = "lbl"
        assert c.label == "lbl"

    def test_state_property(self):
        c = Component()
        assert c.state is None
        c.state = "active"
        assert c.state == "active"

    def test_states_property(self):
        c = Component()
        assert c.states == []
        c.states = ["0", "1", "2"]
        assert c.states == ["0", "1", "2"]

    def test_bonds_property(self):
        c = Component()
        assert c.bonds == []
        c.bonds = ["1", "2"]
        assert c.bonds == ["1", "2"]

    # ── Equality ──────────────────────────────────────────────

    def test_equality_same(self):
        a = make_component("b", state="0", bonds=["1"])
        b = make_component("b", state="0", bonds=["2"])
        assert a == b  # same name, state, label, len(bonds)

    def test_equality_different_name(self):
        a = make_component("b", state="0")
        b = make_component("c", state="0")
        assert a != b

    def test_equality_different_state(self):
        a = make_component("b", state="0")
        b = make_component("b", state="1")
        assert a != b

    def test_equality_different_bond_count(self):
        a = make_component("b", bonds=["1"])
        b = make_component("b", bonds=["1", "2"])
        assert a != b

    def test_equality_different_type(self):
        a = make_component("b")
        assert a != "b"

    # ── add_state / add_bond raise NotImplementedError ────────

    def test_add_state_not_implemented(self):
        c = Component()
        with pytest.raises(NotImplementedError):
            c.add_state()

    def test_add_bond_not_implemented(self):
        c = Component()
        with pytest.raises(NotImplementedError):
            c.add_bond()

    # ── Default init values ───────────────────────────────────

    def test_default_init(self):
        c = Component()
        assert c.name == ""
        assert c.label is None
        assert c.state is None
        assert c.states == []
        assert c.bonds == []
        assert c.canonical_label is None
        assert c.canonical_order is None
        assert c.canonical_bonds is None
        assert c.parent_molecule is None


# ══════════════════════════════════════════════════════════════════
# Molecule tests
# ══════════════════════════════════════════════════════════════════


class TestMolecule:
    # ── __str__ ───────────────────────────────────────────────

    def test_str_no_components(self):
        m = make_molecule("A")
        assert str(m) == "A()"

    def test_str_single_component(self):
        m = make_molecule("A", [make_component("b", state="0")])
        assert str(m) == "A(b~0)"

    def test_str_multiple_components(self):
        m = make_molecule("A", [make_component("b"), make_component("c")])
        assert str(m) == "A(b,c)"

    def test_str_with_compartment(self):
        m = make_molecule("A", compartment="EC")
        assert str(m) == "A()@EC"

    def test_str_with_label(self):
        m = make_molecule("A", label="x")
        assert str(m) == "A()%x"

    def test_str_with_compartment_and_label(self):
        m = make_molecule("A", compartment="EC", label="x")
        assert str(m) == "A()@EC%x"

    def test_str_null_species(self):
        """Null species '0' omits parentheses."""
        m = make_molecule("0")
        assert str(m) == "0"

    def test_repr_matches_str(self):
        m = make_molecule("A", [make_component("b")])
        assert repr(m) == str(m)

    # ── Container protocol ────────────────────────────────────

    def test_getitem(self):
        c0 = make_component("b")
        c1 = make_component("c")
        m = make_molecule("A", [c0, c1])
        assert m[0] is c0
        assert m[1] is c1

    def test_iter(self):
        comps = [make_component("b"), make_component("c")]
        m = make_molecule("A", comps)
        assert list(m) == comps

    def test_contains(self):
        c = make_component("b")
        m = make_molecule("A", [c])
        assert c in m

    def test_contains_missing(self):
        c = make_component("b")
        m = make_molecule("A", [])
        assert c not in m

    # ── Properties ────────────────────────────────────────────

    def test_name_property(self):
        m = make_molecule("A")
        m.name = "B"
        assert m.name == "B"

    def test_compartment_property(self):
        m = make_molecule("A")
        assert m.compartment is None
        m.compartment = "EC"
        assert m.compartment == "EC"

    def test_label_property(self):
        m = make_molecule("A")
        assert m.label is None
        m.label = "x"
        assert m.label == "x"

    def test_components_property(self):
        m = make_molecule("A")
        new_comps = [make_component("b")]
        m.components = new_comps
        assert m.components is new_comps

    # ── add_component ─────────────────────────────────────────

    def test_add_component(self):
        m = make_molecule("A", [])
        m.add_component("b", state="0")
        assert len(m.components) == 1
        assert m.components[0].name == "b"
        assert m.components[0].state == "0"

    def test_add_component_with_states(self):
        m = make_molecule("A", [])
        m.add_component("b", states=["0", "1"])
        assert m.components[0].states == ["0", "1"]

    # ── Equality ──────────────────────────────────────────────

    def test_equality_same(self):
        c1 = make_component("b", state="0")
        c2 = make_component("b", state="0")
        a = make_molecule("A", [c1])
        b = make_molecule("A", [c2])
        assert a == b

    def test_equality_different_name(self):
        a = make_molecule("A")
        b = make_molecule("B")
        assert a != b

    def test_equality_different_type(self):
        a = make_molecule("A")
        assert a != "A"

    def test_equality_different_compartment(self):
        a = make_molecule("A", compartment="EC")
        b = make_molecule("A", compartment="CP")
        assert a != b

    # ── Default init values ───────────────────────────────────

    def test_default_init(self):
        m = Molecule()
        assert m.name == "0"
        assert m.compartment is None
        assert m.label is None
        assert m.canonical_order is None
        assert m.canonical_label is None
        assert m.parent_pattern is None


# ══════════════════════════════════════════════════════════════════
# Pattern tests
# ══════════════════════════════════════════════════════════════════


class TestPattern:
    # ── __str__ ───────────────────────────────────────────────

    def test_str_single_molecule(self):
        p = make_pattern([make_molecule("A", [make_component("b", state="0")])])
        assert str(p) == "A(b~0)"

    def test_str_multiple_molecules(self):
        m1 = make_molecule("A", [make_component("b")])
        m2 = make_molecule("B", [make_component("a")])
        p = make_pattern([m1, m2])
        assert str(p) == "A(b).B(a)"

    def test_str_with_compartment(self):
        p = make_pattern([make_molecule("A")], compartment="EC")
        assert str(p) == "@EC:A()"

    def test_str_with_label(self):
        p = make_pattern([make_molecule("A")], label="x")
        assert str(p) == "%x:A()"

    def test_str_with_compartment_and_label(self):
        p = make_pattern([make_molecule("A")], compartment="EC", label="x")
        assert str(p) == "@EC%x:A()"

    def test_str_fixed(self):
        p = make_pattern([make_molecule("A")])
        p.fixed = True
        assert str(p) == "$A()"

    def test_str_matchonce(self):
        p = make_pattern([make_molecule("A")])
        p.MatchOnce = True
        assert str(p) == "{MatchOnce}A()"

    def test_str_fixed_and_matchonce(self):
        p = make_pattern([make_molecule("A")])
        p.fixed = True
        p.MatchOnce = True
        assert str(p) == "${MatchOnce}A()"

    def test_str_relation_quantity(self):
        p = make_pattern([make_molecule("A")])
        p.relation = "=="
        p.quantity = "5"
        assert str(p) == "A()==5"

    def test_str_relation_leq(self):
        p = make_pattern([make_molecule("A")])
        p.relation = "<="
        p.quantity = "3"
        assert str(p) == "A()<=3"

    def test_str_compartment_with_fixed(self):
        p = make_pattern([make_molecule("A")], compartment="EC")
        p.fixed = True
        assert str(p) == "@EC:$A()"

    def test_str_empty_pattern(self):
        p = make_pattern([])
        assert str(p) == ""

    def test_repr_matches_str(self):
        p = make_pattern([make_molecule("A")])
        assert repr(p) == str(p)

    # ── Bonding in __str__ ────────────────────────────────────

    def test_str_bonded_molecules(self):
        """A(b!1).B(a!1) — two molecules sharing a bond."""
        c_ab = make_component("b", bonds=["1"])
        c_ba = make_component("a", bonds=["1"])
        m1 = make_molecule("A", [c_ab])
        m2 = make_molecule("B", [c_ba])
        p = make_pattern([m1, m2])
        assert str(p) == "A(b!1).B(a!1)"

    # ── Container protocol ────────────────────────────────────

    def test_len(self):
        p = make_pattern([make_molecule("A"), make_molecule("B")])
        # Pattern doesn't define __len__ directly but we test __getitem__/__iter__
        assert len(list(p)) == 2

    def test_getitem(self):
        m0 = make_molecule("A")
        m1 = make_molecule("B")
        p = make_pattern([m0, m1])
        assert p[0] is m0
        assert p[1] is m1

    def test_iter(self):
        mols = [make_molecule("A"), make_molecule("B")]
        p = make_pattern(mols)
        assert list(p) == mols

    def test_contains(self):
        m = make_molecule("A")
        p = make_pattern([m])
        assert m in p

    def test_contains_missing(self):
        m = make_molecule("A")
        p = make_pattern([])
        assert m not in p

    # ── Properties ────────────────────────────────────────────

    def test_compartment_property(self):
        p = make_pattern()
        assert p.compartment is None
        p.compartment = "EC"
        assert p.compartment == "EC"

    def test_label_property(self):
        p = make_pattern()
        assert p.label is None
        p.label = "x"
        assert p.label == "x"

    # ── Default init values ───────────────────────────────────

    def test_default_init(self):
        p = Pattern()
        assert p.molecules == []
        assert p.compartment is None
        assert p.label is None
        assert p.fixed is False
        assert p.MatchOnce is False
        assert p.relation is None
        assert p.quantity is None
        assert p.nautyG is None
        assert p.canonical_certificate is None
        assert p.canonical_label is None

    # ── consolidate_molecule_compartments ─────────────────────

    def test_consolidate_compartments(self):
        """Molecule compartment matching pattern compartment is cleared."""
        m = make_molecule("A", compartment="EC")
        p = make_pattern([m], compartment="EC")
        # __str__ triggers consolidation
        result = str(p)
        assert result == "@EC:A()"
        # After consolidation, molecule compartment is None
        assert m.compartment is None

    def test_consolidate_compartments_no_match(self):
        """Molecule compartment differs from pattern; both printed."""
        m = make_molecule("A", compartment="CP")
        p = make_pattern([m], compartment="EC")
        result = str(p)
        assert result == "@EC:A()@CP"

    # ── Equality ──────────────────────────────────────────────

    def test_equality_same_simple(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        assert a == b

    def test_equality_different_compartment(self):
        a = make_pattern([make_molecule("A")], compartment="EC")
        b = make_pattern([make_molecule("A")], compartment="CP")
        assert a != b

    def test_equality_different_label(self):
        a = make_pattern([make_molecule("A")], label="x")
        b = make_pattern([make_molecule("A")], label="y")
        assert a != b

    def test_equality_different_fixed(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        a.fixed = True
        assert a != b

    def test_equality_different_matchonce(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        a.MatchOnce = True
        assert a != b

    def test_equality_different_relation(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        a.relation = "=="
        a.quantity = "5"
        assert a != b

    def test_equality_different_type(self):
        a = make_pattern([make_molecule("A")])
        assert a != "A()"

    def test_equality_different_molecules(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("B")])
        assert a != b

    # ── Canonicalize (pynauty may or may not be installed) ────

    def test_canonicalize_without_pynauty(self):
        """canonicalize() should not crash when pynauty is unavailable."""
        p = make_pattern([make_molecule("A")])
        # This should either succeed (if pynauty installed) or gracefully return
        p.canonicalize()
        # canonical_label will be set if pynauty is available, None otherwise
        # Either way no exception should be raised

    def test_init_with_canonicalize_flag(self):
        """Pattern(canonicalize=True) calls canonicalize during init."""
        m = make_molecule("A", [make_component("b")])
        _p = Pattern(molecules=[m], canonicalize=True)
        # Should not raise; canonical_label set only if pynauty is available


# ══════════════════════════════════════════════════════════════════
# Integration: complex patterns
# ══════════════════════════════════════════════════════════════════


class TestComplexPatterns:
    def test_full_pattern_string(self):
        """@EC:%x:${MatchOnce}A(b~0!1,c).B(a!1)==5"""
        c_b = make_component("b", state="0", bonds=["1"])
        c_c = make_component("c")
        c_a = make_component("a", bonds=["1"])
        m1 = make_molecule("A", [c_b, c_c])
        m2 = make_molecule("B", [c_a])
        p = make_pattern([m1, m2], compartment="EC", label="x")
        p.fixed = True
        p.MatchOnce = True
        p.relation = "=="
        p.quantity = "5"
        assert str(p) == "@EC%x:${MatchOnce}A(b~0!1,c).B(a!1)==5"

    def test_molecule_with_all_features(self):
        """Molecule with multiple components, compartment, label."""
        c1 = make_component("b", state="0", bonds=["1"])
        c2 = make_component("c", state="1")
        c3 = make_component("d", bonds=["+"])
        m = make_molecule("Rec", [c1, c2, c3], compartment="PM", label="r1")
        assert str(m) == "Rec(b~0!1,c~1,d!+)@PM%r1"

    def test_component_with_all_features(self):
        """Component with state, label, and bond."""
        c = make_component("Y", state="P", label="phos", bonds=["3"])
        assert str(c) == "Y~P%phos!3"

    def test_three_molecule_chain(self):
        """A(b!1).B(a!1,c!2).C(b!2)"""
        ca = make_component("b", bonds=["1"])
        cb1 = make_component("a", bonds=["1"])
        cb2 = make_component("c", bonds=["2"])
        cc = make_component("b", bonds=["2"])
        ma = make_molecule("A", [ca])
        mb = make_molecule("B", [cb1, cb2])
        mc = make_molecule("C", [cc])
        p = make_pattern([ma, mb, mc])
        assert str(p) == "A(b!1).B(a!1,c!2).C(b!2)"

    def test_null_species_in_pattern(self):
        """Null species molecule '0' in a pattern."""
        m = make_molecule("0")
        p = make_pattern([m])
        assert str(p) == "0"


# ══════════════════════════════════════════════════════════════════
# print_canonical tests (manually set canonical_order/canonical_bonds)
# ══════════════════════════════════════════════════════════════════


class TestPrintCanonical:
    """Test print_canonical without requiring pynauty by manually
    setting canonical_order and canonical_bonds attributes."""

    def test_pattern_print_canonical_simple(self):
        c = make_component("b", state="0")
        c.canonical_order = 1
        c.canonical_bonds = None
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m])
        result = p.print_canonical()
        assert result == "A(b~0)"

    def test_pattern_print_canonical_with_compartment_and_label(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m], compartment="EC", label="x")
        result = p.print_canonical()
        assert result == "@EC%x:A(b)"

    def test_pattern_print_canonical_compartment_only(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m], compartment="EC")
        result = p.print_canonical()
        assert result == "@EC:A(b)"

    def test_pattern_print_canonical_label_only(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m], label="x")
        result = p.print_canonical()
        assert result == "%x:A(b)"

    def test_pattern_print_canonical_fixed(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m])
        p.fixed = True
        result = p.print_canonical()
        assert result == "$A(b)"

    def test_pattern_print_canonical_matchonce(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m])
        p.MatchOnce = True
        result = p.print_canonical()
        assert result == "{MatchOnce}A(b)"

    def test_pattern_print_canonical_relation_quantity(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        m.canonical_order = 0
        p = make_pattern([m])
        p.relation = ">="
        p.quantity = "10"
        result = p.print_canonical()
        assert result == "A(b)>=10"

    def test_pattern_print_canonical_two_molecules_reordered(self):
        """Canonical order reverses molecule order."""
        c1 = make_component("a")
        c1.canonical_order = 0
        c2 = make_component("b")
        c2.canonical_order = 0
        m1 = make_molecule("B", [c2])
        m1.canonical_order = 1  # B comes second canonically
        m2 = make_molecule("A", [c1])
        m2.canonical_order = 0  # A comes first canonically
        p = make_pattern([m1, m2])  # stored B first, A second
        result = p.print_canonical()
        # canonical printing should put A first
        assert result == "A(a).B(b)"

    def test_pattern_print_canonical_with_bonds(self):
        c1 = make_component("b")
        c1.canonical_order = 0
        c1.canonical_bonds = ["1"]
        c2 = make_component("a")
        c2.canonical_order = 0
        c2.canonical_bonds = ["1"]
        m1 = make_molecule("A", [c1])
        m1.canonical_order = 0
        m2 = make_molecule("B", [c2])
        m2.canonical_order = 1
        p = make_pattern([m1, m2])
        result = p.print_canonical()
        assert result == "A(b!1).B(a!1)"

    # ── Molecule.print_canonical ──────────────────────────────

    def test_molecule_print_canonical_simple(self):
        c = make_component("b", state="0")
        c.canonical_order = 0
        m = make_molecule("A", [c])
        result = m.print_canonical()
        assert result == "A(b~0)"

    def test_molecule_print_canonical_null_species(self):
        m = make_molecule("0")
        result = m.print_canonical()
        assert result == "0"

    def test_molecule_print_canonical_with_compartment_label(self):
        c = make_component("b")
        c.canonical_order = 0
        m = make_molecule("A", [c], compartment="EC", label="x")
        result = m.print_canonical()
        assert result == "A(b)@EC%x"

    def test_molecule_print_canonical_reordered_components(self):
        """Components printed in canonical order."""
        c1 = make_component("z")
        c1.canonical_order = 1
        c2 = make_component("a")
        c2.canonical_order = 0
        m = make_molecule("M", [c1, c2])  # z stored first, a second
        result = m.print_canonical()
        assert result == "M(a,z)"  # canonical order: a first

    def test_molecule_print_canonical_with_canonical_bonds(self):
        c = make_component("b")
        c.canonical_order = 0
        c.canonical_bonds = ["1"]
        m = make_molecule("A", [c])
        result = m.print_canonical()
        assert result == "A(b!1)"

    # ── Component.print_canonical ─────────────────────────────

    def test_component_print_canonical_simple(self):
        c = make_component("b", state="0")
        result = c.print_canonical()
        assert result == "b~0"

    def test_component_print_canonical_with_label(self):
        c = make_component("b", label="x")
        result = c.print_canonical()
        assert result == "b%x"

    def test_component_print_canonical_with_states(self):
        c = make_component("b", states=["0", "1"])
        result = c.print_canonical()
        assert result == "b~0~1"

    def test_component_print_canonical_with_canonical_bonds(self):
        c = make_component("b")
        c.canonical_bonds = ["1", "2"]
        result = c.print_canonical()
        assert result == "b!1!2"

    def test_component_print_canonical_no_canonical_bonds(self):
        """When canonical_bonds is None, regular bonds are not printed."""
        c = make_component("b", bonds=["1"])
        # canonical_bonds defaults to None
        result = c.print_canonical()
        assert result == "b"


# ══════════════════════════════════════════════════════════════════
# Equality with canonical labels / certificates
# ══════════════════════════════════════════════════════════════════


class TestEqualityWithCanonicalLabels:
    def test_pattern_equal_by_canonical_label(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        a.canonical_label = "A()"
        b.canonical_label = "A()"
        assert a == b

    def test_pattern_not_equal_by_canonical_label(self):
        a = make_pattern([make_molecule("A")])
        b = make_pattern([make_molecule("A")])
        a.canonical_label = "A(b)"
        b.canonical_label = "A(c)"
        assert a != b

    def test_pattern_equal_by_canonical_certificate(self):
        """When canonical_label is None but certificate matches."""
        c = make_component("b")
        a = make_pattern([make_molecule("A", [c])])
        b = make_pattern([make_molecule("A", [c])])
        a.canonical_certificate = (1, 0)
        b.canonical_certificate = (1, 0)
        assert a == b

    def test_pattern_not_equal_by_canonical_certificate(self):
        c = make_component("b")
        a = make_pattern([make_molecule("A", [c])])
        b = make_pattern([make_molecule("A", [c])])
        a.canonical_certificate = (1, 0)
        b.canonical_certificate = (0, 1)
        assert a != b

    def test_molecule_equal_by_canonical_label(self):
        a = make_molecule("A")
        b = make_molecule("A")
        a.canonical_label = "A()"
        b.canonical_label = "A()"
        assert a == b

    def test_molecule_not_equal_by_canonical_label(self):
        a = make_molecule("A")
        b = make_molecule("A")
        a.canonical_label = "A(b)"
        b.canonical_label = "A(c)"
        assert a != b

    def test_component_equal_by_canonical_label(self):
        a = make_component("b", state="0")
        b = make_component("b", state="0")
        a.canonical_label = "b~0"
        b.canonical_label = "b~0"
        assert a == b

    def test_component_not_equal_by_canonical_label(self):
        a = make_component("b", state="0")
        b = make_component("b", state="0")
        a.canonical_label = "b~0!1"
        b.canonical_label = "b~0!2"
        assert a != b

    def test_component_equality_states_length_mismatch(self):
        """Components with different states list lengths are not equal."""
        a = make_component("b")
        b = make_component("b")
        a.states = ["0", "1"]
        b.states = ["0"]
        assert a != b

    def test_molecule_not_equal_different_components(self):
        """Molecules with same name but different components are not equal."""
        a = make_molecule("A", [make_component("b", state="0")])
        b = make_molecule("A", [make_component("c", state="0")])
        assert a != b


# ══════════════════════════════════════════════════════════════════
# Canonicalize with mocked pynauty
# ══════════════════════════════════════════════════════════════════


class _MockGraph:
    """Minimal mock of pynauty.Graph."""

    def __init__(self, n):
        self.n = n
        self.adjacency = {}
        self.vertex_coloring = None

    def connect_vertex(self, v, neighbors):
        if v not in self.adjacency:
            self.adjacency[v] = []
        self.adjacency[v].extend(neighbors)

    def set_vertex_coloring(self, color_sets):
        self.vertex_coloring = color_sets


def _make_mock_pynauty(node_count):
    """Return a mock pynauty module. canon_label returns identity ordering."""
    mod = types.ModuleType("pynauty")
    mod.Graph = _MockGraph

    def certificate(g):
        return tuple(range(g.n))

    def canon_label(g):
        return list(range(g.n))

    mod.certificate = certificate
    mod.canon_label = canon_label
    return mod


@pytest.fixture()
def mock_pynauty():
    """Temporarily inject a mock pynauty into sys.modules."""
    mod = _make_mock_pynauty(10)
    old = sys.modules.get("pynauty")
    sys.modules["pynauty"] = mod
    yield mod
    if old is None:
        sys.modules.pop("pynauty", None)
    else:
        sys.modules["pynauty"] = old


class TestCanonicalizeWithMock:
    def test_canonicalize_single_molecule_no_bonds(self, mock_pynauty):
        """Canonicalize a simple A(b~0) pattern."""
        c = make_component("b", state="0")
        m = make_molecule("A", [c])
        c.parent_molecule = m
        p = make_pattern([m])
        p.canonicalize()
        assert p.nautyG is not None
        assert p.canonical_certificate is not None
        assert p.canonical_label is not None
        # canonical_label should be a valid string representation
        assert "A" in p.canonical_label

    def test_canonicalize_bonded_pair(self, mock_pynauty):
        """Canonicalize A(b!1).B(a!1)."""
        c1 = make_component("b", bonds=["bond1"])
        c2 = make_component("a", bonds=["bond1"])
        m1 = make_molecule("A", [c1])
        m2 = make_molecule("B", [c2])
        c1.parent_molecule = m1
        c2.parent_molecule = m2
        p = make_pattern([m1, m2])
        p.canonicalize()
        assert p.canonical_label is not None
        assert "A" in p.canonical_label
        assert "B" in p.canonical_label
        # Both components should have canonical bonds set
        assert c1.canonical_bonds is not None
        assert c2.canonical_bonds is not None

    def test_canonicalize_duplicate_molecules(self, mock_pynauty):
        """Two identical molecules: A(b).A(b) — exercises the mCopyId branch."""
        c1 = make_component("b")
        c2 = make_component("b")
        m1 = make_molecule("A", [c1])
        m2 = make_molecule("A", [c2])
        c1.parent_molecule = m1
        c2.parent_molecule = m2
        p = make_pattern([m1, m2])
        p.canonicalize()
        assert p.canonical_label is not None

    def test_canonicalize_duplicate_components(self, mock_pynauty):
        """A(b,b) — exercises the cCopyId branch for duplicate component names."""
        c1 = make_component("b")
        c2 = make_component("b")
        m = make_molecule("A", [c1, c2])
        c1.parent_molecule = m
        c2.parent_molecule = m
        p = make_pattern([m])
        p.canonicalize()
        assert p.canonical_label is not None

    def test_canonicalize_dangling_bond(self, mock_pynauty):
        """A bond with only one endpoint triggers a warning, not a crash."""
        c1 = make_component("b", bonds=["dangling"])
        m1 = make_molecule("A", [c1])
        c1.parent_molecule = m1
        p = make_pattern([m1])
        # Should not raise — just warn about dangling bond
        p.canonicalize()
        assert p.canonical_label is not None

    def test_canonicalize_with_compartment(self, mock_pynauty):
        """Pattern with compartment gets it in the canonical label."""
        c = make_component("b")
        m = make_molecule("A", [c])
        c.parent_molecule = m
        p = make_pattern([m], compartment="EC")
        p.canonicalize()
        assert "@EC" in p.canonical_label

    def test_canonicalize_two_bonds(self, mock_pynauty):
        """Two separate bonds: A(b!1).B(a!1) and C(c!2).D(d!2)."""
        c1 = make_component("b", bonds=["bond1"])
        c2 = make_component("a", bonds=["bond1"])
        c3 = make_component("c", bonds=["bond2"])
        c4 = make_component("d", bonds=["bond2"])
        m1 = make_molecule("A", [c1])
        m2 = make_molecule("B", [c2])
        m3 = make_molecule("C", [c3])
        m4 = make_molecule("D", [c4])
        c1.parent_molecule = m1
        c2.parent_molecule = m2
        c3.parent_molecule = m3
        c4.parent_molecule = m4
        p = make_pattern([m1, m2, m3, m4])
        p.canonicalize()
        assert p.canonical_label is not None
        # All bonded components should have canonical_bonds set
        for c in [c1, c2, c3, c4]:
            assert c.canonical_bonds is not None

    def test_canonicalize_multi_bond_append_branch(self, mock_pynauty):
        """Exercise the canonical_bonds.append branch (lines 218, 222).

        Pre-set canonical_bonds on components before canonicalize so the
        bond assignment triggers the append path.
        """
        c1 = make_component("b", bonds=["bond1"])
        c2 = make_component("a", bonds=["bond1"])
        c3 = make_component("c", bonds=["bond2"])
        c4 = make_component("d", bonds=["bond2"])
        m1 = make_molecule("A", [c1])
        m2 = make_molecule("B", [c2])
        m3 = make_molecule("C", [c3])
        m4 = make_molecule("D", [c4])
        c1.parent_molecule = m1
        c2.parent_molecule = m2
        c3.parent_molecule = m3
        c4.parent_molecule = m4
        # Pre-set canonical_bonds so the code hits the append branch
        c1.canonical_bonds = ["0"]
        c2.canonical_bonds = ["0"]
        c3.canonical_bonds = ["0"]
        c4.canonical_bonds = ["0"]
        p = make_pattern([m1, m2, m3, m4])
        p.canonicalize()
        # Each pre-set component should now have appended a bond
        for c in [c1, c2, c3, c4]:
            assert len(c.canonical_bonds) == 2

    def test_init_canonicalize_true_with_mock(self, mock_pynauty):
        """Pattern(canonicalize=True) runs full canonicalize."""
        c = make_component("b")
        m = make_molecule("A", [c])
        c.parent_molecule = m
        p = Pattern(molecules=[m], canonicalize=True)
        assert p.canonical_label is not None
