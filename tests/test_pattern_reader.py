"""Tests for bionetgen/modelapi/pattern_reader.py"""
import pytest
from bionetgen.modelapi.pattern_reader import BNGParsers, BNGPatternReader


class TestBNGParsers:
    def test_init(self):
        p = BNGParsers()
        assert p is not None


class TestBNGPatternReader:
    def test_simple_molecule(self):
        r = BNGPatternReader("A()")
        pat = r.pattern
        assert len(pat.molecules) == 1
        assert pat.molecules[0].name == "A"
        assert len(pat.molecules[0].components) == 0

    def test_molecule_with_component(self):
        r = BNGPatternReader("A(b)")
        pat = r.pattern
        assert len(pat.molecules) == 1
        assert len(pat.molecules[0].components) == 1
        assert pat.molecules[0].components[0].name == "b"

    def test_molecule_with_state(self):
        r = BNGPatternReader("A(b~0)")
        pat = r.pattern
        comp = pat.molecules[0].components[0]
        assert comp.name == "b"
        assert comp.state == "0"

    def test_molecule_with_bond(self):
        r = BNGPatternReader("A(b!1)")
        pat = r.pattern
        comp = pat.molecules[0].components[0]
        assert comp.name == "b"
        assert "1" in comp.bonds

    def test_two_molecules_bonded(self):
        r = BNGPatternReader("A(b!1).B(a!1)")
        pat = r.pattern
        assert len(pat.molecules) == 2
        assert pat.molecules[0].name == "A"
        assert pat.molecules[1].name == "B"

    def test_molecule_with_compartment(self):
        r = BNGPatternReader("A@EC()")
        pat = r.pattern
        assert pat.molecules[0].compartment == "EC"

    def test_molecule_with_tag(self):
        r = BNGPatternReader("A%x()")
        pat = r.pattern
        assert pat.molecules[0].label == "x"

    def test_pattern_compartment(self):
        r = BNGPatternReader("@EC::A()")
        pat = r.pattern
        assert pat.compartment == "EC"

    def test_pattern_label(self):
        r = BNGPatternReader("%x::A()")
        pat = r.pattern
        assert pat.label == "x"

    def test_fixed_pattern(self):
        r = BNGPatternReader("$A()")
        pat = r.pattern
        assert pat.fixed is True

    def test_matchonce_pattern(self):
        r = BNGPatternReader("{MatchOnce}A()")
        pat = r.pattern
        assert pat.MatchOnce is True

    def test_zero_molecule(self):
        r = BNGPatternReader("0")
        pat = r.pattern
        assert len(pat.molecules) == 1
        assert pat.molecules[0].name is None or str(pat.molecules[0]) == "0"

    def test_multiple_components(self):
        r = BNGPatternReader("A(b,c,d)")
        pat = r.pattern
        assert len(pat.molecules[0].components) == 3
        names = [c.name for c in pat.molecules[0].components]
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_wildcard_bond_plus(self):
        r = BNGPatternReader("A(b!+)")
        pat = r.pattern
        comp = pat.molecules[0].components[0]
        assert "+" in comp.bonds

    def test_wildcard_bond_question(self):
        r = BNGPatternReader("A(b!?)")
        pat = r.pattern
        comp = pat.molecules[0].components[0]
        assert "?" in comp.bonds

    def test_state_with_number(self):
        r = BNGPatternReader("A(p~1)")
        pat = r.pattern
        comp = pat.molecules[0].components[0]
        assert comp.state == "1"

    def test_quantifier_equals(self):
        r = BNGPatternReader("A()==5")
        pat = r.pattern
        assert pat.relation == "=="
        assert pat.quantity == 5

    def test_quantifier_less(self):
        r = BNGPatternReader("A()<3")
        pat = r.pattern
        assert pat.relation == "<"
        assert pat.quantity == 3

    def test_quantifier_greater(self):
        r = BNGPatternReader("A()>2")
        pat = r.pattern
        assert pat.relation == ">"
        assert pat.quantity == 2

    def test_three_molecules(self):
        r = BNGPatternReader("A(b!1).B(a!1,c!2).C(b!2)")
        pat = r.pattern
        assert len(pat.molecules) == 3

    def test_complex_pattern(self):
        r = BNGPatternReader("A(b~0!1,c~Y).B(a!1)")
        pat = r.pattern
        assert len(pat.molecules) == 2
        mol_a = pat.molecules[0]
        assert mol_a.name == "A"
        assert len(mol_a.components) == 2

    def test_parsers_defined(self):
        r = BNGPatternReader("A()")
        assert hasattr(r.parsers, "base_name")
        assert hasattr(r.parsers, "state")
        assert hasattr(r.parsers, "bond")
        assert hasattr(r.parsers, "component")
        assert hasattr(r.parsers, "molecule")
        assert hasattr(r.parsers, "pattern")
