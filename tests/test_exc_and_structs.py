"""Tests for bionetgen/core/exc.py, bionetgen/modelapi/structs.py, and bionetgen/modelapi/rulemod.py."""

import pytest

from bionetgen.core.exc import (
    BNGError,
    BNGVersionError,
    BNGPerlError,
    BNGParseError,
    BNGFileError,
    BNGModelError,
    BNGRunError,
    BNGCompileError,
    BNGFormatError,
    BNGSimError,
)
from bionetgen.modelapi.structs import (
    ModelObj,
    Parameter,
    Compartment,
    Observable,
    MoleculeType,
    Species,
    Function,
    Action,
    Rule,
    EnergyPattern,
    PopulationMap,
)
from bionetgen.modelapi.rulemod import RuleMod


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class FakePattern:
    """Lightweight stand-in for Pattern objects used in Observable, Rule, etc."""

    def __init__(self, text, match_once=True):
        self._text = text
        self.MatchOnce = match_once

    def __str__(self):
        return self._text


# ===================================================================
# 1. Exception classes
# ===================================================================

class TestBNGError:
    def test_is_base_exception(self):
        assert issubclass(BNGError, Exception)

    def test_raise_and_catch(self):
        with pytest.raises(BNGError):
            raise BNGError("boom")


class TestBNGVersionError:
    def test_inherits(self):
        assert issubclass(BNGVersionError, BNGError)

    def test_attributes_and_message(self):
        err = BNGVersionError("1.0", "2.0")
        assert err.cur_version == "1.0"
        assert err.req_version == "2.0"
        assert "1.0" in err.message
        assert "2.0" in err.message
        assert "upgrade" in err.message


class TestBNGPerlError:
    def test_inherits(self):
        assert issubclass(BNGPerlError, BNGError)

    def test_message(self):
        err = BNGPerlError()
        assert "Perl" in err.message or "perl" in err.message


class TestBNGParseError:
    def test_inherits(self):
        assert issubclass(BNGParseError, BNGError)

    def test_default_message(self):
        err = BNGParseError()
        assert err.path is None
        assert "parsing" in err.message.lower() or "issue" in err.message.lower()

    def test_with_path_and_message(self):
        err = BNGParseError(bngl_path="/tmp/model.bngl", message=" extra info")
        assert err.path == "/tmp/model.bngl"
        assert "/tmp/model.bngl" in err.message
        assert "extra info" in err.message

    def test_with_path_no_extra_message(self):
        err = BNGParseError(bngl_path="/tmp/m.bngl", message=None)
        assert "/tmp/m.bngl" in err.message


class TestBNGFileError:
    def test_inherits(self):
        assert issubclass(BNGFileError, BNGError)

    def test_attributes(self):
        err = BNGFileError("/some/path.bngl", message="bad file")
        assert err.path == "/some/path.bngl"
        assert err.message == "bad file"

    def test_default_message(self):
        err = BNGFileError("/x")
        assert "BNGL file" in err.message


class TestBNGModelError:
    def test_inherits(self):
        assert issubclass(BNGModelError, BNGError)

    def test_message_formatting(self):
        err = BNGModelError("my_model", message="custom")
        assert "my_model" in err.message
        assert "custom" in err.message

    def test_no_extra_message(self):
        err = BNGModelError("m", message=None)
        assert "m" in err.message


class TestBNGRunError:
    def test_inherits(self):
        assert issubclass(BNGRunError, BNGError)

    def test_without_stdout_stderr(self):
        err = BNGRunError("run cmd")
        assert err.command == "run cmd"
        assert err.stdout is None
        assert err.stderr is None
        assert "run cmd" in err.message

    def test_with_stdout_stderr(self):
        err = BNGRunError("cmd", stdout="out text", stderr="err text")
        assert "out text" in err.message
        assert "err text" in err.message


class TestBNGCompileError:
    def test_inherits(self):
        assert issubclass(BNGCompileError, BNGError)

    def test_attributes(self):
        err = BNGCompileError("model_x", message="compile fail")
        assert err.model == "model_x"
        assert err.message == "compile fail"

    def test_default_message(self):
        err = BNGCompileError("m")
        assert "CVODE" in err.message or "compil" in err.message.lower()


class TestBNGFormatError:
    def test_inherits(self):
        assert issubclass(BNGFormatError, BNGError)

    def test_default_message_with_path(self):
        err = BNGFormatError(path="/tmp/f.txt")
        assert err.path == "/tmp/f.txt"
        assert "/tmp/f.txt" in err.message

    def test_custom_message(self):
        err = BNGFormatError(message="custom format error")
        assert err.message == "custom format error"

    def test_no_args(self):
        err = BNGFormatError()
        assert err.path is None
        assert "None" in err.message or "format" in err.message.lower()


class TestBNGSimError:
    def test_inherits(self):
        assert issubclass(BNGSimError, BNGError)

    def test_default_message(self):
        err = BNGSimError()
        assert "BNGsim" in err.message or "simulation" in err.message.lower()

    def test_custom_message(self):
        err = BNGSimError("oops")
        assert err.message == "oops"


class TestAllInheritFromBNGError:
    @pytest.mark.parametrize(
        "cls",
        [
            BNGVersionError,
            BNGPerlError,
            BNGParseError,
            BNGFileError,
            BNGModelError,
            BNGRunError,
            BNGCompileError,
            BNGFormatError,
            BNGSimError,
        ],
    )
    def test_subclass(self, cls):
        assert issubclass(cls, BNGError)


# ===================================================================
# 2. Struct classes
# ===================================================================

class TestModelObj:
    def test_str_repr_delegate_to_gen_string(self):
        # ModelObj itself has no gen_string, so test via a subclass
        p = Parameter("k1", "0.5")
        assert str(p) == repr(p) == "k1 0.5"

    def test_contains(self):
        p = Parameter("k1", "0.5")
        assert "name" in p
        assert "nonexistent" not in p

    def test_getitem_setitem_delitem(self):
        p = Parameter("k1", "0.5")
        assert p["name"] == "k1"
        p["name"] = "k2"
        assert p.name == "k2"
        del p["name"]
        assert not hasattr(p, "name")

    def test_comment_strips_hash(self):
        p = Parameter("k1", "1")
        p.comment = "#this is a comment"
        assert p.comment == "this is a comment"

    def test_comment_no_hash(self):
        p = Parameter("k1", "1")
        p.comment = "no hash"
        assert p.comment == "no hash"

    def test_line_label_int(self):
        p = Parameter("k1", "1")
        p.line_label = "5"
        assert p.line_label == "5 "

    def test_line_label_string(self):
        p = Parameter("k1", "1")
        p.line_label = "myLabel"
        assert p.line_label == "myLabel: "

    def test_print_line_no_label_no_comment(self):
        p = Parameter("k1", "1")
        result = p.print_line()
        assert result == "  k1 1"

    def test_print_line_with_label_and_comment(self):
        p = Parameter("k1", "1")
        p.line_label = "3"
        p.comment = "rate constant"
        result = p.print_line()
        assert result == "  3 k1 1 #rate constant"


class TestParameter:
    def test_gen_string(self):
        p = Parameter("kf", "0.01")
        assert p.gen_string() == "kf 0.01"
        assert p.name == "kf"
        assert p.value == "0.01"


class TestCompartment:
    def test_without_outside(self):
        c = Compartment("EC", "3", "1e-6")
        assert c.gen_string() == "EC 3 1e-6"

    def test_with_outside(self):
        c = Compartment("PM", "2", "1e-8", outside="EC")
        assert c.gen_string() == "PM 2 1e-8 EC"
        assert c.outside == "EC"


class TestObservable:
    def test_gen_string(self):
        p1 = FakePattern("A(b)")
        p2 = FakePattern("B(a)")
        obs = Observable("obs1", "Molecules", [p1, p2])
        assert obs.gen_string() == "Molecules obs1 A(b),B(a)"

    def test_add_pattern(self):
        obs = Observable("obs1", "Molecules", [])
        obs.add_pattern(FakePattern("C()"))
        assert len(obs.patterns) == 1

    def test_species_type_clears_match_once(self):
        p = FakePattern("A()", match_once=True)
        obs = Observable("obs1", "Species", [p])
        assert p.MatchOnce is False

    def test_add_pattern_species_clears_match_once(self):
        obs = Observable("obs1", "Species", [])
        p = FakePattern("A()", match_once=True)
        obs.add_pattern(p)
        assert p.MatchOnce is False

    def test_single_pattern(self):
        obs = Observable("obs1", "Molecules", [FakePattern("X()")])
        assert obs.gen_string() == "Molecules obs1 X()"


class TestMoleculeType:
    def test_gen_string(self):
        mt = MoleculeType("A", [])
        assert mt.name == "A"
        # Molecule with no components prints "A()"
        result = mt.gen_string()
        assert "A" in result

    def test_molecule_attribute(self):
        mt = MoleculeType("B", [])
        assert mt.molecule is not None


class TestSpecies:
    def test_gen_string(self):
        pat = FakePattern("A(b!1).B(a!1)")
        sp = Species(pattern=pat, count=100)
        assert sp.gen_string() == "A(b!1).B(a!1) 100"
        assert sp.name == "A(b!1).B(a!1)"


class TestFunction:
    def test_without_args(self):
        f = Function("rate", "k1*k2")
        assert f.gen_string() == "rate = k1*k2"

    def test_with_args(self):
        f = Function("rate", "x+y", args=["x", "y"])
        assert f.gen_string() == "rate(x,y) = x+y"


class TestAction:
    def test_normal_type_gen_string(self):
        a = Action(action_type="simulate", action_args={"method": '"ode"', "t_end": "100"})
        s = a.gen_string()
        assert s.startswith("simulate(")
        assert "method" in s
        assert "=>" in s
        assert s.endswith(")")

    def test_no_args(self):
        a = Action(action_type="simulate", action_args={})
        s = a.gen_string()
        # No curly braces when no args for normal_types
        assert s == "simulate()"

    def test_no_setter_syntax(self):
        a = Action(action_type="setConcentration", action_args={"A()": "100"})
        s = a.gen_string()
        assert "=>" not in s
        assert "A()" in s

    def test_square_braces(self):
        a = Action(action_type="saveConcentrations", action_args={})
        s = a.gen_string()
        assert s == "saveConcentrations([])"

    def test_invalid_action_type_raises(self):
        with pytest.raises(Exception):
            Action(action_type="not_a_real_action", action_args={})

    def test_invalid_arg_raises(self):
        with pytest.raises(Exception):
            Action(action_type="simulate", action_args={"bogus_arg_xyz": "1"})

    def test_print_line(self):
        a = Action(action_type="simulate", action_args={"method": '"ode"'})
        line = a.print_line()
        assert "simulate" in line

    def test_print_line_with_comment(self):
        a = Action(action_type="simulate", action_args={})
        a.comment = "run sim"
        line = a.print_line()
        assert "#run sim" in line

    def test_type_attributes(self):
        a = Action(action_type="simulate", action_args={})
        assert isinstance(a.normal_types, list)
        assert isinstance(a.no_setter_syntax, list)
        assert isinstance(a.square_braces, list)
        assert "simulate" in a.normal_types


class TestRule:
    def test_unidirectional(self):
        r = Rule(
            name="r1",
            reactants=[FakePattern("A(b)")],
            products=[FakePattern("B(a)")],
            rate_constants=("kf",),
        )
        assert r.bidirectional is False
        s = r.gen_string()
        assert "->" in s
        assert "<->" not in s
        assert "r1" in s
        assert "kf" in s

    def test_bidirectional(self):
        r = Rule(
            name="r2",
            reactants=[FakePattern("A(b)")],
            products=[FakePattern("B(a)")],
            rate_constants=("kf", "kr"),
        )
        assert r.bidirectional is True
        s = r.gen_string()
        assert "<->" in s
        assert "kf" in s
        assert "kr" in s

    def test_set_rate_constants(self):
        r = Rule(
            name="r",
            reactants=[FakePattern("A()")],
            products=[FakePattern("B()")],
            rate_constants=("k1",),
        )
        assert r.bidirectional is False
        r.set_rate_constants(("k1", "k2"))
        assert r.bidirectional is True
        assert r.rate_constants == ["k1", "k2"]

    def test_side_string(self):
        r = Rule(
            name="r",
            reactants=[],
            products=[],
            rate_constants=("k",),
        )
        result = r.side_string([FakePattern("A()"), FakePattern("B()")])
        assert result == "A() + B()"

    def test_side_string_single(self):
        r = Rule(name="r", reactants=[], products=[], rate_constants=("k",))
        assert r.side_string([FakePattern("X()")]) == "X()"

    def test_side_string_empty(self):
        r = Rule(name="r", reactants=[], products=[], rate_constants=("k",))
        assert r.side_string([]) == ""


class TestEnergyPattern:
    def test_gen_string(self):
        ep = EnergyPattern("ep1", FakePattern("A(b!1).B(a!1)"), "Eab")
        s = ep.gen_string()
        assert s == "A(b!1).B(a!1) Eab"
        assert ep.name == "ep1"


class TestPopulationMap:
    def test_gen_string(self):
        pm = PopulationMap("pm1", FakePattern("A(b~0)"), FakePattern("Apop"), "lump_k")
        s = pm.gen_string()
        assert s == "A(b~0) -> Apop lump_k"
        assert pm.name == "pm1"


# ===================================================================
# 3. RuleMod
# ===================================================================

class TestRuleMod:
    def test_init_none(self):
        rm = RuleMod()
        assert rm.type is None

    def test_init_valid(self):
        rm = RuleMod(mod_type="DeleteMolecules")
        assert rm.type == "DeleteMolecules"

    def test_init_invalid_ignored(self):
        # Invalid type prints a warning and _type is never set,
        # so accessing .type raises AttributeError
        rm = RuleMod(mod_type="InvalidMod")
        with pytest.raises(AttributeError):
            _ = rm.type

    def test_str_none(self):
        rm = RuleMod()
        assert str(rm) == ""

    def test_str_valid(self):
        rm = RuleMod(mod_type="TotalRate")
        assert str(rm) == "TotalRate"

    def test_repr(self):
        rm = RuleMod(mod_type="MoveConnected")
        assert repr(rm) == "Rule modifier of type MoveConnected"

    def test_repr_none(self):
        rm = RuleMod()
        assert repr(rm) == "Rule modifier of type None"

    def test_type_setter_valid(self):
        rm = RuleMod()
        rm.type = "DeleteMolecules"
        assert rm.type == "DeleteMolecules"

    def test_type_setter_invalid(self):
        rm = RuleMod()
        rm.type = "BadType"
        # Invalid assignment is silently rejected; type stays as it was (None)
        assert rm.type is None

    def test_type_setter_none(self):
        rm = RuleMod(mod_type="TotalRate")
        rm.type = None
        assert rm.type is None
