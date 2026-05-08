import pytest
from bionetgen.modelapi.xmlparsers import RuleBlockXML


def test_get_rule_mod():
    parser = RuleBlockXML([])

    xml_totalrate_int = {
        "@name": "test_rule",
        "ListOfOperations": {},
        "RateLaw": {
            "@type": "Function",
            "@totalrate": 1,
            "@id": "rule1",
            "@name": "rate1",
        },
    }
    mod1 = parser.get_rule_mod(xml_totalrate_int)
    assert mod1.type == "TotalRate"
    assert str(mod1.call) == "1"

    xml_totalrate_str = {
        "@name": "test_rule",
        "ListOfOperations": {},
        "RateLaw": {
            "@type": "Function",
            "@totalrate": "1",
            "@id": "rule1",
            "@name": "rate1",
        },
    }
    mod1_str = parser.get_rule_mod(xml_totalrate_str)
    assert mod1_str.type == "TotalRate"
    assert mod1_str.call == "1"

    xml_totalrate_missing = {
        "@name": "test_rule",
        "ListOfOperations": {},
        "RateLaw": {"@type": "Function", "@id": "rule1", "@name": "rate1"},
    }
    mod1_miss = parser.get_rule_mod(xml_totalrate_missing)
    assert mod1_miss.type is None

    xml_totalrate_zero = {
        "@name": "test_rule",
        "ListOfOperations": {},
        "RateLaw": {
            "@type": "Function",
            "@totalrate": "0",
            "@id": "rule1",
            "@name": "rate1",
        },
    }
    mod1_zero = parser.get_rule_mod(xml_totalrate_zero)
    assert mod1_zero.type is None

    xml_delete = {
        "@name": "test_rule",
        "ListOfOperations": {
            "Delete": [{"@DeleteMolecules": "1"}, {"@DeleteMolecules": "1"}]
        },
    }
    mod2 = parser.get_rule_mod(xml_delete)
    assert mod2.type == "DeleteMolecules"

    xml_delete_missing = {"@name": "test_rule", "ListOfOperations": {"Delete": [{}]}}
    mod3 = parser.get_rule_mod(xml_delete_missing)
    assert mod3.type is None

    xml_move = {
        "@name": "test_rule",
        "ListOfOperations": {
            "ChangeCompartment": [
                {
                    "@moveConnected": "1",
                    "@id": "m1",
                    "@source": "s1",
                    "@destination": "d1",
                    "@flipOrientation": "0",
                },
                {
                    "@moveConnected": "1",
                    "@id": "m2",
                    "@source": "s2",
                    "@destination": "d2",
                    "@flipOrientation": "1",
                },
            ]
        },
    }
    mod4 = parser.get_rule_mod(xml_move)
    assert mod4.type == "MoveConnected"
    assert mod4.id == ["m1", "m2"]
    assert mod4.source == ["s1", "s2"]

    xml_move_single = {
        "@name": "test_rule",
        "ListOfOperations": {
            "ChangeCompartment": {
                "@moveConnected": "1",
                "@id": "m1",
                "@source": "s1",
                "@destination": "d1",
                "@flipOrientation": "0",
            }
        },
    }
    mod5 = parser.get_rule_mod(xml_move_single)
    assert mod5.type == "MoveConnected"
    assert mod5.id == "m1"
    assert mod5.source == "s1"

    # Precedence: Delete + RateLaw
    xml_both = {
        "@name": "test_rule",
        "ListOfOperations": {"Delete": [{"@DeleteMolecules": "1"}]},
        "RateLaw": {
            "@type": "Function",
            "@totalrate": "1",
            "@id": "rule1",
            "@name": "rate1",
        },
    }
    mod6 = parser.get_rule_mod(xml_both)
    assert mod6.type == "DeleteMolecules"
