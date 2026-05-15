from collections import OrderedDict

import pytest

from bionetgen.core.exc import BNGParseError
from bionetgen.modelapi.xmlparsers import (
    PatternXML,
    PopulationMapBlockXML,
    RuleBlockXML,
)


def _simple_molecule_xml(name):
    return OrderedDict([("@id", "M1"), ("@name", name)])


def _simple_pattern_xml(molecules, relation=None, quantity=None):
    pattern = OrderedDict()
    if relation is not None and quantity is not None:
        pattern["@relation"] = relation
        pattern["@quantity"] = quantity
    pattern["ListOfMolecules"] = OrderedDict([("Molecule", molecules)])
    return pattern


def _make_rate_law_xml(rate_type, value="0.5"):
    if rate_type == "Function":
        return OrderedDict(
            [("@type", "Function"), ("@id", "rule1"), ("@name", "rate1")]
        )
    return OrderedDict(
        [
            ("@type", rate_type),
            (
                "ListOfRateConstants",
                OrderedDict([("RateConstant", OrderedDict([("@value", value)]))]),
            ),
        ]
    )


def _make_rule_xml(name="r1", reactant="A", product="B", rate_type="Ele", value="0.5"):
    return OrderedDict(
        [
            ("@name", name),
            (
                "ListOfReactantPatterns",
                OrderedDict(
                    [
                        (
                            "ReactantPattern",
                            _simple_pattern_xml(_simple_molecule_xml(reactant)),
                        )
                    ]
                ),
            ),
            (
                "ListOfProductPatterns",
                OrderedDict(
                    [
                        (
                            "ProductPattern",
                            _simple_pattern_xml(_simple_molecule_xml(product)),
                        )
                    ]
                ),
            ),
            ("RateLaw", _make_rate_law_xml(rate_type, value)),
            ("ListOfOperations", OrderedDict()),
        ]
    )


def _make_population_map_xml(rate_type="Ele", value="0.5"):
    return OrderedDict(
        [
            ("@id", "pm1"),
            (
                "StructuredSpecies",
                OrderedDict(
                    [("Species", _simple_pattern_xml(_simple_molecule_xml("A")))]
                ),
            ),
            (
                "PopulationSpecies",
                OrderedDict(
                    [("Species", _simple_pattern_xml(_simple_molecule_xml("Apop")))]
                ),
            ),
            ("RateLaw", _make_rate_law_xml(rate_type, value)),
        ]
    )


def test_pattern_quantity_non_integer_raises_parse_error():
    pattern_xml = _simple_pattern_xml(
        _simple_molecule_xml("A"), relation="==", quantity="1.5"
    )
    with pytest.raises(BNGParseError, match="Pattern quantity must be an integer"):
        PatternXML(pattern_xml)


def test_parse_rule_missing_rate_law_raises_parse_error():
    rule_xml = _make_rule_xml()
    del rule_xml["RateLaw"]
    with pytest.raises(BNGParseError, match="missing a RateLaw entry"):
        RuleBlockXML(rule_xml)


def test_rule_ratelaw_unknown_type_raises_parse_error():
    rule_block = RuleBlockXML(_make_rule_xml())
    with pytest.raises(BNGParseError, match="Unrecognized rate law type"):
        rule_block.resolve_ratelaw(OrderedDict([("@type", "mystery")]))


def test_rule_reaction_side_invalid_xml_raises_parse_error():
    rule_block = RuleBlockXML(_make_rule_xml())
    with pytest.raises(
        BNGParseError,
        match="Reaction side XML must contain ReactantPattern or ProductPattern",
    ):
        rule_block.resolve_rxn_side(OrderedDict([("NotAPattern", OrderedDict())]))


def test_population_map_ratelaw_unknown_type_raises_parse_error():
    population_map = PopulationMapBlockXML(_make_population_map_xml())
    with pytest.raises(BNGParseError, match="Unrecognized rate law type"):
        population_map.resolve_ratelaw(OrderedDict([("@type", "mystery")]))
