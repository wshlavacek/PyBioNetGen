from unittest.mock import patch

import pytest

from bionetgen.core.exc import BNGParseError
from bionetgen.modelapi.rulemod import RuleMod
from bionetgen.modelapi.structs import Parameter, Rule
from bionetgen.network.structs import NetworkObj


class FakePattern:
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


NET_MALFORMED_SPECIES = """\
# NET file
begin species
  1 A(b)
end species
"""


def test_networkparser_malformed_species_line_raises_parse_error(tmp_path):
    net_file = tmp_path / "bad_species.net"
    net_file.write_text(NET_MALFORMED_SPECIES)
    from bionetgen.network import networkparser as networkparser_module
    from bionetgen.network.network import Network

    with patch.object(networkparser_module, "logger") as mock_logger:
        with pytest.raises(BNGParseError, match="Malformed species line"):
            Network(str(net_file))

    mock_logger.error.assert_called_once()
    error_args, error_kwargs = mock_logger.error.call_args
    assert "Malformed species line" in error_args[0]
    assert "expected '<id> <species> <count>'" in error_args[0]
    assert "bad_species.net:3" in error_args[0]
    assert "BNGNetworkParser.parse_network()" in error_kwargs["loc"]


def test_model_struct_line_label_none_uses_string_fallback():
    parameter = Parameter("k1", "1")
    parameter.line_label = None
    assert parameter.line_label == "None: "


def test_network_struct_line_label_none_uses_string_fallback():
    obj = NetworkObj()
    obj.line_label = None
    assert obj.line_label == "None: "


def test_rule_set_rate_constants_invalid_length_raises_parse_error():
    rule = Rule(
        name="r",
        reactants=[FakePattern("A()")],
        products=[FakePattern("B()")],
        rate_constants=("k1",),
    )
    with pytest.raises(BNGParseError, match="1 or 2 rate constants"):
        rule.set_rate_constants(("k1", "k2", "k3"))
    assert rule.bidirectional is False
    assert rule.rate_constants == ["k1"]


def test_rule_init_without_rate_constants_raises_parse_error():
    with pytest.raises(BNGParseError, match="1 or 2 rate constants"):
        Rule(
            name="r",
            reactants=[FakePattern("A()")],
            products=[FakePattern("B()")],
        )


def test_rulemod_invalid_init_raises_parse_error():
    with pytest.raises(BNGParseError, match="Rule modifier type InvalidMod"):
        RuleMod(mod_type="InvalidMod")


def test_rulemod_invalid_setter_raises_parse_error():
    rule_mod = RuleMod()
    with pytest.raises(BNGParseError, match="Rule modifier type BadType"):
        rule_mod.type = "BadType"
    assert rule_mod.type is None
