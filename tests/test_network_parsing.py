"""Tests for bionetgen/network/network.py and networkparser.py.

These modules have module-level `app = BioNetGen()` calls, so they get
imported indirectly. We test by creating fixture .net files.
"""

from unittest.mock import patch

import pytest

from bionetgen.core.exc import BNGParseError
from bionetgen.network.blocks import (
    NetworkGroupBlock,
    NetworkParameterBlock,
    NetworkReactionBlock,
    NetworkSpeciesBlock,
)


NET_CONTENT = """\
# NET file
begin parameters
  1 k1 0.1
  2 k2 0.01
end parameters
begin species
  1 A(b) 100
  2 B(a) 200
end species
begin reactions
  1 1,2 3 k1 #Rule1
end reactions
begin groups
  1 Atot 1
  2 Btot 2
end groups
"""


def _make_network_bypass_init():
    from bionetgen.network.network import Network

    net = object.__new__(Network)
    net.active_blocks = []
    net.block_order = ["parameters", "species", "reactions", "groups"]
    net.network_name = "test"
    net.parameters = NetworkParameterBlock()
    net.species = NetworkSpeciesBlock()
    net.reactions = NetworkReactionBlock()
    net.groups = NetworkGroupBlock()
    return net

NET_MINIMAL = """\
# NET file
begin parameters
  1 kf 1.0
end parameters
begin species
  1 X() 50
end species
begin reactions
  1 1 2 kf #R1
end reactions
begin groups
  1 Xtot 1
end groups
"""

NET_MALFORMED_SPECIES = """\
# NET file
begin species
  1 A(b)
end species
"""


class TestBNGNetworkParser:
    def test_parse_full_network(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.networkparser import BNGNetworkParser
        parser = BNGNetworkParser(str(net_file))
        assert parser.network_name == "test"
        assert len(parser.network_lines) > 0

    def test_parse_network_populates_blocks(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        # Should have parameters, species, reactions, groups
        assert hasattr(net, "parameters")
        assert hasattr(net, "species")
        assert hasattr(net, "reactions")
        assert hasattr(net, "groups")

    def test_parameters_parsed(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        assert "k1" in net.parameters
        assert "k2" in net.parameters

    def test_species_parsed(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        assert len(net.species) > 0

    def test_reactions_parsed(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        assert len(net.reactions) > 0

    def test_groups_parsed(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        assert "Atot" in net.groups
        assert "Btot" in net.groups


class TestNetwork:
    def test_str(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        s = str(net)
        assert "begin parameters" in s
        assert "end parameters" in s

    def test_repr(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        assert repr(net) == "test"

    def test_iter(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        blocks = list(net)
        assert len(blocks) > 0

    def test_write_model(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        out_file = tmp_path / "output.net"
        net.write_model(str(out_file))
        assert out_file.exists()
        content = out_file.read_text()
        assert "begin parameters" in content

    def test_add_block(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_MINIMAL)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        # Parameters block already exists
        assert "parameters" in net.active_blocks

    @pytest.mark.parametrize(
        ("block_cls", "attr_name"),
        [
            (NetworkParameterBlock, "parameters"),
            (NetworkSpeciesBlock, "species"),
            (NetworkReactionBlock, "reactions"),
            (NetworkGroupBlock, "groups"),
        ],
    )
    def test_add_block_dispatches_supported_block(self, block_cls, attr_name):
        net = _make_network_bypass_init()
        block = block_cls()

        net.add_block(block)

        assert getattr(net, attr_name) is block
        assert attr_name in net.active_blocks

    def test_add_empty_block_dispatches_supported_name(self):
        net = _make_network_bypass_init()
        delattr(net, "groups")

        net.add_empty_block("groups")

        assert isinstance(net.groups, NetworkGroupBlock)

    def test_add_block_invalid_name_raises_value_error(self):
        net = _make_network_bypass_init()

        class FakeBlock:
            name = "not a block"

        with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
            net.add_block(FakeBlock())

        assert "not_a_block" not in net.active_blocks
        assert not hasattr(net, "not_a_block")

    def test_add_empty_block_invalid_name_raises_value_error(self):
        net = _make_network_bypass_init()

        with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
            net.add_empty_block("not a block")

        assert "not_a_block" not in net.active_blocks
        assert not hasattr(net, "not_a_block")

    def test_network_block_setattr_propagates_unexpected_float_error(self):
        from bionetgen.network.blocks import NetworkBlock

        class ExplodingFloat:
            def __float__(self):
                raise RuntimeError("boom")

        block = NetworkBlock()
        block.items["k1"] = 1.0

        with pytest.raises(RuntimeError, match="boom"):
            block.k1 = ExplodingFloat()

        assert block.items["k1"] == 1.0
        assert "k1" not in block._changes

    def test_network_block_add_item_logs_unexpected_setattr_failure(self):
        from bionetgen.network import blocks as blocks_module
        from bionetgen.network.blocks import NetworkBlock

        class BrokenSetattrBlock(NetworkBlock):
            def __setattr__(self, name, value) -> None:
                if name == "broken" and hasattr(self, "items") and name in self.items:
                    raise RuntimeError("boom")
                super().__setattr__(name, value)

        block = BrokenSetattrBlock()

        with patch.object(blocks_module, "logger") as mock_logger:
            block.add_item(("broken", "value"))

        mock_logger.warning.assert_called_once()
        warning_args, warning_kwargs = mock_logger.warning.call_args
        assert "Unable to bind attribute 'broken'" in warning_args[0]
        assert "Original error: boom" in warning_args[0]
        assert "NetworkBlock.add_item()" in warning_kwargs["loc"]
        assert block.items["broken"] == "value"

    def test_network_parameter_block_invalid_numeric_assignment_logs_warning(self):
        from bionetgen.network import blocks as blocks_module

        block = NetworkParameterBlock()
        block.add_parameter(1, "k1", "0.5")
        block._changes.clear()

        with patch.object(blocks_module, "logger") as mock_logger:
            block.k1 = object()

        mock_logger.warning.assert_called_once()
        warning_args, warning_kwargs = mock_logger.warning.call_args
        assert "Unable to set parameter 'k1'" in warning_args[0]
        assert "keeping existing value" in warning_args[0]
        assert "NetworkParameterBlock.__setattr__()" in warning_kwargs["loc"]
        assert block.items["k1"]["value"] == "0.5"
        assert len(block._changes) == 0

    def test_empty_blocks_added(self, tmp_path):
        net_file = tmp_path / "test.net"
        # Network with only parameters — need header line so line 0 isn't begin
        net_file.write_text("# header\nbegin parameters\n  1 k1 1.0\nend parameters\n")
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        # Species/reactions/groups should exist as empty blocks
        assert hasattr(net, "species")
        assert hasattr(net, "reactions")
        assert hasattr(net, "groups")

    def test_block_activation_deactivation(self, tmp_path):
        net_file = tmp_path / "test.net"
        net_file.write_text(NET_CONTENT)
        from bionetgen.network.network import Network
        net = Network(str(net_file))
        s1 = str(net)
        assert "begin parameters" in s1
        # Active blocks should include parameters
        assert "parameters" in net.active_blocks

    def test_no_active_blocks_warning(self, tmp_path):
        net_file = tmp_path / "empty.net"
        net_file.write_text("# empty\n")  # file with just a comment
        from bionetgen.network import network as network_module

        with patch.object(network_module, "logger") as mock_logger:
            net = network_module.Network(str(net_file))

        mock_logger.warning.assert_called_once()
        warning_args, warning_kwargs = mock_logger.warning.call_args
        assert "No active blocks" in warning_args[0]
        assert "Network.__init__()" in warning_kwargs["loc"]
        assert len(net.active_blocks) == 0

    @pytest.mark.parametrize(
        ("adder_name", "expected_type_name"),
        [
            ("add_parameters_block", "NetworkParameterBlock"),
            ("add_species_block", "NetworkSpeciesBlock"),
            ("add_groups_block", "NetworkGroupBlock"),
            ("add_reactions_block", "NetworkReactionBlock"),
        ],
    )
    def test_add_block_type_validation(self, adder_name, expected_type_name):
        net = _make_network_bypass_init()
        adder = getattr(net, adder_name)

        with pytest.raises(TypeError, match=expected_type_name):
            adder(object())

    def test_malformed_species_line_raises_parse_error(self, tmp_path):
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
