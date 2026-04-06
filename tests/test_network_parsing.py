"""Tests for bionetgen/network/network.py and networkparser.py.

These modules have module-level `app = BioNetGen()` calls, so they get
imported indirectly. We test by creating fixture .net files.
"""


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

    def test_no_active_blocks_warning(self, tmp_path, capsys):
        net_file = tmp_path / "empty.net"
        net_file.write_text("# empty\n")  # file with just a comment
        from bionetgen.network.network import Network
        _net = Network(str(net_file))
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
