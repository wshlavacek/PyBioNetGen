"""Focused warning-path tests for block objects."""

from unittest.mock import patch

import pytest

from bionetgen.modelapi.blocks import (
    CompartmentBlock,
    ModelBlock,
    ObservableBlock,
    ParameterBlock,
    SpeciesBlock,
)
from bionetgen.modelapi.structs import Species
from bionetgen.network.blocks import (
    NetworkBlock,
    NetworkCompartmentBlock,
    NetworkGroupBlock,
    NetworkParameterBlock,
    NetworkSpeciesBlock,
)
from bionetgen.network.structs import NetworkSpecies


class FakePattern:
    """Minimal mock for Pattern-like objects used by Species and Observable."""

    def __init__(self, name="A()"):
        self.name = name
        self.MatchOnce = False

    def __str__(self):
        return self.name


class ExplodingFloat:
    def __float__(self):
        raise RuntimeError("boom")


def test_model_block_setattr_propagates_unexpected_float_error():
    block = ModelBlock()
    block.items["k1"] = 1.0

    with pytest.raises(RuntimeError, match="boom"):
        block.k1 = ExplodingFloat()

    assert block.items["k1"] == 1.0
    assert "k1" not in block._changes


def test_model_block_add_item_logs_unexpected_setattr_failure():
    from bionetgen.modelapi import blocks as blocks_module

    class BrokenSetattrBlock(ModelBlock):
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
    assert "ModelBlock.add_item()" in warning_kwargs["loc"]
    assert block.items["broken"] == "value"


def test_parameter_block_invalid_numeric_assignment_logs_warning():
    from bionetgen.modelapi import blocks as blocks_module

    block = ParameterBlock()
    block.add_parameter("k1", 0.5)
    block._changes.clear()

    with patch.object(blocks_module, "logger") as mock_logger:
        block.k1 = object()

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set parameter 'k1'" in warning_args[0]
    assert "keeping existing value" in warning_args[0]
    assert "ParameterBlock.__setattr__()" in warning_kwargs["loc"]
    assert block.items["k1"]["value"] == 0.5
    assert len(block._changes) == 0


def test_compartment_block_invalid_numeric_assignment_logs_warning():
    from bionetgen.modelapi import blocks as blocks_module

    block = CompartmentBlock()
    block.add_compartment("EC", 3, 1.0)
    block._changes.clear()

    with patch.object(blocks_module, "logger") as mock_logger:
        block.EC = object()

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set compartment 'EC'" in warning_args[0]
    assert "keeping existing size" in warning_args[0]
    assert "CompartmentBlock.__setattr__()" in warning_kwargs["loc"]
    assert block.items["EC"]["size"] == 1.0
    assert len(block._changes) == 0


def test_observable_block_invalid_type_logs_warning():
    from bionetgen.modelapi import blocks as blocks_module

    block = ObservableBlock()
    block.add_observable("obsA", "Molecules", [FakePattern("A()")])
    block._changes.clear()
    existing_observable = block["obsA"]

    with patch.object(blocks_module, "logger") as mock_logger:
        block.obsA = 42

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set observable 'obsA'" in warning_args[0]
    assert "keeping existing observable" in warning_args[0]
    assert "ObservableBlock.__setattr__()" in warning_kwargs["loc"]
    assert block["obsA"] is existing_observable
    assert len(block._changes) == 0


def test_species_block_invalid_type_logs_warning():
    from bionetgen.modelapi import blocks as blocks_module

    block = SpeciesBlock()
    existing_species = Species(pattern=FakePattern("A()"), count=100)
    block.items["A()"] = existing_species
    block._changes.clear()

    with patch.object(blocks_module, "logger") as mock_logger:
        setattr(block, "A()", 42)

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set species 'A()'" in warning_args[0]
    assert "keeping existing species" in warning_args[0]
    assert "SpeciesBlock.__setattr__()" in warning_kwargs["loc"]
    assert block.items["A()"] is existing_species
    assert len(block._changes) == 0


def test_network_block_setattr_propagates_unexpected_float_error():
    block = NetworkBlock()
    block._changes = {}
    block.items["k1"] = 1.0

    with pytest.raises(RuntimeError, match="boom"):
        block.k1 = ExplodingFloat()

    assert block.items["k1"] == 1.0
    assert "k1" not in block._changes


def test_network_block_add_item_logs_unexpected_setattr_failure():
    from bionetgen.network import blocks as blocks_module

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


def test_network_parameter_block_add_parameter_avoids_warning():
    from bionetgen.network import blocks as blocks_module

    block = NetworkParameterBlock()

    with patch.object(blocks_module, "logger") as mock_logger:
        block.add_parameter(1, "k1", "0.5")

    mock_logger.warning.assert_not_called()


def test_network_parameter_block_invalid_numeric_assignment_logs_warning():
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
    assert block["k1"]["value"] == "0.5"
    assert len(block._changes) == 0


def test_network_compartment_block_invalid_numeric_assignment_logs_warning():
    from bionetgen.network import blocks as blocks_module

    block = NetworkCompartmentBlock()
    block.add_compartment("cytoplasm", 3, "1.0")
    block._changes.clear()

    with patch.object(blocks_module, "logger") as mock_logger:
        block.cytoplasm = object()

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set compartment 'cytoplasm'" in warning_args[0]
    assert "keeping existing size" in warning_args[0]
    assert "NetworkCompartmentBlock.__setattr__()" in warning_kwargs["loc"]
    assert block["cytoplasm"]["size"] == "1.0"
    assert len(block._changes) == 0


def test_network_group_block_invalid_type_logs_warning():
    from bionetgen.network import blocks as blocks_module

    block = NetworkGroupBlock()
    block.add_group(1, "Atot", members=["1"])
    block._changes.clear()
    existing_group = block["Atot"]

    with patch.object(blocks_module, "logger") as mock_logger:
        block.Atot = 42

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set group 'Atot'" in warning_args[0]
    assert "keeping existing group" in warning_args[0]
    assert "NetworkGroupBlock.__setattr__()" in warning_kwargs["loc"]
    assert block["Atot"] is existing_group
    assert len(block._changes) == 0


def test_network_species_block_invalid_type_logs_warning():
    from bionetgen.network import blocks as blocks_module

    block = NetworkSpeciesBlock()
    block._changes = {}
    existing_species = NetworkSpecies(1, "A(b)", count=100)
    block.items["A(b)"] = existing_species

    with patch.object(blocks_module, "logger") as mock_logger:
        setattr(block, "A(b)", 42)

    mock_logger.warning.assert_called_once()
    warning_args, warning_kwargs = mock_logger.warning.call_args
    assert "Unable to set species 'A(b)'" in warning_args[0]
    assert "keeping existing species" in warning_args[0]
    assert "NetworkSpeciesBlock.__setattr__()" in warning_kwargs["loc"]
    assert block["A(b)"] is existing_species
    assert len(block._changes) == 0
