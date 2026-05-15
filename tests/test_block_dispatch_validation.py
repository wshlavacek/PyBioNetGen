"""Focused tests for model and network block dispatch validation."""

import pytest

from bionetgen.modelapi.blocks import (
    ActionBlock,
    CompartmentBlock,
    EnergyPatternBlock,
    FunctionBlock,
    MoleculeTypeBlock,
    ObservableBlock,
    ParameterBlock,
    PopulationMapBlock,
    ProtocolBlock,
    RuleBlock,
    SpeciesBlock,
)
from bionetgen.network.blocks import (
    NetworkGroupBlock,
    NetworkParameterBlock,
    NetworkReactionBlock,
    NetworkSpeciesBlock,
)


def _make_model_bypass_init():
    from bionetgen.modelapi.model import bngmodel

    model = object.__new__(bngmodel)
    model.active_blocks = []
    model._block_order = [
        "parameters",
        "compartments",
        "molecule_types",
        "species",
        "observables",
        "functions",
        "energy_patterns",
        "population_maps",
        "rules",
        "protocol",
        "actions",
    ]
    model.model_name = "test_model"
    model.model_path = "/fake/test.bngl"
    model.parameters = ParameterBlock()
    model.compartments = CompartmentBlock()
    model.molecule_types = MoleculeTypeBlock()
    model.species = SpeciesBlock()
    model.observables = ObservableBlock()
    model.functions = FunctionBlock()
    model.energy_patterns = EnergyPatternBlock()
    model.population_maps = PopulationMapBlock()
    model.rules = RuleBlock()
    model.protocol = ProtocolBlock()
    model.actions = ActionBlock()
    return model


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


@pytest.mark.parametrize(
    ("block_cls", "attr_name"),
    [
        (ParameterBlock, "parameters"),
        (RuleBlock, "rules"),
        (ProtocolBlock, "protocol"),
    ],
)
def test_model_add_block_dispatches_supported_block(block_cls, attr_name):
    model = _make_model_bypass_init()
    block = block_cls()

    model.add_block(block)

    assert getattr(model, attr_name) is block
    assert attr_name in model.active_blocks


@pytest.mark.parametrize(
    ("block_name", "attr_name", "block_cls"),
    [
        ("observables", "observables", ObservableBlock),
        ("reaction_rules", "rules", RuleBlock),
        ("protocol", "protocol", ProtocolBlock),
    ],
)
def test_model_add_empty_block_dispatches_supported_name(
    block_name, attr_name, block_cls
):
    model = _make_model_bypass_init()
    delattr(model, attr_name)

    model.add_empty_block(block_name)

    assert isinstance(getattr(model, attr_name), block_cls)


def test_model_add_block_invalid_name_raises_value_error():
    model = _make_model_bypass_init()

    class FakeBlock:
        name = "not a block"

    with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
        model.add_block(FakeBlock())

    assert "not_a_block" not in model.active_blocks
    assert not hasattr(model, "not_a_block")


def test_model_add_empty_block_invalid_name_raises_value_error():
    model = _make_model_bypass_init()

    with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
        model.add_empty_block("not a block")

    assert "not_a_block" not in model.active_blocks
    assert not hasattr(model, "not_a_block")


@pytest.mark.parametrize(
    ("block_cls", "attr_name"),
    [
        (NetworkParameterBlock, "parameters"),
        (NetworkSpeciesBlock, "species"),
        (NetworkReactionBlock, "reactions"),
        (NetworkGroupBlock, "groups"),
    ],
)
def test_network_add_block_dispatches_supported_block(block_cls, attr_name):
    net = _make_network_bypass_init()
    block = block_cls()

    net.add_block(block)

    assert getattr(net, attr_name) is block
    assert attr_name in net.active_blocks


def test_network_add_empty_block_dispatches_supported_name():
    net = _make_network_bypass_init()
    delattr(net, "groups")

    net.add_empty_block("groups")

    assert isinstance(net.groups, NetworkGroupBlock)


def test_network_add_block_invalid_name_raises_value_error():
    net = _make_network_bypass_init()

    class FakeBlock:
        name = "not a block"

    with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
        net.add_block(FakeBlock())

    assert "not_a_block" not in net.active_blocks
    assert not hasattr(net, "not_a_block")


def test_network_add_empty_block_invalid_name_raises_value_error():
    net = _make_network_bypass_init()

    with pytest.raises(ValueError, match="Unsupported block name 'not a block'"):
        net.add_empty_block("not a block")

    assert "not_a_block" not in net.active_blocks
    assert not hasattr(net, "not_a_block")
