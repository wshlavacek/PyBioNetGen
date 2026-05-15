"""Focused tests for block deletion and invalid-action error contracts."""

import pytest

from bionetgen.core.exc import BNGParseError
from bionetgen.modelapi.blocks import ActionBlock, ModelBlock
from bionetgen.network.blocks import NetworkBlock


def test_model_block_delitem_missing_raises_keyerror():
    block = ModelBlock()

    with pytest.raises(KeyError, match="missing"):
        del block["missing"]


def test_network_block_delitem_missing_raises_keyerror():
    block = NetworkBlock()

    with pytest.raises(KeyError, match="missing"):
        del block["missing"]


def test_action_block_delitem_missing_raises_indexerror():
    block = ActionBlock()

    with pytest.raises(IndexError):
        del block[99]


def test_action_block_add_action_invalid_type_raises_parse_error():
    block = ActionBlock()

    with pytest.raises(
        BNGParseError, match="Action type not_a_real_action not recognized!"
    ):
        block.add_action("not_a_real_action", {})

    assert len(block.items) == 0
