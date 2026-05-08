"""
PyTest Fixtures.
"""

import importlib.util
import os

import pytest
from cement import fs

from bionetgen.core.defaults import BNGDefaults
from bionetgen.core.utils.utils import find_BNG_path


_ATOMIZER_MODULES = ("libsbml", "lxml", "networkx")


def _module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


def _has_bng2():
    search_path = os.environ.get("BNGPATH") or BNGDefaults().bng_path
    _, bngexec = find_BNG_path(search_path)
    return bngexec is not None


@pytest.fixture(scope="function")
def tmp(request):
    """
    Create a `tmp` object that generates a unique temporary directory,
    and file for each test function that requires it
    """
    t = fs.Tmp()
    yield t
    t.remove()


@pytest.fixture(scope="session")
def require_atomizer():
    missing = [name for name in _ATOMIZER_MODULES if not _module_available(name)]
    if missing:
        pytest.skip(
            "requires optional atomizer dependencies; install with "
            f"`bionetgen[atomizer]` (missing: {', '.join(missing)})"
        )


@pytest.fixture(scope="session")
def require_bng2():
    if not _has_bng2():
        pytest.skip(
            "requires BNG2.pl via a vendored bundle, `BNGPATH`, or `PATH`"
        )
