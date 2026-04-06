from .core.defaults import defaults
from .core.tools.bngsim_bridge import BNGSIM_AVAILABLE, BNGSIM_VERSION
from .modelapi import bngmodel
from .modelapi.runner import run
from .simulator import sim_getter

# sympy is an expensive dependency to import. We delay importing the
# SympyOdes helpers until they are actually accessed.

__all__ = [
    "BNGSIM_AVAILABLE",
    "BNGSIM_VERSION",
    "SympyOdes",
    "bngmodel",
    "defaults",
    "export_sympy_odes",
    "run",
    "sim_getter",
]


def __getattr__(name):
    if name in {"SympyOdes", "export_sympy_odes"}:
        from .modelapi.sympy_odes import SympyOdes, export_sympy_odes  # noqa: F401

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
