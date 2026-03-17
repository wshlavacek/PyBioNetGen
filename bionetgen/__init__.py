from .core.defaults import defaults
from .modelapi import bngmodel
from .modelapi.runner import run
from .simulator import sim_getter

# sympy is an expensive dependency to import. We delay importing the
# SympyOdes helpers until they are actually accessed.

__all__ = [
    "defaults",
    "bngmodel",
    "run",
    "sim_getter",
    "SympyOdes",
    "export_sympy_odes",
]


def __getattr__(name):
    if name in {"SympyOdes", "export_sympy_odes"}:
        from .modelapi.sympy_odes import SympyOdes, export_sympy_odes

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
