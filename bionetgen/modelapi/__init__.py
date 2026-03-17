from .model import bngmodel

__all__ = ["bngmodel", "SympyOdes", "export_sympy_odes", "extract_odes_from_mexfile"]


def __getattr__(name):
    if name in {"SympyOdes", "export_sympy_odes", "extract_odes_from_mexfile"}:
        from .sympy_odes import SympyOdes, export_sympy_odes, extract_odes_from_mexfile

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
