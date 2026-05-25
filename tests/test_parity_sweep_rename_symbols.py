"""Unit tests for parity_sweep.rename_symbols — the per-model identifier
rename applied identically to both stacks (see SYMBOL_RENAMES). Covers
whole-word substitution, substring safety, multi-symbol renames, the no-op
passthrough, and the registry contract.

Motivating case: bngsim's NFsim (ExprTk) reserves the name ``frac``; the
three V19xx endemic-infection models carry a parameter literally named
``frac`` (a scalar coefficient, never a .gdat observable), so renaming
``frac``->``fracsym`` is a semantically-null interim rescue (bngsim #64).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


def _load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "parity_sweep.py"
    spec = importlib.util.spec_from_file_location("parity_sweep", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ps = _load()


def test_whole_word_rename_param_and_uses():
    text = (
        "begin parameters\n  frac 0.5\nend parameters\n"
        "begin reaction rules\n  0 -> I() frac * infection_force()\n"
        "  0 -> A() (1 - frac) * infection_force()\nend reaction rules\n"
    )
    out = ps.rename_symbols(text, {"frac": "fracsym"})
    assert not re.search(r"\bfrac\b", out)
    assert out.count("fracsym") == 3


def test_substring_not_touched():
    # "fractions" contains "frac" but must not be renamed (no word boundary).
    text = "# population fractions S+I+A=1\n  frac 0.5\n"
    out = ps.rename_symbols(text, {"frac": "fracsym"})
    assert "fractions" in out
    assert re.search(r"\bfracsym\b", out)
    assert not re.search(r"\bfrac\b", out)


def test_comment_rename_is_harmless_but_consistent():
    # A bare token in a comment is renamed too (BNG strips comments, so this
    # is harmless); the point is the rename is purely lexical and total.
    text = "# Fraction frac of infected\n  frac 0.8\n"
    out = ps.rename_symbols(text, {"frac": "fracsym"})
    assert "Fraction fracsym of infected" in out


def test_multi_symbol_longest_first():
    # Longer source names are applied first so one rename can't partially
    # eat another. Renaming both 'k' and 'kf' must not corrupt 'kf'.
    text = "k 1.0\nkf 2.0\nA()+B()->C() kf\nA()->B() k\n"
    out = ps.rename_symbols(text, {"k": "ksym", "kf": "kfsym"})
    assert re.search(r"\bkfsym\b", out)
    assert re.search(r"\bksym\b", out)
    # 'kf' must have become 'kfsym', never 'ksymf'
    assert "ksymf" not in out


def test_none_and_empty_passthrough():
    text = "frac 0.5\n"
    assert ps.rename_symbols(text, None) == text
    assert ps.rename_symbols(text, {}) == text


def test_registry_well_formed():
    assert isinstance(ps.SYMBOL_RENAMES, dict)
    for fname, mapping in ps.SYMBOL_RENAMES.items():
        assert fname.endswith(".bngl")
        assert isinstance(mapping, dict) and mapping
        for old, new in mapping.items():
            assert old and new and old != new
    # The three V19xx endemic models are the motivating entries.
    for m in (
        "V1988a_endemic_infection.bngl",
        "V1990_cooke_endemic.bngl",
        "V1990_kemper_endemic.bngl",
    ):
        assert ps.SYMBOL_RENAMES[m] == {"frac": "fracsym"}
