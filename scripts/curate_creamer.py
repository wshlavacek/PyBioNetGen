#!/usr/bin/env python3
"""Derive an NFsim-runnable, fast Creamer_2012 parity fixture from the pristine source.

Creamer et al. 2012 is one of the largest rule-based models we have (542
parameters, hundreds of rules, 75 observables) and exercises a lot of BNGL
surface, so it is valuable as a simulator-parity fixture. But the published
model (a) ships with NO actions block, and (b) defines rate constants as
chained parameter EXPRESSIONS (e.g. ``DimEquil2 = Dimkp2/Dimkm2``). NFsim emits
those as composite rate-functions and aborts with "Undefined symbol" — both
BNG2.pl's NFsim and bngsim's NFsim reject it identically (a documented NFsim
limitation, not a bug in either engine).

This script makes it a usable fixture without changing what it tests:
  1. CONSTANT-FOLD every parameter to a numeric literal. The values are
     unchanged (we just pre-evaluate the expression DAG that BNG would compute
     anyway), so NFsim never sees a nested function. Rule/observable/molecule
     structure — the actual test surface — is untouched.
  2. SCALE the seed populations (``*_tot``) by POP_SCALE. The published model
     seeds ~4.8M molecules, which makes a single NFsim run ~80 s; the biology
     is irrelevant here (this is a simulator cross-check, not a reproduction),
     and parity only needs both engines to agree on the SAME system, so a
     smaller population is a faster-but-equally-valid fixture.
  3. APPEND an actions block: a short network-free NFsim run with gml raised to
     the 32-bit max so the (still large) population fits.

Output is committed at tests/models/Creamer_2012.bngl and the parity corpus
symlinks to it. Re-run this script if the source or POP_SCALE changes.
"""
import math
import re
from pathlib import Path

SRC = Path("/Users/wish/Code/RuleHub/Tutorials/NativeTutorials/Creamer2012/Creamer_2012.bngl")
DST = Path(__file__).resolve().parent.parent / "tests" / "models" / "Creamer_2012.bngl"
POP_SCALE = 0.01      # 1/100: ~48k molecules, ~40 s/run, cleaner ensemble stats
TEND = 10
NSTEPS = 20
GML = 2147483647      # 2**31 - 1, NFsim global molecule limit (32-bit)

# Only arithmetic + these names are ever eval'd; no builtins are exposed.
SAFE_ENV = {k: getattr(math, k) for k in
            ("exp", "log", "log10", "log2", "sqrt", "sin", "cos", "tan", "pi", "e")}


def fold_parameters(lines):
    """Return (new_lines, n_folded). Replace each parameter RHS with its value."""
    b = next(i for i, l in enumerate(lines) if re.match(r"\s*begin parameters", l))
    e = next(i for i, l in enumerate(lines) if re.match(r"\s*end parameters", l))
    defs = []          # (line_idx, name)
    pending = {}       # name -> rhs expression
    for i in range(b + 1, e):
        code = lines[i].split("#", 1)[0].strip()
        if not code:
            continue
        m = re.match(r"^([A-Za-z_]\w*)\s*=?\s*(.+)$", code)
        if not m:
            raise ValueError(f"unparsed parameter line {i}: {lines[i]!r}")
        defs.append((i, m.group(1)))
        pending[m.group(1)] = m.group(2).strip()

    vals = {}
    for _ in range(100):                       # iterate the dependency DAG to fixpoint
        progressed = False
        for name, rhs in list(pending.items()):
            try:
                vals[name] = eval(rhs, {"__builtins__": {}}, {**SAFE_ENV, **vals})
            except Exception:
                continue
            del pending[name]
            progressed = True
        if not pending or not progressed:
            break
    if pending:
        raise ValueError(f"could not fold {len(pending)} params: {list(pending)[:5]}")

    out = lines[:]
    for i, name in defs:
        v = vals[name] * POP_SCALE if name.endswith("_tot") else vals[name]
        indent = re.match(r"\s*", lines[i]).group(0)
        out[i] = f"{indent}{name} {v:.15g}"
    return out, len(defs)


def main():
    lines = SRC.read_text().splitlines()
    folded, n = fold_parameters(lines)
    header = (
        "# DERIVED FIXTURE — do not hand-edit; regenerate with scripts/curate_creamer.py\n"
        f"# Source: {SRC}\n"
        f"# Transform: all {n} parameters constant-folded to literals (NFsim cannot\n"
        "#   evaluate the source's chained rate-constant expressions); seed\n"
        f"#   populations (*_tot) scaled x{POP_SCALE}; NFsim actions block appended.\n"
        "# Purpose: simulator-parity test fixture (subprocess BNG2.pl vs bngsim).\n"
    )
    actions = (
        "\nbegin actions\n"
        f'\tsimulate({{method=>"nf",t_end=>{TEND},n_steps=>{NSTEPS},gml=>{GML}}})\n'
        "end actions\n"
    )
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(header + "\n".join(folded) + "\n" + actions)
    print(f"folded {n} params, scaled *_tot x{POP_SCALE}; wrote {DST}")


if __name__ == "__main__":
    main()
