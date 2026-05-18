"""In-process BNGsim driver for the ``parameter_scan`` action.

A fast-path optimization layered over the BNG2.pl backend-hook route.
For a ``generate_network`` followed by a single ``ode`` ``parameter_scan``,
BNG2.pl generates the network once and this driver then owns the scan
loop, driving BNGsim *in-process*: build the model once, vary the
scanned parameter, and re-integrate. That avoids the N process / socket
/ JSON boundary crossings the backend-hook route pays per scan point.

Correctness never depends on this path. :func:`detect_inprocess_scan`
returns ``None`` for anything it cannot handle conservatively, and
``run_bngl_with_bngsim`` falls back to the backend-hook route on a
``None`` decision or on any exception raised here.

Scope (Phase 1): ``ode``/``cvode`` ``parameter_scan`` only, with a
``par_min``/``par_max``/``n_scan_pts`` value range and a trivial action
sequence (``generate_network`` + a single trailing ``parameter_scan``,
optionally preceded by ``setParameter``). ``ssa``/``nf``/``rm`` scans,
``bifurcate``, ``par_scan_vals``, ``sample_times``, ``steady_state``,
``reset_conc=>0`` and non-trivial sequences all fall back.
"""

import logging
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("bionetgen.bngsim_bridge")

# parameter_scan arg keys this driver knows how to honor in-process.
_SCAN_SUPPORTED_KEYS = frozenset({
    "parameter", "par_min", "par_max", "n_scan_pts", "log_scale",
    "method", "t_start", "t_end", "n_steps", "suffix", "prefix",
    "reset_conc", "atol", "rtol", "print_CDAT",
})
# Keys that are recognized but not handled in-process: their presence
# (when meaningful) forces a fallback rather than a silent wrong answer.
_SCAN_FALLBACK_KEYS = frozenset({
    "par_scan_vals", "sample_times", "steady_state", "continue",
    "print_functions",
})
# Keys that are recognized and safe to ignore (output is unaffected).
_SCAN_IGNORED_KEYS = frozenset({
    "parallel", "num_cores", "verbose", "get_final_state",
})

# Action types allowed in a fast-path scan sequence.
_SCAN_ALLOWED_ACTIONS = frozenset({
    "generate_network", "parameter_scan", "setParameter",
})

_ODE_METHODS = frozenset({"ode", "cvode"})

# A backslash that is NOT a clean end-of-line continuation. BNGL line
# continuation is ``\`` at end of line; a ``\`` followed by anything else
# (e.g. ``101,\log_scale=>1``) is malformed. PyBioNetGen's action parser
# silently absorbs such a stray ``\`` while BNG2.pl treats the token after
# it as a differently-named (unrecognized) key -- so the two parsers
# disagree on the action's meaning. The fast path defers to BNG2.pl by
# declining whenever the scan action text carries one.
_BAD_BACKSLASH_RE = re.compile(r"\\(?![ \t]*\r?\n)")

_PARAMETER_SCAN_RE = re.compile(
    r"\bparameter_scan\s*\(\s*\{[^}]*\}\s*\)", re.DOTALL | re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"#.*")


def _scan_action_text_is_clean(bngl_text):
    """True if the ``parameter_scan`` action text has no parser ambiguity.

    Returns ``False`` when the action carries a stray (non-line-continuation)
    backslash, which PyBioNetGen and BNG2.pl parse differently — the fast
    path then declines so BNG2.pl's interpretation governs.
    """
    text = _COMMENT_RE.sub("", bngl_text)
    match = _PARAMETER_SCAN_RE.search(text)
    if match is None:
        return False
    return _BAD_BACKSLASH_RE.search(match.group(0)) is None


@dataclass(frozen=True)
class ScanRequest:
    """A parameter_scan reduced to the inputs the in-process driver needs."""

    parameter: str
    par_min: float
    par_max: float
    n_scan_pts: int
    log_scale: bool
    t_start: float
    t_end: float
    n_steps: int
    suffix: str | None
    prefix: str | None
    reset_conc: bool
    atol: float | None
    rtol: float | None
    print_cdat: bool


def _unquote(value):
    """Strip one layer of matching single/double quotes and whitespace."""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _as_float(value):
    """Parse a BNGL numeric action argument as a float (raises on failure)."""
    return float(_unquote(value))


def _as_truthy(value):
    """Parse a BNGL boolean-ish action argument (``0``/``1``)."""
    return _as_float(value) != 0.0


def detect_inprocess_scan(actions_items, bngl_text=None):
    """Classify whether a BNGL action list is an in-process-scan fast path.

    Returns a :class:`ScanRequest` when the action sequence is exactly a
    ``generate_network`` plus a single trailing ``ode`` ``parameter_scan``
    (an optional ``setParameter`` preamble is allowed), and the scan uses
    only options this driver honors. Returns ``None`` for anything else —
    the caller then uses the backend-hook route, which stays correct.

    When ``bngl_text`` (the raw BNGL source) is supplied, the scan action
    text is also checked for parser ambiguity (a stray backslash); an
    ambiguous action declines so BNG2.pl's reading governs.
    """
    if not actions_items:
        return None

    if bngl_text is not None and not _scan_action_text_is_clean(bngl_text):
        logger.debug("scan fast path declined: ambiguous backslash in action")
        return None

    types = [getattr(a, "type", None) for a in actions_items]
    if any(t not in _SCAN_ALLOWED_ACTIONS for t in types):
        return None
    if types.count("parameter_scan") != 1:
        return None
    if types[-1] != "parameter_scan":
        return None
    if "generate_network" not in types:
        return None
    if types.index("generate_network") > types.index("parameter_scan"):
        return None

    scan_action = actions_items[types.index("parameter_scan")]
    args = {k: v for k, v in (getattr(scan_action, "args", None) or {}).items()}

    for key in args:
        if key not in _SCAN_SUPPORTED_KEYS and key not in _SCAN_FALLBACK_KEYS \
                and key not in _SCAN_IGNORED_KEYS:
            logger.debug("scan fast path declined: unknown option %r", key)
            return None

    try:
        # Options that, when present and meaningful, are out of Phase 1 scope.
        if "par_scan_vals" in args:
            return None
        if "sample_times" in args:
            return None
        if "continue" in args and _as_truthy(args["continue"]):
            return None
        if "steady_state" in args and _as_truthy(args["steady_state"]):
            return None
        if "print_functions" in args and _as_truthy(args["print_functions"]):
            return None

        method = _unquote(args.get("method", "ode")).lower()
        if method not in _ODE_METHODS:
            return None

        reset_conc = True
        if "reset_conc" in args:
            reset_conc = _as_truthy(args["reset_conc"])
        if not reset_conc:
            # reset_conc=>0 chains each point from the prior end state
            # (hysteresis-like); that is Phase 2 bifurcate territory.
            return None

        for required in ("parameter", "par_min", "par_max", "n_scan_pts",
                         "t_end", "n_steps"):
            if required not in args:
                return None

        parameter = _unquote(args["parameter"])
        if not parameter:
            return None
        par_min = _as_float(args["par_min"])
        par_max = _as_float(args["par_max"])
        n_scan_pts = int(_as_float(args["n_scan_pts"]))
        log_scale = _as_truthy(args["log_scale"]) if "log_scale" in args else False
        t_start = _as_float(args["t_start"]) if "t_start" in args else 0.0
        t_end = _as_float(args["t_end"])
        n_steps = int(_as_float(args["n_steps"]))
        suffix = _unquote(args["suffix"]) if "suffix" in args else None
        prefix = _unquote(args["prefix"]) if "prefix" in args else None
        atol = _as_float(args["atol"]) if "atol" in args else None
        rtol = _as_float(args["rtol"]) if "rtol" in args else None
        print_cdat = True
        if "print_CDAT" in args:
            print_cdat = _as_truthy(args["print_CDAT"])
    except (ValueError, TypeError) as exc:
        logger.debug("scan fast path declined: unparseable option (%s)", exc)
        return None

    # Range sanity — mirror BNG2.pl's parameter_scan checks.
    if n_scan_pts < 1:
        return None
    if par_max != par_min and n_scan_pts <= 1:
        return None
    if log_scale and (par_min <= 0.0 or par_max <= 0.0):
        return None
    if n_steps < 1 or t_end <= t_start:
        return None

    return ScanRequest(
        parameter=parameter,
        par_min=par_min,
        par_max=par_max,
        n_scan_pts=n_scan_pts,
        log_scale=log_scale,
        t_start=t_start,
        t_end=t_end,
        n_steps=n_steps,
        suffix=suffix,
        prefix=prefix,
        reset_conc=reset_conc,
        atol=atol,
        rtol=rtol,
        print_cdat=print_cdat,
    )


def scan_values(request):
    """Compute the N scanned parameter values.

    Matches BNG2.pl's ``parameter_scan``: linear spacing, or geometric
    spacing (uniform in ``log``) when ``log_scale`` is set. Endpoints are
    inclusive.
    """
    n = request.n_scan_pts
    lo, hi = request.par_min, request.par_max
    if request.log_scale:
        lo, hi = math.log(lo), math.log(hi)
    if n == 1:
        return [math.exp(lo) if request.log_scale else lo]
    delta = (hi - lo) / (n - 1)
    out = []
    for k in range(n):
        v = lo + k * delta
        out.append(math.exp(v) if request.log_scale else v)
    return out


def _make_network_gen_bngl(bngl_path, model_name, work_dir):
    """Write a temp BNGL whose actions stop at ``generate_network``.

    BNGL comments are stripped (harmless for network generation) and the
    trailing ``parameter_scan`` action is removed, leaving the model
    definition plus its ``generate_network`` (and any ``setParameter``
    preamble). The copy keeps the model basename so BNG2.pl emits
    ``<model_name>.net``.
    """
    with open(bngl_path, "r", errors="replace") as fh:
        text = fh.read()
    text = _COMMENT_RE.sub("", text)
    text, n_sub = _PARAMETER_SCAN_RE.subn("", text)
    if n_sub != 1:
        raise ValueError(
            f"expected exactly one parameter_scan action, found {n_sub}"
        )
    gen_path = os.path.join(work_dir, f"{model_name}.bngl")
    with open(gen_path, "w") as fh:
        fh.write(text)
    return gen_path


def _parse_net_initial_concentrations(net_path):
    """Return the ``.net`` species block as ordered ``(pattern, init_token)``.

    Each ``.net`` species line is ``<idx> <pattern> <init>`` where ``init``
    is either a literal number or a single parameter name (BNG2.pl emits
    synthesized ``_InitialConcN`` params for compound expressions). The
    list is ordered to match BNGsim's ``species_names`` index for index.
    """
    rows = []
    in_species = False
    with open(net_path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == "begin species":
                in_species = True
                continue
            if s == "end species":
                break
            if in_species:
                parts = s.split()
                if len(parts) >= 3:
                    rows.append((parts[1], parts[2]))
    return rows


def _is_literal_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _write_scan_file(scan_path, parameter, observable_names, rows):
    """Write a BNG2.pl-format ``.scan`` file.

    Mirrors BNGAction.pm's ``parameter_scan`` writer: a ``# <param> <obs>
    ...`` header followed by one ``%16.8e``-formatted row per scan point
    (the parameter value, then each observable at ``t_end``).
    """
    with open(scan_path, "w") as fh:
        header = "# " + f"{parameter:>14}"
        for name in observable_names:
            header += " " + f"{name:>16}"
        fh.write(header + "\n")
        for par_value, obs in rows:
            line = f"{par_value:16.8e}"
            for x in obs:
                line += " " + f"{x:16.8e}"
            fh.write(line + "\n")


def run_parameter_scan_with_bngsim(
    bngl_path,
    output_dir,
    bngpath,
    request,
    model_name,
    suppress=False,
    log_file=None,
    timeout=None,
    app=None,
):
    """Run an ``ode`` ``parameter_scan`` in-process through BNGsim.

    BNG2.pl generates the reaction network once; this driver then loops
    over the scan values in-process — building the BNGsim model once and
    re-integrating per point — and writes BNG2.pl-compatible output
    (``<basename>.scan`` plus per-point ``.gdat``/``.cdat`` files under
    ``<basename>/``).

    Raises on any failure so the caller can fall back to the backend hook.
    """
    import bngsim

    from bionetgen.core.tools.bngsim_bridge import (
        _run_bngl_subprocess,
        _write_bngsim_results,
        _make_bng_result,
    )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    basename = (request.prefix or model_name)
    basename += "_" + (request.suffix or request.parameter)
    work_dir = os.path.join(output_dir, basename)
    scan_path = os.path.join(output_dir, basename + ".scan")

    gen_dir = tempfile.mkdtemp(prefix="bngsim_scan_gen_")
    try:
        gen_bngl = _make_network_gen_bngl(bngl_path, model_name, gen_dir)
        _run_bngl_subprocess(
            gen_bngl, gen_dir, bngpath,
            suppress=suppress, log_file=log_file, timeout=timeout, app=app,
        )
        net_path = os.path.join(gen_dir, f"{model_name}.net")
        if not os.path.isfile(net_path):
            raise FileNotFoundError(
                f"network generation produced no {model_name}.net"
            )

        init_tokens = _parse_net_initial_concentrations(net_path)
        model = bngsim.Model.from_net(net_path)
        sim = bngsim.Simulator(model)

        species_names = list(model.species_names)
        if len(init_tokens) != len(species_names):
            raise ValueError(
                "species count mismatch between .net "
                f"({len(init_tokens)}) and BNGsim model ({len(species_names)})"
            )
        # Species whose initial concentration is a parameter expression:
        # BNGsim freezes init concentrations as literals at load time and
        # reset() does not re-derive them, so the driver must re-apply
        # them per scan point from the (now updated) parameter value.
        param_linked = [
            (species_names[i], token)
            for i, (_pat, token) in enumerate(init_tokens)
            if not _is_literal_number(token)
        ]

        if request.parameter not in model.param_names:
            raise ValueError(
                f"scanned parameter {request.parameter!r} is not a "
                "model parameter in the generated network"
            )

        values = scan_values(request)
        run_kwargs = {}
        if request.atol is not None:
            run_kwargs["atol"] = request.atol
        if request.rtol is not None:
            run_kwargs["rtol"] = request.rtol

        os.makedirs(work_dir, exist_ok=True)
        observable_names = list(model.observable_names)
        scan_rows = []
        for k, value in enumerate(values):
            model.set_param(request.parameter, value)
            model.reset()
            for sp_name, token in param_linked:
                model.set_concentration(sp_name, model.get_param(token))
            result = sim.run(
                t_span=(request.t_start, request.t_end),
                n_points=request.n_steps + 1,
                **run_kwargs,
            )
            point_name = f"{basename}_{k + 1:05d}"
            _write_bngsim_results(
                result, work_dir, point_name, print_cdat=request.print_cdat,
            )
            scan_rows.append((value, list(result.observables[-1, :])))

        _write_scan_file(scan_path, request.parameter, observable_names, scan_rows)
        logger.info(
            "parameter_scan fast path: %d points for %r via in-process BNGsim",
            len(values), request.parameter,
        )
        return _make_bng_result(output_dir, "ode")
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
