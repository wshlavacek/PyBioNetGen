"""In-process BNGsim driver for the ``parameter_scan`` and ``bifurcate`` actions.

A fast-path optimization layered over the BNG2.pl backend-hook route.
For a ``generate_network`` followed by a single ``parameter_scan`` (or
``bifurcate``), BNG2.pl generates the network once and this driver then
owns the scan loop, driving BNGsim *in-process*: build the model once,
vary the scanned parameter, and re-integrate. That avoids the N process
/ socket / JSON boundary crossings the backend-hook route pays per scan
point.

Correctness never depends on this path. :func:`detect_inprocess_scan`
returns ``None`` for anything it cannot handle conservatively, and
``run_bngl_with_bngsim`` falls back to the backend-hook route on a
``None`` decision or on any exception raised here.

Scope:

* ``parameter_scan`` — ``ode``/``cvode`` or ``ssa`` method, a
  ``par_min``/``par_max``/``n_scan_pts`` value range, ``reset_conc``
  either ``1`` (reset each point) or ``0`` (carry the prior point's end
  state).
* ``bifurcate`` — two ``parameter_scan`` passes (ascending then
  descending, ``reset_conc`` forced to ``0``) merged per-observable into
  ``<prefix>_bifurcation_<obs>.scan`` files.

``print_functions=>1`` is honored: BNGL functions (BNGsim "expressions")
are appended after the observable columns in both the per-point
``.gdat`` and the merged ``.scan``, exactly as BNG2.pl does — and using
the same ``Result.expressions`` the backend-hook route already trusts
for network jobs.

The action sequence must be trivial: ``generate_network`` plus a single
trailing ``parameter_scan``/``bifurcate``, optionally preceded by
``setParameter``. ``nf``/``rm``/``pla`` methods, ``par_scan_vals``,
``sample_times``, ``steady_state`` and non-trivial sequences all fall
back.

For ``ssa`` the driver rounds param-linked initial concentrations to the
nearest integer (matching BNG2.pl's ``run_network``, which simulates
integer molecule counts); see the Phase 2 spike findings.
"""

import logging
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("bionetgen.bngsim_bridge")

# Scan/bifurcate arg keys this driver knows how to honor in-process.
_SCAN_SUPPORTED_KEYS = frozenset(
    {
        "parameter",
        "par_min",
        "par_max",
        "n_scan_pts",
        "log_scale",
        "method",
        "t_start",
        "t_end",
        "n_steps",
        "suffix",
        "prefix",
        "reset_conc",
        "atol",
        "rtol",
        "print_CDAT",
        "print_functions",
        "seed",
    }
)
# Keys that are recognized but not handled in-process: their presence
# (when meaningful) forces a fallback rather than a silent wrong answer.
_SCAN_FALLBACK_KEYS = frozenset(
    {
        "par_scan_vals",
        "sample_times",
        "steady_state",
        "continue",
    }
)
# Keys that are recognized and safe to ignore (output is unaffected).
_SCAN_IGNORED_KEYS = frozenset(
    {
        "parallel",
        "num_cores",
        "verbose",
        "get_final_state",
    }
)

# Action types allowed in a fast-path scan sequence.
_SCAN_ALLOWED_ACTIONS = frozenset(
    {
        "generate_network",
        "parameter_scan",
        "bifurcate",
        "setParameter",
    }
)
# The two workflow actions the driver can drive in-process.
_WORKFLOW_ACTIONS = frozenset({"parameter_scan", "bifurcate"})

_ODE_METHODS = frozenset({"ode", "cvode"})
_SSA_METHODS = frozenset({"ssa"})
_SUPPORTED_METHODS = _ODE_METHODS | _SSA_METHODS

# A backslash that is NOT a clean end-of-line continuation. BNGL line
# continuation is ``\`` at end of line; a ``\`` followed by anything else
# (e.g. ``101,\log_scale=>1``) is malformed. PyBioNetGen's action parser
# silently absorbs such a stray ``\`` while BNG2.pl treats the token after
# it as a differently-named (unrecognized) key -- so the two parsers
# disagree on the action's meaning. The fast path defers to BNG2.pl by
# declining whenever the scan action text carries one.
_BAD_BACKSLASH_RE = re.compile(r"\\(?![ \t]*\r?\n)")

_WORKFLOW_ACTION_RE = re.compile(
    r"\b(?:parameter_scan|bifurcate)\s*\(\s*\{[^}]*\}\s*\)",
    re.DOTALL | re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"#.*")


def _scan_action_text_is_clean(bngl_text):
    """True if the scan/bifurcate action text has no parser ambiguity.

    Returns ``False`` when the action carries a stray (non-line-continuation)
    backslash, which PyBioNetGen and BNG2.pl parse differently — the fast
    path then declines so BNG2.pl's interpretation governs.
    """
    text = _COMMENT_RE.sub("", bngl_text)
    match = _WORKFLOW_ACTION_RE.search(text)
    if match is None:
        return False
    return _BAD_BACKSLASH_RE.search(match.group(0)) is None


@dataclass(frozen=True)
class ScanRequest:
    """A parameter_scan/bifurcate reduced to the in-process driver's inputs."""

    action: str  # "parameter_scan" or "bifurcate"
    parameter: str
    par_min: float
    par_max: float
    n_scan_pts: int
    log_scale: bool
    method: str  # "ode" or "ssa" (normalized; "cvode" -> "ode")
    t_start: float
    t_end: float
    n_steps: int
    suffix: str | None
    prefix: str | None
    reset_conc: bool
    seed: int | None
    atol: float | None
    rtol: float | None
    print_cdat: bool
    print_functions: bool


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
    ``generate_network`` plus a single trailing ``parameter_scan`` or
    ``bifurcate`` (an optional ``setParameter`` preamble is allowed), the
    method is ``ode``/``cvode``/``ssa``, and the action uses only options
    this driver honors. Returns ``None`` for anything else — the caller
    then uses the backend-hook route, which stays correct.

    When ``bngl_text`` (the raw BNGL source) is supplied, the action text
    is also checked for parser ambiguity (a stray backslash); an ambiguous
    action declines so BNG2.pl's reading governs.
    """
    if not actions_items:
        return None

    if bngl_text is not None and not _scan_action_text_is_clean(bngl_text):
        logger.debug("scan fast path declined: ambiguous backslash in action")
        return None

    types = [getattr(a, "type", None) for a in actions_items]
    if any(t not in _SCAN_ALLOWED_ACTIONS for t in types):
        return None
    if sum(1 for t in types if t in _WORKFLOW_ACTIONS) != 1:
        return None
    if types[-1] not in _WORKFLOW_ACTIONS:
        return None
    if "generate_network" not in types:
        return None
    # The workflow action is types[-1] (checked above), so a present
    # generate_network necessarily precedes it.

    scan_action = actions_items[-1]
    action_type = types[-1]
    args = {k: v for k, v in (getattr(scan_action, "args", None) or {}).items()}

    for key in args:
        if (
            key not in _SCAN_SUPPORTED_KEYS
            and key not in _SCAN_FALLBACK_KEYS
            and key not in _SCAN_IGNORED_KEYS
        ):
            logger.debug("scan fast path declined: unknown option %r", key)
            return None

    try:
        # Options that, when present and meaningful, are out of scope.
        if "par_scan_vals" in args:
            return None
        if "sample_times" in args:
            return None
        if "continue" in args and _as_truthy(args["continue"]):
            return None
        if "steady_state" in args and _as_truthy(args["steady_state"]):
            return None

        method = _unquote(args.get("method", "ode")).lower()
        if method not in _SUPPORTED_METHODS:
            return None
        method = "ode" if method in _ODE_METHODS else "ssa"

        # bifurcate always carries each point from the prior end state;
        # for parameter_scan reset_conc defaults to 1 and may be 0.
        if action_type == "bifurcate":
            reset_conc = False
        else:
            reset_conc = True
            if "reset_conc" in args:
                reset_conc = _as_truthy(args["reset_conc"])

        for required in ("parameter", "par_min", "par_max", "n_scan_pts", "t_end", "n_steps"):
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
        seed = int(_as_float(args["seed"])) if "seed" in args else None
        atol = _as_float(args["atol"]) if "atol" in args else None
        rtol = _as_float(args["rtol"]) if "rtol" in args else None
        print_cdat = True
        if "print_CDAT" in args:
            print_cdat = _as_truthy(args["print_CDAT"])
        print_functions = False
        if "print_functions" in args:
            print_functions = _as_truthy(args["print_functions"])
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
        action=action_type,
        parameter=parameter,
        par_min=par_min,
        par_max=par_max,
        n_scan_pts=n_scan_pts,
        log_scale=log_scale,
        method=method,
        t_start=t_start,
        t_end=t_end,
        n_steps=n_steps,
        suffix=suffix,
        prefix=prefix,
        reset_conc=reset_conc,
        seed=seed,
        atol=atol,
        rtol=rtol,
        print_cdat=print_cdat,
        print_functions=print_functions,
    )


def scan_values(request):
    """Compute the N scanned parameter values (ascending par_min→par_max).

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
    trailing ``parameter_scan``/``bifurcate`` action is removed, leaving
    the model definition plus its ``generate_network`` (and any
    ``setParameter`` preamble). The copy keeps the model basename so
    BNG2.pl emits ``<model_name>.net``.
    """
    with open(bngl_path, "r", errors="replace") as fh:
        text = fh.read()
    text = _COMMENT_RE.sub("", text)
    text, n_sub = _WORKFLOW_ACTION_RE.subn("", text)
    if n_sub != 1:
        raise ValueError(f"expected exactly one parameter_scan/bifurcate action, found {n_sub}")
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


def _write_scan_file(scan_path, parameter, column_names, rows):
    """Write a BNG2.pl-format ``.scan`` file.

    Mirrors BNGAction.pm's ``parameter_scan`` writer: a ``# <param> <col>
    ...`` header followed by one ``%16.8e``-formatted row per scan point
    (the parameter value, then each column at ``t_end``). ``column_names``
    is the observables, plus the BNGL functions when ``print_functions``
    is set — matching the per-point ``.gdat`` column set.
    """
    with open(scan_path, "w") as fh:
        header = "# " + f"{parameter:>14}"
        for name in column_names:
            header += " " + f"{name:>16}"
        fh.write(header + "\n")
        for par_value, cols in rows:
            line = f"{par_value:16.8e}"
            for x in cols:
                line += " " + f"{x:16.8e}"
            fh.write(line + "\n")


def _write_bifurcation_file(scan_path, parameter, col_name, fwd_col, bwd_col):
    """Write one BNG2.pl-format ``_bifurcation_<col>.scan`` file.

    Mirrors BNGAction.pm's ``bifurcate`` merge writer: a 3-column file
    ``# <param> <col>_fwd <col>_bwd`` whose rows pair the forward column
    with the backward column reversed onto the same ascending parameter
    axis (``backward[N-1-i]``). ``col_name`` is an observable, or a BNGL
    function when ``print_functions`` is set.
    """
    n = len(fwd_col)
    with open(scan_path, "w") as fh:
        fh.write(
            "# "
            + f"{parameter:>14}"
            + " "
            + f"{col_name + '_fwd':>16}"
            + " "
            + f"{col_name + '_bwd':>16}"
            + "\n"
        )
        for i in range(n):
            par_value, fwd = fwd_col[i]
            bwd = bwd_col[n - 1 - i][1]
            fh.write(f"{par_value:16.8e} {fwd:16.8e} {bwd:16.8e}\n")


def _build_model_and_metadata(net_path, request):
    """Load the BNGsim model from ``.net`` and gather scan metadata.

    Returns ``(model, species_names, observable_names, param_linked)``.
    ``param_linked`` is the list of ``(species_name, init_param_token)``
    pairs whose initial concentration is a parameter expression — BNGsim
    freezes init concentrations as literals at load time and ``reset()``
    does not re-derive them, so the driver must re-apply those per scan
    point from the (updated) parameter value.
    """
    import bngsim

    init_tokens = _parse_net_initial_concentrations(net_path)
    model = bngsim.Model.from_net(net_path)
    species_names = list(model.species_names)
    if len(init_tokens) != len(species_names):
        raise ValueError(
            "species count mismatch between .net "
            f"({len(init_tokens)}) and BNGsim model ({len(species_names)})"
        )
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
    return model, species_names, list(model.observable_names), param_linked


def _run_scan_loop(
    model,
    sim,
    request,
    values,
    species_names,
    param_linked,
    work_dir,
    basename,
    write_results,
):
    """Drive one in-process scan pass.

    For each scanned ``value``: set the parameter; either reset to initial
    concentrations and re-derive param-linked initial concentrations
    (``reset_conc=>1``), or carry the prior point's end state
    (``reset_conc=>0`` / ``bifurcate``); integrate; write the per-point
    ``.gdat``/``.cdat``; collect each output column at ``t_end``.

    Returns ``(scan_rows, expression_names)``. Each ``scan_rows`` entry is
    ``(parameter_value, columns)`` where ``columns`` is the observables at
    ``t_end``, followed by the BNGL functions when ``print_functions`` is
    set. ``expression_names`` is the matching function-column name list
    (empty unless ``print_functions`` and the model has functions).

    ``write_results`` is :func:`bngsim_bridge._write_bngsim_results`,
    passed in to avoid a circular import.
    """
    is_ssa = request.method == "ssa"
    run_kwargs = {}
    if request.method == "ode":
        if request.atol is not None:
            run_kwargs["atol"] = request.atol
        if request.rtol is not None:
            run_kwargs["rtol"] = request.rtol
    if is_ssa and request.seed is not None:
        run_kwargs["seed"] = request.seed

    os.makedirs(work_dir, exist_ok=True)
    scan_rows = []
    expression_names = []
    for k, value in enumerate(values):
        model.set_param(request.parameter, value)
        if request.reset_conc:
            model.reset()
            for sp_name, token in param_linked:
                conc = model.get_param(token)
                if is_ssa:
                    # BNG2.pl's run_network simulates integer molecule
                    # counts; round so bngsim SSA gets the same input
                    # (a fractional count rides the whole trajectory —
                    # bngsim issue #43).
                    conc = math.floor(conc + 0.5)
                model.set_concentration(sp_name, conc)
        # else: species hold the prior scan point's end state, which
        # sim.run already left in the model; the explicit carry-over
        # below keeps that contract robust against future API changes.
        result = sim.run(
            t_span=(request.t_start, request.t_end),
            n_points=request.n_steps + 1,
            **run_kwargs,
        )
        point_name = f"{basename}_{k + 1:05d}"
        write_results(
            result,
            work_dir,
            point_name,
            print_functions=request.print_functions,
            print_cdat=request.print_cdat,
        )
        if not request.reset_conc:
            for i, sp_name in enumerate(species_names):
                model.set_concentration(sp_name, result.species[-1, i])
        # The .scan columns mirror the per-point .gdat: observables at
        # t_end, then the BNGL functions when print_functions is set. The
        # function set is fixed across scan points — capture it once.
        if k == 0 and request.print_functions:
            expression_names = list(result.expression_names)
        columns = list(result.observables[-1, :])
        if expression_names:
            columns += list(result.expressions[-1, :])
        scan_rows.append((value, columns))
    return scan_rows, expression_names


def _generate_network(
    bngl_path, model_name, bngpath, gen_dir, run_subprocess, suppress, log_file, timeout, app
):
    """Run BNG2.pl once to emit ``<model_name>.net``; return its path."""
    gen_bngl = _make_network_gen_bngl(bngl_path, model_name, gen_dir)
    run_subprocess(
        gen_bngl,
        gen_dir,
        bngpath,
        suppress=suppress,
        log_file=log_file,
        timeout=timeout,
        app=app,
    )
    net_path = os.path.join(gen_dir, f"{model_name}.net")
    if not os.path.isfile(net_path):
        raise FileNotFoundError(f"network generation produced no {model_name}.net")
    return net_path


def _new_simulator(model, request):
    """Construct a BNGsim ``Simulator`` for the request's method."""
    import bngsim

    if request.method == "ssa":
        return bngsim.Simulator(model, method="ssa")
    return bngsim.Simulator(model)


def run_inprocess_scan(
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
    """Dispatch an in-process ``parameter_scan`` or ``bifurcate`` run.

    Raises on any failure so the caller can fall back to the backend hook.
    """
    if request.action == "bifurcate":
        return run_bifurcate_with_bngsim(
            bngl_path,
            output_dir,
            bngpath,
            request,
            model_name,
            suppress=suppress,
            log_file=log_file,
            timeout=timeout,
            app=app,
        )
    return run_parameter_scan_with_bngsim(
        bngl_path,
        output_dir,
        bngpath,
        request,
        model_name,
        suppress=suppress,
        log_file=log_file,
        timeout=timeout,
        app=app,
    )


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
    """Run a ``parameter_scan`` in-process through BNGsim.

    BNG2.pl generates the reaction network once; this driver then loops
    over the scan values in-process — building the BNGsim model once and
    re-integrating per point — and writes BNG2.pl-compatible output
    (``<basename>.scan`` plus per-point ``.gdat``/``.cdat`` files under
    ``<basename>/``).

    Raises on any failure so the caller can fall back to the backend hook.
    """
    from bionetgen.core.tools.bngsim_bridge import (
        _run_bngl_subprocess,
        _write_bngsim_results,
        _make_bng_result,
    )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    basename = request.prefix or model_name
    basename += "_" + (request.suffix or request.parameter)
    work_dir = os.path.join(output_dir, basename)
    scan_path = os.path.join(output_dir, basename + ".scan")

    gen_dir = tempfile.mkdtemp(prefix="bngsim_scan_gen_")
    try:
        net_path = _generate_network(
            bngl_path,
            model_name,
            bngpath,
            gen_dir,
            _run_bngl_subprocess,
            suppress,
            log_file,
            timeout,
            app,
        )
        model, species_names, observable_names, param_linked = _build_model_and_metadata(
            net_path, request
        )
        sim = _new_simulator(model, request)

        values = scan_values(request)
        scan_rows, expression_names = _run_scan_loop(
            model,
            sim,
            request,
            values,
            species_names,
            param_linked,
            work_dir,
            basename,
            _write_bngsim_results,
        )
        column_names = observable_names + expression_names
        _write_scan_file(scan_path, request.parameter, column_names, scan_rows)
        logger.info(
            "parameter_scan fast path: %d points for %r via in-process BNGsim (%s)",
            len(values),
            request.parameter,
            request.method,
        )
        return _make_bng_result(output_dir, request.method)
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)


def run_bifurcate_with_bngsim(
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
    """Run a ``bifurcate`` in-process through BNGsim.

    A ``bifurcate`` is two ``parameter_scan`` passes (``reset_conc`` forced
    to ``0``): a forward pass ``par_min→par_max`` then a backward pass
    ``par_max→par_min``, both carrying state across every point *and*
    across the forward→backward boundary (one continuous ``Simulator``).
    The two passes are merged per observable into
    ``<prefix>_bifurcation_<obs>.scan`` files; per-point ``.gdat``/``.cdat``
    artifacts are written under ``<prefix>_forward/`` and
    ``<prefix>_backward/`` (BNG2.pl keeps those, deleting only the
    intermediate ``.scan`` files — which this driver simply never writes).

    Raises on any failure so the caller can fall back to the backend hook.
    """
    from bionetgen.core.tools.bngsim_bridge import (
        _run_bngl_subprocess,
        _write_bngsim_results,
        _make_bng_result,
    )

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prefix = request.prefix or model_name
    if request.suffix:
        prefix += "_" + request.suffix
    fwd_base = prefix + "_forward"
    bwd_base = prefix + "_backward"

    gen_dir = tempfile.mkdtemp(prefix="bngsim_bifurcate_gen_")
    try:
        net_path = _generate_network(
            bngl_path,
            model_name,
            bngpath,
            gen_dir,
            _run_bngl_subprocess,
            suppress,
            log_file,
            timeout,
            app,
        )
        model, species_names, observable_names, param_linked = _build_model_and_metadata(
            net_path, request
        )
        # One Simulator drives both passes so concentrations carry across
        # every point and across the forward→backward boundary.
        sim = _new_simulator(model, request)

        fwd_values = scan_values(request)
        bwd_values = list(reversed(fwd_values))

        fwd_rows, expression_names = _run_scan_loop(
            model,
            sim,
            request,
            fwd_values,
            species_names,
            param_linked,
            os.path.join(output_dir, fwd_base),
            fwd_base,
            _write_bngsim_results,
        )
        bwd_rows, _ = _run_scan_loop(
            model,
            sim,
            request,
            bwd_values,
            species_names,
            param_linked,
            os.path.join(output_dir, bwd_base),
            bwd_base,
            _write_bngsim_results,
        )

        # Merge: one file per output column (observables, then BNGL
        # functions when print_functions is set), backward column reversed
        # onto the ascending parameter axis (BNGAction.pm sub bifurcate).
        column_names = observable_names + expression_names
        for j, col_name in enumerate(column_names):
            fwd_col = [(par, cols[j]) for par, cols in fwd_rows]
            bwd_col = [(par, cols[j]) for par, cols in bwd_rows]
            out_path = os.path.join(output_dir, f"{prefix}_bifurcation_{col_name}.scan")
            _write_bifurcation_file(
                out_path,
                request.parameter,
                col_name,
                fwd_col,
                bwd_col,
            )
        logger.info(
            "bifurcate fast path: %d points for %r via in-process BNGsim (%s)",
            len(fwd_values),
            request.parameter,
            request.method,
        )
        return _make_bng_result(output_dir, request.method)
    finally:
        shutil.rmtree(gen_dir, ignore_errors=True)
