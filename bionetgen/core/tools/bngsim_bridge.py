"""Bridge module for optional BNGsim integration.

BNGsim is a high-performance C++ simulation engine with Python bindings
that can replace run_network and NFsim for in-process simulation.
This module handles availability detection, input format detection,
and routing simulation requests to BNGsim when available.
"""

import concurrent.futures
import logging
import os

from bionetgen.core.exc import BNGFormatError, BNGSimError

logger = logging.getLogger("bionetgen.bngsim_bridge")

# ─── Availability detection ────────────────────────────────────────

try:
    if os.environ.get("BIONETGEN_NO_BNGSIM"):
        raise ImportError("BIONETGEN_NO_BNGSIM is set")
    import bngsim

    BNGSIM_AVAILABLE = True
except ImportError:
    bngsim = None
    BNGSIM_AVAILABLE = False

BNGSIM_HAS_NFSIM = False
if BNGSIM_AVAILABLE:
    try:
        from bngsim._bngsim_core import HAS_NFSIM

        BNGSIM_HAS_NFSIM = bool(HAS_NFSIM)
    except (ImportError, AttributeError):
        BNGSIM_HAS_NFSIM = False

BNGSIM_VERSION = None
if BNGSIM_AVAILABLE:
    BNGSIM_VERSION = getattr(bngsim, "__version__", "unknown")


# ─── Format constants ──────────────────────────────────────────────

FORMAT_BNGL = "bngl"
FORMAT_NET = "net"
FORMAT_SBML = "sbml"
FORMAT_BNG_XML = "bng-xml"
FORMAT_ANTIMONY = "antimony"

VALID_FORMATS = {FORMAT_BNGL, FORMAT_NET, FORMAT_SBML, FORMAT_BNG_XML, FORMAT_ANTIMONY}

# Formats that require BNGsim (no subprocess fallback)
BNGSIM_REQUIRED_FORMATS = {FORMAT_SBML, FORMAT_ANTIMONY}

# Formats that have subprocess fallbacks
FALLBACK_FORMATS = {FORMAT_BNGL, FORMAT_NET, FORMAT_BNG_XML}


# ─── Format detection ──────────────────────────────────────────────


def _sniff_xml_format(file_path):
    """Sniff an XML file to determine if it is SBML or BioNetGen XML.

    Reads the first ~4KB of the file and looks for distinguishing markers.

    Returns
    -------
    str or None
        FORMAT_SBML, FORMAT_BNG_XML, or None if ambiguous.
    """
    try:
        with open(file_path, "r", errors="replace") as f:
            head = f.read(4096)
    except OSError as e:
        raise BNGFormatError(file_path, f"Could not read file for format detection: {e}") from e

    head_lower = head.lower()

    is_sbml = "<sbml" in head_lower or "www.sbml.org" in head_lower
    is_bng = (
        "<listofmoleculetypes" in head_lower
        or "<listofspeciestypes" in head_lower
        # BNG XML typically has a <model> tag inside a <sbml> root but
        # with BNG-specific children. Check for BNG-specific structures.
        or "<listofobservables" in head_lower
        or "bionetgen" in head_lower
    )

    if is_sbml and not is_bng:
        return FORMAT_SBML
    if is_bng and not is_sbml:
        return FORMAT_BNG_XML
    if is_bng and is_sbml:
        # BNG XML can also have an sbml namespace. If BNG-specific tags are
        # present, treat it as BNG XML.
        return FORMAT_BNG_XML
    return None


def detect_input_format(file_path, explicit_format=None):
    """Detect the input file format, optionally validating against an explicit hint.

    Parameters
    ----------
    file_path : str
        Path to the input file.
    explicit_format : str or None
        User-provided format hint (e.g. 'sbml', 'bng-xml', 'net', 'bngl', 'antimony').

    Returns
    -------
    str
        One of the FORMAT_* constants.

    Raises
    ------
    BNGFormatError
        If the format cannot be determined or the explicit hint conflicts
        with auto-detection.
    """
    if explicit_format is not None:
        explicit_format = explicit_format.lower().strip()
        if explicit_format not in VALID_FORMATS:
            raise BNGFormatError(
                file_path,
                f"Unknown format '{explicit_format}'. "
                f"Valid formats: {', '.join(sorted(VALID_FORMATS))}",
            )

    # Extension-based detection
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".bngl":
        detected = FORMAT_BNGL
    elif ext == ".net":
        detected = FORMAT_NET
    elif ext == ".ant":
        detected = FORMAT_ANTIMONY
    elif ext == ".xml":
        detected = _sniff_xml_format(file_path)
    else:
        detected = None

    # Reconcile explicit vs detected
    if explicit_format is not None:
        if detected is not None and explicit_format != detected:
            raise BNGFormatError(
                file_path,
                f"Format conflict: you specified --format={explicit_format} "
                f"but auto-detection suggests '{detected}'. "
                f"Please verify the file and correct the --format flag.",
            )
        return explicit_format

    if detected is None:
        if ext == ".xml":
            raise BNGFormatError(
                file_path,
                "Could not determine whether this XML file is SBML or BioNetGen XML. "
                "Please specify --format=sbml or --format=bng-xml.",
            )
        raise BNGFormatError(
            file_path,
            f"Unrecognized file extension '{ext}'. "
            f"Supported extensions: .bngl, .net, .xml, .ant. "
            f"Or specify --format explicitly.",
        )

    return detected


# ─── BNGsim simulation dispatch ────────────────────────────────────


def _is_nf_method(method):
    """Return True if the method string is a network-free method."""
    return method in ("nf", "nf_reject", "nfsim")


def _normalize_method(method, poplevel=None):
    """Normalize simulation method, matching BNG2.pl conventions.

    BNG2.pl auto-promotes ``method=>"ssa"`` to PSA when ``poplevel`` is
    defined. BNGsim also supports ``method=>"psa"`` directly. This
    function handles both conventions.

    Returns
    -------
    (method, poplevel) : (str, float or None)
    """
    method = method.strip().lower()

    # BNG2.pl compat: ssa + poplevel → psa
    if method == "ssa" and poplevel is not None:
        return "psa", poplevel

    # Direct psa: default poplevel to 100 if not specified (BNG2.pl default)
    if method == "psa":
        if poplevel is None or poplevel <= 1.0:
            poplevel = 100.0
        return "psa", poplevel

    return method, poplevel


def _write_bng_dat(path, time, data_2d, col_names):
    """Write a BNG-format data file (space-separated with # header).

    Parameters
    ----------
    path : str
        Output file path.
    time : numpy.ndarray
        1D array of time values.
    data_2d : numpy.ndarray
        2D array (n_times x n_cols).
    col_names : list of str
        Column names (excluding 'time').
    """

    headers = ["time"] + list(col_names)
    with open(path, "w") as f:
        f.write("# " + "  ".join(f"{h:>18s}" for h in headers) + "\n")
        for i in range(len(time)):
            vals = [time[i]] + [data_2d[i, j] for j in range(data_2d.shape[1])]
            f.write("  ".join(f"{v:22.12e}" for v in vals) + "\n")


def _write_bngsim_results(result, output_dir, model_name, print_functions=False):
    """Write BNGsim Result to .gdat and .cdat files.

    Parameters
    ----------
    result : bngsim.Result
        The simulation result.
    output_dir : str
        Directory to write output files.
    model_name : str
        Base name for output files (without extension).
    print_functions : bool
        If True, include BNGL functions (BNGsim "expressions") in .gdat
        output. Matches BNG2.pl's ``print_functions=>1`` behavior.
        Default False, matching BNG2.pl's default.
    """
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    gdat_path = os.path.join(output_dir, f"{model_name}.gdat")
    cdat_path = os.path.join(output_dir, f"{model_name}.cdat")

    # .cdat: species concentrations
    result.to_cdat(cdat_path)

    # .gdat: observables (from "begin groups"), and optionally
    # BNGL functions (from "begin functions") when print_functions is set.
    # BNGsim stores BNGL functions as "expressions" in its Result object.
    obs_names = list(result.observable_names)
    obs_array = np.asarray(result.observables) if result.n_observables > 0 else np.empty((result.n_times, 0))

    if print_functions:
        core = result._core
        func_names = list(core.expression_names)
        func_array = np.asarray(core.expression_data)
        has_funcs = len(func_names) > 0 and func_array.ndim == 2 and func_array.shape[1] > 0
    else:
        has_funcs = False

    if result.n_observables > 0 or has_funcs:
        if has_funcs:
            combined = np.hstack([obs_array, func_array])
            combined_names = obs_names + func_names
        else:
            combined = obs_array
            combined_names = obs_names
        _write_bng_dat(gdat_path, result.time, combined, combined_names)


def _make_bng_result(output_dir, method):
    """Load a BNGResult from an output directory."""
    from bionetgen.core.tools.result import BNGResult

    bng_result = BNGResult(path=output_dir)
    bng_result.process_return = 0
    bng_result.output = [f"BNGsim simulation completed: method={method}"]
    return bng_result


def run_nfsim(
    xml_path,
    output_dir,
    t_span=None,
    n_points=None,
    seed=None,
    gml=None,
    model_name=None,
    param_overrides=None,
    conc_overrides=None,
):
    """Run a network-free simulation using BNGsim's NfsimSimulator.

    Uses the low-level NfsimSimulator directly with a BioNetGen XML file.
    No .net file or Model object is needed.

    Parameters
    ----------
    xml_path : str
        Path to BioNetGen XML file.
    output_dir : str
        Directory for output files.
    t_span : tuple of (float, float) or None
        Time span (t_start, t_end). Defaults to (0, 100).
    n_points : int or None
        Number of output time points. Defaults to 101.
    seed : int or None
        Random seed. Defaults to 42.
    gml : int or None
        Global molecule limit.
    model_name : str or None
        Base name for output files. Derived from xml_path if None.
    param_overrides : dict or None
        Parameter name → value overrides to apply via
        ``NfsimSimulator.set_param()`` before initialization.
        Used to propagate ``setParameter`` calls to NFsim.
    conc_overrides : dict or None
        Species pattern → absolute molecule count overrides to apply
        after initialization via ``NfsimSimulator.add_molecules()``.
        Used to propagate ``setConcentration``/``addConcentration``
        calls to NFsim.

    Returns
    -------
    BNGResult
    """
    if not BNGSIM_AVAILABLE:
        raise BNGSimError("BNGsim is required for NFsim but is not installed.")
    if not BNGSIM_HAS_NFSIM:
        raise BNGSimError(
            "BNGsim NFsim support is not available in this build. "
            "Rebuild bngsim with -DBNGSIM_BUILD_NFSIM=ON."
        )

    if t_span is None:
        t_span = (0.0, 100.0)
    if n_points is None:
        n_points = 101
    if seed is None:
        seed = 42

    xml_path = os.path.abspath(xml_path)
    output_dir = os.path.abspath(output_dir)
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(xml_path))[0]

    try:
        from bngsim._bngsim_core import NfsimSimulator

        nfsim = NfsimSimulator(xml_path)
        if gml is not None:
            nfsim.set_molecule_limit(int(gml))

        # Apply parameter overrides from setParameter actions
        if param_overrides:
            for pname, pval in param_overrides.items():
                try:
                    nfsim.set_param(pname, float(pval))
                except Exception:
                    pass  # param may not exist in NFsim model

        nfsim.initialize(seed)

        # Apply concentration overrides from setConcentration/addConcentration.
        # Must happen after initialize() so molecule counts are available.
        if conc_overrides:
            for species_pattern, target_count in conc_overrides.items():
                mol_type = species_pattern.split("(")[0]
                try:
                    current = nfsim.get_molecule_count(mol_type)
                    to_add = target_count - current
                    if to_add > 0:
                        nfsim.add_molecules(mol_type, to_add)
                    elif to_add < 0:
                        logger.warning(
                            "NFsim: cannot decrease %s from %d to %d; "
                            "leaving count unchanged",
                            mol_type, current, target_count,
                        )
                except Exception as e:
                    logger.warning(
                        "NFsim: conc override for %s failed: %s",
                        species_pattern, e,
                    )

        core_result = nfsim.simulate(t_span[0], t_span[1], n_points)
        result = bngsim.Result(core_result)

        _write_bngsim_results(result, output_dir, model_name)

        try:
            nfsim.destroy_session()
        except Exception:
            pass

        return _make_bng_result(output_dir, method="nf")

    except Exception as e:
        if isinstance(e, (BNGSimError, BNGFormatError)):
            raise
        raise BNGSimError(f"BNGsim NFsim simulation failed: {e}") from e


def run_with_bngsim(
    input_path,
    output_dir,
    fmt=None,
    method="ode",
    t_span=None,
    n_points=None,
    **sim_kwargs,
):
    """Run a simulation using BNGsim.

    This handles .net, SBML .xml, BNG .xml, and .ant files directly.
    For .bngl files, use run_bngl_with_bngsim() instead.

    Parameters
    ----------
    input_path : str
        Path to the input file.
    output_dir : str
        Directory for output files.
    fmt : str
        Detected format (one of FORMAT_* constants).
    method : str
        Simulation method ('ode', 'ssa', 'psa', 'nf', etc.).
    t_span : tuple of (float, float) or None
        Time span (t_start, t_end). If None, defaults to (0, 100).
    n_points : int or None
        Number of output time points. If None, defaults to 101.
    **sim_kwargs
        Additional keyword arguments passed to bngsim.Simulator
        (e.g. poplevel for PSA).

    Returns
    -------
    BNGResult
        Result loaded from the written .gdat/.cdat files.

    Raises
    ------
    BNGSimError
        If BNGsim is not available or simulation fails.
    """
    if not BNGSIM_AVAILABLE:
        raise BNGSimError(
            f"BNGsim is required for format '{fmt}' but is not installed. "
            "Install with: pip install bngsim"
        )

    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    model_name = os.path.splitext(os.path.basename(input_path))[0]

    # BNG XML → NFsim path (no Model needed)
    if fmt == FORMAT_BNG_XML:
        if not _is_nf_method(method) and method != "ode":
            raise BNGSimError(
                f"BioNetGen XML files are for network-free simulation, "
                f"but method='{method}' was requested. "
                f"Use method='nf' or provide a .net file for ODE/SSA/PSA."
            )
        return run_nfsim(
            input_path,
            output_dir,
            t_span=t_span,
            n_points=n_points,
            seed=sim_kwargs.pop("seed", None),
            gml=sim_kwargs.pop("gml", None),
            model_name=model_name,
        )

    # Network-based methods: .net, SBML, Antimony
    if _is_nf_method(method):
        # NF with a .net file requires an xml_path kwarg
        xml_path = sim_kwargs.pop("xml_path", None)
        if xml_path:
            return run_nfsim(
                xml_path,
                output_dir,
                t_span=t_span,
                n_points=n_points,
                seed=sim_kwargs.pop("seed", None),
                gml=sim_kwargs.pop("gml", None),
                model_name=model_name,
            )
        raise BNGSimError(
            f"Network-free method '{method}' requires a BioNetGen XML file. "
            "Provide a .xml file or use method='ode'/'ssa'/'psa' with a .net file."
        )

    if t_span is None:
        t_span = (0.0, 100.0)
    if n_points is None:
        n_points = 101

    try:
        # Load model based on format
        if fmt == FORMAT_NET:
            model = bngsim.Model.from_net(input_path)
        elif fmt == FORMAT_SBML:
            model = bngsim.Model.from_sbml(input_path)
        elif fmt == FORMAT_ANTIMONY:
            model = bngsim.Model.from_antimony(input_path)
        else:
            raise BNGSimError(f"Unsupported format for BNGsim: '{fmt}'")

        # Create simulator
        sim = bngsim.Simulator(model, method=method, **sim_kwargs)

        # Run simulation
        result = sim.run(t_span=t_span, n_points=n_points)

        # Write results to files for downstream compatibility
        _write_bngsim_results(result, output_dir, model_name)
        return _make_bng_result(output_dir, method=method)

    except Exception as e:
        if isinstance(e, (BNGSimError, BNGFormatError)):
            raise
        raise BNGSimError(f"BNGsim simulation failed: {e}") from e


# ─── Action parsing helpers ────────────────────────────────────────

# Actions handled by BNG2.pl preprocessing — skip during BNGsim execution
_BNG2PL_ACTIONS = frozenset({
    "generate_network", "generate_hybrid_model",
    "writeXML", "writeSBML", "writeModel", "writeNetwork", "writeFile",
    "writeMfile", "writeCPYfile", "writeMexfile", "writeMDL",
    "readFile", "visualize",
    "setModelName", "substanceUnits", "setOption", "version", "quit",
})

_SIMULATE_METHOD_MAP = {
    "simulate": "ode",
    "simulate_ode": "ode",
    "simulate_ssa": "ssa",
    "simulate_psa": "psa",
    "simulate_nf": "nf",
    "simulate_pla": "pla",
}


def _strip_quotes(s):
    """Strip surrounding single or double quotes from a string."""
    if s and len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _safe_math_namespace(extra=None):
    """Build a safe namespace for evaluating numeric expressions.

    Includes standard math functions (matching BNGL's expression language)
    and optionally extra name-value pairs (e.g., model parameters).
    """
    import math

    # Start from user-supplied parameters, then overlay math builtins so
    # that reserved names (exp, log, sqrt, …) can never be shadowed.
    ns = dict(extra) if extra else {}
    ns.update({
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
        "min": min,
        "max": max,
        "pi": math.pi,
        "_pi": math.pi,
        "_e": math.e,
    })
    ns["__builtins__"] = {}
    return ns


def _eval_numeric(expr_str, extra_ns=None):
    """Safely evaluate a numeric expression string.

    Handles plain floats and arithmetic/math expressions like
    ``((1/52)*50000/0.04)`` or ``exp(k) * 100``. If *extra_ns* is provided,
    those names are available during evaluation (e.g., model parameters).
    """
    try:
        return float(expr_str)
    except (ValueError, TypeError):
        pass
    try:
        ns = _safe_math_namespace(extra_ns)
        return float(eval(expr_str, ns))
    except Exception:
        raise ValueError(f"Cannot evaluate numeric expression: {expr_str!r}") from None


# ─── Species initializer re-evaluation ─────────────────────────────


def _parse_net_species_initializers(net_path):
    """Parse (species_name, init_expression) pairs from a .net file.

    In a .net file, species lines look like::

        1 @b::X(p~0,y) 5000
        2 @b::X(p~1,y) k_init*100

    The third field may be a numeric literal or a parameter expression.
    Returns a list of (species_name, expression_string) tuples.
    """
    import re

    initializers = []
    in_species = False
    pattern = re.compile(r"\s*\d+\s+(\S+)\s+(.+?)\s*$")

    try:
        with open(net_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("begin species"):
                    in_species = True
                    continue
                if stripped.startswith("end species"):
                    break
                if in_species:
                    m = pattern.match(stripped)
                    if m:
                        initializers.append((m.group(1), m.group(2)))
    except OSError:
        pass
    return initializers


def _sync_species_concentrations(bngsim_model, initializers):
    """Re-evaluate species initial concentrations from .net expressions.

    Called after parameter changes so that derived species concentrations
    (e.g., ``S0 = I0 * kfactor``) track parameter updates.
    """
    if not initializers:
        return

    # Build namespace from current model parameters
    param_values = {}
    for pname in bngsim_model.param_names:
        try:
            param_values[pname] = bngsim_model.get_param(pname)
        except Exception:
            pass

    ns = _safe_math_namespace(param_values)

    for species_name, expr_text in initializers:
        try:
            value = float(eval(expr_text, ns))
        except Exception:
            continue
        try:
            bngsim_model.set_concentration(species_name, value)
        except Exception:
            pass

    bngsim_model.save_concentrations()


# ─── Codegen helpers ───────────────────────────────────────────────


def _try_prepare_codegen(net_path):
    """Attempt to compile a code-generated RHS for ODE simulation.

    Returns the path to the compiled shared library, or "" if codegen
    is unavailable or disabled via BIONETGEN_NO_CODEGEN env var.
    """
    if os.environ.get("BIONETGEN_NO_CODEGEN"):
        return ""
    try:
        from bngsim._codegen import prepare_codegen

        so_path = str(prepare_codegen(net_path))
        logger.debug("Codegen compiled: %s", so_path)
        return so_path
    except Exception as e:
        logger.warning("Codegen compilation failed (%s); falling back to interpreted ODE RHS (slower)", e)
        return ""


def _extract_positional_args(action):
    """Extract (name, value) from a no-setter-syntax action.

    The parser stores ``setParameter("kf", 1.0)`` as
    ``args={'"kf"': None, '1.0': None}``.
    Returns (name_str, value_str) with quotes stripped.
    """
    keys = list(action.args.keys())
    name = _strip_quotes(keys[0].strip()) if len(keys) > 0 else ""
    value = _strip_quotes(keys[1].strip()) if len(keys) > 1 else "0"
    return name, value


def _resolve_sample_times(args):
    """Extract and validate sample_times from parsed action args.

    Parameters
    ----------
    args : dict
        Action argument dict. ``sample_times`` is expected to be a string
        like ``"[1,5,10,20,50]"`` (from the BNGL parser's list handling).

    Returns
    -------
    list[float] or None
        Sorted list of sample times, or None if not specified or invalid.
        Returns None if ``n_steps`` or ``n_output_steps`` is also present
        (those take precedence, matching BNG2.pl behavior).
    """
    raw = args.get("sample_times")
    if raw is None:
        return None

    # Parse from string "[1,5,10,20]" to list of floats
    if isinstance(raw, str):
        raw = raw.strip().strip("[]")
        if not raw:
            return None
        try:
            sample_times = sorted(float(v.strip()) for v in raw.split(","))
        except (ValueError, TypeError):
            logger.warning("sample_times: could not parse %r — ignoring", raw)
            return None
    elif isinstance(raw, (list, tuple)):
        sample_times = sorted(float(t) for t in raw)
    else:
        return None

    if len(sample_times) < 3:
        logger.warning(
            "sample_times must contain at least 3 points, got %d — ignoring",
            len(sample_times),
        )
        return None

    # n_steps takes precedence over sample_times (BioNetGen compat)
    if "n_steps" in args or "n_output_steps" in args:
        precedence_key = "n_steps" if "n_steps" in args else "n_output_steps"
        logger.warning(
            "%s and sample_times both defined. %s takes precedence.",
            precedence_key,
            precedence_key,
        )
        return None

    # If t_end is also specified, append it (BioNetGen compat)
    if "t_end" in args:
        t_end = float(args["t_end"])
        if t_end > sample_times[-1]:
            sample_times.append(t_end)

    return sample_times


def _parse_simulate_params(action):
    """Extract simulation parameters from a simulate_* Action.

    Returns dict with all simulation-relevant keys, or None if the
    action type is not a recognized simulate variant.  Applies
    BNG2.pl-compatible method normalization (ssa + poplevel → psa).
    """
    atype = action.type
    args = action.args

    method = _SIMULATE_METHOD_MAP.get(atype)
    if method is None:
        return None
    if atype == "simulate" and "method" in args:
        method = _strip_quotes(args["method"].strip())

    poplevel = float(args["poplevel"]) if "poplevel" in args else None
    method, poplevel = _normalize_method(method, poplevel)

    return {
        "method": method,
        "t_start": float(args.get("t_start", 0)),
        "t_end": float(args.get("t_end", 100)),
        "n_steps": int(float(args.get("n_steps", 100))),
        "suffix": _strip_quotes(args["suffix"].strip()) if "suffix" in args else None,
        "poplevel": poplevel,
        "continue_flag": bool(int(float(args.get("continue", 0)))),
        "atol": float(args["atol"]) if "atol" in args else None,
        "rtol": float(args["rtol"]) if "rtol" in args else None,
        "seed": int(float(args["seed"])) if "seed" in args else None,
        "print_functions": bool(int(float(args.get("print_functions", 0)))),
        "stop_if": _strip_quotes(args["stop_if"].strip()) if "stop_if" in args else None,
        "sample_times": _resolve_sample_times(args),
        "gml": int(float(args["gml"])) if "gml" in args else None,
    }


def _resolve_scan_points(args):
    """Build scan point array from parameter_scan action args."""
    import numpy as np

    par_scan_vals = args.get("par_scan_vals")
    if par_scan_vals is not None:
        raw = par_scan_vals.strip().strip("[]")
        return np.array([float(v.strip()) for v in raw.split(",")])

    par_min = float(args.get("par_min", 0))
    par_max = float(args.get("par_max", 1))
    n_scan_pts = int(float(args.get("n_scan_pts", 10)))
    log_scale = int(float(args.get("log_scale", 0)))

    if log_scale:
        return np.logspace(np.log10(par_min), np.log10(par_max), n_scan_pts)
    return np.linspace(par_min, par_max, n_scan_pts)


def _write_scan_file(scan_path, param_name, col_names, rows):
    """Write a .scan file (same format as .gdat: # header + space-separated data).

    Parameters
    ----------
    scan_path : str
        Output file path.
    param_name : str
        Name of the scanned parameter (first column).
    col_names : list of str
        Column names after the parameter (observables + expressions).
    rows : list of array-like
        One row per scan point.
    """
    headers = [param_name] + list(col_names)
    with open(scan_path, "w") as f:
        f.write("# " + "  ".join(f"{h:>18s}" for h in headers) + "\n")
        for row in rows:
            f.write("  ".join(f"{v:22.12e}" for v in row) + "\n")


def _actions_need_network(actions_items):
    """Return True if any action requires a .net file (network-based simulation)."""
    for a in actions_items:
        if a.type in _SIMULATE_METHOD_MAP:
            sp = _parse_simulate_params(a)
            if sp and not _is_nf_method(sp["method"]):
                return True
        if a.type in ("parameter_scan", "bifurcate"):
            m = _strip_quotes(a.args.get("method", "ode").strip())
            if not _is_nf_method(m):
                return True
    return True  # default: generate network


def _actions_need_xml(actions_items):
    """Return True if any action requires BNG XML (NFsim)."""
    for a in actions_items:
        if a.type in _SIMULATE_METHOD_MAP:
            sp = _parse_simulate_params(a)
            if sp and _is_nf_method(sp["method"]):
                return True
        if a.type == "writeXML":
            return True
        if a.type in ("parameter_scan", "bifurcate"):
            m = _strip_quotes(a.args.get("method", "ode").strip())
            if _is_nf_method(m):
                return True
    return False


# ─── Action executor ───────────────────────────────────────────────


def _scan_result_to_row(result, scan_value, print_functions=False):
    """Extract the final time point from a Result as a scan row.

    Returns (row_array, obs_names, func_names) where row is
    [scan_value, obs1, obs2, ..., func1, func2, ...].

    BNGL functions (BNGsim "expressions") are only included when
    *print_functions* is True, matching BNG2.pl's default behavior.
    """
    import numpy as np

    obs_names = list(result.observable_names)
    obs_array = np.asarray(result.observables)
    final_obs = (
        obs_array[-1, :]
        if obs_array.ndim == 2 and obs_array.shape[0] > 0
        else np.array([])
    )

    func_names = []
    final_funcs = np.array([])
    if print_functions:
        # BNGsim "expressions" = BNGL functions (from "begin functions" block)
        core = result._core
        func_names = list(core.expression_names)
        func_array = np.asarray(core.expression_data)
        if func_array.ndim == 2 and func_array.shape[0] > 0 and func_array.shape[1] > 0:
            final_funcs = func_array[-1, :]

    row = np.concatenate((
        np.array([scan_value], dtype=float),
        np.asarray(final_obs, dtype=float),
        np.asarray(final_funcs, dtype=float),
    ))
    return row, obs_names, func_names


def _run_protocol(
    bngsim_model, protocol_lines, codegen_so="", net_path=None,
):
    """Execute a protocol: a sequence of action lines on a BNGsim model.

    A protocol is a ``begin protocol...end protocol`` block from BNGL that
    contains simulate, setParameter, setConcentration, resetConcentrations,
    and saveConcentrations actions. It is used with
    ``parameter_scan({method=>"protocol", ...})``.

    Parameters
    ----------
    bngsim_model : bngsim.Model
        The model to execute on (typically a clone per scan point).
    protocol_lines : list of str
        Raw action lines from the protocol block.
    codegen_so : str
        Path to codegen shared library, or "" if unavailable.
    net_path : str or None
        Path to .net file (for codegen).

    Returns
    -------
    bngsim.Result or None
        Result from the last simulate action, or None if the protocol
        contains no simulate actions.
    """
    import re

    codegen_kw = {}
    if codegen_so and net_path:
        codegen_kw["codegen"] = True
        codegen_kw["net_path"] = net_path

    sim = bngsim.Simulator(bngsim_model, method="ode", **codegen_kw)
    current_method = "ode"
    current_poplevel = None
    current_time = 0.0
    last_result = None

    # Manual parameter save/restore for protocol context
    saved_params = {}
    for pname in bngsim_model.param_names:
        try:
            saved_params[pname] = bngsim_model.get_param(pname)
        except Exception:
            pass

    # Simple regex parsers for protocol action lines
    _sim_re = re.compile(
        r"simulate(?:_(\w+))?\s*\(\s*\{(.*)\}\s*\)", re.DOTALL
    )
    _setparam_re = re.compile(
        r'setParameter\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)'
    )
    _setconc_re = re.compile(
        r'setConcentration\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)'
    )
    _resetconc_re = re.compile(r"resetConcentrations\s*\(")
    _saveconc_re = re.compile(r"saveConcentrations\s*\(")
    _saveparam_re = re.compile(r"saveParameters\s*\(")
    _resetparam_re = re.compile(r"resetParameters\s*\(")

    def _parse_kvargs(argstr):
        """Parse ``key=>value, key=>value`` into a dict."""
        kv = {}
        for m in re.finditer(r'(\w+)\s*=>\s*(?:"([^"]*)"|(\S+?))\s*(?:,|$)', argstr):
            key = m.group(1)
            val = m.group(2) if m.group(2) is not None else m.group(3)
            kv[key] = val
        return kv

    for raw_line in protocol_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # ── simulate ──
        sm = _sim_re.search(line)
        if sm:
            method_suffix = sm.group(1)  # e.g. "ode" from simulate_ode
            kvargs = _parse_kvargs(sm.group(2))

            if method_suffix:
                method = method_suffix
            else:
                method = kvargs.get("method", "ode")

            is_continue = int(kvargs.get("continue", 0))
            if is_continue and "t_start" not in kvargs:
                t_start = current_time
            else:
                t_start = float(kvargs.get("t_start", 0))
            t_end = float(kvargs.get("t_end", 100))
            n_steps = int(kvargs.get("n_steps", 100))

            # Resolve sample_times
            st_raw = kvargs.get("sample_times")
            sample_times = None
            if st_raw is not None:
                sample_times = _resolve_sample_times({"sample_times": st_raw})

            # Parse poplevel and normalize method (ssa + poplevel → psa)
            poplevel = float(kvargs["poplevel"]) if "poplevel" in kvargs else None
            method, poplevel = _normalize_method(method, poplevel)

            # Rebuild simulator if method changed
            if method == "psa":
                if current_method != "psa" or current_poplevel != poplevel:
                    sim = bngsim.Simulator(
                        bngsim_model, method="psa", poplevel=poplevel
                    )
                    current_method = "psa"
                    current_poplevel = poplevel
            elif current_method != method:
                if method == "ode":
                    sim = bngsim.Simulator(bngsim_model, method="ode", **codegen_kw)
                else:
                    sim = bngsim.Simulator(bngsim_model, method=method)
                current_method = method
                current_poplevel = None

            if sample_times is not None:
                last_result = sim.run(
                    t_span=(sample_times[0], sample_times[-1]),
                    n_points=len(sample_times),
                    sample_times=sample_times,
                )
                current_time = sample_times[-1]
            else:
                last_result = sim.run(
                    t_span=(t_start, t_end),
                    n_points=n_steps + 1,
                )
                current_time = t_end
            continue

        # ── setConcentration ──
        sc = _setconc_re.search(line)
        if sc:
            species_name = sc.group(1)
            conc_str = sc.group(2).strip()
            try:
                bngsim_model.set_concentration(species_name, _eval_numeric(conc_str))
            except Exception:
                logger.warning("protocol: setConcentration(%s, %s) failed", species_name, conc_str)
            continue

        # ── setParameter ──
        sp = _setparam_re.search(line)
        if sp:
            param_name = sp.group(1)
            param_str = sp.group(2).strip()
            try:
                bngsim_model.set_param(param_name, _eval_numeric(param_str))
            except Exception:
                logger.warning("protocol: setParameter(%s, %s) failed", param_name, param_str)
            continue

        # ── resetConcentrations ──
        if _resetconc_re.search(line):
            bngsim_model.reset()
            continue

        # ── saveConcentrations ──
        if _saveconc_re.search(line):
            bngsim_model.save_concentrations()
            continue

        # ── saveParameters ──
        if _saveparam_re.search(line):
            saved_params = {}
            for pname in bngsim_model.param_names:
                try:
                    saved_params[pname] = bngsim_model.get_param(pname)
                except Exception:
                    pass
            continue

        # ── resetParameters ──
        if _resetparam_re.search(line):
            for pname, pval in saved_params.items():
                try:
                    bngsim_model.set_param(pname, pval)
                except Exception:
                    pass
            # Invalidate simulator — params changed
            sim = bngsim.Simulator(bngsim_model, method=current_method, **codegen_kw)
            continue

        logger.debug("protocol: skipping unrecognized command: %s", line)

    return last_result


def _run_nfsim_scan(
    xml_path, action, output_dir, model_name, is_bifurcate=False,
    param_overrides=None,
):
    """Execute a parameter_scan with NFsim: fresh NfsimSimulator per scan point.

    NFsim is stateless (no .net model to clone), so each scan point gets a
    fresh simulator loaded from the BNG XML file.
    """
    from bngsim._bngsim_core import NfsimSimulator

    args = action.args
    param_name = _strip_quotes(args.get("parameter", "").strip())
    t_start = float(args.get("t_start", 0))
    t_end = float(args.get("t_end", 100))
    n_steps = int(float(args.get("n_steps", 100)))
    suffix = _strip_quotes(args.get("suffix", "").strip()) or "scan"
    print_funcs = bool(int(float(args.get("print_functions", 0))))
    gml = int(float(args["gml"])) if "gml" in args else None
    base_seed = int(float(args.get("seed", 42)))

    points = _resolve_scan_points(args)
    rows = []
    obs_names = None
    func_names = None

    for i, value in enumerate(points):
        nfsim = NfsimSimulator(xml_path)
        try:
            if gml is not None:
                nfsim.set_molecule_limit(gml)
            # Apply parameter overrides from prior setParameter actions
            if param_overrides:
                for pname, pval in param_overrides.items():
                    try:
                        nfsim.set_param(pname, float(pval))
                    except Exception:
                        pass
            if param_name:
                try:
                    nfsim.set_param(param_name, float(value))
                except Exception:
                    logger.warning(
                        "NFsim scan: could not set %s=%s", param_name, value
                    )
            nfsim.initialize((base_seed + i) % (2**31))
            core_result = nfsim.simulate(t_start, t_end, n_steps + 1)
            result = bngsim.Result(core_result)

            row, row_obs, row_funcs = _scan_result_to_row(
                result, float(value), print_functions=print_funcs,
            )
            rows.append(row)
            if obs_names is None:
                obs_names = row_obs
                func_names = row_funcs
        finally:
            try:
                nfsim.destroy_session()
            except Exception:
                pass

    col_names = (obs_names or []) + (func_names or [])
    scan_path = os.path.join(output_dir, f"{model_name}_{suffix}.scan")
    _write_scan_file(scan_path, param_name or "scan_param", col_names, rows)


def _prepare_scan_point(base_model, param_name, value, species_initializers):
    """Clone the base model, apply the scan parameter, and refresh initials."""
    point_model = base_model.clone()
    if param_name:
        point_model.set_param(param_name, _eval_numeric(str(value)))
    if species_initializers:
        _sync_species_concentrations(point_model, species_initializers)
    point_model.reset()
    return point_model


def _run_ss_scan_threaded(
    base_model, param_name, points, species_initializers,
    make_sim_fn, codegen_so, net_path, t_start, t_end, print_funcs,
    max_workers=4,
):
    """Run steady-state parameter scan with threaded parallelism.

    Prepares all point models sequentially (species initializer sync is not
    thread-safe), then submits steady_state() calls to a thread pool.
    Falls back to long time-course per point on non-convergence or error.
    """
    n_workers = min(len(points), max_workers)
    rows = []
    obs_names = None
    func_names = None

    # Prepare models and simulators sequentially (not thread-safe)
    point_models = []
    point_sims = []
    for value in points:
        pm = _prepare_scan_point(base_model, param_name, value, species_initializers)
        ps = make_sim_fn(pm)
        point_models.append(pm)
        point_sims.append(ps)

    # Run steady_state() in parallel
    def _solve_ss(idx):
        try:
            ss_result = point_sims[idx].steady_state()
            return (idx, ss_result, None)
        except Exception as exc:
            return (idx, None, exc)

    ss_outcomes = [None] * len(points)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_ss, i): i for i in range(len(points))}
        for fut in concurrent.futures.as_completed(futures):
            idx, ss_result, exc = fut.result()
            ss_outcomes[idx] = (ss_result, exc)

    # Process results and handle fallbacks
    for i, value in enumerate(points):
        ss_result, exc = ss_outcomes[i]
        ss_ok = False

        if exc is not None:
            logger.warning(
                "steady-state solver failed for %s=%s: %s. "
                "Falling back to long time-course.",
                param_name, value, exc,
            )
        elif ss_result.converged:
            point_model = point_models[i]
            for j, sname in enumerate(ss_result.species_names):
                point_model.set_concentration(sname, ss_result.concentrations[j])
            point_model.save_concentrations()
            point_model.reset()
            eval_kw = {}
            if codegen_so and net_path:
                eval_kw["codegen"] = True
                eval_kw["net_path"] = net_path
            eval_sim = bngsim.Simulator(point_model, method="ode", **eval_kw)
            result = eval_sim.run(t_span=(0, 1e-10), n_points=2)
            ss_ok = True
        else:
            residual = getattr(ss_result, "residual", None)
            res_str = f" (residual={residual:.2e})" if residual is not None else ""
            logger.warning(
                "steady-state solver did not converge for %s=%s%s. "
                "Falling back to long time-course.",
                param_name, value, res_str,
            )

        if not ss_ok:
            fb_model = _prepare_scan_point(
                base_model, param_name, value, species_initializers,
            )
            fb_sim = make_sim_fn(fb_model)
            result = fb_sim.run(t_span=(t_start, t_end), n_points=2)

        row, row_obs, row_funcs = _scan_result_to_row(
            result, float(value), print_functions=print_funcs,
        )
        rows.append(row)
        if obs_names is None:
            obs_names = row_obs
            func_names = row_funcs

    return rows, obs_names, func_names


def _run_parameter_scan_bngsim(
    bngsim_model, action, output_dir, model_name, is_bifurcate=False,
    codegen_so="", net_path=None, species_initializers=None,
    protocol_lines=None, xml_path=None, nf_param_overrides=None,
):
    """Execute a parameter_scan or bifurcate action via BNGsim.

    Supports time-course scans, steady-state scans (``steady_state=>1``),
    protocol scans (``method=>"protocol"``), and NFsim scans.
    Uses codegen for ODE acceleration and re-evaluates species initial
    concentrations when parameters change.
    """

    args = action.args
    param_name = _strip_quotes(args.get("parameter", "").strip())
    t_start = float(args.get("t_start", 0))
    t_end = float(args.get("t_end", 100))
    n_steps = int(float(args.get("n_steps", 100)))
    suffix = _strip_quotes(args.get("suffix", "").strip()) or "scan"
    reset_conc = not is_bifurcate and bool(int(float(args.get("reset_conc", 1))))
    use_ss = bool(int(float(args.get("steady_state", 0))))

    method = _strip_quotes(args.get("method", "ode").strip())
    is_protocol = method == "protocol"

    # Normalize method (ssa + poplevel → psa, psa default poplevel, etc.)
    poplevel = float(args["poplevel"]) if "poplevel" in args else None
    method, poplevel = _normalize_method(method, poplevel)

    # NFsim parameter scan: entirely different path
    if _is_nf_method(method):
        if not BNGSIM_HAS_NFSIM:
            raise BNGSimError(
                "NFsim parameter_scan requires BNGsim with NFsim support."
            )
        if xml_path is None or not os.path.isfile(xml_path):
            raise BNGSimError(
                f"NFsim parameter_scan requires BNG XML but none found at {xml_path}"
            )
        return _run_nfsim_scan(
            xml_path, action, output_dir, model_name,
            is_bifurcate=is_bifurcate,
            param_overrides=nf_param_overrides,
        )

    if is_protocol:
        if not protocol_lines:
            raise BNGSimError(
                'parameter_scan method=>"protocol" but no '
                '"begin protocol...end protocol" block found in the BNGL file.'
            )
        sim_method = "ode"  # protocol handles its own method dispatch
    elif method in _SIMULATE_METHOD_MAP.values():
        sim_method = method
    else:
        sim_method = _SIMULATE_METHOD_MAP.get(f"simulate_{method}", method)

    if use_ss and sim_method != "ode" and not is_protocol:
        logger.warning(
            "steady_state=>1 only supported for ODE. "
            "Falling back to time-course scan for method=%s.",
            sim_method,
        )
        use_ss = False

    print_funcs = bool(int(float(args.get("print_functions", 0))))

    # Resolve sample_times
    sample_times = _resolve_sample_times(args)
    if sample_times is not None:
        t_start = sample_times[0]
        t_end = sample_times[-1]

    points = _resolve_scan_points(args)
    rows = []
    obs_names = None
    func_names = None  # BNGL functions, only if print_functions=>1

    bngsim_model.save_concentrations()

    def _make_sim(mdl):
        kw = {}
        if sim_method == "psa" and poplevel is not None:
            kw["poplevel"] = poplevel
        if sim_method == "ode" and codegen_so and net_path:
            kw["codegen"] = True
            kw["net_path"] = net_path
        return bngsim.Simulator(mdl, method=sim_method, **kw)

    # ── Threaded steady-state path ──────────────────────────────────
    if use_ss and not species_initializers and len(points) >= 4:
        rows, obs_names, func_names = _run_ss_scan_threaded(
            bngsim_model, param_name, points, species_initializers,
            _make_sim, codegen_so, net_path, t_start, t_end, print_funcs,
        )
        col_names = (obs_names or []) + (func_names or [])
        scan_path = os.path.join(output_dir, f"{model_name}_{suffix}.scan")
        _write_scan_file(scan_path, param_name or "scan_param", col_names, rows)
        return None

    # ── Batch time-course path ──────────────────────────────────────
    use_batch = (
        not use_ss
        and not is_protocol
        and reset_conc
        and not species_initializers
        and sample_times is None
        and len(points) >= 4
        and hasattr(bngsim.Simulator, "run_batch")
    )
    if use_batch:
        params = [{param_name: float(v)} for v in points]
        n_workers = min(len(points), 4)
        batch_sim = _make_sim(bngsim_model)
        try:
            batch_results = batch_sim.run_batch(
                t_span=(t_start, t_end),
                n_points=2,
                params=params,
                num_processors=n_workers,
            )
        except Exception:
            logger.warning(
                "run_batch() failed; falling back to sequential scan.",
                exc_info=True,
            )
            use_batch = False

    if use_batch:
        for i, value in enumerate(points):
            row, row_obs, row_funcs = _scan_result_to_row(
                batch_results[i], float(value), print_functions=print_funcs,
            )
            rows.append(row)
            if obs_names is None:
                obs_names = row_obs
                func_names = row_funcs
        col_names = (obs_names or []) + (func_names or [])
        scan_path = os.path.join(output_dir, f"{model_name}_{suffix}.scan")
        _write_scan_file(scan_path, param_name or "scan_param", col_names, rows)
        return None

    # ── Sequential fallback (protocol, SS with few points, etc.) ────
    for value in points:
        if reset_conc:
            point_model = bngsim_model.clone()
        else:
            point_model = bngsim_model

        if param_name:
            point_model.set_param(param_name, _eval_numeric(str(value)))

        # Re-evaluate species ICs that depend on parameters
        if species_initializers:
            _sync_species_concentrations(point_model, species_initializers)

        if reset_conc:
            point_model.reset()

        if is_protocol:
            # Protocol route: run entire protocol per scan point
            result = _run_protocol(
                point_model, protocol_lines,
                codegen_so=codegen_so, net_path=net_path,
            )
            if result is None:
                raise BNGSimError(
                    "protocol contains no simulate actions"
                )
        elif use_ss:
            # Steady-state scan: find equilibrium, then evaluate observables
            ss_sim = _make_sim(point_model)
            ss_ok = False
            try:
                ss_result = ss_sim.steady_state()
                if ss_result.converged:
                    for j, sname in enumerate(ss_result.species_names):
                        point_model.set_concentration(sname, ss_result.concentrations[j])
                    point_model.save_concentrations()
                    point_model.reset()
                    # Brief evaluation run to compute observables/functions at SS
                    eval_kw = {}
                    if codegen_so and net_path:
                        eval_kw["codegen"] = True
                        eval_kw["net_path"] = net_path
                    eval_sim = bngsim.Simulator(point_model, method="ode", **eval_kw)
                    result = eval_sim.run(t_span=(0, 1e-10), n_points=2)
                    ss_ok = True
                else:
                    residual = getattr(ss_result, "residual", None)
                    res_str = f" (residual={residual:.2e})" if residual is not None else ""
                    logger.warning(
                        "steady-state solver did not converge for %s=%s%s. "
                        "Falling back to long time-course.",
                        param_name, value, res_str,
                    )
            except Exception as exc:
                logger.warning(
                    "steady-state solver failed for %s=%s: %s. "
                    "Falling back to long time-course.",
                    param_name, value, exc,
                )
            if not ss_ok:
                # Re-prepare the model from the saved base state
                if reset_conc:
                    point_model = bngsim_model.clone()
                else:
                    point_model = bngsim_model
                if param_name:
                    point_model.set_param(param_name, _eval_numeric(str(value)))
                if species_initializers:
                    _sync_species_concentrations(point_model, species_initializers)
                if reset_conc:
                    point_model.reset()
                fallback_sim = _make_sim(point_model)
                if sample_times is not None:
                    result = fallback_sim.run(
                        t_span=(sample_times[0], sample_times[-1]),
                        n_points=len(sample_times),
                        sample_times=sample_times,
                    )
                else:
                    result = fallback_sim.run(t_span=(t_start, t_end), n_points=n_steps + 1)
        else:
            # Time-course scan: simulate to t_end
            sim = _make_sim(point_model)
            if sample_times is not None:
                result = sim.run(
                    t_span=(sample_times[0], sample_times[-1]),
                    n_points=len(sample_times),
                    sample_times=sample_times,
                )
            else:
                result = sim.run(t_span=(t_start, t_end), n_points=n_steps + 1)

        row, row_obs, row_funcs = _scan_result_to_row(
            result, float(value), print_functions=print_funcs,
        )
        rows.append(row)

        if obs_names is None:
            obs_names = row_obs
            func_names = row_funcs

    col_names = (obs_names or []) + (func_names or [])
    scan_path = os.path.join(output_dir, f"{model_name}_{suffix}.scan")
    _write_scan_file(scan_path, param_name or "scan_param", col_names, rows)
    return None


def _execute_bngsim_actions(
    actions_items, bngsim_model, output_dir, model_name,
    xml_path=None, net_path=None, protocol_lines=None,
):
    """Walk through BNGL actions in order, executing each via BNGsim.

    Handles all state-affecting BNGL actions: simulate_*, parameter_scan,
    bifurcate, setParameter, setConcentration, addConcentration,
    save/resetConcentrations, save/resetParameters. Also supports
    codegen acceleration, species IC re-evaluation, continue=>1,
    sample_times for non-uniform time output, and method=>"protocol"
    in parameter_scan.

    Parameters
    ----------
    actions_items : list of Action
        Parsed actions from the original bngmodel.
    bngsim_model : bngsim.Model
        Loaded BNGsim model (from .net file).
    output_dir : str
        Output directory for result files.
    model_name : str
        Base name for output files.
    xml_path : str or None
        Path to BNG XML file (needed for simulate_nf).
    net_path : str or None
        Path to .net file (needed for codegen).
    protocol_lines : list of str or None
        Raw action lines from a ``begin protocol...end protocol`` block.
        Required when parameter_scan uses ``method=>"protocol"``.

    Returns
    -------
    BNGResult
    """
    current_method = None
    current_sim = None
    current_poplevel = None
    # Track model time for continue=>1 support
    model_time = 0.0
    # Track parameter overrides for NFsim propagation.
    # NFsim loads from XML and doesn't share state with bngsim_model,
    # so setParameter changes must be explicitly forwarded.
    nf_param_overrides = {}
    # Track concentration overrides for NFsim propagation.
    # setConcentration/addConcentration modify the .net model but NFsim
    # loads from XML, so concentration changes must be forwarded separately.
    # Keys are species patterns (e.g. "A(b)"), values are absolute counts.
    nf_conc_overrides = {}

    # Codegen: compile ODE RHS once, reuse for all ODE simulations.
    # Set BIONETGEN_NO_CODEGEN=1 to disable.
    codegen_so = ""
    if net_path and BNGSIM_AVAILABLE:
        codegen_so = _try_prepare_codegen(net_path)

    # Species IC re-evaluation: parse (species, expression) pairs from
    # the .net file so derived concentrations track parameter changes.
    species_initializers = []
    if net_path:
        species_initializers = _parse_net_species_initializers(net_path)

    def _make_ode_kwargs():
        """Build kwargs for ODE Simulator construction, including codegen."""
        kw = {}
        if codegen_so:
            kw["codegen"] = True
            kw["net_path"] = net_path
        return kw

    # Manual parameter save/restore (BNGsim has no saveParameters API)
    saved_params = {}
    for pname in bngsim_model.param_names:
        try:
            saved_params[pname] = bngsim_model.get_param(pname)
        except Exception:
            pass

    for action in actions_items:
        atype = action.type

        # Skip actions handled by BNG2.pl preprocessing
        if atype in _BNG2PL_ACTIONS:
            continue

        # ── simulate_* ──────────────────────────────────────────
        if atype.startswith("simulate"):
            sp = _parse_simulate_params(action)
            if sp is None:
                if atype == "simulate_pla":
                    raise BNGSimError(
                        "simulate_pla is not supported by BNGsim. "
                        "Use simulate_ode, simulate_ssa, or simulate_psa."
                    )
                logger.warning("Unrecognized simulate action: %s", atype)
                continue

            method = sp["method"]
            if method == "pla":
                raise BNGSimError(
                    "method='pla' is not supported by BNGsim. "
                    "Use 'ode', 'ssa', or 'psa' instead."
                )

            t_start, t_end = sp["t_start"], sp["t_end"]
            n_steps = sp["n_steps"]
            suffix = sp["suffix"]
            poplevel = sp["poplevel"]
            continue_flag = sp["continue_flag"]
            atol = sp["atol"]
            rtol = sp["rtol"]
            seed = sp["seed"]
            print_funcs = sp["print_functions"]
            stop_if = sp["stop_if"]
            sample_times = sp["sample_times"]
            gml = sp["gml"]
            out_name = f"{model_name}_{suffix}" if suffix else model_name

            # continue=>1: use current model time as t_start
            if continue_flag:
                t_start = model_time

            if _is_nf_method(method):
                if sample_times is not None:
                    logger.warning("sample_times is not supported for NFsim — ignoring")
                if xml_path is None or not os.path.isfile(xml_path):
                    raise BNGSimError(
                        f"simulate_nf requires BNG XML but none found at {xml_path}"
                    )
                run_nfsim(
                    xml_path,
                    output_dir,
                    t_span=(t_start, t_end),
                    n_points=n_steps + 1,
                    seed=seed,
                    gml=gml,
                    model_name=out_name,
                    param_overrides=nf_param_overrides or None,
                    conc_overrides=nf_conc_overrides or None,
                )
            else:
                # Rebuild simulator if method/poplevel changed, or if
                # it was invalidated by a parameter change
                if current_sim is None or method != current_method or (
                    method == "psa" and poplevel != current_poplevel
                ):
                    sim_kwargs = _make_ode_kwargs() if method == "ode" else {}
                    if method == "psa" and poplevel is not None:
                        sim_kwargs["poplevel"] = poplevel
                    current_sim = bngsim.Simulator(
                        bngsim_model, method=method, **sim_kwargs
                    )
                    current_method = method
                    current_poplevel = poplevel

                # Register stop_if condition if specified
                current_sim.clear_stop_conditions()
                if stop_if:
                    current_sim.add_stop_condition(stop_if, label=stop_if)

                # Pass atol/rtol/seed to the run call
                run_kwargs = {}
                if atol is not None:
                    run_kwargs["atol"] = atol
                if rtol is not None:
                    run_kwargs["rtol"] = rtol
                if seed is not None:
                    run_kwargs["seed"] = seed

                # Use sample_times for non-uniform time sampling
                if sample_times is not None:
                    run_kwargs["sample_times"] = sample_times
                    run_t_span = (sample_times[0], sample_times[-1])
                    run_n_points = len(sample_times)
                else:
                    run_t_span = (t_start, t_end)
                    run_n_points = n_steps + 1

                try:
                    result = current_sim.run(
                        t_span=run_t_span, n_points=run_n_points,
                        **run_kwargs,
                    )
                except bngsim.StopConditionMet as e:
                    # Stop condition triggered — use the truncated result
                    result = e.result
                    logger.info("stop_if triggered: %s", stop_if)

                _write_bngsim_results(
                    result, output_dir, out_name,
                    print_functions=print_funcs,
                )

            # Update model time for continue=>1 support
            model_time = t_end
            continue

        # ── parameter_scan ──────────────────────────────────────
        if atype == "parameter_scan":
            _run_parameter_scan_bngsim(
                bngsim_model, action, output_dir, model_name,
                is_bifurcate=False, codegen_so=codegen_so, net_path=net_path,
                species_initializers=species_initializers,
                protocol_lines=protocol_lines, xml_path=xml_path,
                nf_param_overrides=nf_param_overrides or None,
            )
            continue

        # ── bifurcate ───────────────────────────────────────────
        if atype == "bifurcate":
            _run_parameter_scan_bngsim(
                bngsim_model, action, output_dir, model_name,
                is_bifurcate=True, codegen_so=codegen_so, net_path=net_path,
                species_initializers=species_initializers,
                protocol_lines=protocol_lines, xml_path=xml_path,
                nf_param_overrides=nf_param_overrides or None,
            )
            continue

        # ── setParameter ────────────────────────────────────────
        if atype == "setParameter":
            name, value = _extract_positional_args(action)
            numeric_value = _eval_numeric(value)
            try:
                bngsim_model.set_param(name, numeric_value)
                logger.debug("setParameter(%s, %s)", name, value)
            except Exception as e:
                logger.warning("setParameter(%s, %s) failed: %s", name, value, e)
            # Track for NFsim propagation
            nf_param_overrides[name] = numeric_value
            # Invalidate simulator cache — params changed
            current_sim = None
            current_method = None
            continue

        # ── setConcentration ────────────────────────────────────
        if atype == "setConcentration":
            name, value = _extract_positional_args(action)
            numeric_value = _eval_numeric(value)
            try:
                bngsim_model.set_concentration(name, numeric_value)
                logger.debug("setConcentration(%s, %s)", name, value)
            except Exception as e:
                logger.warning("setConcentration(%s, %s) failed: %s", name, value, e)
            # Track for NFsim propagation (absolute count)
            nf_conc_overrides[name] = round(numeric_value)
            continue

        # ── addConcentration ────────────────────────────────────
        if atype == "addConcentration":
            name, value = _extract_positional_args(action)
            try:
                current = bngsim_model.get_concentration(name)
                new_val = current + _eval_numeric(value)
                bngsim_model.set_concentration(name, new_val)
                logger.debug("addConcentration(%s, %s)", name, value)
                # Track for NFsim propagation (absolute count)
                nf_conc_overrides[name] = round(new_val)
            except Exception as e:
                logger.warning("addConcentration(%s, %s) failed: %s", name, value, e)
            continue

        # ── saveConcentrations ──────────────────────────────────
        if atype == "saveConcentrations":
            bngsim_model.save_concentrations()
            continue

        # ── resetConcentrations ─────────────────────────────────
        if atype == "resetConcentrations":
            bngsim_model.reset()
            nf_conc_overrides.clear()
            continue

        # ── saveParameters ──────────────────────────────────────
        if atype == "saveParameters":
            saved_params = {}
            for pname in bngsim_model.param_names:
                try:
                    saved_params[pname] = bngsim_model.get_param(pname)
                except Exception:
                    pass
            continue

        # ── resetParameters ─────────────────────────────────────
        if atype == "resetParameters":
            for pname, pval in saved_params.items():
                try:
                    bngsim_model.set_param(pname, pval)
                except Exception:
                    pass
            # Clear NFsim overrides — params restored to initial
            nf_param_overrides.clear()
            # Invalidate simulator cache — params changed
            current_sim = None
            current_method = None
            continue

        logger.warning("Unhandled action: %s", atype)

    return _make_bng_result(output_dir, method=current_method or "ode")


# ─── Table function support ───────────────────────────────────────


def _parse_table_functions(bngl_path):
    """Parse table function definitions from a BNGL file's functions block.

    Finds ``tfun(...)`` calls within the ``begin functions...end functions``
    block and extracts the function name, data source (file path or inline
    arrays), index variable, and interpolation method.

    Parameters
    ----------
    bngl_path : str
        Path to the .bngl file.

    Returns
    -------
    list of dict
        Each dict has keys: ``name``, and either ``file`` or
        ``times``/``values``, plus ``index`` and ``method``.
    """
    import re

    tfun_specs = []
    bngl_dir = os.path.dirname(os.path.abspath(bngl_path))

    in_functions = False
    try:
        with open(bngl_path, "r", errors="replace") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                # Strip comments for block detection
                comment_idx = stripped.find("#")
                clean = stripped[:comment_idx].strip() if comment_idx >= 0 else stripped

                if re.match(r"begin\s+functions", clean):
                    in_functions = True
                    continue
                if re.match(r"end\s+functions", clean):
                    in_functions = False
                    continue
                if not in_functions:
                    continue

                # Look for: funcname(...) = ... tfun(...) ...
                # or: funcname = ... tfun(...) ...
                if "tfun(" not in clean:
                    continue

                # Extract function name (before '=')
                eq_match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*=", clean)
                if not eq_match:
                    continue
                func_name = eq_match.group(1)

                # Extract the tfun(...) call arguments
                tfun_match = re.search(r"tfun\((.+)\)", clean)
                if not tfun_match:
                    continue
                tfun_body = tfun_match.group(1)

                spec = _parse_tfun_args(func_name, tfun_body, bngl_dir)
                if spec is not None:
                    tfun_specs.append(spec)
    except OSError:
        pass

    return tfun_specs


def _parse_tfun_args(func_name, tfun_body, bngl_dir):
    """Parse the arguments of a single ``tfun(...)`` call.

    Handles two forms:
    - File-based: ``tfun('filename.tfun', index_var)``
    - Inline data: ``tfun([x1,x2,...], [y1,y2,...], index_var)``

    Optional trailing ``method=>"linear|step"`` is supported.

    Returns a dict with ``name``, ``index``, ``method``, and either
    ``file`` or ``times``/``values``.
    """
    import re

    # Default values
    index = "time"
    method = "linear"

    # Extract method=>"..." if present
    method_match = re.search(r'method\s*=>\s*"(\w+)"', tfun_body)
    if method_match:
        method = method_match.group(1).lower()
        # Remove the method=>... from the body for simpler parsing
        tfun_body = tfun_body[:method_match.start()] + tfun_body[method_match.end():]

    # Clean up trailing commas/whitespace
    tfun_body = tfun_body.strip().rstrip(",").strip()

    # Check for inline array form: [x1,x2,...], [y1,y2,...], index
    array_match = re.match(
        r"\[([^\]]+)\]\s*,\s*\[([^\]]+)\](?:\s*,\s*(\w+))?",
        tfun_body,
    )
    if array_match:
        try:
            times = [float(v.strip()) for v in array_match.group(1).split(",")]
            values = [float(v.strip()) for v in array_match.group(2).split(",")]
        except ValueError:
            logger.warning("tfun: could not parse inline data for %s", func_name)
            return None
        if array_match.group(3):
            index = array_match.group(3)
        return {
            "name": func_name,
            "times": times,
            "values": values,
            "index": index,
            "method": method,
        }

    # Check for file-based form: 'filename.tfun', index
    # or: "filename.tfun", index
    file_match = re.match(
        r"""['"]([^'"]+)['"]\s*(?:,\s*(\w+))?""",
        tfun_body,
    )
    if file_match:
        tfun_file = file_match.group(1)
        if file_match.group(2):
            index = file_match.group(2)
        # Resolve path relative to BNGL directory
        if not os.path.isabs(tfun_file):
            tfun_file = os.path.join(bngl_dir, tfun_file)
        return {
            "name": func_name,
            "file": tfun_file,
            "index": index,
            "method": method,
        }

    logger.warning("tfun: could not parse arguments for %s: %s", func_name, tfun_body)
    return None


def _add_table_functions(bngsim_model, tfun_specs):
    """Add parsed table function specifications to a BNGsim model.

    Parameters
    ----------
    bngsim_model : bngsim.Model
        The loaded model.
    tfun_specs : list of dict
        Table function specifications from ``_parse_table_functions``.
    """
    for spec in tfun_specs:
        name = spec["name"]
        index = spec.get("index", "time")
        method = spec.get("method", "linear")
        try:
            if "file" in spec:
                bngsim_model.add_table_function(
                    name, file=spec["file"], index=index, method=method,
                )
            elif "times" in spec and "values" in spec:
                bngsim_model.add_table_function(
                    name, times=spec["times"], values=spec["values"],
                    index=index, method=method,
                )
            logger.debug("Added table function: %s (index=%s, method=%s)", name, index, method)
        except Exception as e:
            logger.warning("Failed to add table function %s: %s", name, e)


# ─── Protocol block parsing ───────────────────────────────────────


def _parse_protocol_block(bngl_path):
    """Extract raw action lines from a ``begin protocol...end protocol`` block.

    Parameters
    ----------
    bngl_path : str
        Path to the .bngl file.

    Returns
    -------
    list of str
        Action lines from the protocol block (empty list if no block found).
    """
    import re

    protocol_lines = []
    in_protocol = False

    try:
        with open(bngl_path, "r", errors="replace") as f:
            for raw_line in f:
                # Handle line continuations
                line = raw_line.rstrip("\n")
                stripped = line.strip()

                # Remove comments for block detection
                comment_idx = stripped.find("#")
                clean = stripped[:comment_idx].strip() if comment_idx >= 0 else stripped

                if re.match(r"begin\s+protocol", clean):
                    in_protocol = True
                    continue
                if re.match(r"end\s+protocol", clean):
                    in_protocol = False
                    continue
                if in_protocol:
                    protocol_lines.append(raw_line.rstrip("\n"))
    except OSError:
        pass

    return protocol_lines


# ─── BNGL hybrid path ─────────────────────────────────────────────


def run_bngl_with_bngsim(
    bngl_path,
    output_dir,
    bngpath,
    method=None,
    t_span=None,
    n_points=None,
    suppress=False,
    log_file=None,
    timeout=None,
    app=None,
    **sim_kwargs,
):
    """Run a .bngl file: BNG2.pl for network generation, then BNGsim for simulation.

    This is the hybrid path:
    1. Parse the BNGL to get the full action list
    2. Write a modified BNGL with only generate_network / writeXML
    3. Run BNG2.pl on the modified file to produce .net / .xml
    4. Load the .net into BNGsim and execute all actions in order

    Parameters
    ----------
    bngl_path : str
        Path to the .bngl file.
    output_dir : str
        Directory for output files.
    bngpath : str
        Path to BioNetGen directory containing BNG2.pl.
    method : str or None
        Simulation method override. If None, uses methods from BNGL actions.
    t_span : tuple or None
        Time span override.
    n_points : int or None
        Number of output time points override.
    suppress : bool
        Suppress BNG2.pl output.
    log_file : str or None
        Path to log file.
    timeout : int or None
        Timeout in seconds.
    app : cement.App or None
        Cement application for logging.
    **sim_kwargs
        Additional kwargs for bngsim.Simulator.

    Returns
    -------
    BNGResult
    """
    if not BNGSIM_AVAILABLE:
        raise BNGSimError("BNGsim is not available.")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: Extract protocol block and table functions from the BNGL file
    # before parsing. The bngmodel parser does not handle these constructs.
    protocol_lines = _parse_protocol_block(bngl_path)
    tfun_specs = _parse_table_functions(bngl_path)

    # Step 1: Parse the BNGL file and save original actions
    import bionetgen.modelapi.model as mdl

    model = mdl.bngmodel(bngl_path)
    model_name = model.model_name
    original_actions = list(model.actions.items)

    # Step 2: Determine what BNG2.pl needs to produce
    needs_network = _actions_need_network(original_actions)
    needs_xml = _actions_need_xml(original_actions)

    # If CLI overrides method to NF, we need XML
    if method is not None and _is_nf_method(method):
        needs_xml = True
    # If CLI overrides method to network-based, we need network
    if method is not None and not _is_nf_method(method):
        needs_network = True

    # Write modified BNGL for BNG2.pl preprocessing.
    # Keep write/output actions (writeSBML, writeMfile, visualize, etc.)
    # so BNG2.pl produces those outputs. Only strip simulate/scan/state
    # actions that BNGsim will handle.
    _BNGSIM_HANDLED = frozenset({
        "simulate", "simulate_ode", "simulate_ssa", "simulate_psa",
        "simulate_pla", "simulate_nf",
        "parameter_scan", "bifurcate",
        "setParameter", "setConcentration", "addConcentration",
        "saveConcentrations", "resetConcentrations",
        "saveParameters", "resetParameters",
    })
    preserved_actions = [a for a in original_actions if a.type not in _BNGSIM_HANDLED]
    model.actions.clear_actions()
    if needs_network:
        model.add_action("generate_network", {"overwrite": 1})
    if needs_xml:
        model.add_action("writeXML", {})
    # Re-add write/output actions that BNG2.pl should execute
    for a in preserved_actions:
        if a.type not in ("generate_network", "writeXML"):
            model.actions.items.append(a)

    gen_bngl_path = os.path.join(output_dir, f"{model_name}.bngl")
    model.write_model(gen_bngl_path)

    # Step 3: Run BNG2.pl
    from bionetgen.core.tools.cli import BNGCLI

    cli = BNGCLI(
        gen_bngl_path,
        output_dir,
        bngpath,
        suppress=suppress,
        log_file=log_file,
        timeout=timeout,
        app=app,
    )
    cli.run()
    if cli.result is None:
        raise BNGSimError("BNG2.pl failed. Cannot proceed with BNGsim.")

    # Step 4: Load .net into BNGsim and execute actions
    net_path = os.path.join(output_dir, f"{model_name}.net")
    xml_path = os.path.join(output_dir, f"{model_name}.xml")

    # If CLI provided method/t_span/n_points but there are no simulation
    # actions in the BNGL, create a synthetic simulate action
    has_sim_actions = any(
        a.type.startswith("simulate") or a.type in ("parameter_scan", "bifurcate")
        for a in original_actions
    )
    if not has_sim_actions and (method or t_span or n_points):
        from bionetgen.modelapi.structs import Action

        sim_method = method or "ode"
        t0 = t_span[0] if t_span else 0.0
        t1 = t_span[1] if t_span else 100.0
        np_ = n_points or 101
        synthetic = Action(
            action_type=f"simulate_{sim_method}" if sim_method != "nf" else "simulate_nf",
            action_args={
                "t_start": str(t0),
                "t_end": str(t1),
                "n_steps": str(np_ - 1),
            },
        )
        original_actions.append(synthetic)
    elif not has_sim_actions:
        # No simulation actions and no CLI overrides — just return BNG2.pl result
        return cli.result

    # Apply CLI overrides by modifying action parameters in-place
    if method is not None or t_span is not None or n_points is not None:
        for action in original_actions:
            if action.type.startswith("simulate"):
                if method is not None:
                    # Change the action type to match the override
                    mapped = f"simulate_{method}" if method != "nf" else "simulate_nf"
                    if mapped in _SIMULATE_METHOD_MAP:
                        action.type = mapped
                        action.name = mapped
                if t_span is not None:
                    action.args["t_start"] = str(t_span[0])
                    action.args["t_end"] = str(t_span[1])
                if n_points is not None:
                    action.args["n_steps"] = str(n_points - 1)

    # Load model for network-based actions
    if os.path.isfile(net_path):
        bngsim_model = bngsim.Model.from_net(net_path)
        # Add table functions parsed from the original BNGL
        if tfun_specs:
            _add_table_functions(bngsim_model, tfun_specs)
    elif needs_network:
        raise BNGSimError(
            f"Expected .net file at {net_path} but it was not generated."
        )
    else:
        # Pure NF — no .net needed, but we need a dummy model for
        # parameter tracking. Create from the XML if possible.
        bngsim_model = None

    xml_arg = xml_path if os.path.isfile(xml_path) else None
    net_arg = net_path if os.path.isfile(net_path) else None

    if bngsim_model is not None:
        return _execute_bngsim_actions(
            original_actions,
            bngsim_model,
            output_dir,
            model_name,
            xml_path=xml_arg,
            net_path=net_arg,
            protocol_lines=protocol_lines,
        )
    else:
        # Pure NF path — execute NF actions directly
        return _execute_bngsim_actions(
            original_actions,
            None,
            output_dir,
            model_name,
            xml_path=xml_arg,
            net_path=net_arg,
            protocol_lines=protocol_lines,
        )
