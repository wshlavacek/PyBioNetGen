"""Bridge module for optional BNGsim integration.

BNGsim is a high-performance C++ simulation engine with Python bindings
that can replace run_network and NFsim for in-process simulation.
This module handles availability detection, input format detection,
and routing simulation requests to BNGsim when available.
"""

import ast
import concurrent.futures
import inspect
import logging
import operator
import os
import re

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
        from bngsim import HAS_NFSIM

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
    # BNG XML always reuses an SBML root but has BNG-specific child elements
    # (capitalization differs: BNG uses "ListOfMoleculeTypes" while SBML uses
    # "listOfReactions"). Match those tag names directly. Don't match the
    # bare string "bionetgen" — BNG2.pl writes a "Created by BioNetGen"
    # comment into its SBML output too, so that substring is ambiguous.
    is_bng = (
        "<listofmoleculetypes" in head_lower
        or "<listofspeciestypes" in head_lower
        or "<listofobservables" in head_lower
        or "<listofreactionrules" in head_lower
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


def _append_bng_dat_rows(path, time, data_2d, skip_first=True):
    """Append data rows to an existing .gdat/.cdat file (no header).

    Used for ``continue=>1`` to extend a prior segment's output. The first
    row of *time* is normally the previous segment's t_end (a duplicate),
    so it is skipped by default — matching BNG2.pl's run_network ``-x``.
    """
    start = 1 if (skip_first and len(time) > 0) else 0
    with open(path, "a") as f:
        for i in range(start, len(time)):
            vals = [time[i]] + [data_2d[i, j] for j in range(data_2d.shape[1])]
            f.write("  ".join(f"{v:22.12e}" for v in vals) + "\n")


def _append_cdat_rows(cdat_path, result):
    """Append rows from a fresh BNGsim Result to an existing .cdat file."""
    import numpy as np

    species = np.asarray(result.species)
    time = np.asarray(result.time)
    if species.ndim != 2 or species.shape[0] == 0:
        return
    _append_bng_dat_rows(cdat_path, time, species, skip_first=True)


def _write_bngsim_results(
    result, output_dir, model_name,
    print_functions=False, append=False,
    bngmodel=None, bngmodel_params=None, param_overrides=None,
):
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
    append : bool
        If True and the target files already exist, append rows from
        *result* (skipping its first row, which duplicates the prior
        segment's t_end). Used for ``continue=>1``. If the files do not
        yet exist, falls back to a fresh write so the first segment of
        a continuation chain still produces complete output.
    bngmodel, bngmodel_params, param_overrides
        Used only as a fallback when *print_functions* is True and
        BNGsim's ``Result.expressions`` is empty (the NFsim path), so
        BNGL functions can be evaluated post-hoc per time point. Ignored
        for non-NF results that already carry function columns.
    """
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    gdat_path = os.path.join(output_dir, f"{model_name}.gdat")
    cdat_path = os.path.join(output_dir, f"{model_name}.cdat")

    do_append = append and os.path.exists(gdat_path) and os.path.exists(cdat_path)

    # Build the optional functions block once for both write/append paths
    obs_names = list(result.observable_names)
    obs_array = np.asarray(result.observables) if result.n_observables > 0 else np.empty((result.n_times, 0))

    func_names = []
    func_array = np.empty((result.n_times, 0))
    has_funcs = False
    if print_functions:
        bngsim_func_names = list(result.expression_names)
        bngsim_func_array = np.asarray(result.expressions)
        if (
            len(bngsim_func_names) > 0
            and bngsim_func_array.ndim == 2
            and bngsim_func_array.shape[1] > 0
        ):
            func_names = bngsim_func_names
            func_array = bngsim_func_array
            has_funcs = True
        elif bngmodel is not None and obs_array.shape[0] > 0:
            # NFsim leaves expressions empty — recompute per time point
            # from the parsed bngmodel.
            func_names, func_array = _evaluate_functions_per_timepoint(
                bngmodel, bngmodel_params, param_overrides,
                obs_names, obs_array,
            )
            has_funcs = func_array.shape[1] > 0

    if has_funcs:
        combined = np.hstack([obs_array, func_array])
        combined_names = obs_names + func_names
    else:
        combined = obs_array
        combined_names = obs_names

    if do_append:
        _append_cdat_rows(cdat_path, result)
        if result.n_observables > 0 or has_funcs:
            _append_bng_dat_rows(gdat_path, result.time, combined, skip_first=True)
        return

    # Fresh write (default and first-segment-of-continuation path)
    result.to_cdat(cdat_path)
    if result.n_observables > 0 or has_funcs:
        _write_bng_dat(gdat_path, result.time, combined, combined_names)


def _make_bng_result(output_dir, method):
    """Load a BNGResult from an output directory."""
    from bionetgen.core.tools.result import BNGResult

    bng_result = BNGResult(path=output_dir)
    bng_result.process_return = 0
    bng_result.output = [f"BNGsim simulation completed: method={method}"]
    return bng_result


def _collapse_nfsim_concentration_changes(
    conc_overrides=None,
    conc_deltas=None,
):
    """Collapse concentration changes to NFsim's molecule-type granularity."""
    collapsed_overrides = {}
    collapsed_deltas = {}

    if conc_overrides:
        for species_pattern, target_count in conc_overrides.items():
            mol_type = str(species_pattern).split("(", 1)[0]
            try:
                collapsed_overrides[mol_type] = (
                    collapsed_overrides.get(mol_type, 0) + int(target_count)
                )
            except Exception as e:
                logger.warning(
                    "NFsim: conc override for %s failed: %s",
                    species_pattern, e,
                )

    if conc_deltas:
        for species_pattern, delta_count in conc_deltas.items():
            mol_type = str(species_pattern).split("(", 1)[0]
            try:
                collapsed_deltas[mol_type] = (
                    collapsed_deltas.get(mol_type, 0) + int(delta_count)
                )
            except Exception as e:
                logger.warning(
                    "NFsim: conc delta for %s failed: %s",
                    species_pattern, e,
                )

    return collapsed_overrides, collapsed_deltas


def _apply_nfsim_concentration_changes(
    nfsim,
    conc_overrides=None,
    conc_deltas=None,
):
    """Apply recorded concentration changes to a fresh NFsim session."""
    if (
        callable(getattr(nfsim, "set_species_count", None))
        and callable(getattr(nfsim, "add_species", None))
        and callable(getattr(nfsim, "remove_species", None))
    ):
        remaining_deltas = {
            str(species_pattern): delta for species_pattern, delta in (conc_deltas or {}).items()
        }

        if conc_overrides:
            for species_pattern, target_count in conc_overrides.items():
                try:
                    pattern = str(species_pattern)
                    desired_count = int(target_count) + int(remaining_deltas.pop(pattern, 0))
                    nfsim.set_species_count(pattern, desired_count)
                except Exception as e:
                    logger.warning(
                        "NFsim: concentration replay for %s failed: %s",
                        species_pattern, e,
                    )

        for species_pattern, delta in remaining_deltas.items():
            try:
                pattern = str(species_pattern)
                delta = int(delta)
                if delta > 0:
                    nfsim.add_species(pattern, delta)
                elif delta < 0:
                    nfsim.remove_species(pattern, -delta)
            except Exception as e:
                logger.warning(
                    "NFsim: concentration replay for %s failed: %s",
                    species_pattern, e,
                )
        return

    collapsed_overrides, collapsed_deltas = _collapse_nfsim_concentration_changes(
        conc_overrides=conc_overrides,
        conc_deltas=conc_deltas,
    )

    for mol_type, target_count in collapsed_overrides.items():
        try:
            desired_count = target_count + collapsed_deltas.pop(mol_type, 0)
            current = nfsim.get_molecule_count(mol_type)
            to_add = desired_count - current
            if to_add > 0:
                nfsim.add_molecules(mol_type, to_add)
            elif to_add < 0:
                logger.warning(
                    "NFsim: cannot decrease %s from %d to %d; "
                    "leaving count unchanged",
                    mol_type, current, desired_count,
                )
        except Exception as e:
            logger.warning(
                "NFsim: concentration replay for %s failed: %s",
                mol_type, e,
            )

    for mol_type, delta in collapsed_deltas.items():
        try:
            if delta > 0:
                nfsim.add_molecules(mol_type, delta)
            elif delta < 0:
                logger.warning(
                    "NFsim: cannot decrease %s by %d; leaving count unchanged",
                    mol_type, -delta,
                )
        except Exception as e:
            logger.warning(
                "NFsim: concentration replay for %s failed: %s",
                mol_type, e,
            )


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
    conc_deltas=None,
    print_functions=False,
    bngmodel=None,
    bngmodel_params=None,
    nf_params=None,
):
    """Run a network-free simulation using BNGsim's NfsimSession.

    Uses the public NfsimSession API with a BioNetGen XML file.
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
        ``NfsimSession.set_param()`` before initialization.
        Used to propagate ``setParameter`` calls to NFsim.
    conc_overrides : dict or None
        Species pattern → absolute molecule count overrides to apply
        after initialization via ``NfsimSession.set_species_count()`` when
        available, with a molecule-type fallback for older bngsim builds.
        Used to propagate ``setConcentration``/``addConcentration``
        calls to NFsim.
    conc_deltas : dict or None
        Species pattern → relative molecule count deltas to apply after
        initialization. Used for ``addConcentration`` replay when no
        generated network model is available.

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
        nf_kwargs = _nfsim_session_kwargs(nf_params)
        with bngsim.NfsimSession(xml_path, molecule_limit=gml, **nf_kwargs) as nfsim:
            # Apply parameter overrides from setParameter actions
            if param_overrides:
                for pname, pval in param_overrides.items():
                    try:
                        nfsim.set_param(pname, float(pval))
                    except Exception as exc:
                        logger.debug("NFsim: set_param(%s, %s) skipped: %s", pname, pval, exc)

            nfsim.initialize(seed)
            _apply_nfsim_concentration_changes(
                nfsim,
                conc_overrides=conc_overrides,
                conc_deltas=conc_deltas,
            )

            result = nfsim.simulate(t_span[0], t_span[1], n_points)

        _write_bngsim_results(
            result, output_dir, model_name,
            print_functions=print_functions,
            bngmodel=bngmodel,
            bngmodel_params=bngmodel_params,
            param_overrides=param_overrides,
        )

        return _make_bng_result(output_dir, method="nf")

    except Exception as e:
        if isinstance(e, (BNGSimError, BNGFormatError)):
            raise
        raise BNGSimError(f"BNGsim NFsim simulation failed: {e}") from e


def run_with_bngsim(
    input_path,
    output_dir,
    fmt=None,
    method=None,
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
    method : str or None
        Simulation method ('ode', 'ssa', 'psa', 'nf', etc.). If None,
        direct non-BNG-XML inputs default to ``'ode'`` while direct
        BioNetGen XML inputs default to ``'nf'``.
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
        if method is None:
            method = "nf"
        if not _is_nf_method(method):
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

    # BNGL handling lives in run_bngl_with_bngsim(); for other direct
    # inputs, preserve historical behavior by defaulting to ODE when no
    # explicit method override was provided.
    if method is None:
        method = "ode"

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

_NF_ONLY_STATE_ACTIONS = frozenset({
    "setParameter", "setConcentration", "addConcentration",
    "saveConcentrations", "resetConcentrations",
    "saveParameters", "resetParameters",
})

_NF_SAFE_BNG2PL_ACTIONS = frozenset({
    "writeXML",
    "setModelName", "substanceUnits", "setOption", "version", "quit",
})


def _strip_quotes(s):
    """Strip surrounding single or double quotes from a string."""
    if s and len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


_NFSIM_PARAM_FLAG_RE = re.compile(r"-bscb|-cb|-utl\s*(\d+)")


def _parse_nfsim_param_string(args):
    """Map a BNGL ``simulate({...,param=>"-bscb -utl N"})`` flag string to
    NfsimSession kwargs.

    Only flags the BNGL explicitly requests are returned — when ``param=>``
    is absent we leave the BNGsim defaults alone, since some models rely
    on the binding's default complex-bookkeeping behavior to simulate at
    all (forcing it off here triggers ``IndexError: vector`` on rules
    that pattern-match bond state).

    The action parser strips internal whitespace from quoted args, so
    ``"-bscb -utl 5"`` arrives here as ``"-bscb-utl5"``. Match flag tokens
    via regex so both spaced and collapsed forms work.
    """
    raw = args.get("param") if isinstance(args, dict) else None
    raw = _strip_quotes(raw.strip()) if raw else ""
    flags = {}
    if not raw:
        return flags
    for m in _NFSIM_PARAM_FLAG_RE.finditer(raw):
        if m.group(1):
            flags["traversal_limit"] = int(m.group(1))
        else:
            flags["block_same_complex_binding"] = True
    return flags


def _nfsim_session_kwargs(nf_params):
    """Translate parsed param=> flags into NfsimSession kwargs, dropping any
    keys the installed BNGsim build doesn't accept so older wheels keep
    working.
    """
    if not nf_params:
        return {}
    try:
        accepted = set(inspect.signature(bngsim.NfsimSession.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set(nf_params)
    return {k: v for k, v in nf_params.items() if k in accepted}


def _is_pla_action(action):
    """Return True if this action requests PLA simulation.

    BNGsim does not implement PLA, so PLA simulations and PLA-method
    parameter scans are deferred to BNG2.pl: the action is preserved in
    the BNGL handed to BNG2.pl and skipped during BNGsim execution.
    """
    if action.type == "simulate_pla":
        return True
    args = action.args or {}
    method_raw = args.get("method")
    if not isinstance(method_raw, str):
        return False
    method = _strip_quotes(method_raw.strip())
    if action.type == "simulate" and method == "pla":
        return True
    if action.type in ("parameter_scan", "bifurcate") and method == "pla":
        return True
    return False


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
        # BNGL's eager ternary — used by BNG2.pl in derived parameter
        # expressions like ``use_excess = if(LT/RT>=100, 1, 0)``. Stored
        # under a sanitized name because ``if`` is a Python keyword and
        # can't be parsed as a Name token; ``_safe_eval_expr`` rewrites
        # ``if(`` → ``_BNG_IF_(`` before parsing.
        "_BNG_IF_": lambda cond, t, f: t if cond else f,
        "pi": math.pi,
        "_pi": math.pi,
        "_e": math.e,
    })
    ns["__builtins__"] = {}
    return ns


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
# BNGL's ``if(cond, t, f)`` uses the keyword ``if`` as a function name,
# which Python's parser rejects. Rewrite to a sanitized identifier that
# resolves through the safe namespace. The trailing ``\b(?=\()`` keeps
# words like ``ifx`` or ``stiff`` untouched.
_BNG_IF_RE = re.compile(r"\bif\s*(?=\()")


def _safe_eval_expr(expr_str, ns):
    """Evaluate ``expr_str`` against ``ns`` using a whitelisted AST walker.

    Supports arithmetic (``+ - * / // % **``), unary ``+``/``-``, name
    lookup, and calls to whitelisted callables already present in ``ns``.
    Rejects every other syntax form (attribute access, subscripts,
    comprehensions, lambdas, comparisons, string/bool/None constants,
    keyword arguments, etc.) by raising ``ValueError``.
    """
    msg = f"Cannot evaluate numeric expression: {expr_str!r}"

    try:
        tree = ast.parse(_BNG_IF_RE.sub("_BNG_IF_", expr_str), mode="eval")
    except SyntaxError:
        raise ValueError(msg) from None

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(msg)
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in ns:
                raise ValueError(msg)
            return ns[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Compare):
            # BNG2.pl emits chains like ``a >= b`` inside if(...). Support
            # standard comparisons; reject membership/identity ops.
            left = walk(node.left)
            for op_node, comp in zip(node.ops, node.comparators):
                op = _CMP_OPS.get(type(op_node))
                if op is None:
                    raise ValueError(msg)
                right = walk(comp)
                if not op(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name):
                raise ValueError(msg)
            func = walk(node.func)
            if not callable(func):
                raise ValueError(msg)
            return func(*(walk(arg) for arg in node.args))
        raise ValueError(msg)

    return walk(tree)


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
    ns = _safe_math_namespace(extra_ns)
    try:
        return float(_safe_eval_expr(expr_str, ns))
    except Exception:
        raise ValueError(f"Cannot evaluate numeric expression: {expr_str!r}") from None


def _model_param_namespace(bngsim_model, fallback=None):
    """Build a {param_name: float} dict from a BNGsim model.

    Used as ``extra_ns`` for ``_eval_numeric`` so that BNGL actions like
    ``setConcentration("S0", I0 * kfactor)`` can resolve the parameter
    names that appear in the value expression.

    If *bngsim_model* is None (pure-NF runs that never load a .net), the
    optional *fallback* mapping is returned instead — typically the
    parameter values resolved from the BNGL parameter block.
    """
    if bngsim_model is None:
        return dict(fallback) if fallback else None
    ns = {}
    for pname in bngsim_model.param_names:
        try:
            ns[pname] = bngsim_model.get_param(pname)
        except Exception as exc:
            logger.debug("param ns: get_param(%s) failed: %s", pname, exc)
    return ns


def _resolve_bngmodel_params(bngmodel, overrides=None):
    """Resolve a parsed bngmodel's parameter block to {name: float}.

    BNGL parameter values are stored as expression strings (e.g. ``"30"``,
    ``"2*RT"``, ``"koff"``). Iteratively evaluate each one using the
    already-resolved parameters as a namespace until either all resolve
    or no further progress is made. Unresolvable parameters are skipped
    with a debug log — they may be referenced indirectly or be intentional
    deferred-evaluation expressions.

    When *overrides* is provided, those name→value pairs are applied
    first and used as the starting namespace; expressions referring to
    overridden names then re-resolve transitively (e.g., a scan parameter
    flows through to derived seed-species expressions like ``LT = AT_nM*…``).
    Overridden names are not re-evaluated from their BNGL expressions.
    """
    if bngmodel is None or not getattr(bngmodel, "parameters", None):
        return dict(overrides or {})
    items = getattr(bngmodel.parameters, "items", None) or {}
    resolved = {}
    if overrides:
        for name, val in overrides.items():
            try:
                resolved[name] = float(val)
            except (TypeError, ValueError):
                continue
    pending = {
        name: getattr(p, "value", None)
        for name, p in items.items()
        if name not in resolved
    }

    while pending:
        progressed = False
        for name in list(pending):
            expr = pending[name]
            if expr is None:
                pending.pop(name)
                continue
            try:
                resolved[name] = _eval_numeric(str(expr), extra_ns=resolved)
            except ValueError:
                continue
            pending.pop(name)
            progressed = True
        if not progressed:
            break

    for name, expr in pending.items():
        logger.debug("bngmodel param %s=%r unresolved", name, expr)
    return resolved


def _evaluate_bngmodel_functions(bngmodel, base_params, obs_dict):
    """Evaluate parameterless BNGL functions for a single time point.

    BNGsim's NFsim binding returns an empty ``Result.expressions`` array, so
    NFsim parameter_scan output drops any ``begin functions`` columns even
    when the action requests ``print_functions=>1``. This helper recomputes
    those columns by evaluating each function's expression against a
    namespace built from resolved parameters and the observable values at
    the requested time point.

    Functions that take arguments are skipped — they require user-supplied
    arg values and don't appear as scan columns. Function-to-function
    references (``f() = 1 + g()``) resolve via fixed-point iteration.

    Returns ``(names, values)`` in declaration order; functions that fail
    to evaluate are silently dropped.
    """
    if bngmodel is None:
        return [], []
    fb = getattr(bngmodel, "functions", None)
    items = getattr(fb, "items", None) if fb is not None else None
    if not items:
        return [], []

    pending = {}
    for name, fn in items.items():
        if getattr(fn, "args", None):
            continue
        expr = getattr(fn, "expr", None)
        if not expr:
            continue
        pending[name] = str(expr)

    if not pending:
        return [], []

    func_names_set = set(items.keys())
    ns = dict(base_params or {})
    ns.update(obs_dict or {})

    resolved = {}
    while pending:
        progressed = False
        for name in list(pending):
            expr = _strip_zero_arg_calls(pending[name], func_names_set)
            try:
                val = float(_safe_eval_expr(expr, _safe_math_namespace(ns)))
            except Exception:
                continue
            resolved[name] = val
            ns[name] = val
            del pending[name]
            progressed = True
        if not progressed:
            break

    ordered = [n for n in items if n in resolved]
    return ordered, [resolved[n] for n in ordered]


def _evaluate_functions_per_timepoint(
    bngmodel, bngmodel_params, param_overrides, obs_names, obs_array,
):
    """Evaluate parameterless BNGL functions at every time-course row.

    Returns ``(col_names, col_array)`` where *col_names* are the function
    names rendered as ``name()`` (matching BNG2.pl/NFsim header style) and
    *col_array* has one column per function and one row per time point.
    Functions that fail to evaluate at the first row are dropped from all
    subsequent rows so the output stays rectangular.
    """
    import numpy as np

    effective_params = dict(bngmodel_params or {})
    for k, v in (param_overrides or {}).items():
        try:
            effective_params[k] = float(v)
        except (TypeError, ValueError):
            continue

    n_times = obs_array.shape[0]
    rows = []
    locked_names = None
    for i in range(n_times):
        obs_dict = dict(zip(obs_names, obs_array[i]))
        names, vals = _evaluate_bngmodel_functions(
            bngmodel, effective_params, obs_dict,
        )
        if locked_names is None:
            locked_names = names
            rows.append(vals)
        else:
            # Restrict to the first row's column set so the result is rectangular.
            d = dict(zip(names, vals))
            rows.append([d.get(n, float("nan")) for n in locked_names])

    if not locked_names:
        return [], np.empty((n_times, 0))
    arr = np.asarray(rows, dtype=float)
    return [f"{n}()" for n in locked_names], arr


def _strip_zero_arg_calls(expr_str, func_names):
    """Replace ``name()`` with ``name`` for each name in *func_names*.

    The safe expression walker treats ``f()`` as a call (looks up *f* in
    the namespace and calls it with no args). When *f* is a model function
    we have only a numeric value, not a callable, so rewrite the call form
    to a bare name lookup before evaluation.
    """
    import re

    out = expr_str
    for name in func_names:
        out = re.sub(rf"\b{re.escape(name)}\s*\(\s*\)", name, out)
    return out


# ─── Species initializer re-evaluation ─────────────────────────────


def _parse_net_species_initializers(net_path):
    """Parse (species_name, init_expression) pairs from a .net file.

    In a .net file, species lines look like::

        1 @b::X(p~0,y) 5000
        2 @b::X(p~1,y) k_init*100

    Only species whose initial concentration is a parameter expression
    (i.e., not a numeric literal) are returned. Constant initializers
    don't need re-evaluation when scan parameters change, and including
    them would force the slow sequential scan path and clobber any
    snapshot saved by ``saveConcentrations()`` or by parameter_scan
    itself (see ``_sync_species_concentrations``).

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
                        expr = m.group(2)
                        try:
                            float(expr)
                        except ValueError:
                            initializers.append((m.group(1), expr))
    except OSError as exc:
        logger.debug("could not read .net for species initializers (%s): %s", net_path, exc)
    return initializers


def _sync_species_concentrations(bngsim_model, initializers):
    """Re-evaluate species initial concentrations from .net expressions.

    Called after parameter changes so that derived species concentrations
    (e.g., ``S0 = I0 * kfactor``) track parameter updates. Sets the new
    values on the model directly; callers must apply this *after* any
    ``reset()`` so the overlay is not undone, and must not rely on
    ``save_concentrations()`` to make the values stick — that would
    overwrite any snapshot of post-time-course state held for the scan.
    """
    if not initializers:
        return

    # Build namespace from current model parameters
    param_values = {}
    for pname in bngsim_model.param_names:
        try:
            param_values[pname] = bngsim_model.get_param(pname)
        except Exception as exc:
            logger.warning("conc-sync: get_param(%s) failed: %s", pname, exc)

    ns = _safe_math_namespace(param_values)

    for species_name, expr_text in initializers:
        try:
            value = float(_safe_eval_expr(expr_text, ns))
        except ValueError as exc:
            logger.warning("conc-sync: could not evaluate %r for %s: %s", expr_text, species_name, exc)
            continue
        try:
            bngsim_model.set_concentration(species_name, value)
        except Exception as exc:
            logger.warning("conc-sync: set_concentration(%s, %s) failed: %s", species_name, value, exc)


def _parse_bngmodel_seed_species_initializers(bngmodel):
    """Parse (pattern, init_expression) pairs from a parsed bngmodel.

    NF parameter_scan does not generate a .net file (no network
    expansion), so seed species expressions must come from the parsed
    BNGL. Only species whose initial count is a parameter expression
    (not a numeric literal) are returned — constant initializers don't
    need re-evaluation.

    Returns a list of (species_pattern, expression_string) tuples.
    """
    if bngmodel is None or getattr(bngmodel, "species", None) is None:
        return []
    items = getattr(bngmodel.species, "items", None) or {}
    initializers = []
    for sp in items.values():
        pattern = getattr(sp, "pattern", None)
        count = getattr(sp, "count", None)
        if pattern is None or count is None:
            continue
        count_str = str(count)
        try:
            float(count_str)
        except ValueError:
            initializers.append((str(pattern), count_str))
    return initializers


def _parse_xml_parameter_table(xml_path):
    """Read every ``<Parameter id="..." expr="..."/>`` entry from a BNG XML.

    BNG2.pl emits both the user's BNGL parameters AND auto-generated
    ``_rateLaw*`` rate-law constants whose ``value=`` attribute is
    pre-computed at XML-export time. Returns ``[(name, expr), ...]`` in
    document order (so dependency order is roughly correct for the
    fixed-point resolver).
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, OSError) as exc:
        logger.debug("XML parse failed for %s: %s", xml_path, exc)
        return []
    root = tree.getroot()
    rows = []
    # Strip any namespace from tag for simple matching
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag != "Parameter":
            continue
        name = elem.get("id")
        expr = elem.get("expr") or elem.get("value")
        if name and expr is not None:
            rows.append((name, expr))
    return rows


def _resolve_xml_params(xml_param_table, overrides):
    """Iteratively evaluate XML parameter expressions against an override
    namespace until fixpoint. Returns ``{name: float}``."""
    resolved = {}
    if overrides:
        for n, v in overrides.items():
            try:
                resolved[n] = float(v)
            except (TypeError, ValueError):
                continue
    pending = [(n, e) for n, e in xml_param_table if n not in resolved]
    while pending:
        progressed = False
        next_pending = []
        for name, expr in pending:
            try:
                resolved[name] = _eval_numeric(str(expr), extra_ns=resolved)
                progressed = True
            except ValueError:
                next_pending.append((name, expr))
        pending = next_pending
        if not progressed:
            break
    return resolved


def _apply_nfsim_derived_params(
    nfsim, baseline_xml_params, xml_param_table, bngmodel, param_overrides,
    scan_param=None,
):
    """Push parameters whose values transitively change with a scan/override.

    NFsim loads rate-law parameter values from the BNG XML's pre-computed
    ``value=`` attribute, ignoring ``expr=`` for derived rate constants.
    So ``set_param("LT_conc_M", v)`` updates ``LT_conc_M`` but leaves
    ``kf1_pseudo = Ka_1*koff*LT_conc_M`` AND the auto-generated
    ``_rateLaw* = kf1_pseudo*use_excess`` pinned to their XML-time values.

    Solution: re-evaluate every ``<Parameter expr=...>`` in the XML
    (including ``_rateLaw*``) against the new override namespace and push
    every changed value via ``set_param``.

    *baseline_xml_params* is ``_resolve_xml_params(xml_param_table, {})``
    evaluated once by the caller and reused across scan points.
    """
    overrides = {}
    if param_overrides:
        for pname, pval in param_overrides.items():
            try:
                overrides[pname] = float(pval)
            except (TypeError, ValueError):
                continue
    if scan_param:
        sname, sval = scan_param
        if sname:
            try:
                overrides[sname] = float(sval)
            except (TypeError, ValueError):
                pass
    if not overrides:
        return

    point_params = _resolve_xml_params(xml_param_table, overrides)
    for name, new_val in point_params.items():
        base_val = baseline_xml_params.get(name)
        if base_val is None or base_val == new_val:
            continue
        denom = max(abs(base_val), abs(new_val), 1.0)
        if abs(base_val - new_val) / denom < 1e-12:
            continue
        try:
            nfsim.set_param(name, float(new_val))
        except Exception as exc:
            logger.debug(
                "NFsim derived: set_param(%s, %s) skipped: %s",
                name, new_val, exc,
            )


def _apply_nfsim_seed_species_initializers(
    nfsim, initializers, bngmodel, param_overrides, scan_param=None,
):
    """Re-evaluate parameter-derived seed species and apply to NFsim.

    The BNG XML hard-codes seed species counts at network-generation
    time, so a fresh ``NfsimSession`` initialized after a scan parameter
    change still carries the XML-time counts (e.g., ``LT = AT_nM*1e-9*NA*V``
    evaluated at the original ``AT_nM`` literal). Calling
    ``set_param("AT_nM", new_value)`` updates the rate-law expression
    engine but does not re-evaluate seed species. This helper closes
    that gap by re-resolving the bngmodel parameter block with
    *param_overrides* and *scan_param* applied — so derived parameters
    like ``LT`` flow through transitively — then evaluating each
    parameter-dependent seed species expression against that namespace
    and pushing the new count via
    ``NfsimSession.set_species_count(pattern, count)``.

    *scan_param*, when provided, is ``(name, value)`` for the current
    scan point.
    """
    if not initializers:
        return

    overrides = {}
    if param_overrides:
        for pname, pval in param_overrides.items():
            try:
                overrides[pname] = float(pval)
            except (TypeError, ValueError):
                continue
    if scan_param:
        sname, sval = scan_param
        if sname:
            try:
                overrides[sname] = float(sval)
            except (TypeError, ValueError):
                pass

    point_params = _resolve_bngmodel_params(bngmodel, overrides=overrides)
    ns = _safe_math_namespace(point_params)

    for pattern, expr_text in initializers:
        try:
            new_count = int(round(float(_safe_eval_expr(expr_text, ns))))
        except (ValueError, TypeError) as exc:
            logger.warning(
                "NFsim scan: could not evaluate %r for %s: %s",
                expr_text, pattern, exc,
            )
            continue
        if new_count < 0:
            logger.warning(
                "NFsim scan: %s evaluated to negative count %d; skipping",
                pattern, new_count,
            )
            continue
        try:
            nfsim.set_species_count(pattern, new_count)
        except Exception as exc:
            logger.warning(
                "NFsim scan: set_species_count(%s, %d) failed: %s",
                pattern, new_count, exc,
            )


# ─── Codegen helpers ───────────────────────────────────────────────


def _net_has_tfun(net_path):
    """Return True if the .net file's function block uses tfun(...).

    Codegen+tfun is fragile in current BNGsim (the compiled .so calls a
    callback that segfaults if the table function dispatch is not wired
    up exactly right at runtime). The interpreted RHS handles tfun
    correctly, so we route models containing tfun there until BNGsim's
    codegen path stabilizes.
    """
    try:
        with open(net_path, "r", errors="replace") as f:
            in_functions = False
            for line in f:
                s = line.strip()
                if s.startswith("begin functions"):
                    in_functions = True
                    continue
                if s.startswith("end functions"):
                    return False
                if in_functions and "tfun(" in s.lower():
                    return True
    except OSError as exc:
        logger.debug("could not scan .net for tfun (%s): %s", net_path, exc)
    return False


def _try_prepare_codegen(net_path):
    """Attempt to compile a code-generated RHS for ODE simulation.

    Returns the path to the compiled shared library, or "" if codegen
    is unavailable, disabled via BIONETGEN_NO_CODEGEN env var, or skipped
    because the model uses ``tfun(...)`` (BNGsim codegen+tfun is unstable;
    the interpreted RHS handles tfun correctly).
    """
    if os.environ.get("BIONETGEN_NO_CODEGEN"):
        return ""
    if _net_has_tfun(net_path):
        logger.info(
            "Codegen disabled for model with tfun() function; "
            "using interpreted ODE RHS (codegen+tfun is currently unstable in BNGsim)"
        )
        return ""
    try:
        from bngsim import prepare_codegen

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
        "nf_params": _parse_nfsim_param_string(args),
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
            if sp is None or not _is_nf_method(sp["method"]):
                return True
            continue
        if a.type in ("parameter_scan", "bifurcate"):
            m = _strip_quotes(a.args.get("method", "ode").strip())
            if m == "protocol" or not _is_nf_method(m):
                return True
            continue
        if a.type in _NF_ONLY_STATE_ACTIONS or a.type in _NF_SAFE_BNG2PL_ACTIONS:
            continue
        return True
    return False


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
        func_names = list(result.expression_names)
        func_array = np.asarray(result.expressions)
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
        except Exception as exc:
            logger.debug("protocol: initial get_param(%s) failed: %s", pname, exc)

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
                conc_val = _eval_numeric(
                    conc_str, extra_ns=_model_param_namespace(bngsim_model),
                )
                bngsim_model.set_concentration(species_name, conc_val)
            except Exception as exc:
                logger.warning("protocol: setConcentration(%s, %s) failed: %s", species_name, conc_str, exc)
            # Force a Simulator rebuild on the next simulate line, mirroring
            # the defensive pattern in _execute_bngsim_actions.
            current_method = None
            current_poplevel = None
            continue

        # ── setParameter ──
        sp = _setparam_re.search(line)
        if sp:
            param_name = sp.group(1)
            param_str = sp.group(2).strip()
            try:
                param_val = _eval_numeric(
                    param_str, extra_ns=_model_param_namespace(bngsim_model),
                )
                bngsim_model.set_param(param_name, param_val)
            except Exception as exc:
                logger.warning("protocol: setParameter(%s, %s) failed: %s", param_name, param_str, exc)
            # Force a Simulator rebuild on the next simulate line, mirroring
            # the defensive pattern in _execute_bngsim_actions.
            current_method = None
            current_poplevel = None
            continue

        # ── resetConcentrations ──
        if _resetconc_re.search(line):
            bngsim_model.reset()
            # Force a Simulator rebuild on the next simulate line, mirroring
            # the defensive pattern in _execute_bngsim_actions.
            current_method = None
            current_poplevel = None
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
                except Exception as exc:
                    logger.debug("protocol: saveParameters get_param(%s) failed: %s", pname, exc)
            continue

        # ── resetParameters ──
        if _resetparam_re.search(line):
            for pname, pval in saved_params.items():
                try:
                    bngsim_model.set_param(pname, pval)
                except Exception as exc:
                    logger.warning("protocol: resetParameters set_param(%s, %s) failed: %s", pname, pval, exc)
            # Defer Simulator rebuild to the next simulate line. The previous
            # eager rebuild called ``bngsim.Simulator(model, method=current_method,
            # **codegen_kw)``, which crashed unconditionally when current_method
            # was "psa" (BNGsim raises ValueError because poplevel is required)
            # or when codegen was active for ssa/psa (BNGsim rejects codegen=True
            # for non-ODE). Lazy rebuild routes through the simulate-line logic
            # which branches correctly on method/poplevel/codegen.
            current_method = None
            current_poplevel = None
            continue

        logger.debug("protocol: skipping unrecognized command: %s", line)

    return last_result


def _run_nfsim_scan(
    xml_path, action, output_dir, model_name, is_bifurcate=False,
    param_overrides=None,
    conc_overrides=None,
    conc_deltas=None,
    bngmodel=None,
    bngmodel_params=None,
    nf_params=None,
):
    """Execute a parameter_scan with NFsim: fresh NfsimSession per scan point.

    NFsim is stateless (no .net model to clone), so each scan point gets a
    fresh session loaded from the BNG XML file.

    When the action requests ``print_functions=>1``, BNGL functions are
    evaluated post-hoc from the parsed *bngmodel* (BNGsim's NFsim binding
    does not surface function values directly).

    Parameter-derived seed species (e.g., ``L(r1,r2) LT`` where
    ``LT = AT_nM*1e-9*NA*V``) have their counts re-evaluated per scan
    point and pushed via ``set_species_count``: the BNG XML hard-codes
    seed counts at network-generation time, so without this they would
    stay pinned to the XML-time value for every scan point regardless
    of the scan parameter.
    """
    import numpy as np

    args = action.args
    param_name = _strip_quotes(args.get("parameter", "").strip())
    t_start = float(args.get("t_start", 0))
    t_end = float(args.get("t_end", 100))
    n_steps = int(float(args.get("n_steps", 100)))
    suffix = _strip_quotes(args.get("suffix", "").strip()) or "scan"
    print_funcs = bool(int(float(args.get("print_functions", 0))))
    gml = int(float(args["gml"])) if "gml" in args else None
    base_seed = int(float(args.get("seed", 42)))

    seed_initializers = _parse_bngmodel_seed_species_initializers(bngmodel)
    xml_param_table = _parse_xml_parameter_table(xml_path)
    baseline_xml_params = _resolve_xml_params(xml_param_table, overrides=None)

    points = _resolve_scan_points(args)
    rows = []
    obs_names = None
    func_names = None

    nf_kwargs = _nfsim_session_kwargs(nf_params)
    for i, value in enumerate(points):
        with bngsim.NfsimSession(xml_path, molecule_limit=gml, **nf_kwargs) as nfsim:
            # Apply parameter overrides from prior setParameter actions
            if param_overrides:
                for pname, pval in param_overrides.items():
                    try:
                        nfsim.set_param(pname, float(pval))
                    except Exception as exc:
                        logger.debug("NFsim scan: set_param(%s, %s) skipped: %s", pname, pval, exc)
            if param_name:
                try:
                    nfsim.set_param(param_name, float(value))
                except Exception as exc:
                    logger.warning(
                        "NFsim scan: could not set %s=%s: %s", param_name, value, exc
                    )
            # Push derived rate constants (including auto-generated
            # _rateLaw*) whose values transitively change with the scan
            # parameter — NFsim doesn't re-evaluate XML ``expr=`` strings
            # on set_param, so e.g. kf_pseudo = Ka*koff*X stays pinned to
            # its XML-time value otherwise.
            _apply_nfsim_derived_params(
                nfsim, baseline_xml_params, xml_param_table, bngmodel,
                param_overrides,
                scan_param=(param_name, value) if param_name else None,
            )
            nfsim.initialize((base_seed + i) % (2**31))
            _apply_nfsim_seed_species_initializers(
                nfsim, seed_initializers,
                bngmodel, param_overrides,
                scan_param=(param_name, value) if param_name else None,
            )
            _apply_nfsim_concentration_changes(
                nfsim,
                conc_overrides=conc_overrides,
                conc_deltas=conc_deltas,
            )
            result = nfsim.simulate(t_start, t_end, n_steps + 1)

            row, row_obs, row_funcs = _scan_result_to_row(
                result, float(value), print_functions=print_funcs,
            )
            # NFsim exposes obs values but not BNGL functions; recompute
            # them from the parsed bngmodel using this point's params + obs.
            if print_funcs and not row_funcs and bngmodel is not None:
                obs_dict = dict(zip(row_obs, row[1:1 + len(row_obs)]))
                point_params = dict(bngmodel_params or {})
                if param_name:
                    try:
                        point_params[param_name] = float(value)
                    except (TypeError, ValueError):
                        pass
                if param_overrides:
                    for pname, pval in param_overrides.items():
                        try:
                            point_params[pname] = float(pval)
                        except (TypeError, ValueError):
                            continue
                fn_names, fn_vals = _evaluate_bngmodel_functions(
                    bngmodel, point_params, obs_dict,
                )
                if fn_names:
                    row = np.concatenate(
                        (row, np.asarray(fn_vals, dtype=float))
                    )
                    row_funcs = [f"{n}()" for n in fn_names]
            rows.append(row)
            if obs_names is None:
                obs_names = row_obs
                func_names = row_funcs

    col_names = (obs_names or []) + (func_names or [])
    scan_path = os.path.join(output_dir, f"{model_name}_{suffix}.scan")
    _write_scan_file(scan_path, param_name or "scan_param", col_names, rows)


def _prepare_scan_point(base_model, param_name, value, species_initializers):
    """Clone the base model, apply the scan parameter, and refresh initials.

    The clone inherits the snapshot the caller holds for the scan's
    starting state (typically the post-time-course concentrations). We
    ``reset()`` first to restore that snapshot, then overlay the scan
    parameter and any expression-based initial concentrations on top —
    constant initializers were already filtered out by
    ``_parse_net_species_initializers``.
    """
    point_model = base_model.clone()
    point_model.reset()
    if param_name:
        point_model.set_param(param_name, _eval_numeric(str(value)))
    if species_initializers:
        _sync_species_concentrations(point_model, species_initializers)
    return point_model


_DEFAULT_SS_WORKERS = 4


def _resolve_ss_workers(default=_DEFAULT_SS_WORKERS):
    """Resolve the steady-state scan thread pool worker count.

    Reads ``BIONETGEN_SS_WORKERS`` from the environment; falls back to
    *default* when unset, malformed, or non-positive.
    """
    raw = os.environ.get("BIONETGEN_SS_WORKERS")
    if not raw:
        return default
    try:
        n = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "BIONETGEN_SS_WORKERS=%r is not an integer; using default %d",
            raw, default,
        )
        return default
    if n < 1:
        logger.warning(
            "BIONETGEN_SS_WORKERS=%d must be >= 1; using default %d",
            n, default,
        )
        return default
    return n


def _run_ss_scan_threaded(
    base_model, param_name, points, species_initializers,
    make_sim_fn, codegen_so, net_path, t_start, t_end, print_funcs,
    max_workers=None,
):
    """Run steady-state parameter scan with threaded parallelism.

    Prepares all point models sequentially (species initializer sync is not
    thread-safe), then submits steady_state() calls to a thread pool.
    Falls back to long time-course per point on non-convergence or error.

    The worker count comes from ``BIONETGEN_SS_WORKERS`` (env var) when
    *max_workers* is None, falling back to ``_DEFAULT_SS_WORKERS``.
    """
    if max_workers is None:
        max_workers = _resolve_ss_workers()
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
    nf_conc_overrides=None, nf_conc_deltas=None,
    bngmodel=None, bngmodel_params=None,
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
            conc_overrides=nf_conc_overrides,
            conc_deltas=nf_conc_deltas,
            bngmodel=bngmodel,
            bngmodel_params=bngmodel_params,
            nf_params=_parse_nfsim_param_string(args),
        )

    if bngsim_model is None:
        raise BNGSimError(
            f"method='{method}' requires a generated network model."
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
            # Restore the snapshot saved at scan entry (line above) so the
            # scan starts from the post-time-course state, then overlay the
            # scan parameter and any expression-based initial concentrations
            # on top. Resetting after IC sync would clobber the overlay.
            point_model.reset()
        else:
            point_model = bngsim_model

        if param_name:
            point_model.set_param(param_name, _eval_numeric(str(value)))

        # Re-evaluate species ICs that depend on parameters
        if species_initializers:
            _sync_species_concentrations(point_model, species_initializers)

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
                    point_model.reset()
                else:
                    point_model = bngsim_model
                if param_name:
                    point_model.set_param(param_name, _eval_numeric(str(value)))
                if species_initializers:
                    _sync_species_concentrations(point_model, species_initializers)
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


def _read_scan_file(path):
    """Parse a .scan file written by ``_write_scan_file``.

    Returns ``(header_cols, rows)`` where ``header_cols`` is the list of
    column names (param name + observables/functions) and ``rows`` is a
    list of float lists, one per scan point.
    """
    header_cols = []
    rows = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if not header_cols:
                    header_cols = stripped.lstrip("#").split()
                continue
            rows.append([float(v) for v in stripped.split()])
    return header_cols, rows


def _run_bifurcate_bngsim(
    bngsim_model, action, output_dir, model_name,
    codegen_so="", net_path=None, species_initializers=None,
    protocol_lines=None, xml_path=None,
    nf_param_overrides=None, nf_conc_overrides=None, nf_conc_deltas=None,
    bngmodel=None, bngmodel_params=None,
):
    """Execute a bifurcate action: forward + backward scans, then split.

    Mirrors BNG2.pl's ``BNGAction::bifurcate``: runs the parameter scan
    once forward (par_min → par_max) and once backward (par_max →
    par_min), then for each observable/function writes one
    ``<model>_<suffix>_bifurcation_<obs>.scan`` file with three columns
    ``param``, ``<obs>_fwd``, ``<obs>_bwd``. Both passes share the
    underlying BNGsim model with ``reset_conc=False`` so each scan point
    continues from the previous point's final state — that's what
    surfaces hysteresis in bistable systems.

    The two intermediate forward/backward scan files are deleted after
    splitting, matching BNG2.pl behavior.
    """
    import copy

    args = action.args
    base_suffix = _strip_quotes(args.get("suffix", "").strip()) or "bif"

    par_scan_vals = args.get("par_scan_vals")

    # Forward pass: original par_min/par_max, suffix becomes "<suffix>_forward"
    fwd_action = copy.copy(action)
    fwd_action.args = dict(args)
    fwd_action.args["suffix"] = base_suffix + "_forward"

    # Backward pass: par_min/par_max swapped (or par_scan_vals reversed),
    # suffix becomes "<suffix>_backward"
    bwd_action = copy.copy(action)
    bwd_action.args = dict(args)
    bwd_action.args["suffix"] = base_suffix + "_backward"
    if par_scan_vals is not None:
        raw = par_scan_vals.strip().strip("[]")
        vals = [v.strip() for v in raw.split(",")]
        bwd_action.args["par_scan_vals"] = "[" + ",".join(reversed(vals)) + "]"
    else:
        bwd_action.args["par_min"] = args["par_max"]
        bwd_action.args["par_max"] = args["par_min"]

    common_kwargs = dict(
        codegen_so=codegen_so, net_path=net_path,
        species_initializers=species_initializers,
        protocol_lines=protocol_lines, xml_path=xml_path,
        nf_param_overrides=nf_param_overrides,
        nf_conc_overrides=nf_conc_overrides,
        nf_conc_deltas=nf_conc_deltas,
        bngmodel=bngmodel, bngmodel_params=bngmodel_params,
    )

    _run_parameter_scan_bngsim(
        bngsim_model, fwd_action, output_dir, model_name,
        is_bifurcate=True, **common_kwargs,
    )
    _run_parameter_scan_bngsim(
        bngsim_model, bwd_action, output_dir, model_name,
        is_bifurcate=True, **common_kwargs,
    )

    fwd_path = os.path.join(output_dir, f"{model_name}_{base_suffix}_forward.scan")
    bwd_path = os.path.join(output_dir, f"{model_name}_{base_suffix}_backward.scan")
    fwd_header, fwd_rows = _read_scan_file(fwd_path)
    bwd_header, bwd_rows = _read_scan_file(bwd_path)

    if len(fwd_rows) != len(bwd_rows):
        raise BNGSimError(
            f"bifurcate: forward ({len(fwd_rows)}) and backward "
            f"({len(bwd_rows)}) scans produced different row counts"
        )
    if fwd_header != bwd_header:
        raise BNGSimError(
            "bifurcate: forward and backward scans produced different columns"
        )

    n = len(fwd_rows)
    param_col = fwd_header[0] if fwd_header else "scan_param"

    # Per-observable bifurcation files: param, obs_fwd, obs_bwd. Backward
    # rows are aligned by reversing the index so each output row reports
    # the forward and backward observable values at the same param value.
    for j, obs in enumerate(fwd_header[1:], start=1):
        rows = [
            [fwd_rows[i][0], fwd_rows[i][j], bwd_rows[n - 1 - i][j]]
            for i in range(n)
        ]
        out_path = os.path.join(
            output_dir, f"{model_name}_{base_suffix}_bifurcation_{obs}.scan"
        )
        _write_scan_file(out_path, param_col, [f"{obs}_fwd", f"{obs}_bwd"], rows)

    # Drop intermediate scans — only the per-observable bifurcation files
    # are part of bifurcate's contract.
    for path in (fwd_path, bwd_path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.debug("bifurcate: could not remove %s: %s", path, exc)


def _execute_bngsim_actions(
    actions_items, bngsim_model, output_dir, model_name,
    xml_path=None, net_path=None, protocol_lines=None,
    bngmodel_params=None, bngmodel=None,
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
    # Track which simulate output basenames have been written this run, so
    # continue=>1 segments append instead of clobbering the prior segment.
    written_out_names = set()
    # For pure-NF runs (bngsim_model is None) we cannot read live parameter
    # values from a BNGsim Model — track them ourselves so set/setParameter
    # actions can resolve names like ``setParameter("LT_current","LT_low")``.
    live_nf_params = dict(bngmodel_params) if bngmodel_params else {}
    # Track parameter overrides for NFsim propagation.
    # NFsim loads from XML and doesn't share state with bngsim_model,
    # so setParameter changes must be explicitly forwarded.
    nf_param_overrides = {}
    # Track absolute concentration targets for NFsim propagation.
    nf_conc_overrides = {}
    # Track additive concentration deltas when no absolute target is known.
    nf_conc_deltas = {}

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
    saved_nf_param_overrides = {}
    saved_nf_conc_overrides = {}
    saved_nf_conc_deltas = {}
    if bngsim_model is not None:
        for pname in bngsim_model.param_names:
            try:
                saved_params[pname] = bngsim_model.get_param(pname)
            except Exception as exc:
                logger.debug("actions: initial get_param(%s) failed: %s", pname, exc)

    for action in actions_items:
        atype = action.type

        # Skip actions handled by BNG2.pl preprocessing
        if atype in _BNG2PL_ACTIONS:
            continue

        # PLA is not implemented in BNGsim — BNG2.pl ran it during preprocessing.
        if _is_pla_action(action):
            continue

        # ── simulate_* ──────────────────────────────────────────
        if atype.startswith("simulate"):
            sp = _parse_simulate_params(action)
            if sp is None:
                logger.warning("Unrecognized simulate action: %s", atype)
                continue

            method = sp["method"]

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
                    conc_deltas=nf_conc_deltas or None,
                    print_functions=print_funcs,
                    bngmodel=bngmodel,
                    bngmodel_params=bngmodel_params,
                    nf_params=sp.get("nf_params"),
                )
                current_method = "nf"
                current_poplevel = None
            else:
                if bngsim_model is None:
                    raise BNGSimError(
                        f"method='{method}' requires a generated network model."
                    )
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
                    append=continue_flag and out_name in written_out_names,
                )
                written_out_names.add(out_name)

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
                nf_conc_overrides=nf_conc_overrides or None,
                nf_conc_deltas=nf_conc_deltas or None,
                bngmodel=bngmodel,
                bngmodel_params=bngmodel_params,
            )
            continue

        # ── bifurcate ───────────────────────────────────────────
        if atype == "bifurcate":
            _run_bifurcate_bngsim(
                bngsim_model, action, output_dir, model_name,
                codegen_so=codegen_so, net_path=net_path,
                species_initializers=species_initializers,
                protocol_lines=protocol_lines, xml_path=xml_path,
                nf_param_overrides=nf_param_overrides or None,
                nf_conc_overrides=nf_conc_overrides or None,
                nf_conc_deltas=nf_conc_deltas or None,
                bngmodel=bngmodel,
                bngmodel_params=bngmodel_params,
            )
            continue

        # ── setParameter ────────────────────────────────────────
        if atype == "setParameter":
            name, value = _extract_positional_args(action)
            numeric_value = _eval_numeric(
                value,
                extra_ns=_model_param_namespace(bngsim_model, fallback=live_nf_params),
            )
            if bngsim_model is not None:
                try:
                    bngsim_model.set_param(name, numeric_value)
                    logger.debug("setParameter(%s, %s)", name, value)
                except Exception as e:
                    logger.warning("setParameter(%s, %s) failed: %s", name, value, e)
            else:
                live_nf_params[name] = numeric_value
            # Track for NFsim propagation
            nf_param_overrides[name] = numeric_value
            # Invalidate simulator cache — params changed
            current_sim = None
            current_method = None
            continue

        # ── setConcentration ────────────────────────────────────
        if atype == "setConcentration":
            name, value = _extract_positional_args(action)
            numeric_value = _eval_numeric(
                value,
                extra_ns=_model_param_namespace(bngsim_model, fallback=live_nf_params),
            )
            if bngsim_model is not None:
                try:
                    bngsim_model.set_concentration(name, numeric_value)
                    logger.debug("setConcentration(%s, %s)", name, value)
                except Exception as e:
                    logger.warning("setConcentration(%s, %s) failed: %s", name, value, e)
            nf_conc_overrides[name] = round(numeric_value)
            nf_conc_deltas.pop(name, None)
            continue

        # ── addConcentration ────────────────────────────────────
        if atype == "addConcentration":
            name, value = _extract_positional_args(action)
            numeric_delta = _eval_numeric(
                value,
                extra_ns=_model_param_namespace(bngsim_model, fallback=live_nf_params),
            )
            if bngsim_model is not None:
                try:
                    current = bngsim_model.get_concentration(name)
                    new_val = current + numeric_delta
                    bngsim_model.set_concentration(name, new_val)
                    logger.debug("addConcentration(%s, %s)", name, value)
                except Exception as e:
                    logger.warning("addConcentration(%s, %s) failed: %s", name, value, e)
            # Track for NFsim propagation as a delta. NFsim's live count can
            # diverge from the network model (separate stochastic trajectory),
            # so derive the NFsim target additively rather than from the
            # network model's concentration.
            rounded_delta = round(numeric_delta)
            if name in nf_conc_overrides:
                nf_conc_overrides[name] += rounded_delta
            else:
                new_delta = nf_conc_deltas.get(name, 0) + rounded_delta
                if new_delta:
                    nf_conc_deltas[name] = new_delta
                else:
                    nf_conc_deltas.pop(name, None)
            continue

        # ── saveConcentrations ──────────────────────────────────
        if atype == "saveConcentrations":
            if bngsim_model is not None:
                bngsim_model.save_concentrations()
            saved_nf_conc_overrides = dict(nf_conc_overrides)
            saved_nf_conc_deltas = dict(nf_conc_deltas)
            continue

        # ── resetConcentrations ─────────────────────────────────
        if atype == "resetConcentrations":
            if bngsim_model is not None:
                bngsim_model.reset()
            nf_conc_overrides = dict(saved_nf_conc_overrides)
            nf_conc_deltas = dict(saved_nf_conc_deltas)
            continue

        # ── saveParameters ──────────────────────────────────────
        if atype == "saveParameters":
            if bngsim_model is not None:
                saved_params = {}
                for pname in bngsim_model.param_names:
                    try:
                        saved_params[pname] = bngsim_model.get_param(pname)
                    except Exception as exc:
                        logger.debug("saveParameters: get_param(%s) failed: %s", pname, exc)
            saved_nf_param_overrides = dict(nf_param_overrides)
            continue

        # ── resetParameters ─────────────────────────────────────
        if atype == "resetParameters":
            if bngsim_model is not None:
                for pname, pval in saved_params.items():
                    try:
                        bngsim_model.set_param(pname, pval)
                    except Exception as exc:
                        logger.warning("resetParameters: set_param(%s, %s) failed: %s", pname, pval, exc)
            nf_param_overrides = dict(saved_nf_param_overrides)
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
            raw_lines = list(f)
    except OSError as exc:
        logger.debug("could not read BNGL for table functions (%s): %s", bngl_path, exc)
        return tfun_specs

    # Join logical lines: BNGL allows backslash line continuation, and the
    # inline-array tfun() form often spans multiple physical lines for
    # readability. Walk through raw lines, glue any line that ends with '\'
    # to the next, and produce a list of (logical_line, in_functions_block)
    # tuples to scan for tfun() calls.
    logical_lines = []
    pending = ""
    for raw_line in raw_lines:
        # Detect backslash continuation. We want to drop the trailing '\'
        # (and any whitespace before the newline) and concatenate with the
        # next line's leading content.
        stripped_eol = raw_line.rstrip("\n").rstrip("\r")
        if stripped_eol.rstrip().endswith("\\"):
            # Pull off the trailing backslash and append; do not flush yet.
            pending += stripped_eol.rstrip()[:-1] + " "
            continue
        logical_lines.append(pending + stripped_eol)
        pending = ""
    if pending:
        logical_lines.append(pending)

    for raw_line in logical_lines:
        stripped = raw_line.strip()
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

        if "tfun(" not in clean:
            continue

        eq_match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*=", clean)
        if not eq_match:
            continue
        func_name = eq_match.group(1)

        tfun_match = re.search(r"tfun\((.+)\)", clean)
        if not tfun_match:
            continue
        tfun_body = tfun_match.group(1)

        spec = _parse_tfun_args(func_name, tfun_body, bngl_dir)
        if spec is not None:
            tfun_specs.append(spec)

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
    except OSError as exc:
        logger.debug("could not read BNGL for protocol block (%s): %s", bngl_path, exc)

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
        Simulation method override. If None, preserves the BNGL file's
        declared ``simulate_*`` methods.
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
    # PLA actions stay in the BNGL: BNGsim has no PLA, so BNG2.pl runs them.
    preserved_actions = [
        a for a in original_actions
        if a.type not in _BNGSIM_HANDLED or _is_pla_action(a)
    ]
    model.actions.clear_actions()
    if needs_network:
        # Preserve original generate_network args (e.g. max_stoich, max_iter,
        # check_iso). Without this, models that rely on max_stoich to bound
        # network expansion would have BNG2.pl run unbounded.
        gen_net_args = {}
        for a in original_actions:
            if a.type == "generate_network" and a.args:
                gen_net_args.update(a.args)
                break
        gen_net_args["overwrite"] = 1
        model.add_action("generate_network", gen_net_args)
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

    # Resolve the BNGL parameter block once. For pure-NF runs this is the
    # only source of parameter values; for network-based runs the BNGsim
    # Model is the source of truth and the dict is used only as a fallback
    # if a name is missing (defensive — should not happen in practice).
    bngmodel_params = _resolve_bngmodel_params(model)

    if bngsim_model is not None:
        return _execute_bngsim_actions(
            original_actions,
            bngsim_model,
            output_dir,
            model_name,
            xml_path=xml_arg,
            net_path=net_arg,
            protocol_lines=protocol_lines,
            bngmodel_params=bngmodel_params,
            bngmodel=model,
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
            bngmodel_params=bngmodel_params,
            bngmodel=model,
        )
