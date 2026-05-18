"""Bridge module for optional BNGsim integration.

BNGsim is a high-performance C++ simulation engine with Python bindings
that can replace run_network and NFsim for in-process simulation.
This module handles availability detection, input format detection,
and routing simulation requests to BNGsim when available.
"""

import inspect
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

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

BNGSIM_HAS_RULEMONKEY = False
if BNGSIM_AVAILABLE:
    try:
        from bngsim import HAS_RULEMONKEY

        BNGSIM_HAS_RULEMONKEY = bool(HAS_RULEMONKEY)
    except (ImportError, AttributeError):
        BNGSIM_HAS_RULEMONKEY = bool(getattr(bngsim, "RuleMonkeySession", None))

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

ROUTE_DIRECT_BNGSIM = "direct-bngsim"
ROUTE_BNGL_BNGSIM = "bngl-bngsim"
ROUTE_SUBPROCESS = "subprocess"
ROUTE_ERROR = "error"


@dataclass(frozen=True)
class BngsimRouteDecision:
    """Conservative routing decision for optional BNGsim use."""

    route: str
    reason: str
    method: str | None = None


@dataclass(frozen=True)
class BngsimDirectJob:
    """Normalized direct BNGsim job for non-BNGL artifacts."""

    input_path: str
    input_format: str
    method: str
    t_span: tuple[float, float]
    n_points: int
    output_dir: str
    output_root: str
    bngsim_options: dict | None = None
    result_options: dict | None = None


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


def _truncate_cdat_to_endpoints(cdat_path):
    """Reduce a .cdat to its comment header plus first and last data rows.

    Matches BNG2.pl's ``print_CDAT=>0`` behavior: BNG2.pl still emits a
    .cdat (the final row carries the end-state used for concentration
    write-back) but only the initial and final concentration rows, not
    the full trajectory.
    """
    with open(cdat_path) as handle:
        lines = handle.readlines()
    header = [ln for ln in lines if ln.lstrip().startswith("#")]
    data = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if len(data) <= 2:
        return
    with open(cdat_path, "w") as handle:
        handle.writelines(header + [data[0], data[-1]])


def _write_bngsim_results(
    result, output_dir, model_name,
    print_functions=False, append=False, print_cdat=True,
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
    print_cdat : bool
        If False, the .cdat is reduced to its initial and final rows,
        matching BNG2.pl's ``print_CDAT=>0`` behavior. Default True
        (full trajectory), matching BNG2.pl's default.
    Function columns are written only when BNGsim supplies them in
    ``Result.expressions``. BNGL-owned function semantics are handled by
    BNG2.pl before invoking the backend helper.
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
        if not print_cdat:
            _truncate_cdat_to_endpoints(cdat_path)
        return

    # Fresh write (default and first-segment-of-continuation path)
    result.to_cdat(cdat_path)
    if not print_cdat:
        _truncate_cdat_to_endpoints(cdat_path)
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


def _load_direct_bngsim_model(input_path, fmt):
    """Load a direct network-backed artifact into a BNGsim Model."""
    if fmt == FORMAT_NET:
        return bngsim.Model.from_net(input_path)
    if fmt == FORMAT_SBML:
        return bngsim.Model.from_sbml(input_path)
    if fmt == FORMAT_ANTIMONY:
        return bngsim.Model.from_antimony(input_path)
    raise BNGSimError(f"Unsupported format for BNGsim: '{fmt}'")


# Options the BNG2.pl backend hook / direct job may carry, split by which
# BNGsim entry point consumes them. ``bngsim.Simulator.__init__`` takes
# model-construction options; ``bngsim.Simulator.run`` takes per-run
# integration options. Anything else (e.g. ``print_CDAT``, an output-format
# flag BNG2.pl always emits) is not a BNGsim argument and is dropped here.
_SIMULATOR_INIT_OPTIONS = frozenset({
    "poplevel", "gml", "connectivity", "nfsim_v1143_compat",
    "block_same_complex_binding", "traversal_limit", "jacobian",
    "codegen", "net_path", "strict_ssa",
})
_SIMULATOR_RUN_OPTIONS = frozenset({
    "seed", "rtol", "atol", "max_steps", "sample_times",
})


def _partition_simulator_options(sim_options):
    """Split direct-job options into Simulator __init__ vs run kwargs.

    Returns ``(init_kwargs, run_kwargs)``. Options BNGsim's network
    Simulator does not accept are dropped: ``print_CDAT`` is an output
    flag (the .cdat is always written for network models), and ``sparse``
    / ``steady_state`` are not part of the BNGsim Simulator API — a
    ``steady_state`` request is surfaced as a warning since it would
    otherwise silently run as a plain time course.
    """
    init_kwargs = {}
    run_kwargs = {}
    dropped = []
    for key, value in sim_options.items():
        if value is None:
            continue
        if key in _SIMULATOR_INIT_OPTIONS:
            init_kwargs[key] = value
        elif key in _SIMULATOR_RUN_OPTIONS:
            run_kwargs[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.debug("Direct BNGsim job: ignoring non-Simulator options %s", sorted(dropped))
    if sim_options.get("steady_state"):
        logger.warning(
            "Direct BNGsim route does not support steady_state; "
            "running a plain time course instead."
        )
    return init_kwargs, run_kwargs


def _run_rulemonkey_job(job, input_path, output_dir, sim_options, result_options):
    """Execute a network-free RuleMonkey job from a BioNetGen XML artifact.

    BNG2.pl has no ``rm`` method, so ``method=>"rm"`` BNGL is rewritten to
    ``nf`` before BNG2.pl runs (see :func:`_rewrite_rm_method_to_nf`); the
    ``simulate_nf`` backend hook fires and the helper restores ``rm`` from
    ``BIONETGEN_BNGSIM_BACKEND_METHOD``. This adapter drives BNGsim's
    ``RuleMonkeySession`` instead of ``NfsimSession``.
    """
    if not BNGSIM_HAS_RULEMONKEY:
        raise BNGSimError(
            "BNGsim RuleMonkey support is not available in this build."
        )

    seed = sim_options.pop("seed", None)
    if seed is None:
        seed = 42
    gml = sim_options.pop("gml", None)
    param_overrides = sim_options.pop("param_overrides", None)
    # conc overrides/deltas come from setConcentration in multi-segment
    # workflows; the simulate_nf hook does not forward them and
    # RuleMonkeySession exposes no equivalent — warn if ever present.
    for key in ("conc_overrides", "conc_deltas", "nf_params"):
        if sim_options.pop(key, None):
            logger.warning(
                "RuleMonkey job: '%s' is not supported and was ignored", key
            )

    with bngsim.RuleMonkeySession(input_path, molecule_limit=gml) as rm_session:
        if param_overrides:
            for pname, pval in param_overrides.items():
                try:
                    rm_session.set_param(pname, float(pval))
                except Exception as exc:
                    logger.debug(
                        "RuleMonkey: set_param(%s, %s) skipped: %s", pname, pval, exc
                    )
        rm_session.initialize(seed)
        result = rm_session.simulate(job.t_span[0], job.t_span[1], job.n_points)

    _write_bngsim_results(result, output_dir, job.output_root, **result_options)
    return _make_bng_result(output_dir, method=job.method)


def execute_bngsim_direct_job(job):
    """Execute a normalized direct BNGsim job and write BNG-compatible files.

    The caller owns BNGL parsing/action semantics and supplies a fully
    normalized artifact job. This adapter only loads the direct artifact,
    dispatches to BNGsim, and writes the direct-run result files.
    """
    if not BNGSIM_AVAILABLE:
        raise BNGSimError(
            f"BNGsim is required for format '{job.input_format}' but is not installed. "
            "Install with: pip install bngsim"
        )

    input_path = os.path.abspath(job.input_path)
    output_dir = os.path.abspath(job.output_dir)
    sim_options = dict(job.bngsim_options or {})
    result_options = dict(job.result_options or {})

    if job.input_format == FORMAT_BNG_XML:
        if job.method == "rm":
            return _run_rulemonkey_job(
                job, input_path, output_dir, sim_options, result_options
            )
        if not _is_nf_method(job.method):
            raise BNGSimError(
                f"BioNetGen XML files are for network-free simulation, "
                f"but method='{job.method}' was requested. "
                f"Use method='nf' or provide a .net file for ODE/SSA/PSA."
            )
        if not BNGSIM_HAS_NFSIM:
            raise BNGSimError(
                "BNGsim NFsim support is not available in this build. "
                "Rebuild bngsim with -DBNGSIM_BUILD_NFSIM=ON."
            )

        seed = sim_options.pop("seed", None)
        if seed is None:
            seed = 42
        gml = sim_options.pop("gml", None)
        nf_params = sim_options.pop("nf_params", None)
        param_overrides = sim_options.pop("param_overrides", None)
        conc_overrides = sim_options.pop("conc_overrides", None)
        conc_deltas = sim_options.pop("conc_deltas", None)

        nf_kwargs = _nfsim_session_kwargs(nf_params)
        with bngsim.NfsimSession(input_path, molecule_limit=gml, **nf_kwargs) as nfsim:
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
            result = nfsim.simulate(job.t_span[0], job.t_span[1], job.n_points)

        _write_bngsim_results(
            result, output_dir, job.output_root,
            **result_options,
        )
        return _make_bng_result(output_dir, method=job.method)

    if _is_nf_method(job.method):
        raise BNGSimError(
            f"Network-free method '{job.method}' requires a BioNetGen XML file. "
            "Provide a .xml file or use method='ode'/'ssa'/'psa' with a .net file."
        )

    model = _load_direct_bngsim_model(input_path, job.input_format)
    init_kwargs, run_kwargs = _partition_simulator_options(sim_options)
    sim = bngsim.Simulator(model, method=job.method, **init_kwargs)
    result = sim.run(t_span=job.t_span, n_points=job.n_points, **run_kwargs)

    _write_bngsim_results(result, output_dir, job.output_root, **result_options)
    return _make_bng_result(output_dir, method=job.method)


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

    xml_path = os.path.abspath(xml_path)
    output_dir = os.path.abspath(output_dir)
    if model_name is None:
        model_name = os.path.splitext(os.path.basename(xml_path))[0]

    try:
        job = BngsimDirectJob(
            input_path=xml_path,
            input_format=FORMAT_BNG_XML,
            method="nf",
            t_span=t_span,
            n_points=n_points,
            output_dir=output_dir,
            output_root=model_name,
            bngsim_options={
                "seed": seed,
                "gml": gml,
                "nf_params": nf_params,
                "param_overrides": param_overrides,
                "conc_overrides": conc_overrides,
                "conc_deltas": conc_deltas,
            },
            result_options={
                "print_functions": print_functions,
            },
        )
        return execute_bngsim_direct_job(job)

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
        job = BngsimDirectJob(
            input_path=input_path,
            input_format=fmt,
            method=method,
            t_span=t_span,
            n_points=n_points,
            output_dir=output_dir,
            output_root=model_name,
            bngsim_options=sim_kwargs,
        )
        return execute_bngsim_direct_job(job)

    except Exception as e:
        if isinstance(e, (BNGSimError, BNGFormatError)):
            raise
        raise BNGSimError(f"BNGsim simulation failed: {e}") from e


# ─── BNGL routing helpers ──────────────────────────────────────────

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


_DIRECT_BNGSIM_FORMATS = {
    FORMAT_NET,
    FORMAT_SBML,
    FORMAT_BNG_XML,
    FORMAT_ANTIMONY,
}

_BNGSIM_NETWORK_METHODS = frozenset({"ode", "ssa", "psa", "rm"})

_BNGL_ROUTING_COMPLEX_ACTIONS = frozenset({
    "parameter_scan", "bifurcate",
    "setParameter", "setConcentration", "addConcentration",
    "saveConcentrations", "resetConcentrations",
    "saveParameters", "resetParameters",
    "writeXML", "writeSBML", "writeModel", "writeNetwork", "writeFile",
    "writeMfile", "writeCPYfile", "writeMexfile", "writeMDL",
    "readFile", "visualize", "setModelName",
})

_BNGL_ROUTING_PASSTHROUGH_ACTIONS = frozenset({
    "generate_network", "generate_hybrid_model",
})


def _method_supported_by_bngsim_for_routing(method, bngsim_has_nfsim=None):
    """Return True if a normalized method can be handed to BNGsim."""
    if method in _BNGSIM_NETWORK_METHODS:
        return True
    if _is_nf_method(method):
        if bngsim_has_nfsim is None:
            bngsim_has_nfsim = BNGSIM_HAS_NFSIM
        return bool(bngsim_has_nfsim)
    return False


def _bngl_action_method_for_routing(action):
    """Extract only the method hint needed for conservative routing.

    This deliberately avoids evaluating BNGL expressions. For legacy
    ``ssa`` plus ``poplevel`` syntax, the classifier reports the effective
    backend method as ``psa`` so BNGsim is never requested as
    ``ssa`` with a PSA-only option.
    """
    method = _SIMULATE_METHOD_MAP.get(action.type)
    if method is None:
        return None
    args = action.args or {}
    if action.type == "simulate" and "method" in args:
        method = _strip_quotes(str(args["method"]).strip()).lower()
    else:
        method = method.lower()
    if method == "ssa" and "poplevel" in args:
        return "psa"
    return method


def _bngl_workflow_method_for_routing(action):
    """Return the backend method implied by a BNG2.pl-owned workflow action."""
    atype = getattr(action, "type", None)
    args = getattr(action, "args", None) or {}
    if atype not in {"parameter_scan", "bifurcate"}:
        return None
    method = args.get("method")
    if method is None:
        return None
    method = _strip_quotes(str(method).strip()).lower()
    if method == "ode":
        return "ode"
    if method == "ssa" and "poplevel" in args:
        return "psa"
    return method


def _bngl_has_protocol_block(bngl_path):
    """Return True when a BNGL file declares a protocol block."""
    try:
        with open(bngl_path, "r", errors="replace") as f:
            for raw_line in f:
                clean = raw_line.split("#", 1)[0].strip()
                if re.match(r"begin\s+protocol\b", clean):
                    return True
    except OSError as exc:
        logger.debug("could not inspect BNGL protocol blocks (%s): %s", bngl_path, exc)
    return False


def _load_bngl_actions_for_routing(bngl_path):
    """Parse BNGL actions for routing only.

    Parse failures fall back to BNG2.pl rather than blocking the legacy path.
    """
    try:
        import bionetgen.modelapi.model as mdl

        model = mdl.bngmodel(bngl_path)
    except Exception as exc:
        logger.debug("could not parse BNGL for BNGsim routing (%s): %s", bngl_path, exc)
        return None
    try:
        return list(model.actions.items)
    except Exception as exc:
        logger.debug("could not read BNGL actions for BNGsim routing (%s): %s", bngl_path, exc)
        return None


def _classify_bngl_actions_for_bngsim(
    actions_items,
    method=None,
    has_protocol=False,
    bngsim_has_nfsim=None,
):
    """Classify whether BNGL can use the BNG2.pl-owned BNGsim backend hook.

    This routing pass only reads action names and method hints. It does not
    evaluate BNGL expressions or replay any action semantics in Python.
    """
    if actions_items is None:
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            "BNGL actions could not be inspected safely",
        )

    sim_actions = []
    has_backend_hook_workflow = bool(has_protocol)
    workflow_methods = []
    for action in actions_items:
        atype = getattr(action, "type", None)
        args = getattr(action, "args", None) or {}

        if atype in _BNGL_ROUTING_PASSTHROUGH_ACTIONS:
            continue

        if atype in _BNGL_ROUTING_COMPLEX_ACTIONS:
            has_backend_hook_workflow = True
            workflow_method = _bngl_workflow_method_for_routing(action)
            if workflow_method is not None:
                workflow_methods.append(workflow_method)
            continue

        if atype in _SIMULATE_METHOD_MAP:
            if (
                args.get("prefix") is not None
                or args.get("suffix") is not None
                or args.get("continue") is not None
            ):
                has_backend_hook_workflow = True
            sim_actions.append(action)
            continue

        if atype is not None:
            return BngsimRouteDecision(
                ROUTE_SUBPROCESS,
                f"BNGL action '{atype}' is not a conservative BNGsim route",
            )

    if len(sim_actions) > 1:
        has_backend_hook_workflow = True

    if any(workflow_method == "pla" for workflow_method in workflow_methods):
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            "BNGL PLA is not supported by BNGsim",
            method="pla",
        )

    if method is not None:
        method_name = _strip_quotes(str(method).strip()).lower()
        if sim_actions:
            if any(_bngl_action_method_for_routing(action) == "pla" for action in sim_actions):
                return BngsimRouteDecision(
                    ROUTE_SUBPROCESS,
                    "BNGL PLA is not supported by BNGsim",
                    method="pla",
                )
            action_method = _bngl_action_method_for_routing(sim_actions[0])
            if action_method == "pla":
                return BngsimRouteDecision(
                    ROUTE_SUBPROCESS,
                    "BNGL PLA is not supported by BNGsim",
                    method="pla",
                )
            if method_name == "ssa" and "poplevel" in (sim_actions[0].args or {}):
                method_name = "psa"
        if method_name == "pla":
            return BngsimRouteDecision(
                ROUTE_SUBPROCESS,
                "BNGL PLA is not supported by BNGsim",
                method="pla",
            )
        if _method_supported_by_bngsim_for_routing(method_name, bngsim_has_nfsim):
            return BngsimRouteDecision(
                ROUTE_BNGL_BNGSIM,
                "BNGL method override is a BNGsim-supported simulation",
                method=method_name,
            )
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            f"BNGL method '{method_name}' is not supported by the BNGsim route",
            method=method_name,
        )

    candidate_methods = []
    for action in sim_actions:
        method_name = _bngl_action_method_for_routing(action)
        if method_name == "pla":
            return BngsimRouteDecision(
                ROUTE_SUBPROCESS,
                "BNGL PLA is not supported by BNGsim",
                method="pla",
            )
        candidate_methods.append(method_name)

    for method_name in workflow_methods:
        if method_name == "pla":
            return BngsimRouteDecision(
                ROUTE_SUBPROCESS,
                "BNGL PLA is not supported by BNGsim",
                method="pla",
            )
        if method_name != "protocol":
            candidate_methods.append(method_name)

    if not candidate_methods and not has_backend_hook_workflow:
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            "BNGL has no simulation action that needs BNGsim",
        )

    for method_name in candidate_methods:
        if not _method_supported_by_bngsim_for_routing(method_name, bngsim_has_nfsim):
            return BngsimRouteDecision(
                ROUTE_SUBPROCESS,
                f"BNGL method '{method_name}' is not supported by the BNGsim route",
                method=method_name,
            )

    if has_backend_hook_workflow:
        return BngsimRouteDecision(
            ROUTE_BNGL_BNGSIM,
            "BNGL workflow is owned by BNG2.pl with BNGsim backend jobs",
            method=candidate_methods[0] if candidate_methods else None,
        )

    method_name = candidate_methods[0]
    return BngsimRouteDecision(
        ROUTE_BNGL_BNGSIM,
        "BNGL action is an atomic BNGsim-supported simulation",
        method=method_name,
    )


def classify_bngsim_route(
    input_path,
    fmt,
    simulator="auto",
    method=None,
    bngsim_available=None,
    bngsim_has_nfsim=None,
    bngl_actions=None,
    has_protocol=None,
):
    """Choose the conservative Stage 1 route for a simulation request."""
    if bngsim_available is None:
        bngsim_available = BNGSIM_AVAILABLE
    if bngsim_has_nfsim is None:
        bngsim_has_nfsim = BNGSIM_HAS_NFSIM

    if simulator not in {"auto", "bngsim", "subprocess"}:
        raise ValueError(
            f"Unknown simulator '{simulator}'. "
            "Valid options: 'auto', 'bngsim', 'subprocess'."
        )

    if simulator == "bngsim" and not bngsim_available:
        return BngsimRouteDecision(
            ROUTE_ERROR,
            "simulator='bngsim' was requested but BNGsim is not installed. "
            "Install with: pip install bngsim",
        )

    if simulator == "subprocess":
        if fmt in BNGSIM_REQUIRED_FORMATS:
            return BngsimRouteDecision(
                ROUTE_ERROR,
                f"Format '{fmt}' requires BNGsim but subprocess was requested. "
                "Install BNGsim and omit --no-bngsim for this format.",
            )
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            "subprocess simulator was requested",
        )

    if not bngsim_available:
        if fmt in BNGSIM_REQUIRED_FORMATS:
            return BngsimRouteDecision(
                ROUTE_ERROR,
                f"Format '{fmt}' requires BNGsim but it is not available. "
                "Install with: pip install bngsim",
            )
        return BngsimRouteDecision(
            ROUTE_SUBPROCESS,
            "BNGsim is unavailable; using legacy subprocess route",
        )

    if fmt in _DIRECT_BNGSIM_FORMATS:
        if fmt == FORMAT_BNG_XML:
            method_name = _strip_quotes(str(method).strip()).lower() if method else "nf"
            if not _is_nf_method(method_name):
                return BngsimRouteDecision(
                    ROUTE_ERROR,
                    f"BioNetGen XML files require method='nf', got '{method_name}'",
                    method=method_name,
                )
            if not bngsim_has_nfsim:
                if simulator == "bngsim":
                    return BngsimRouteDecision(
                        ROUTE_ERROR,
                        "BioNetGen XML direct routing requires BNGsim NFsim support",
                        method=method_name,
                    )
                return BngsimRouteDecision(
                    ROUTE_SUBPROCESS,
                    "BNGsim NFsim support is unavailable; using legacy subprocess route",
                    method=method_name,
                )
            return BngsimRouteDecision(
                ROUTE_DIRECT_BNGSIM,
                "BioNetGen XML routes directly to BNGsim NFsim",
                method=method_name,
            )
        method_name = _strip_quotes(str(method).strip()).lower() if method else None
        if method_name == "pla":
            if fmt in FALLBACK_FORMATS:
                return BngsimRouteDecision(
                    ROUTE_SUBPROCESS,
                    "PLA is not supported by the direct BNGsim route",
                    method=method_name,
                )
            return BngsimRouteDecision(
                ROUTE_ERROR,
                f"Format '{fmt}' requires BNGsim but method='pla' is not supported",
                method=method_name,
            )
        return BngsimRouteDecision(
            ROUTE_DIRECT_BNGSIM,
            f"Format '{fmt}' routes directly to BNGsim",
            method=method_name,
        )

    if fmt != FORMAT_BNGL:
        return BngsimRouteDecision(
            ROUTE_ERROR,
            f"No simulation backend available for format '{fmt}'",
        )

    if has_protocol is None:
        has_protocol = _bngl_has_protocol_block(input_path)
    if bngl_actions is None:
        bngl_actions = _load_bngl_actions_for_routing(input_path)
    return _classify_bngl_actions_for_bngsim(
        bngl_actions,
        method=method,
        has_protocol=has_protocol,
        bngsim_has_nfsim=bngsim_has_nfsim,
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


def _run_bngl_subprocess(
    bngl_path,
    output_dir,
    bngpath,
    suppress=False,
    log_file=None,
    timeout=None,
    app=None,
):
    """Run the original BNGL through the legacy BNG2.pl subprocess stack."""
    from bionetgen.core.tools.cli import BNGCLI

    cli = BNGCLI(
        bngl_path,
        output_dir,
        bngpath,
        suppress=suppress,
        log_file=log_file,
        timeout=timeout,
        app=app,
    )
    cli.run()
    if cli.result is None:
        raise BNGSimError("BNG2.pl failed.")
    return cli.result


_RM_QUOTED_RE = re.compile(r'(method\s*=>\s*)(["\'])rm\2', re.IGNORECASE)
_RM_BARE_RE = re.compile(r'(method\s*=>\s*)rm(?=[\s,)}\]])', re.IGNORECASE)


def _bngl_network_free_methods(actions_items):
    """Return the set of network-free methods (``nf``/``rm``) a BNGL uses."""
    methods = set()
    for action in actions_items or []:
        method = _bngl_action_method_for_routing(action)
        if method == "rm":
            methods.add("rm")
        elif _is_nf_method(method):
            methods.add("nf")
    return methods


def _rewrite_rm_method_to_nf(bngl_path):
    """Write a temp BNGL copy with ``method=>"rm"`` rewritten to ``"nf"``.

    BNG2.pl has no ``rm`` method, so rewriting to ``nf`` makes its
    ``simulate_nf`` path (and the BNGsim backend hook) fire; the helper
    restores ``rm`` from ``BIONETGEN_BNGSIM_BACKEND_METHOD``. The copy keeps
    the original basename so BNG2.pl's output-file naming is unchanged.

    Returns ``(run_path, temp_dir)``; the caller removes ``temp_dir``.
    """
    with open(bngl_path, "r", errors="replace") as f:
        text = f.read()
    rewritten = _RM_QUOTED_RE.sub(r"\1\2nf\2", text)
    rewritten = _RM_BARE_RE.sub(r'\1"nf"', rewritten)
    temp_dir = tempfile.mkdtemp(prefix="bngsim_rm_")
    run_path = os.path.join(temp_dir, os.path.basename(bngl_path))
    with open(run_path, "w") as f:
        f.write(rewritten)
    return run_path, temp_dir


def run_bngl_with_bngsim_backend_hook(
    bngl_path,
    output_dir,
    bngpath,
    suppress=False,
    log_file=None,
    timeout=None,
    app=None,
    bngsim_backend_helper=None,
    backend_method=None,
):
    """Run BNGL through BNG2.pl with the BNGsim backend helper enabled.

    This Stage 4 path keeps BNG2.pl as the BNGL action driver. A hook-capable
    BNG2.pl may delegate atomic simulation jobs to the helper advertised in
    the environment by :class:`BNGCLI`. ``backend_method`` carries an
    out-of-band method override (currently only ``rm``) to the helper.
    """
    from bionetgen.core.tools.cli import BNGCLI

    cli = BNGCLI(
        bngl_path,
        output_dir,
        bngpath,
        suppress=suppress,
        log_file=log_file,
        timeout=timeout,
        app=app,
        bngsim_backend=True,
        bngsim_backend_helper=bngsim_backend_helper,
        bngsim_backend_method=backend_method,
    )
    cli.run()
    if cli.result is None:
        raise BNGSimError("BNG2.pl failed.")
    return cli.result


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
    """Run a BNGL file through the Stage 6 BNGsim route.

    BNG2.pl owns BNGL parsing, action semantics, workflows, protocol blocks,
    scans, bifurcations, and output naming. Supported BNGL requests use the
    BNG2.pl backend hook so only normalized direct BNGsim jobs cross into
    Python. Unsupported BNGL requests keep the legacy subprocess route.
    """
    if not BNGSIM_AVAILABLE:
        raise BNGSimError("BNGsim is not available.")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    decision = classify_bngsim_route(
        bngl_path,
        FORMAT_BNGL,
        simulator="auto",
        method=method,
        bngsim_available=BNGSIM_AVAILABLE,
        bngsim_has_nfsim=BNGSIM_HAS_NFSIM,
    )
    if decision.route != ROUTE_BNGL_BNGSIM:
        logger.info("%s; using subprocess route.", decision.reason)
        return _run_bngl_subprocess(
            bngl_path,
            output_dir,
            bngpath,
            suppress=suppress,
            log_file=log_file,
            timeout=timeout,
            app=app,
        )

    if t_span is not None or n_points is not None:
        logger.info(
            "BNGL time-span and point-count overrides are interpreted by BNG2.pl "
            "on the backend-hook route."
        )

    # ``rm`` (RuleMonkey) has no BNG2.pl method. Rewrite ``method=>"rm"`` to
    # ``"nf"`` on a temp copy so the simulate_nf hook fires, and tell the
    # helper the real method out of band.
    run_path = bngl_path
    backend_method = None
    rm_temp_dir = None
    nf_methods = _bngl_network_free_methods(_load_bngl_actions_for_routing(bngl_path))
    if "rm" in nf_methods:
        if not BNGSIM_HAS_RULEMONKEY:
            logger.info(
                "BNGL uses method=>\"rm\" but BNGsim RuleMonkey is unavailable; "
                "using subprocess route."
            )
            return _run_bngl_subprocess(
                bngl_path, output_dir, bngpath,
                suppress=suppress, log_file=log_file, timeout=timeout, app=app,
            )
        if "nf" in nf_methods:
            logger.info(
                "BNGL mixes nf and rm methods, which the single-method backend "
                "override cannot disambiguate; using subprocess route."
            )
            return _run_bngl_subprocess(
                bngl_path, output_dir, bngpath,
                suppress=suppress, log_file=log_file, timeout=timeout, app=app,
            )
        run_path, rm_temp_dir = _rewrite_rm_method_to_nf(bngl_path)
        backend_method = "rm"

    logger.info("%s; using BNG2.pl-owned BNGsim backend hook.", decision.reason)
    try:
        return run_bngl_with_bngsim_backend_hook(
            run_path,
            output_dir,
            bngpath,
            suppress=suppress,
            log_file=log_file,
            timeout=timeout,
            app=app,
            bngsim_backend_helper=sim_kwargs.get("bngsim_backend_helper"),
            backend_method=backend_method,
        )
    finally:
        if rm_temp_dir and os.path.isdir(rm_temp_dir):
            shutil.rmtree(rm_temp_dir, ignore_errors=True)
