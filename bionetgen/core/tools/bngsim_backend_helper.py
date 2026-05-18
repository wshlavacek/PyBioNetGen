"""JSON helper for BNG2.pl-owned BNGsim backend jobs.

This module is the process boundary for the Stage 4 BNG2.pl simulator
backend hook. BNG2.pl remains responsible for BNGL parsing, action
semantics, state, scans, and output naming; it passes one already-normalized
atomic simulation job here as JSON.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from bionetgen.core.exc import BNGSimError
from bionetgen.core.tools.bngsim_bridge import (
    BngsimDirectJob,
    FORMAT_BNG_XML,
    FORMAT_NET,
    execute_bngsim_direct_job,
)


NETWORK_METHOD_ALIASES = {
    "cvode": "ode",
    "ode": "ode",
    "ssa": "ssa",
    "psa": "psa",
    "rm": "rm",
}

NF_METHOD_ALIASES = {
    "nf": "nf",
    "nfsim": "nf",
    "nf_reject": "nf",
}

# Network-free methods (canonical names). BNG2.pl runs both ``nf`` and
# rm-rewritten-to-``nf`` BNGL through ``sub simulate_nf``, which reports
# timepoints as time elapsed since ``t_start`` (the output axis starts at
# 0). The network methods (ode/ssa/psa) instead honor ``t_start`` via
# ``run_network -i``. See ``direct_job_from_backend_job``.
NETWORK_FREE_METHODS = frozenset({"nf", "rm"})

ARTIFACT_FORMAT_ALIASES = {
    "net": FORMAT_NET,
    ".net": FORMAT_NET,
    "bng-xml": FORMAT_BNG_XML,
    "bng_xml": FORMAT_BNG_XML,
    "xml": FORMAT_BNG_XML,
    ".xml": FORMAT_BNG_XML,
}


@dataclass(frozen=True)
class BackendHelperJob:
    """Machine-readable job supplied by a BNG2.pl simulator hook."""

    artifact_path: str
    artifact_format: str
    method: str
    simulation_options: dict[str, Any]
    output_dir: str
    output_root: str
    backend_flags: dict[str, Any]


def _as_number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    number = _as_number(value)
    if number is None:
        return default
    return int(number)


def _normalize_method(method: Any) -> str:
    method_name = str(method or "").strip().lower()
    if method_name in NETWORK_METHOD_ALIASES:
        return NETWORK_METHOD_ALIASES[method_name]
    if method_name in NF_METHOD_ALIASES:
        return NF_METHOD_ALIASES[method_name]
    raise BNGSimError(f"Unsupported BNGsim backend method: {method!r}")


def _normalize_artifact_format(raw_format: Any, artifact_path: str) -> str:
    if raw_format is not None:
        fmt = ARTIFACT_FORMAT_ALIASES.get(str(raw_format).strip().lower())
        if fmt is None:
            raise BNGSimError(f"Unsupported BNGsim backend artifact format: {raw_format!r}")
        return fmt

    ext = os.path.splitext(artifact_path)[1].lower()
    fmt = ARTIFACT_FORMAT_ALIASES.get(ext)
    if fmt is None:
        raise BNGSimError(
            f"Could not infer BNGsim backend artifact format from {artifact_path!r}"
        )
    return fmt


def _output_dir_and_root(payload: dict[str, Any]) -> tuple[str, str]:
    output_prefix = payload.get("output_prefix") or payload.get("output_path_root")
    if output_prefix:
        output_prefix = os.path.abspath(str(output_prefix))
        output_dir = os.path.dirname(output_prefix) or os.getcwd()
        output_root = os.path.basename(output_prefix)
        return output_dir, output_root

    output_dir = os.path.abspath(str(payload.get("output_dir") or os.getcwd()))
    output_root = payload.get("output_root")
    if not output_root:
        artifact_path = str(payload.get("artifact_path") or payload.get("input_path") or "")
        output_root = os.path.splitext(os.path.basename(artifact_path))[0]
    return output_dir, str(output_root)


def load_backend_job(payload: dict[str, Any]) -> BackendHelperJob:
    """Validate and normalize one BNG2.pl backend job payload."""
    artifact_path = payload.get("artifact_path") or payload.get("input_path")
    if not artifact_path:
        raise BNGSimError("BNGsim backend job is missing artifact_path")
    artifact_path = os.path.abspath(str(artifact_path))

    method = _normalize_method(payload.get("method"))
    # BNG2.pl has no ``rm`` method, so ``method=>"rm"`` BNGL is rewritten to
    # ``nf`` before BNG2.pl runs and the real method is carried out of band
    # in BIONETGEN_BNGSIM_BACKEND_METHOD. Restore it here. The override
    # applies only to network-free jobs (the simulate_nf hook always sends
    # ``nf``); network jobs (ode/ssa/psa) in the same run are left alone.
    method_override = os.environ.get(
        "BIONETGEN_BNGSIM_BACKEND_METHOD", ""
    ).strip().lower()
    if method_override == "rm" and method == "nf":
        method = "rm"
    artifact_format = _normalize_artifact_format(
        payload.get("artifact_format") or payload.get("input_format"),
        artifact_path,
    )
    output_dir, output_root = _output_dir_and_root(payload)

    sim_options = dict(
        payload.get("simulation_options")
        or payload.get("options")
        or {}
    )
    backend_flags = dict(payload.get("backend_flags") or {})

    return BackendHelperJob(
        artifact_path=artifact_path,
        artifact_format=artifact_format,
        method=method,
        simulation_options=sim_options,
        output_dir=output_dir,
        output_root=output_root,
        backend_flags=backend_flags,
    )


def direct_job_from_backend_job(job: BackendHelperJob) -> BngsimDirectJob:
    """Convert a hook job into the Stage 2 direct BNGsim executor contract."""
    opts = dict(job.simulation_options)
    t_start = _as_number(opts.pop("t_start", None), 0.0)
    t_end = _as_number(opts.pop("t_end", None), 100.0)

    # Network-free methods follow NFsim's output convention: BNG2.pl's
    # ``sub simulate_nf`` reports timepoints as elapsed time since
    # ``t_start`` (axis starts at 0) and warns the user it does so. Rebase
    # the span to start at 0 so the BNGsim run matches BNG2.pl output --
    # both the time column and any ``time()``-dependent rate laws, which
    # would otherwise evaluate over the wrong interval. Network methods
    # (ode/ssa/psa) keep ``t_start``; BNG2.pl honors it there.
    if job.method in NETWORK_FREE_METHODS and t_start != 0.0:
        t_end = t_end - t_start
        t_start = 0.0

    # Explicit output times. BNG2.pl's ``sub simulate`` honors a
    # ``sample_times`` array; the backend hook forwards it. BNG2.pl emits
    # the initial (t_start) state row followed by the explicit sample
    # times, whereas BNGsim's ``sample_times`` yields exactly the listed
    # times -- so prepend t_start for output parity. ``n_points`` is then
    # the row count; BNGsim ignores it when ``sample_times`` is given.
    sample_times = opts.pop("sample_times", None)
    n_points = _as_int(opts.pop("n_points", None))
    n_steps = _as_int(opts.pop("n_steps", opts.pop("n_output_steps", None)))
    if sample_times:
        sample_times = [float(t_start)] + [float(t) for t in sample_times]
        opts["sample_times"] = sample_times
        n_points = len(sample_times)
    elif n_points is None:
        n_points = (n_steps if n_steps is not None else 100) + 1

    result_options = {}
    if "print_functions" in opts:
        result_options["print_functions"] = bool(_as_int(opts.pop("print_functions"), 0))
    # ``print_CDAT=>0`` keeps only the initial and final .cdat rows.
    if "print_CDAT" in opts:
        result_options["print_cdat"] = bool(_as_int(opts.pop("print_CDAT"), 1))
    # ``continue=>1`` segments append to the prior segment's output files
    # (skipping the duplicated t_start row) instead of overwriting them.
    if "continue" in opts:
        result_options["append"] = bool(_as_int(opts.pop("continue"), 0))

    bngsim_options = {}
    for key in (
        "seed",
        "poplevel",
        "atol",
        "rtol",
        "gml",
        "nf_params",
        "param_overrides",
        "conc_overrides",
        "conc_deltas",
    ):
        if key in opts and opts[key] is not None:
            bngsim_options[key] = opts.pop(key)
    bngsim_options.update(opts)

    return BngsimDirectJob(
        input_path=job.artifact_path,
        input_format=job.artifact_format,
        method=job.method,
        t_span=(float(t_start), float(t_end)),
        n_points=int(n_points),
        output_dir=job.output_dir,
        output_root=job.output_root,
        bngsim_options=bngsim_options,
        result_options=result_options,
    )


def execute_backend_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one JSON payload and return a JSON-serializable status."""
    helper_job = load_backend_job(payload)
    direct_job = direct_job_from_backend_job(helper_job)
    result = execute_bngsim_direct_job(direct_job)
    return {
        "success": True,
        "method": helper_job.method,
        "artifact_path": helper_job.artifact_path,
        "output_dir": helper_job.output_dir,
        "output_root": helper_job.output_root,
        "process_return": getattr(result, "process_return", 0),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(
            json.dumps({
                "success": False,
                "error": "usage: python -m bionetgen.core.tools.bngsim_backend_helper JOB.json",
            }),
            file=sys.stderr,
        )
        return 2

    try:
        with open(argv[0], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        status = execute_backend_payload(payload)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        status = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(status, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
