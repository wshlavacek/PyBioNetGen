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

    n_points = _as_int(opts.pop("n_points", None))
    if n_points is None:
        n_steps = _as_int(opts.pop("n_steps", opts.pop("n_output_steps", None)), 100)
        n_points = n_steps + 1

    result_options = {}
    if "print_functions" in opts:
        result_options["print_functions"] = bool(_as_int(opts.pop("print_functions"), 0))

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
