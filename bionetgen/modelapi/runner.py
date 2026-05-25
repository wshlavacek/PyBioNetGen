import os
from tempfile import TemporaryDirectory

from bionetgen.core.tools import BNGCLI
from bionetgen.main import get_conf


def run(
    inp,
    out=None,
    suppress=False,
    timeout=None,
    simulator="auto",
    format=None,
    method=None,
    t_span=None,
    n_points=None,
):
    """
    Convenience function to run a simulation as a library.

    Supports BNGL, .net, SBML (.xml), BioNetGen XML, and Antimony (.ant)
    files. When BNGsim is available in the environment, it is used for
    in-process simulation. Otherwise, falls back to BNG2.pl subprocess.

    Usage: run(path_to_input_file, output_folder)

    Arguments
    ---------
    inp : str
        Path to an input file (.bngl, .net, .xml, or .ant).
    out : str, optional
        Output folder for results. If None, a temp directory is used.
    suppress : bool
        Suppress output from BNG2.pl.
    timeout : int, optional
        Timeout in seconds for BNG2.pl subprocess.
    simulator : str
        Simulation backend: 'auto' (use BNGsim if available, else subprocess),
        'bngsim' (require BNGsim, error if missing), or 'subprocess' (force
        BNG2.pl/run_network path).
    format : str, optional
        Explicit input format hint: 'bngl', 'net', 'sbml', 'bng-xml', 'antimony'.
        If None, auto-detected from file extension and content.
    method : str, optional
        Optional simulation method override: 'ode', 'ssa', 'psa', 'nf', etc.
        For BNGL inputs, if omitted, the model's existing ``simulate_*``
        actions are preserved when routing through BNGsim. For direct
        BioNetGen XML inputs, if omitted, the method defaults to ``nf``
        and network-based methods are rejected.
    t_span : tuple of (float, float), optional
        Time span (t_start, t_end). If None, defaults to (0, 100).
    n_points : int, optional
        Number of output time points. If None, defaults to 101.

    Returns
    -------
    BNGResult
        Simulation results.
    """
    from bionetgen.core.tools.bngsim_bridge import (
        FORMAT_BNGL,
        ROUTE_BNGL_BNGSIM,
        ROUTE_DIRECT_BNGSIM,
        ROUTE_ERROR,
        ROUTE_SUBPROCESS,
        classify_bngsim_route,
        detect_input_format,
        run_bngl_with_bngsim,
        run_with_bngsim,
    )

    # Detect input format
    fmt = detect_input_format(inp, explicit_format=format)

    route = classify_bngsim_route(
        inp,
        fmt,
        simulator=simulator,
        method=method,
    )
    if route.route == ROUTE_ERROR:
        from bionetgen.core.exc import BNGSimError

        raise BNGSimError(route.reason)

    cur_dir = os.getcwd()

    def _run_with_output_dir(output_dir):
        try:
            if route.route == ROUTE_BNGL_BNGSIM and fmt == FORMAT_BNGL:
                conf = get_conf()
                result = run_bngl_with_bngsim(
                    inp,
                    output_dir,
                    conf["bngpath"],
                    method=method,
                    t_span=t_span,
                    n_points=n_points,
                    suppress=suppress,
                    log_file=None,
                    timeout=timeout,
                )
            elif route.route == ROUTE_DIRECT_BNGSIM:
                result = run_with_bngsim(
                    inp,
                    output_dir,
                    fmt=fmt,
                    method=method,
                    t_span=t_span,
                    n_points=n_points,
                )
            elif route.route == ROUTE_SUBPROCESS:
                conf = get_conf()
                # Subprocess path — only for .bngl, .net, .bng-xml
                cli = BNGCLI(
                    inp,
                    output_dir,
                    conf["bngpath"],
                    suppress=suppress,
                    timeout=timeout,
                )
                cli.run()
                result = cli.result
            else:
                from bionetgen.core.exc import BNGSimError

                raise BNGSimError(route.reason)
            return result
        finally:
            os.chdir(cur_dir)

    if out is None:
        with TemporaryDirectory() as out:
            return _run_with_output_dir(out)
    else:
        return _run_with_output_dir(out)
