import os
from tempfile import TemporaryDirectory

from bionetgen.core.tools import BNGCLI
from bionetgen.main import BioNetGen

# This allows access to the CLIs config setup
app = BioNetGen()
app.setup()
conf = app.config["bionetgen"]  # type: ignore[index]


def run(
    inp,
    out=None,
    suppress=False,
    timeout=None,
    simulator="auto",
    format=None,
    method="ode",
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
    method : str
        Simulation method: 'ode', 'ssa', 'psa', 'nf', etc. Default 'ode'.
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
        BNGSIM_AVAILABLE,
        BNGSIM_REQUIRED_FORMATS,
        FORMAT_BNGL,
        detect_input_format,
        run_bngl_with_bngsim,
        run_with_bngsim,
    )

    # Detect input format
    fmt = detect_input_format(inp, explicit_format=format)

    # Determine whether to use BNGsim
    use_bngsim = False
    if simulator == "bngsim":
        if not BNGSIM_AVAILABLE:
            from bionetgen.core.exc import BNGSimError

            raise BNGSimError(
                "simulator='bngsim' was requested but BNGsim is not installed. "
                "Install with: pip install bngsim"
            )
        use_bngsim = True
    elif simulator == "auto":
        use_bngsim = BNGSIM_AVAILABLE
    elif simulator == "subprocess":
        use_bngsim = False
    else:
        raise ValueError(
            f"Unknown simulator '{simulator}'. "
            "Valid options: 'auto', 'bngsim', 'subprocess'."
        )

    # Formats that require BNGsim have no subprocess fallback
    if not use_bngsim and fmt in BNGSIM_REQUIRED_FORMATS:
        from bionetgen.core.exc import BNGSimError

        raise BNGSimError(
            f"Format '{fmt}' requires BNGsim but it is not available. "
            "Install with: pip install bngsim"
        )

    cur_dir = os.getcwd()

    def _run_with_output_dir(output_dir):
        try:
            if use_bngsim and fmt == FORMAT_BNGL:
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
            elif use_bngsim:
                result = run_with_bngsim(
                    inp,
                    output_dir,
                    fmt=fmt,
                    method=method,
                    t_span=t_span,
                    n_points=n_points,
                )
            else:
                # Subprocess path — only for .bngl, .net, .bng-xml
                cli = BNGCLI(
                    inp, output_dir, conf["bngpath"],
                    suppress=suppress, timeout=timeout,
                )
                cli.run()
                result = cli.result
            os.chdir(cur_dir)
            return result
        except Exception:
            os.chdir(cur_dir)
            raise

    if out is None:
        with TemporaryDirectory() as out:
            return _run_with_output_dir(out)
    else:
        return _run_with_output_dir(out)
