import os
import sys

from bionetgen.core.exc import BNGFileError, BNGRunError
from bionetgen.core.utils.logging import BNGLogger


class BNGCLI:
    """
    Command Line Interface class to run BNG2.pl on a given
    model.

    Usage: BNGCLI(inp_file, output, bngpath)

    Arguments
    ---------
    inp_file : str
        path to the the BNGL file to run
    output : str
        path to the output folder to run the model in
    bngpath : str
        path to BioNetGen folder where BNG2.pl lives

    Methods
    -------
    run()
        runs the model in the given output folder
    """

    def __init__(
        self,
        inp_file,
        output,
        bngpath,
        suppress=False,
        log_file=None,
        timeout=None,
        app=None,
        bngsim_backend=False,
        bngsim_backend_helper=None,
        bngsim_backend_method=None,
    ):
        self.app = app
        self.logger = BNGLogger(app=self.app)
        self.logger.debug(
            "Setting up BNGCLI object", loc=f"{__file__} : BNGCLI.__init__()"
        )
        self.inp_file = inp_file
        import bionetgen.modelapi.model as mdl

        if isinstance(inp_file, mdl.bngmodel):
            self.is_bngmodel = True
        else:
            self.is_bngmodel = False
            # ensure correct path to the input file
            self.inp_path = os.path.abspath(self.inp_file)
        # pull other arugments out
        if log_file is not None:
            self.log_file = os.path.abspath(log_file)
        else:
            self.log_file = None
        self._set_output(output)
        # sedml_file = sedml
        # Resolve BioNetGen executable path. Historically this code assumed
        # `bngpath` was a directory containing BNG2.pl, but on Windows installs
        # and some deployments we may need to honor $BNGPATH or accept a direct
        # path to BNG2.pl.
        from bionetgen.core.utils.utils import find_BNG_path

        try:
            resolved_dir, resolved_exec = find_BNG_path(bngpath)
        except Exception as exc:
            msg = (
                "Unable to resolve BNG2.pl. "
                "Set the BNGPATH environment variable to the BioNetGen folder containing BNG2.pl. "
                f"Details: {exc}"
            )
            self.logger.error(msg, loc=f"{__file__} : BNGCLI.__init__()")
            raise BNGFileError(bngpath, message=msg) from exc

        self.bngpath = resolved_dir
        self.bng_exec = resolved_exec
        if "BNGPATH" in os.environ:
            self.old_bngpath = os.environ["BNGPATH"]
        else:
            self.old_bngpath = None
        if self.bngpath is not None:
            os.environ["BNGPATH"] = self.bngpath
        self.result = None
        self.stdout = "PIPE"
        self.stderr = "STDOUT"
        self.suppress = suppress
        self.timeout = timeout
        self.bngsim_backend = bngsim_backend
        self.bngsim_backend_helper = bngsim_backend_helper
        self.bngsim_backend_method = bngsim_backend_method
        self._old_bngsim_backend_env = {}

    def _install_bngsim_backend_env(self):
        """Expose the BNGsim backend helper contract to hook-capable BNG2.pl."""
        keys = (
            "BIONETGEN_BNGSIM_BACKEND",
            "BIONETGEN_BNGSIM_BACKEND_HELPER",
            "BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON",
            "BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE",
            "BIONETGEN_BNGSIM_BACKEND_METHOD",
        )
        self._old_bngsim_backend_env = {key: os.environ.get(key) for key in keys}
        if not self.bngsim_backend and self.bngsim_backend_helper is None:
            return

        helper = self.bngsim_backend_helper
        os.environ["BIONETGEN_BNGSIM_BACKEND"] = "1"
        if helper is not None:
            os.environ["BIONETGEN_BNGSIM_BACKEND_HELPER"] = helper
        else:
            os.environ.pop("BIONETGEN_BNGSIM_BACKEND_HELPER", None)
        os.environ["BIONETGEN_BNGSIM_BACKEND_HELPER_PYTHON"] = sys.executable
        os.environ["BIONETGEN_BNGSIM_BACKEND_HELPER_MODULE"] = (
            "bionetgen.core.tools.bngsim_backend_helper"
        )
        # ``rm`` BNGL is rewritten to ``nf`` so BNG2.pl's simulate_nf hook
        # fires; this carries the real method to the helper out of band.
        if self.bngsim_backend_method:
            os.environ["BIONETGEN_BNGSIM_BACKEND_METHOD"] = self.bngsim_backend_method
        else:
            os.environ.pop("BIONETGEN_BNGSIM_BACKEND_METHOD", None)

    def _restore_bngsim_backend_env(self):
        for key, value in self._old_bngsim_backend_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _set_output(self, output):
        self.logger.debug(
            "Setting up output path", loc=f"{__file__} : BNGCLI._set_output()"
        )
        # setting up output area
        self.output = os.path.abspath(output)
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

    def run(self):
        self.logger.debug("Running", loc=f"{__file__} : BNGCLI.run()")
        self._install_bngsim_backend_env()
        # If BNG2.pl is not available, fall back to an empty result so that
        # library users can still instantiate and inspect models without a
        # full BioNetGen install.
        if self.bng_exec is None:
            from bionetgen.core.tools import BNGResult

            self.result = BNGResult(self.output)
            self.result.process_return = 0
            self.result.output = []
            if self.old_bngpath is not None:
                os.environ["BNGPATH"] = self.old_bngpath
            else:
                if "BNGPATH" in os.environ:
                    del os.environ["BNGPATH"]
            self._restore_bngsim_backend_env()
            return

        from bionetgen.core.utils.utils import run_command

        # run BNG2.pl
        if self.is_bngmodel:
            self.logger.debug(
                "The given model is a bngmodel object", loc=f"{__file__} : BNGCLI.run()"
            )
            self.logger.debug(
                "Writing the model to a file", loc=f"{__file__} : BNGCLI.run()"
            )
            write_to = os.path.join(self.output, self.inp_file.model_name + ".bngl")
            write_to = os.path.abspath(write_to)
            if os.path.isfile(write_to):
                self.logger.warning(
                    f"Overwriting file {write_to}", loc=f"{__file__} : BNGCLI.run()"
                )
            with open(write_to, "w") as tfile:
                tfile.write(str(self.inp_file))
            command = ["perl", self.bng_exec, write_to]
        else:
            self.logger.debug(
                "The given model is a file", loc=f"{__file__} : BNGCLI.run()"
            )
            fname = os.path.basename(self.inp_path)
            fname = fname.replace(".bngl", "")
            command = ["perl", self.bng_exec, self.inp_path]
        self.logger.debug("Running command", loc=f"{__file__} : BNGCLI.run()")
        rc, out = run_command(
            command, suppress=self.suppress, timeout=self.timeout, cwd=self.output
        )
        if self.log_file is not None:
            self.logger.debug("Setting up log file", loc=f"{__file__} : BNGCLI.run()")
            # If log_file already points to an existing file or directory, use
            # that path directly. Otherwise, resolve it relative to the output
            # directory for this run.
            if os.path.exists(self.log_file):
                # file or folder exists, check if folder
                if os.path.isdir(self.log_file):
                    fname = os.path.basename(self.inp_path)
                    fname = fname.replace(".bngl", "")
                    full_log_path = os.path.join(self.log_file, fname + ".log")
                else:
                    # it's intended to be file, so we keep it as is
                    full_log_path = self.log_file
            else:
                # doesn't exist, so we assume it's a file
                # and we keep it as is
                full_log_path = self.log_file
            self.logger.debug("Writing log file", loc=f"{__file__} : BNGCLI.run()")
            log_parent = os.path.dirname(os.path.abspath(full_log_path))
            if not os.path.exists(log_parent):
                os.makedirs(log_parent, exist_ok=True)
            with open(full_log_path, "w") as f:
                f.write("\n".join(out))
        if rc == 0:
            self.logger.debug(
                "Command ran successfully", loc=f"{__file__} : BNGCLI.run()"
            )
            from bionetgen.core.tools import BNGResult

            # load in the result
            self.result = BNGResult(self.output)
            self.result.process_return = rc
            self.result.output = out
            # set BNGPATH back
            if self.old_bngpath is not None:
                os.environ["BNGPATH"] = self.old_bngpath
            else:
                if "BNGPATH" in os.environ:
                    del os.environ["BNGPATH"]
            self._restore_bngsim_backend_env()
        else:
            self.logger.error("Command failed to run", loc=f"{__file__} : BNGCLI.run()")
            self.result = None
            # set BNGPATH back
            if self.old_bngpath is not None:
                os.environ["BNGPATH"] = self.old_bngpath
            else:
                if "BNGPATH" in os.environ:
                    del os.environ["BNGPATH"]
            self._restore_bngsim_backend_env()
            stdout_str = None
            stderr_str = None
            if getattr(out, "stdout", None) is not None:
                stdout_str = out.stdout.decode("utf-8")
            if getattr(out, "stderr", None) is not None:
                stderr_str = out.stderr.decode("utf-8")
            raise BNGRunError(command, stdout=stdout_str, stderr=stderr_str)
