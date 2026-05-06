import ctypes
import os
import tempfile

import numpy as np

import bionetgen
from bionetgen.core.exc import BNGCompileError, BNGFormatError, BNGSimError
from bionetgen.core.utils.logging import BNGLogger
from bionetgen.main import BioNetGen

from .bngsimulator import BNGSimulator


def _new_ccompiler():
    """Return a distutils ccompiler instance, sourced from setuptools.

    distutils was removed from the stdlib in Python 3.12; setuptools vendors
    the same API as ``setuptools._distutils``. Imported lazily so that
    ``import bionetgen`` does not require setuptools at runtime — only
    instantiating ``CSimulator`` does.
    """
    try:
        from setuptools._distutils import ccompiler
    except ImportError as exc:
        raise BNGCompileError(
            None,
            "CSimulator requires the distutils ccompiler API, which was "
            "removed from the stdlib in Python 3.12. Install setuptools "
            "(pip install setuptools) to provide it.",
        ) from exc
    return ccompiler.new_compiler()

# This allows access to the CLIs config setup
app = BioNetGen()
app.setup()
conf = app.config["bionetgen"]  # type: ignore[index]
def_bng_path = conf["bngpath"]
logger = BNGLogger()


class RESULT(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("n_observables", ctypes.c_int),
        ("n_species", ctypes.c_int),
        ("n_tpts", ctypes.c_int),
        ("obs_name_len", ctypes.c_int),
        ("spcs_name_len", ctypes.c_int),
        ("observables", ctypes.POINTER(ctypes.c_double)),
        ("species", ctypes.POINTER(ctypes.c_double)),
        ("obs_names", ctypes.POINTER(ctypes.c_char)),
        ("spcs_names", ctypes.POINTER(ctypes.c_char)),
    ]


class CSimWrapper:
    """
    Wrapper class for the compiled C simulator shared library.

    The class loads the compiled C shared library and passes
    pointers to the initial species arrays and parameter arrays
    to the shared library, runs the simulation and returns the
    results as numpy named arrays.
    """

    def __init__(self, lib_path, num_params=None, num_spec_init=None):
        # we need the result struct to reconstruct the object
        self.return_struct = RESULT
        # load the shared library
        self.lib = ctypes.CDLL(lib_path)
        # set the return type of the simulate function to a pointer
        # we'll use it to reconstruct the RESULT object from it later
        self.lib.simulate.restype = ctypes.c_void_p
        # set number of parameters
        self.num_params = num_params
        # set number of initial species values
        self.num_spec_init = num_spec_init

    def set_species_init(self, arr):
        """
        Set the initial species values array
        """
        if len(arr) != self.num_spec_init:
            raise ValueError(
                "Expected "
                f"{self.num_spec_init} initial species values, got {len(arr)}"
            )
        self.species_init = np.array(arr, dtype=np.float64)

    def set_parameters(self, arr):
        """
        Set the parameter values array
        """
        if len(arr) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameter values, got {len(arr)}"
            )
        self.parameters = np.array(arr, dtype=np.float64)

    def simulate(self, t_start=0, t_end=100, n_steps=100):
        """
        Run the simulate command of the shared C library.

        This function will construct a timepoint array from the given
        arguments and then pass the timepoints, parameters and
        initial species to the simulate command. Take the result pointer
        and convert the pointer back to a result struct and then
        construct named numpy arrays to return observable and species
        values over time.
        """
        # generate the time point array
        del_t = (t_end - t_start) / float(n_steps)
        timepoints = np.arange(t_start, t_end + 1, del_t)
        ntpts = len(timepoints)
        # call the simulate command
        self.result = self.return_struct.from_address(
            self.lib.simulate(
                ntpts,
                ctypes.c_void_p(timepoints.ctypes.data),
                self.num_spec_init,
                ctypes.c_void_p(self.species_init.ctypes.data),
                self.num_params,
                ctypes.c_void_p(self.parameters.ctypes.data),
            )
        )
        # we need to pull the observable names and get a list of them
        obs_names = ctypes.cast(
            self.result.obs_names,
            ctypes.POINTER(ctypes.c_char * self.result.obs_name_len),
        )[0].value.decode()
        obs_names = obs_names.split("/")[:-1]
        # same thing with species names
        spcs_names = ctypes.cast(
            self.result.spcs_names,
            ctypes.POINTER(ctypes.c_char * self.result.spcs_name_len),
        )[0].value.decode()
        spcs_names = spcs_names.split("/")[:-1]
        # cast the observables into a named numpy array
        buffer_as_ctypes_arr_obs = ctypes.cast(
            self.result.observables,
            ctypes.POINTER(ctypes.c_double * ntpts * self.result.n_observables),
        )[0]
        observables = np.frombuffer(buffer_as_ctypes_arr_obs, np.float64)
        fmt = ["f8"] * len(obs_names)
        obs_all = np.reshape(observables, (self.result.n_observables, ntpts))
        obs_all = np.core.records.fromarrays(obs_all, names=obs_names, formats=fmt)
        # cast the species into a named numpy array
        buffer_as_ctypes_arr_spc = ctypes.cast(
            self.result.species,
            ctypes.POINTER(ctypes.c_double * ntpts * self.result.n_species),
        )[0]
        species = np.frombuffer(buffer_as_ctypes_arr_spc, np.float64)
        fmt = ["f8"] * len(spcs_names)
        spcs_all = np.reshape(species, (self.result.n_species, ntpts))
        spcs_all = np.core.records.fromarrays(spcs_all, names=spcs_names, formats=fmt)
        # free the memory used for the results struct
        self.lib.free_result(ctypes.byref(self.result))
        del self.result
        # return named numpy arrays
        return (timepoints, obs_all, spcs_all)


class CSimulator(BNGSimulator):
    """
    Object that bridges the BNG model object and the CSimWrapper object.

    The point of this object is to deal with the compilation of the shared library
    and pass the correct parameter and initial species values to the wrapper object.
    """

    def __init__(self, model_file, generate_network=False):
        # check cvode library paths
        if (conf.get("cvode_include") is None) or (conf.get("cvode_lib") is None):
            logger.warning(
                "CVODE include and library paths are not set, compilation won't work",
                loc=f"{__file__} : CSimulator.__init__()",
            )
        # let's load the model first
        if isinstance(model_file, str):
            # load model file
            self.model = bionetgen.bngmodel(
                model_file, generate_network=generate_network
            )
        elif isinstance(model_file, bionetgen.bngmodel):
            # loaded model
            self.model = model_file
            cd = os.getcwd()
            with tempfile.TemporaryDirectory() as tmpdirname:
                os.chdir(tmpdirname)
                self.model.actions.clear_actions()
                self.model.write_model(f"{self.model.model_name}_cpy.bngl")
                self.model = bionetgen.bngmodel(
                    f"{self.model.model_name}_cpy.bngl",
                    generate_network=generate_network,
                )
            os.chdir(cd)
        else:
            msg = (
                "CSimulator model input must be a BNGL path or bngmodel instance, "
                f"got {type(model_file).__name__}"
            )
            logger.error(msg, loc=f"{__file__} : CSimulator.__init__()")
            raise BNGFormatError(message=msg)
        # set compiler
        self.compiler = _new_ccompiler()
        self.compiler.add_include_dir(conf.get("cvode_include"))
        self.compiler.add_library_dir(conf.get("cvode_lib"))
        # compile shared library
        self.compile_shared_lib()
        # setup simulator
        self.simulator = self.lib_file

    def __str__(self):
        return f"C/Python Simulator, params: {self.model.parameters} \ninit species: {self.model.species}"

    def __repr__(self):
        return str(self)

    def compile_shared_lib(self):
        # run and get CPY file
        # make sure we don't have actions
        self.model.actions.clear_actions()
        self.model.actions.add_action("generate_network", {"overwrite": 1})
        self.model.actions.add_action("writeCPYfile", {})
        # for now run and write the .c file in the current folder
        bionetgen.run(self.model, out=os.path.abspath(os.getcwd()))
        # compile CPY file
        c_file = f"{self.model.model_name}_cvode_py.c"
        obj_file = f"{self.model.model_name}_cvode_py.o"
        lib_file = f"{self.model.model_name}_cvode_py"
        # compile objects with fPIC for the shared lib we'll link
        self.compiler.compile([c_file], extra_preargs=["-fPIC"])
        # now link cvode and nvecserial and make a shared lib
        self.compiler.link_shared_lib(
            [obj_file], lib_file, libraries=["sundials_cvode", "sundials_nvecserial"]
        )
        # # keep a record of what we got
        self.cfile = os.path.abspath(c_file)
        self.obj_file = os.path.abspath(obj_file)
        # compiler tacks on the lib at the beginning and .so at the end
        lib_file = f"lib{self.model.model_name}_cvode_py.so"
        self.lib_file = os.path.abspath(lib_file)

    def _get_numeric_parameter_values(self):
        valid_params = []
        for pname in self.model.parameters:
            if pname.startswith("_"):
                continue
            val = self.model.parameters[pname]
            try:
                valid_params.append(float(val.expr))
            except (AttributeError, TypeError, ValueError):
                continue
        return valid_params

    def _resolve_species_count(self, spc_name):
        count_value = self.model.species[spc_name].count
        try:
            return float(count_value)
        except (TypeError, ValueError):
            try:
                return float(self.model.parameters[count_value].value)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                msg = (
                    f"Could not resolve initial species value for '{spc_name}' "
                    f"from '{count_value}'"
                )
                logger.error(msg, loc=f"{__file__} : CSimulator.simulate()")
                raise BNGSimError(msg) from exc

    @property
    def simulator(self):
        """
        simulator attribute that stores
        the instantiated simulator object
        and also saves the SBML text in the
        sbml attribute
        """
        return self._simulator

    @simulator.setter
    def simulator(self, lib_file):
        # use CSimWrapper under the hood
        try:
            valid_params = self._get_numeric_parameter_values()
            n_param = len(valid_params)
            self._simulator = CSimWrapper(
                os.path.abspath(lib_file),
                num_params=n_param,
                num_spec_init=len(self.model.species),
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
            logger.error(
                f"Failed to initialize C simulator wrapper: {exc}",
                loc=f"{__file__} : CSimulator.simulator.setter()",
            )
            raise BNGCompileError(self.model) from exc

    def simulate(self, t_start=0, t_end=10, n_steps=10):
        # set parameters and initial species values
        spcs = [self._resolve_species_count(spc_name) for spc_name in self.model.species]
        self.simulator.set_species_init(spcs)
        params = self._get_numeric_parameter_values()
        self.simulator.set_parameters(params)
        # now that we have CSimWrapper setup correctly, run the simulation
        timepoints, obs_all, spcs_all = self.simulator.simulate(t_start, t_end, n_steps)
        # return our results
        return (timepoints, obs_all, spcs_all)
