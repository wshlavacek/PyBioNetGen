# A simple CLI for BioNetGen 

[![BNG CLI build status](https://github.com/RuleWorld/PyBioNetGen/workflows/bng-cli-tests/badge.svg)](https://github.com/RuleWorld/PyBioNetGen/actions)
[![Open in Remote - Containers](https://img.shields.io/static/v1?label=Remote%20-%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/RuleWorld/PyBioNetGen)
[![Documentation Status](https://readthedocs.org/projects/pybionetgen/badge/?version=latest)](https://pybionetgen.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/bionetgen)](https://pepy.tech/project/bionetgen)
[![Downloads](https://static.pepy.tech/badge/bionetgen/week)](https://pepy.tech/project/bionetgen)

This is a simple CLI and a library for [BioNetGen modeling language](http://bionetgen.org/). PyBioNetGen optionally includes a heavily updated version of [Atomizer](https://github.com/RuleWorld/atomizer) which allows for conversion of models written in [Systems Biology Markup Language (SBML)](https://synonym.caltech.edu/) into BioNetGen language (BNGL) format. 

Please see the [documentation](https://pybionetgen.readthedocs.io/en/latest/) to learn how to use PyBioNetGen. 

## Installation

You will need both python (3.8 and above) and perl installed. Once both are available you can use the following pip command to install the package

```
$ pip install bionetgen
```

To include atomizer (SBML-to-BNGL conversion), install with the optional extra:

```
$ pip install bionetgen[atomizer]
```

### Features 

PyBioNetGen comes with a command line interface (CLI), based on [cement framework](https://builtoncement.com/), as well as a functional library that can be imported. The CLI can be used to run BNGL models, generate Jupyter notebooks and do rudimentary plotting. 

The library side provides a simple BNGL model runner as well as a model object that can be manipulated and used to get libRoadRunner simulators for the model. 

**BNGsim integration:** When [BNGsim](https://github.com/lanl/bngsim) is installed in the same environment (`pip install bngsim`), PyBioNetGen automatically uses it for high-performance in-process simulation, replacing the subprocess-based `run_network` and `NFsim` backends. BNGsim also enables direct simulation of SBML (`.xml`) and Antimony (`.ant`) files in addition to BNGL. BNGsim is optional — without it, PyBioNetGen works exactly as before.

**Supported input formats:**
- `.bngl` — BioNetGen Language (always processed by BNG2.pl, then simulated via BNGsim or `run_network`)
- `.net` — BioNetGen network files (direct simulation via BNGsim or `run_network`)
- `.xml` — SBML files (requires BNGsim) or BioNetGen XML files (for network-free simulation)
- `.ant` — Antimony files (requires BNGsim)

**Atomizer (optional):** PyBioNetGen includes a heavily updated version of [Atomizer](https://github.com/RuleWorld/atomizer) for conversion of SBML models into BNGL format. Atomizer can also automatically infer the internal structure of SBML species during conversion; see [here](https://pybionetgen.readthedocs.io/en/latest/atomizer.html) for more information. Install with `pip install bionetgen[atomizer]`. Please note that this version of Atomizer is the main supported version and the version distributed with BioNetGen will eventually be deprecated. 

The model object requires a system call to BioNetGen so the initialization can be relatively costly. For parallel applications, use [libRoadRunner](http://libroadrunner.org/) or BNGsim instead.

### Usage 

Sample CLI usage

```
$ bionetgen -h # help on every subcommand
$ bionetgen run -h # help on run subcommand
$ bionetgen run -i mymodel.bngl -o output_folder # run a BNGL model
$ bionetgen run -i mymodel.net -o output_folder # run a .net file directly
$ bionetgen run -i mymodel.xml --format sbml -o output_folder # run an SBML model (requires BNGsim)
$ bionetgen run -i mymodel.bngl --no-bngsim -o output_folder # force subprocess path
$ bionetgen run -i mymodel.net --method ssa -o output_folder # use SSA method via BNGsim
```

Sample library usage

```
import bionetgen 

# Check if BNGsim is available
print(bionetgen.BNGSIM_AVAILABLE)

ret = bionetgen.run("/path/to/mymodel.bngl", out="/path/to/output/folder")
# out keyword is optional, if not given, 
# generated files will be deleted after running
res = ret.results['mymodel']
# res will be a numpy record array of your gdat results

# Run with explicit simulator and method options
ret = bionetgen.run("model.net", out="output/", simulator="bngsim", method="ode")

# Run SBML or Antimony files (requires BNGsim)
ret = bionetgen.run("model.xml", out="output/", format="sbml")
ret = bionetgen.run("model.ant", out="output/")

model = bionetgen.bngmodel("/path/to/mymodel.bngl")
# model will be a python object that contains all model information
print(model.parameters) # this will print only the parameter block in BNGL format
print(model) # this will print the entire BNGL
model.parameters.k = 1 # setting parameter k to 1
with open("new_model.bngl", "w") as f:
    f.write(str(model)) # writes the changed model to new_model file

# this will give you a libRoadRunner instance of the model
librr_sim = model.setup_simulator()
```

You can find more tutorials [here](https://pybionetgen.readthedocs.io/en/latest/tutorials.html).

### Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run bionetgen cli application

$ bionetgen --help


### run pytest / coverage

$ make test
```

### Docker

Included is a basic `Dockerfile` for building and distributing `BioNetGen CLI`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it bionetgen --help
```

### Publishing to PyPI

You can use `make dist` command to make the distribution and push to PyPI with

```
python -m twine upload dist/*
```

You'll need to have a PyPI API token created, see [here](https://packaging.python.org/tutorials/packaging-projects/) for more information. 
