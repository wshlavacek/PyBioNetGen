import glob
import os

import pytest

import bionetgen as bng
from bionetgen.main import BioNetGenTest

tfold = os.path.dirname(__file__)


def test_bionetgen_model():
    fpath = os.path.join(tfold, "test.bngl")
    fpath = os.path.abspath(fpath)
    _m = bng.bngmodel(fpath)


def test_bionetgen_all_model_loading():
    # tests library model loading using many models
    mpattern = os.path.join(tfold, "models") + os.sep + "*.bngl"
    models = glob.glob(mpattern)
    succ = []
    fail = []
    success = 0
    fails = 0
    for model in models:
        try:
            _m = bng.bngmodel(model)
            success += 1
            _mstr = str(_m)
            succ.append(model)
        except:
            print(f"can't load model {model}")
            fails += 1
            fail.append(model)
    print(f"succ: {success}")
    print(sorted(succ))
    print(f"fail: {fails}")
    print(sorted(fail))
    assert fails == 0


def test_action_loading():
    # tests a BNGL file containing all BNG actions
    all_action_model = os.path.join(*[tfold, "models", "actions", "all_actions.bngl"])
    m1 = bng.bngmodel(all_action_model)
    assert len(m1.actions) + len(m1.actions.before_model) == 31

    no_action_model = os.path.join(*[tfold, "models", "actions", "no_actions.bngl"])
    m2 = bng.bngmodel(no_action_model)
    assert len(m2.actions) == 0


SKIPPED_MODEL_MARKERS = ("test_tfun",)


def _is_skipped_model(path):
    name = os.path.basename(path)
    return any(marker in name for marker in SKIPPED_MODEL_MARKERS)


# Sweeping every model through BNG2.pl + libroadrunner is an integration
# test that depends on a working perl + BNG vendor + system perf, and it
# can run for many minutes per model on some builds. Gate it behind an
# opt-in env var so the default `pytest` is fast and reliable.
_RUN_MODEL_SWEEPS = os.environ.get("BNG_RUN_MODEL_SWEEPS") == "1"
_skip_model_sweeps = pytest.mark.skipif(
    not _RUN_MODEL_SWEEPS,
    reason="set BNG_RUN_MODEL_SWEEPS=1 to run the full model integration sweep",
)


@_skip_model_sweeps
def test_model_running_CLI(tmp_path):
    # tests running a list of models using the CLI
    mpattern = os.path.join(tfold, "models") + os.sep + "*.bngl"
    models = glob.glob(mpattern)
    succ = []
    fail = []
    success = 0
    fails = 0
    for model in models:
        if _is_skipped_model(model):
            continue
        model_name = os.path.basename(model).replace(".bngl", "")
        try:
            argv = [
                "run",
                "-i",
                model,
                "-o",
                str(tmp_path / model_name),
            ]
            with BioNetGenTest(argv=argv) as app:
                app.run()
                assert app.exit_code == 0
            success += 1
            model = os.path.split(model)
            model = model[1]
            succ.append(model)
        except Exception as e:
            print(e)
            print(f"can't run model {model}")
            fails += 1
            model = os.path.split(model)
            model = model[1]
            fail.append(model)
    print(f"succ: {success}")
    print(sorted(succ))
    print(f"fail: {fails}")
    print(sorted(fail))
    assert fails == 0


@_skip_model_sweeps
def test_model_running_lib():
    # test running a list of models using the library
    mpattern = os.path.join(tfold, "models") + os.sep + "*.bngl"
    models = glob.glob(mpattern)
    succ = []
    fail = []
    success = 0
    fails = 0
    for model in models:
        if _is_skipped_model(model):
            continue
        try:
            bng.run(model)
            success += 1
            model = os.path.split(model)
            model = model[1]
            succ.append(model)
        except:
            print(f"can't run model {model}")
            fails += 1
            model = os.path.split(model)
            model = model[1]
            fail.append(model)
    print(f"succ: {success}")
    print(sorted(succ))
    print(f"fail: {fails}")
    print(sorted(fail))
    assert fails == 0


def test_setup_simulator():
    pytest.importorskip("roadrunner")
    fpath = os.path.abspath(os.path.join(tfold, "test.bngl"))
    m = bng.bngmodel(fpath)
    librr_simulator = m.setup_simulator()
    res = librr_simulator.simulate(0, 1, 10)
    assert res is not None
