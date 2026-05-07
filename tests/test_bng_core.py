import os

from pytest import raises

from bionetgen.main import BioNetGenTest

tfold = os.path.dirname(__file__)


def test_bionetgen_help():
    # tests basic command help
    with raises(SystemExit):
        argv = ["--help"]
        with BioNetGenTest(argv=argv) as app:
            app.run()
            assert app.exit_code == 0


def test_bionetgen_input(tmp_path):
    out_dir = tmp_path / "test"
    argv = [
        "run",
        "-i",
        os.path.join(tfold, "test.bngl"),
        "-o",
        str(out_dir),
    ]
    with BioNetGenTest(argv=argv) as app:
        app.run()
        assert app.exit_code == 0
        produced = set(os.listdir(out_dir))
        assert {"test.xml", "test.cdat", "test.gdat", "test.net"} <= produced


def test_bionetgen_plot(tmp_path):
    # generate a fresh .gdat first so the test does not depend on prior runs
    out_dir = tmp_path / "test"
    run_argv = [
        "run",
        "-i",
        os.path.join(tfold, "test.bngl"),
        "-o",
        str(out_dir),
    ]
    with BioNetGenTest(argv=run_argv) as app:
        app.run()
        assert app.exit_code == 0

    gdat = out_dir / "test.gdat"
    png = out_dir / "test.png"
    assert gdat.is_file()

    plot_argv = [
        "plot",
        "-i",
        str(gdat),
        "-o",
        str(png),
    ]
    with BioNetGenTest(argv=plot_argv) as app:
        app.run()
        assert app.exit_code == 0
        assert png.is_file()


def test_bionetgen_info():
    # tests info subcommand
    argv = ["info"]
    with BioNetGenTest(argv=argv) as app:
        app.run()
        assert app.exit_code == 0
