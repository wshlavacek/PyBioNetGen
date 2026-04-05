"""Tests for plot.py, visualize.py, csimulator.py, and other remaining gaps."""
import os
import pytest
import numpy as np
from unittest import mock


# ── BNGPlotter tests ──────────────────────────────────────────────

class TestBNGPlotter:
    def test_init(self, tmp_path):
        gdat = tmp_path / "test.gdat"
        gdat.write_text("# time A B\n0.0 1.0 2.0\n1.0 3.0 4.0\n")
        from bionetgen.core.tools.plot import BNGPlotter
        p = BNGPlotter(str(gdat), str(tmp_path / "out.png"))
        assert p.inp == str(gdat)
        assert p.result is not None

    def test_plot_gdat(self, tmp_path):
        gdat = tmp_path / "test.gdat"
        gdat.write_text("# time A B\n0.0 1.0 2.0\n1.0 3.0 4.0\n")
        out = tmp_path / "out.png"
        from bionetgen.core.tools.plot import BNGPlotter
        p = BNGPlotter(str(gdat), str(out))
        mock_sbrn = mock.MagicMock()
        mock_plt = mock.MagicMock()
        mock_ax = mock.MagicMock()
        mock_fig = mock.MagicMock()
        mock_gca = mock.MagicMock()
        mock_gca.get_xlim.return_value = (0, 1)
        mock_gca.get_ylim.return_value = (0, 4)
        mock_gca.legend.return_value = mock.MagicMock()
        mock_fig.gca.return_value = mock_gca
        mock_ax.get_figure.return_value = mock_fig
        mock_sbrn.lineplot.return_value = mock_ax
        with mock.patch.dict("sys.modules", {"seaborn": mock_sbrn, "matplotlib.pyplot": mock_plt}):
            p._datplot()
            mock_plt.savefig.assert_called_once()

    def test_plot_scan(self, tmp_path):
        scan = tmp_path / "test.scan"
        scan.write_text("# param A\n0.1 1.0\n0.2 2.0\n")
        out = tmp_path / "out.png"
        from bionetgen.core.tools.plot import BNGPlotter
        p = BNGPlotter(str(scan), str(out))
        mock_sbrn = mock.MagicMock()
        mock_plt = mock.MagicMock()
        mock_ax = mock.MagicMock()
        mock_fig = mock.MagicMock()
        mock_gca = mock.MagicMock()
        mock_gca.get_xlim.return_value = (0, 1)
        mock_gca.get_ylim.return_value = (0, 4)
        mock_gca.legend.return_value = mock.MagicMock()
        mock_fig.gca.return_value = mock_gca
        mock_ax.get_figure.return_value = mock_fig
        mock_sbrn.lineplot.return_value = mock_ax
        with mock.patch.dict("sys.modules", {"seaborn": mock_sbrn, "matplotlib.pyplot": mock_plt}):
            p._datplot()
            mock_plt.savefig.assert_called_once()

    def test_plot_unknown_raises(self, tmp_path):
        dat = tmp_path / "test.xyz"
        dat.write_text("# a b\n1 2\n")
        from bionetgen.core.tools.plot import BNGPlotter
        p = BNGPlotter.__new__(BNGPlotter)
        p.logger = mock.MagicMock()
        p.result = mock.MagicMock()
        p.result.file_extension = ".xyz"
        with pytest.raises(NotImplementedError):
            p.plot()

    def test_plot_with_kwargs(self, tmp_path):
        gdat = tmp_path / "test.gdat"
        gdat.write_text("# time A\n0.0 1.0\n1.0 3.0\n")
        out = tmp_path / "out.png"
        from bionetgen.core.tools.plot import BNGPlotter
        p = BNGPlotter(str(gdat), str(out), legend=True, xlabel="Time", ylabel="Conc", title="Test")
        mock_sbrn = mock.MagicMock()
        mock_plt = mock.MagicMock()
        mock_ax = mock.MagicMock()
        mock_fig = mock.MagicMock()
        mock_gca = mock.MagicMock()
        mock_gca.get_xlim.return_value = (0, 1)
        mock_gca.get_ylim.return_value = (0, 4)
        mock_gca.legend.return_value = mock.MagicMock()
        mock_fig.gca.return_value = mock_gca
        mock_ax.get_figure.return_value = mock_fig
        mock_sbrn.lineplot.return_value = mock_ax
        with mock.patch.dict("sys.modules", {"seaborn": mock_sbrn, "matplotlib.pyplot": mock_plt}):
            p._datplot()
            mock_plt.xlabel.assert_called()
            mock_plt.ylabel.assert_called()
            mock_plt.title.assert_called()


# ── BNGVisualize tests ──────────────────────────────────────────────

class TestBNGVisualize:
    def test_init_default_vtype(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl")
        assert v.vtype == "contactmap"
        assert v.input == "test.bngl"

    def test_init_custom_vtype(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl", vtype="regulatory")
        assert v.vtype == "regulatory"

    def test_init_all_vtype(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl", vtype="all")
        assert v.vtype == "all"

    def test_init_invalid_vtype_raises(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        with pytest.raises(ValueError):
            BNGVisualize("test.bngl", vtype="invalid_type")

    def test_init_empty_vtype(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl", vtype="")
        assert v.vtype == "contactmap"

    def test_init_with_output(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl", output="/tmp/out")
        assert v.output == "/tmp/out"

    def test_run_calls_normal_mode(self):
        from bionetgen.core.tools.visualize import BNGVisualize
        v = BNGVisualize("test.bngl")
        with mock.patch.object(v, "_normal_mode", return_value=mock.MagicMock()) as m:
            v.run()
            m.assert_called_once()

    # _normal_mode requires complex mocking of bionetgen.modelapi.bngmodel
    # which is a module-level import — skip to focus on bngsim_bridge


# ── VisResult tests ──────────────────────────────────────────────

class TestVisResult:
    def test_init(self, tmp_path):
        from bionetgen.core.tools.visualize import VisResult
        # Create a graphml file in tmp_path
        gml = tmp_path / "test.graphml"
        gml.write_text("<graphml></graphml>")
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            vr = VisResult(str(tmp_path), name="test")
            assert len(vr.files) == 1
            assert "test.graphml" in vr.files[0]
        finally:
            os.chdir(old_cwd)

    def test_dump_files(self, tmp_path):
        from bionetgen.core.tools.visualize import VisResult
        gml = tmp_path / "test.graphml"
        gml.write_text("<graphml>data</graphml>")
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            vr = VisResult(str(tmp_path))
            assert len(vr.files) > 0
            out_dir = tmp_path / "output"
            out_dir.mkdir()
            vr._dump_files(str(out_dir))
            assert (out_dir / "test.graphml").exists()
        finally:
            os.chdir(old_cwd)


# ── CSimWrapper tests ──────────────────────────────────────────────

class TestCSimWrapper:
    def test_result_struct_fields(self):
        from bionetgen.simulator.csimulator import RESULT
        import ctypes
        # Verify struct fields exist
        field_names = [f[0] for f in RESULT._fields_]
        assert "status" in field_names
        assert "n_observables" in field_names
        assert "n_species" in field_names
        assert "n_tpts" in field_names
        assert "observables" in field_names
        assert "species" in field_names


# ── runner.py tests ──────────────────────────────────────────────

class TestRunner:
    # runner.run() has complex import-time dependencies; tested via integration tests
    pass


# ── main.py (BNGBase / CLI) tests ──────────────────────────────────

class TestBioNetGenApp:
    def test_app_import(self):
        from bionetgen.main import BioNetGen
        app = BioNetGen()
        app.setup()
        assert app.config is not None

    def test_app_config_has_bngpath(self):
        from bionetgen.main import BioNetGen
        app = BioNetGen()
        app.setup()
        assert "bngpath" in app.config["bionetgen"]


# ── librrsimulator tests ──────────────────────────────────────────

class TestLibRRSimulator:
    def test_simulator_setter(self):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        sim = libRRSimulator.__new__(libRRSimulator)
        mock_rr = mock.MagicMock()
        mock_rr_sim = mock.MagicMock()
        mock_rr.RoadRunner.return_value = mock_rr_sim
        with mock.patch.dict("sys.modules", {"roadrunner": mock_rr}):
            sim.simulator = "/fake/model.bngl"
            mock_rr.RoadRunner.assert_called_once_with("/fake/model.bngl")

    def test_simulator_setter_no_roadrunner(self, capsys):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        sim = libRRSimulator.__new__(libRRSimulator)
        # Simulate ImportError for roadrunner
        with mock.patch.dict("sys.modules", {"roadrunner": None}):
            sim.simulator = "/fake/model.bngl"
            captured = capsys.readouterr()
            assert "not installed" in captured.out

    def test_simulate(self):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        sim = libRRSimulator.__new__(libRRSimulator)
        sim._simulator = mock.MagicMock()
        sim._simulator.simulate.return_value = "result"
        result = sim.simulate(t_start=0, t_end=10, n_steps=5)
        sim._simulator.simulate.assert_called_once_with(t_start=0, t_end=10, n_steps=5)

    def test_sbml_property(self):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        sim = libRRSimulator.__new__(libRRSimulator)
        sim._simulator = mock.MagicMock()
        sim._simulator.getCurrentSBML.return_value = "<sbml/>"
        assert sim.sbml == "<sbml/>"

    def test_sbml_setter(self):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        sim = libRRSimulator.__new__(libRRSimulator)
        sim.sbml = "<custom_sbml/>"
        assert sim._sbml == "<custom_sbml/>"

    def test_init_with_model_file(self):
        from bionetgen.simulator.librrsimulator import libRRSimulator
        mock_rr = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"roadrunner": mock_rr}):
            sim = libRRSimulator(model_file="/fake/model.bngl")
            mock_rr.RoadRunner.assert_called()
