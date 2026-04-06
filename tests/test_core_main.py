"""Tests for bionetgen/core/main.py — thin wrappers around tool classes."""
from unittest import mock

import pytest


class TestRunCLI:
    def test_runs_bngcli(self):
        from bionetgen.core.main import runCLI
        app = mock.MagicMock()
        app.pargs.input = "test.bngl"
        app.pargs.output = "/tmp/out"
        app.pargs.log_file = None
        app.pargs.traceback_depth = 0
        app.config.get.return_value = "/fake/bng"
        with mock.patch("bionetgen.core.main.BNGCLI") as MockCLI:
            mock_cli = mock.MagicMock()
            MockCLI.return_value = mock_cli
            runCLI(app)
            MockCLI.assert_called_once()
            mock_cli.run.assert_called_once()


class TestPlotDAT:
    def test_plot_gdat(self):
        from bionetgen.core.main import plotDAT
        app = mock.MagicMock()
        app.pargs.input = "test.gdat"
        app.pargs.output = "/tmp/out.png"
        app.pargs._get_kwargs.return_value = []
        with mock.patch("bionetgen.core.tools.BNGPlotter") as MockPlotter:
            mock_plotter = mock.MagicMock()
            MockPlotter.return_value = mock_plotter
            plotDAT(app)
            MockPlotter.assert_called_once()
            mock_plotter.plot.assert_called_once()

    def test_plot_dot_output(self):
        from bionetgen.core.main import plotDAT
        app = mock.MagicMock()
        app.pargs.input = "/some/path/test.gdat"
        app.pargs.output = "."
        app.pargs._get_kwargs.return_value = []
        with mock.patch("bionetgen.core.tools.BNGPlotter") as MockPlotter:
            mock_plotter = mock.MagicMock()
            MockPlotter.return_value = mock_plotter
            plotDAT(app)
            # output should be resolved to /some/path/test.png
            call_args = MockPlotter.call_args
            assert call_args[0][1].endswith("test.png")

    def test_invalid_extension_raises(self):
        from bionetgen.core.main import plotDAT
        app = mock.MagicMock()
        app.pargs.input = "test.txt"
        with pytest.raises(AssertionError):
            plotDAT(app)


class TestPrintInfo:
    def test_runs_info(self):
        from bionetgen.core.main import printInfo
        app = mock.MagicMock()
        with mock.patch("bionetgen.core.main.BNGInfo") as MockInfo:
            mock_info = mock.MagicMock()
            MockInfo.return_value = mock_info
            printInfo(app)
            mock_info.gatherInfo.assert_called_once()
            mock_info.messageGeneration.assert_called_once()
            mock_info.run.assert_called_once()


class TestVisualizeModel:
    def test_runs_visualize(self):
        from bionetgen.core.main import visualizeModel
        app = mock.MagicMock()
        app.pargs.input = "test.bngl"
        app.pargs.output = "/tmp/out"
        app.pargs.type = "contactmap"
        app.config.get.return_value = "/fake/bng"
        with mock.patch("bionetgen.core.main.BNGVisualize") as MockViz:
            mock_viz = mock.MagicMock()
            MockViz.return_value = mock_viz
            visualizeModel(app)
            MockViz.assert_called_once()
            mock_viz.run.assert_called_once()


class TestGraphDiff:
    def test_runs_gdiff(self):
        from bionetgen.core.main import graphDiff
        app = mock.MagicMock()
        app.pargs.input = "g1.graphml"
        app.pargs.input2 = "g2.graphml"
        app.pargs.output = "out.graphml"
        app.pargs.output2 = None
        app.pargs.mode = "matrix"
        app.pargs.colors = None
        with mock.patch("bionetgen.core.main.BNGGdiff") as MockGdiff:
            mock_gdiff = mock.MagicMock()
            MockGdiff.return_value = mock_gdiff
            graphDiff(app)
            MockGdiff.assert_called_once()
            mock_gdiff.run.assert_called_once()


class TestGenerateNotebook:
    def test_with_input(self):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = "model.bngl"
        app.pargs.output = ""
        app.pargs.open = False
        app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        # bionetgen is imported inside the function via `import bionetgen`
        mock_bng = mock.MagicMock()
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB, \
             mock.patch.dict("sys.modules", {"bionetgen": mock_bng}):
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            generate_notebook(app)
            MockNB.assert_called_once()
            mock_nb.write.assert_called_once()

    def test_without_input(self):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = None
        app.pargs.output = ""
        app.pargs.open = False
        app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB:
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            generate_notebook(app)
            MockNB.assert_called_once_with("/tmp/nb.ipynb")
            mock_nb.write.assert_called_once()

    def test_with_dir_output_and_input(self, tmp_path):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = "mymodel.bngl"
        app.pargs.output = str(tmp_path)
        app.pargs.open = False
        app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        mock_bng = mock.MagicMock()
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB, \
             mock.patch.dict("sys.modules", {"bionetgen": mock_bng}):
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            generate_notebook(app)
            call_args = mock_nb.write.call_args[0][0]
            assert "mymodel" in call_args

    def test_with_open(self):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = None
        app.pargs.output = "nb.ipynb"
        app.pargs.open = True
        app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        with mock.patch("bionetgen.core.main.BNGNotebook") as MockNB, \
             mock.patch("bionetgen.core.main.run_command") as mock_rc:
            mock_nb = mock.MagicMock()
            MockNB.return_value = mock_nb
            mock_rc.return_value = (0, None)
            generate_notebook(app)
            mock_rc.assert_called_once()

    def test_bad_extension_raises(self):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = "model.txt"
        with pytest.raises(AssertionError):
            generate_notebook(app)

    def test_failed_load_raises(self):
        from bionetgen.core.main import generate_notebook
        app = mock.MagicMock()
        app.pargs.input = "model.bngl"
        app.pargs.output = ""
        app.pargs.open = False
        app.config = {"bionetgen": {
            "notebook": {
                "path": "/tmp/nb.ipynb",
                "template": "/tmp/nb_tmpl.ipynb",
                "name": "bng_notebook.ipynb",
            },
            "stdout": "DEVNULL",
            "stderr": "DEVNULL",
        }}
        mock_bng = mock.MagicMock()
        mock_bng.bngmodel.side_effect = Exception("fail")
        with mock.patch.dict("sys.modules", {"bionetgen": mock_bng}):
            with pytest.raises(RuntimeError):
                generate_notebook(app)
