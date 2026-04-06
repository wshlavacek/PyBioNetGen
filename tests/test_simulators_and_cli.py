"""
Tests for BNGSimulator base class, sim_getter, BNGCLI, and BNGInfo.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. BNGSimulator base class
# ---------------------------------------------------------------------------


class TestBNGSimulator:
    """Tests for bionetgen.simulator.bngsimulator.BNGSimulator."""

    def test_init_no_args(self):
        """__init__ with model_file=None, model_str=None: no crash."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        sim = BNGSimulator()
        # Neither _model_file nor _model_str should be set
        assert not hasattr(sim, "_model_file")
        assert not hasattr(sim, "_model_str")

    def test_init_with_model_file(self):
        """__init__ with model_file sets _model_file and calls simulator setter."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        # BNGSimulator doesn't define a `simulator` property -- the setter on
        # model_file just does ``self.simulator = mfile``, which creates an
        # instance attribute.  We can verify that.
        sim = BNGSimulator(model_file="/some/model.bngl")
        assert sim._model_file == "/some/model.bngl"
        # The setter assigned self.simulator = mfile
        assert sim.simulator == "/some/model.bngl"

    def test_init_with_model_str(self):
        """__init__ with model_str sets _model_str and calls simulator setter."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        sim = BNGSimulator(model_str="begin model\nend model\n")
        assert sim._model_str == "begin model\nend model\n"
        # The setter assigned self.simulator = mstr
        assert sim.simulator == "begin model\nend model\n"

    def test_model_file_property_getter_setter(self):
        """model_file property: setter stores value and sets simulator."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        sim = BNGSimulator()
        sim.model_file = "/new/path.bngl"
        assert sim.model_file == "/new/path.bngl"
        assert sim.simulator == "/new/path.bngl"

    def test_model_str_property_getter_setter(self):
        """model_str property: setter stores value and sets simulator."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        sim = BNGSimulator()
        sim.model_str = "some model string"
        assert sim.model_str == "some model string"
        assert sim.simulator == "some model string"

    def test_simulate_raises_not_implemented(self):
        """simulate() raises NotImplementedError."""
        from bionetgen.simulator.bngsimulator import BNGSimulator

        sim = BNGSimulator()
        with pytest.raises(NotImplementedError):
            sim.simulate()


# ---------------------------------------------------------------------------
# 2. sim_getter function
# ---------------------------------------------------------------------------


class TestSimGetter:
    """Tests for bionetgen.simulator.simulators.sim_getter."""

    @patch("bionetgen.simulator.simulators.CSimulator")
    @patch("bionetgen.simulator.simulators.libRRSimulator")
    def test_model_file_libRR(self, mock_libRR, mock_CSim):
        """model_file + sim_type='libRR' returns libRRSimulator."""
        from bionetgen.simulator.simulators import sim_getter

        sentinel = MagicMock(name="libRR_instance")
        mock_libRR.return_value = sentinel

        result = sim_getter(model_file="/path/model.bngl", sim_type="libRR")
        mock_libRR.assert_called_once_with(model_file="/path/model.bngl")
        assert result is sentinel

    @patch("bionetgen.simulator.simulators.CSimulator")
    @patch("bionetgen.simulator.simulators.libRRSimulator")
    def test_model_file_cpy(self, mock_libRR, mock_CSim):
        """model_file + sim_type='cpy' returns CSimulator."""
        from bionetgen.simulator.simulators import sim_getter

        sentinel = MagicMock(name="CSim_instance")
        mock_CSim.return_value = sentinel

        result = sim_getter(model_file="/path/model.bngl", sim_type="cpy")
        mock_CSim.assert_called_once_with(
            model_file="/path/model.bngl", generate_network=True
        )
        assert result is sentinel

    @patch("bionetgen.simulator.simulators.CSimulator")
    @patch("bionetgen.simulator.simulators.libRRSimulator")
    def test_model_file_unsupported(self, mock_libRR, mock_CSim, capsys):
        """Unsupported sim_type prints a message."""
        from bionetgen.simulator.simulators import sim_getter

        result = sim_getter(model_file="/path/model.bngl", sim_type="unknown")
        captured = capsys.readouterr()
        assert "unknown" in captured.out
        assert "not supported" in captured.out
        assert result is None

    @patch("bionetgen.simulator.simulators.CSimulator")
    @patch("bionetgen.simulator.simulators.libRRSimulator")
    def test_model_str_creates_temp_file_libRR(self, mock_libRR, mock_CSim):
        """model_str creates a temp file and calls libRRSimulator."""
        from bionetgen.simulator.simulators import sim_getter

        sentinel = MagicMock(name="libRR_instance")
        mock_libRR.return_value = sentinel

        result = sim_getter(model_str="begin model\nend model\n", sim_type="libRR")
        mock_libRR.assert_called_once()
        # The call should include model_file keyword
        kw = mock_libRR.call_args
        assert "model_file" in kw.kwargs or len(kw.args) > 0
        assert result is sentinel

    @patch("bionetgen.simulator.simulators.CSimulator")
    @patch("bionetgen.simulator.simulators.libRRSimulator")
    def test_model_str_creates_temp_file_cpy(self, mock_libRR, mock_CSim):
        """model_str creates a temp file and calls CSimulator."""
        from bionetgen.simulator.simulators import sim_getter

        sentinel = MagicMock(name="CSim_instance")
        mock_CSim.return_value = sentinel

        result = sim_getter(model_str="begin model\nend model\n", sim_type="cpy")
        mock_CSim.assert_called_once()
        kw = mock_CSim.call_args
        assert kw.kwargs.get("generate_network") is True
        assert result is sentinel


# ---------------------------------------------------------------------------
# 3. BNGCLI class
# ---------------------------------------------------------------------------


def _make_bngl(tmp_path):
    """Helper: create a minimal .bngl file and return its path."""
    bngl = tmp_path / "test_model.bngl"
    bngl.write_text("begin model\nend model\n")
    return str(bngl)


def _create_cli(inp_file, output, bngpath, **kwargs):
    """Import and create a BNGCLI instance."""
    from bionetgen.core.tools.cli import BNGCLI
    return BNGCLI(inp_file, output, bngpath, **kwargs)


class TestBNGCLI:
    """Tests for bionetgen.core.tools.cli.BNGCLI."""

    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", "/fake/bng/BNG2.pl"))
    def test_init_sets_paths(self, mock_find, tmp_path):
        """__init__ sets up paths and creates output directory."""
        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        cli = _create_cli(bngl, output_dir, "/fake/bng")

        assert cli.bngpath == "/fake/bng"
        assert cli.bng_exec == "/fake/bng/BNG2.pl"
        assert os.path.isdir(output_dir)
        assert cli.inp_path == os.path.abspath(bngl)

    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", "/fake/bng/BNG2.pl"))
    def test_set_output_creates_directory(self, mock_find, tmp_path):
        """_set_output creates the output directory."""
        bngl = _make_bngl(tmp_path)
        nested = str(tmp_path / "a" / "b" / "c")

        cli = _create_cli(bngl, nested, "/fake/bng")

        assert os.path.isdir(nested)
        assert cli.output == os.path.abspath(nested)

    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", None))
    def test_run_bng_exec_none_returns_empty_result(self, mock_find, tmp_path):
        """When bng_exec is None, run() returns empty BNGResult with rc=0."""
        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        with patch("bionetgen.core.tools.BNGResult") as MockResult:
            mock_res = MagicMock()
            MockResult.return_value = mock_res

            cli = _create_cli(bngl, output_dir, "/fake/bng")
            cli.run()

            assert cli.result is mock_res
            assert cli.result.process_return == 0
            assert cli.result.output == []

    @patch("bionetgen.core.utils.utils.run_command", return_value=(0, ["line1"]))
    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", "/fake/bng/BNG2.pl"))
    def test_run_success(self, mock_find, mock_run_cmd, tmp_path):
        """With bng_exec set, run() calls run_command and handles success."""
        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        cli = _create_cli(bngl, output_dir, "/fake/bng")

        with patch("bionetgen.core.tools.BNGResult") as MockResult:
            mock_res = MagicMock()
            MockResult.return_value = mock_res

            cli.run()

            assert cli.result is mock_res
            assert cli.result.process_return == 0
            assert cli.result.output == ["line1"]

    @patch("bionetgen.core.utils.utils.run_command", return_value=(1, []))
    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", "/fake/bng/BNG2.pl"))
    def test_run_failure_raises_bngrun_error(self, mock_find, mock_run_cmd, tmp_path):
        """run() raises BNGRunError when rc != 0."""
        from bionetgen.core.exc import BNGRunError

        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        cli = _create_cli(bngl, output_dir, "/fake/bng")

        with pytest.raises(BNGRunError):
            cli.run()

    @patch("bionetgen.core.utils.utils.run_command", return_value=(0, []))
    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", "/fake/bng/BNG2.pl"))
    def test_bngpath_env_restored_after_run(self, mock_find, mock_run_cmd, tmp_path):
        """BNGPATH environment variable is restored after run()."""
        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        original_val = "ORIGINAL_BNG_PATH"
        os.environ["BNGPATH"] = original_val

        try:
            cli = _create_cli(bngl, output_dir, "/fake/bng")
            assert cli.old_bngpath == original_val

            with patch("bionetgen.core.tools.BNGResult") as MockResult:
                MockResult.return_value = MagicMock()
                cli.run()

            # After run, BNGPATH should be restored to original
            assert os.environ.get("BNGPATH") == original_val
        finally:
            os.environ.pop("BNGPATH", None)

    @patch("bionetgen.core.utils.utils.find_BNG_path", return_value=("/fake/bng", None))
    def test_bngpath_env_deleted_when_not_originally_set(self, mock_find, tmp_path):
        """BNGPATH is removed from env after run() if it wasn't set before."""
        bngl = _make_bngl(tmp_path)
        output_dir = str(tmp_path / "output")

        # Make sure BNGPATH is not set
        os.environ.pop("BNGPATH", None)

        cli = _create_cli(bngl, output_dir, "/fake/bng")
        assert cli.old_bngpath is None

        with patch("bionetgen.core.tools.BNGResult") as MockResult:
            MockResult.return_value = MagicMock()
            cli.run()

        assert "BNGPATH" not in os.environ


# ---------------------------------------------------------------------------
# 4. BNGInfo class
# ---------------------------------------------------------------------------


class TestBNGInfo:
    """Tests for bionetgen.core.tools.info.BNGInfo."""

    def test_init_stores_config_args_app(self):
        """__init__ stores config, args, app."""
        from bionetgen.core.tools.info import BNGInfo

        config = MagicMock()
        args = MagicMock()
        app = MagicMock()

        info = BNGInfo(config, args=args, app=app)

        assert info.config is config
        assert info.args is args
        assert info.app is app

    def test_message_generation_formats_correctly(self):
        """messageGeneration formats self.info dict into a readable string."""
        from bionetgen.core.tools.info import BNGInfo

        config = MagicMock()
        info_obj = BNGInfo(config)
        # Pre-set the info dict directly (skip gatherInfo)
        info_obj.info = {
            "BNG version": "2.8.0",
            "CLI version": "1.0.0",
            "numpy version": "1.24.0",
        }

        message = info_obj.messageGeneration()

        assert "BNG version: 2.8.0" in message
        assert "CLI version: 1.0.0" in message
        assert "numpy version: 1.24.0" in message
        # Each entry should end with a newline
        assert message.count("\n") >= 3

    def test_message_generation_empty_values(self):
        """messageGeneration handles entries with empty string values."""
        from bionetgen.core.tools.info import BNGInfo

        config = MagicMock()
        info_obj = BNGInfo(config)
        info_obj.info = {
            "\nSection header": "",
            "Some key": "some value",
        }

        message = info_obj.messageGeneration()

        assert "\nSection header: \n" in message
        assert "Some key: some value" in message

    def test_run_prints_message(self, capsys):
        """run() calls print with self.message."""
        from bionetgen.core.tools.info import BNGInfo

        config = MagicMock()
        info_obj = BNGInfo(config)
        info_obj.info = {"key": "value"}
        info_obj.messageGeneration()

        info_obj.run()

        captured = capsys.readouterr()
        assert "key: value" in captured.out
