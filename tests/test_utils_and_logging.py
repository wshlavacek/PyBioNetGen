"""Tests for bionetgen/core/utils/utils.py and bionetgen/core/utils/logging.py"""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from bionetgen.core.exc import BNGPerlError
from bionetgen.core.utils.logging import BNGLogger
from bionetgen.core.utils.utils import (
    ActionList,
    find_BNG_path,
    run_command,
)
from bionetgen.core.utils.utils import (
    test_bngexec as check_bngexec,
)
from bionetgen.core.utils.utils import (
    test_perl as check_perl,
)

# ======================================================================
# ActionList tests
# ======================================================================


class TestActionList:
    def test_init_creates_correct_lists(self):
        al = ActionList()
        # normal_types is a non-empty list with known members
        assert isinstance(al.normal_types, list)
        assert "generate_network" in al.normal_types
        assert "simulate" in al.normal_types
        assert "parameter_scan" in al.normal_types
        assert "visualize" in al.normal_types

    def test_no_setter_syntax(self):
        al = ActionList()
        assert isinstance(al.no_setter_syntax, list)
        assert "setConcentration" in al.no_setter_syntax
        assert "setParameter" in al.no_setter_syntax
        assert "quit" in al.no_setter_syntax
        assert "setModelName" in al.no_setter_syntax

    def test_square_braces(self):
        al = ActionList()
        assert isinstance(al.square_braces, list)
        assert "saveConcentrations" in al.square_braces
        assert "resetConcentrations" in al.square_braces
        assert "resetParameters" in al.square_braces
        assert "saveParameters" in al.square_braces

    def test_before_model(self):
        al = ActionList()
        expected = ["setModelName", "substanceUnits", "version", "setOption"]
        assert al.before_model == expected

    def test_possible_types_is_union(self):
        al = ActionList()
        # possible_types = normal_types + no_setter_syntax + square_braces
        expected = al.normal_types + al.no_setter_syntax + al.square_braces
        assert al.possible_types == expected

    def test_arg_dict_has_all_action_types(self):
        al = ActionList()
        for action in al.possible_types:
            assert action in al.arg_dict, f"{action} missing from arg_dict"

    def test_irregular_args(self):
        al = ActionList()
        assert al.irregular_args["max_stoich"] == "dict"
        assert al.irregular_args["actions"] == "list"
        assert al.irregular_args["sample_times"] == "list"
        assert al.irregular_args["par_scan_vals"] == "list"
        assert al.irregular_args["blocks"] == "list"
        assert al.irregular_args["opts"] == "list"

    def test_is_before_model_true(self):
        al = ActionList()
        for name in ["setModelName", "substanceUnits", "version", "setOption"]:
            assert al.is_before_model(name) is True

    def test_is_before_model_false(self):
        al = ActionList()
        for name in ["simulate", "generate_network", "writeFile", "quit"]:
            assert al.is_before_model(name) is False


# ======================================================================
# find_BNG_path tests
# ======================================================================


class TestFindBNGPath:
    @patch("bionetgen.core.utils.utils.test_bngexec", return_value=True)
    def test_explicit_bngpath_directory(self, mock_exec):
        result = find_BNG_path(BNGPATH="/some/bng/dir")
        assert result == ("/some/bng/dir", os.path.join("/some/bng/dir", "BNG2.pl"))
        mock_exec.assert_called_once_with(os.path.join("/some/bng/dir", "BNG2.pl"))

    @patch("bionetgen.core.utils.utils.test_bngexec", return_value=True)
    def test_explicit_bngpath_file(self, mock_exec):
        """When BNGPATH points directly to BNG2.pl file."""
        result = find_BNG_path(BNGPATH="/some/bng/dir/BNG2.pl")
        assert result == ("/some/bng/dir", "/some/bng/dir/BNG2.pl")
        mock_exec.assert_called_once_with("/some/bng/dir/BNG2.pl")

    @patch("bionetgen.core.utils.utils.test_bngexec", return_value=True)
    @patch.dict(os.environ, {"BNGPATH": "/env/bng/dir"})
    def test_env_var_bngpath(self, mock_exec):
        result = find_BNG_path()
        assert result == ("/env/bng/dir", os.path.join("/env/bng/dir", "BNG2.pl"))

    @patch("bionetgen.core.utils.utils.test_bngexec", return_value=True)
    @patch("bionetgen.core.utils.utils.spawn.find_executable", return_value="/usr/bin/BNG2.pl")
    @patch.dict(os.environ, {}, clear=True)
    def test_bng_on_path(self, mock_find, mock_exec):
        result = find_BNG_path()
        assert result == ("/usr/bin", "/usr/bin/BNG2.pl")
        mock_find.assert_called_once_with("BNG2.pl")

    @patch("bionetgen.core.utils.utils.test_bngexec", return_value=False)
    @patch("bionetgen.core.utils.utils.spawn.find_executable", return_value=None)
    @patch.dict(os.environ, {}, clear=True)
    def test_nothing_found(self, mock_find, mock_exec):
        result = find_BNG_path()
        assert result == (None, None)


# ======================================================================
# test_perl tests
# ======================================================================


class TestTestPerl:
    @patch("bionetgen.core.utils.utils.run_command", return_value=(0, None))
    @patch("bionetgen.core.utils.utils.spawn.find_executable", return_value="/usr/bin/perl")
    def test_success(self, mock_find, mock_run):
        # Should not raise
        check_perl()
        mock_run.assert_called_once_with(["/usr/bin/perl", "-v"])

    @patch("bionetgen.core.utils.utils.spawn.find_executable", return_value=None)
    def test_perl_not_found_raises(self, mock_find):
        with pytest.raises(BNGPerlError):
            check_perl()

    @patch("bionetgen.core.utils.utils.run_command", return_value=(1, None))
    @patch("bionetgen.core.utils.utils.spawn.find_executable", return_value="/usr/bin/perl")
    def test_perl_fails_raises(self, mock_find, mock_run):
        with pytest.raises(BNGPerlError):
            check_perl()

    @patch("bionetgen.core.utils.utils.run_command", return_value=(0, None))
    def test_explicit_perl_path(self, mock_run):
        check_perl(perl_path="/custom/perl")
        mock_run.assert_called_once_with(["/custom/perl", "-v"])


# ======================================================================
# test_bngexec tests
# ======================================================================


class TestTestBngexec:
    @patch("bionetgen.core.utils.utils.run_command", return_value=(0, None))
    def test_returns_true_on_success(self, mock_run):
        assert check_bngexec("/path/to/BNG2.pl") is True
        mock_run.assert_called_once_with(["perl", "/path/to/BNG2.pl"])

    @patch("bionetgen.core.utils.utils.run_command", return_value=(1, None))
    def test_returns_false_on_failure(self, mock_run):
        assert check_bngexec("/path/to/BNG2.pl") is False


# ======================================================================
# run_command tests
# ======================================================================


class TestRunCommand:
    @patch("bionetgen.core.utils.utils.subprocess.run")
    def test_timeout_suppress(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        rc, result = run_command(["cmd"], suppress=True, timeout=30)
        assert rc == 0
        assert result is mock_result
        mock_run.assert_called_once_with(
            ["cmd"],
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=None,
        )

    @patch("bionetgen.core.utils.utils.subprocess.run")
    def test_timeout_no_suppress(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        rc, result = run_command(["cmd"], suppress=False, timeout=30)
        assert rc == 0
        mock_run.assert_called_once_with(
            ["cmd"],
            timeout=30,
            capture_output=True,
            cwd=None,
        )

    @patch("bionetgen.core.utils.utils.subprocess.Popen")
    def test_no_timeout_suppress(self, mock_popen):
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        rc, result = run_command(["cmd"], suppress=True, timeout=None)
        assert rc == 0
        assert result is mock_process
        mock_popen.assert_called_once_with(
            ["cmd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=-1,
            cwd=None,
        )

    @patch("bionetgen.core.utils.utils.subprocess.Popen")
    def test_no_timeout_no_suppress(self, mock_popen):
        """Reads stdout line by line until empty + poll is not None."""
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        # readline returns lines then empty string
        mock_stdout.readline.side_effect = ["line1\n", "line2\n", ""]
        mock_process.stdout = mock_stdout
        # poll is only called when readline returns "", at which point process is done
        mock_process.poll.return_value = 0
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        rc, result = run_command(["cmd"], suppress=False, timeout=None)
        assert rc == 0
        assert isinstance(result, list)
        assert "line1" in result
        assert "line2" in result
        mock_popen.assert_called_once_with(
            ["cmd"],
            stdout=subprocess.PIPE,
            encoding="utf8",
            cwd=None,
        )

    @patch("bionetgen.core.utils.utils.subprocess.run")
    def test_cwd_is_forwarded(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        run_command(["cmd"], suppress=True, timeout=10, cwd="/tmp")
        mock_run.assert_called_once_with(
            ["cmd"],
            timeout=10,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/tmp",
        )


# ======================================================================
# BNGLogger tests
# ======================================================================


class TestBNGLoggerInit:
    def test_default_level(self):
        logger = BNGLogger()
        assert logger.level == "INFO"
        assert logger.app is None
        assert logger.loc is None

    def test_custom_level(self):
        logger = BNGLogger(level="DEBUG")
        assert logger.level == "DEBUG"

    @patch("bionetgen.core.utils.logging.log_level", "CRITICAL")
    def test_global_log_level_overrides(self):
        logger = BNGLogger(level="DEBUG")
        assert logger.level == "CRITICAL"

    def test_loc_stored(self):
        logger = BNGLogger(loc="somefile.py : MyClass.method")
        assert logger.loc == "somefile.py : MyClass.method"


class TestBNGLoggerGetLogger:
    def test_with_loc_string(self):
        logger_obj = BNGLogger()
        py_logger = logger_obj.get_logger(loc="somefile.py : ClassName.method")
        assert py_logger.name == "ClassName"

    def test_with_self_loc(self):
        logger_obj = BNGLogger(loc="somefile.py : SelfClass.method")
        py_logger = logger_obj.get_logger()
        assert py_logger.name == "SelfClass"

    def test_without_loc_returns_root(self):
        logger_obj = BNGLogger()
        py_logger = logger_obj.get_logger()
        assert py_logger.name == "root"

    def test_adds_handler_when_no_handlers(self):
        """When a logger has no handlers at all (including parent), a handler is added."""
        import colorlog

        unique_name = "UniqueTestLoggerXYZ456"
        test_logger = colorlog.getLogger(unique_name)
        test_logger.handlers.clear()
        # Temporarily disable propagation so hasHandlers() returns False
        test_logger.propagate = False
        try:
            assert not test_logger.hasHandlers()
            logger_obj = BNGLogger()
            py_logger = logger_obj.get_logger(loc=f"file.py : {unique_name}.method")
            assert len(py_logger.handlers) > 0
        finally:
            test_logger.propagate = True

    def test_skips_handler_when_parent_has_handlers(self):
        """When parent logger has handlers, hasHandlers() is True and no handler is added."""
        import colorlog

        unique_name = "UniqueTestLoggerXYZ789"
        test_logger = colorlog.getLogger(unique_name)
        test_logger.handlers.clear()
        # With propagation on (default), root logger's handlers make hasHandlers() True
        assert test_logger.hasHandlers()

        logger_obj = BNGLogger()
        py_logger = logger_obj.get_logger(loc=f"file.py : {unique_name}.method")
        # No handler added directly to this logger since parent has handlers
        assert len(py_logger.handlers) == 0


class TestBNGLoggerMethods:
    """Test debug/info/warning/error/critical log methods."""

    def _make_logger(self, level="DEBUG"):
        return BNGLogger(level=level)

    @pytest.mark.parametrize("method_name", ["debug", "info", "warning", "error", "critical"])
    def test_with_loc_arg(self, method_name):
        """Calling with explicit loc= argument."""
        logger_obj = self._make_logger()
        with patch.object(logger_obj, "get_logger") as mock_get:
            mock_py_logger = MagicMock()
            mock_get.return_value = mock_py_logger
            method = getattr(logger_obj, method_name)
            method("test message", loc="file.py : Cls.meth")
            mock_get.assert_called_once_with(loc="file.py : Cls.meth")
            log_call = getattr(mock_py_logger, method_name)
            log_call.assert_called_once()
            # The message should contain the loc prefix
            call_msg = log_call.call_args[0][0]
            assert "file.py : Cls.meth" in call_msg

    @pytest.mark.parametrize("method_name", ["debug", "info", "warning", "error", "critical"])
    def test_with_self_loc(self, method_name):
        """Calling without loc= but with self.loc set."""
        logger_obj = BNGLogger(level="DEBUG", loc="file.py : SelfCls.meth")
        with patch.object(logger_obj, "get_logger") as mock_get:
            mock_py_logger = MagicMock()
            mock_get.return_value = mock_py_logger
            method = getattr(logger_obj, method_name)
            method("test message")
            mock_get.assert_called_once_with(loc=None)
            log_call = getattr(mock_py_logger, method_name)
            call_msg = log_call.call_args[0][0]
            assert "SelfCls.meth" in call_msg

    @pytest.mark.parametrize("method_name", ["debug", "info", "warning", "error", "critical"])
    def test_without_any_loc(self, method_name):
        """Calling with no loc and no self.loc -- plain message."""
        logger_obj = self._make_logger()
        with patch.object(logger_obj, "get_logger") as mock_get:
            mock_py_logger = MagicMock()
            mock_get.return_value = mock_py_logger
            method = getattr(logger_obj, method_name)
            method("plain message")
            log_call = getattr(mock_py_logger, method_name)
            call_msg = log_call.call_args[0][0]
            assert call_msg == "plain message"

    @pytest.mark.parametrize("method_name", ["debug", "info", "warning", "error", "critical"])
    def test_with_app(self, method_name):
        """When app is provided, delegates to app.log.<method>."""
        mock_app = MagicMock()
        # Need pargs to not trigger debug/log_level branches
        mock_app.pargs.debug = False
        mock_app.pargs.log_level = None
        mock_app.log.get_level.return_value = "INFO"

        logger_obj = BNGLogger(app=mock_app)
        method = getattr(logger_obj, method_name)
        method("app message", loc="some.loc")
        app_log_method = getattr(mock_app.log, method_name)
        app_log_method.assert_called_once_with("app message", "some.loc")
