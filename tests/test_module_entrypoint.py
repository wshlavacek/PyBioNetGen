import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "bionetgen", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_m_bionetgen_help():
    result = _run_module("--help")

    assert result.returncode == 0
    assert "usage:" in (result.stdout + result.stderr).lower()


def test_python_m_bionetgen_require_help():
    result = _run_module("-req", "0.5.0", "--help")

    assert result.returncode == 0
    assert "usage:" in (result.stdout + result.stderr).lower()
