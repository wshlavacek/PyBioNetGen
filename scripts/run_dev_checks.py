#!/usr/bin/env python3
"""Run the default developer checks in the same order locally and in CI."""

import subprocess
import sys


def main(argv):
    pytest_args = argv or ["tests/"]
    subprocess.run([sys.executable, "-m", "pytest", *pytest_args], check=True)
    subprocess.run([sys.executable, "-m", "mypy", "bionetgen", "tests"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
