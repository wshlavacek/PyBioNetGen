"""Tests for the developer-check launcher script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_run_dev_checks():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dev_checks.py"
    spec = importlib.util.spec_from_file_location("run_dev_checks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_local_bngsim_checkout_prefers_env_override(monkeypatch, tmp_path):
    module = _load_run_dev_checks()
    env_checkout = tmp_path / "env-bngsim"
    env_checkout.mkdir()
    (env_checkout / "pyproject.toml").write_text("[project]\nname='bngsim'\n", encoding="utf-8")

    default_checkout = tmp_path / "default-bngsim"
    default_checkout.mkdir()
    (default_checkout / "pyproject.toml").write_text("[project]\nname='bngsim'\n", encoding="utf-8")

    monkeypatch.setattr(module, "DEFAULT_BNGSIM_CHECKOUT", default_checkout)
    monkeypatch.setenv("PYBNG_DEV_BNGSIM_PATH", str(env_checkout))

    assert module.resolve_local_bngsim_checkout() == env_checkout.resolve()


def test_build_uv_command_uses_editable_checkout(monkeypatch):
    module = _load_run_dev_checks()
    checkout = Path("/tmp/bngsim")
    monkeypatch.setattr(module, "resolve_local_bngsim_checkout", lambda: checkout)

    command = module.build_uv_command("uv")

    assert command[:4] == ["uv", "run", "--no-project", "--with-requirements"]
    assert "--with-editable" in command
    # build_uv_command inserts str(checkout); on Windows that is \tmp\bngsim.
    assert str(checkout) in command
    assert ["--with", "lxml", "--with", "networkx"] == command[-4:]


def test_build_uv_command_falls_back_to_published_bngsim(monkeypatch):
    module = _load_run_dev_checks()
    monkeypatch.setattr(module, "resolve_local_bngsim_checkout", lambda: None)

    command = module.build_uv_command("uv")

    assert "--with-editable" not in command
    assert command.count("--with") == 3
    assert "bngsim" in command
    assert "lxml" in command
    assert "networkx" in command


def test_validate_bng_candidate_accepts_dir_and_file(monkeypatch, tmp_path):
    module = _load_run_dev_checks()
    bng_dir = tmp_path / "bng"
    bng_dir.mkdir()
    bng_exec = bng_dir / "BNG2.pl"
    bng_exec.write_text("#!/usr/bin/perl\n", encoding="utf-8")
    monkeypatch.setattr(module, "bng_exec_works", lambda path: path == bng_exec)

    assert module.validate_bng_candidate(bng_dir) == bng_dir.resolve()
    assert module.validate_bng_candidate(bng_exec) == bng_dir.resolve()


def test_release_asset_url_selects_platform_asset(tmp_path):
    module = _load_run_dev_checks()
    release_json = tmp_path / "ghapi.json"
    release_json.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "BioNetGen-2.9.3-linux.tar.gz",
                        "browser_download_url": "https://example.invalid/linux.tar.gz",
                    },
                    {
                        "name": "BioNetGen-2.9.3-mac.tar.gz",
                        "browser_download_url": "https://example.invalid/mac.tar.gz",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert module.release_asset_url(release_json, "linux") == "https://example.invalid/linux.tar.gz"
    assert module.release_asset_url(release_json, "mac") == "https://example.invalid/mac.tar.gz"


def test_parse_args_preserves_pytest_flags():
    module = _load_run_dev_checks()

    args = module.parse_args(["--skip-bng-download", "tests/test_bng_core.py", "-q"])

    assert args.skip_bng_download is True
    assert args.pytest_args == ["tests/test_bng_core.py", "-q"]
