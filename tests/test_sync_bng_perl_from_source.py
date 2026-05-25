"""Smoke tests for ``scripts/sync_bng_perl_from_source.py``.

The script's job is to refresh the Perl side of the bng-mac/bng-linux/bng-win
bundles from a local upstream BioNetGen checkout, when an upstream release tag
is not yet available with the changes you need (e.g. inline-array tfun()
parsing landed post-2.9.3 but no 2.9.4 tag exists yet). The release-driven
``vendor_bionetgen_assets.py`` doesn't support that case.

These tests build a minimal fake source tree (BNG2.pl + Perl2/) and exercise
the script's per-bundle copy + line-ending normalization in a temp directory
so the real PyBioNetGen bundles aren't touched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_bng_perl_from_source.py"


@pytest.fixture
def script_module(monkeypatch, tmp_path):
    """Load the script as a module with BUNDLE_ROOT pointed at a tmp dir.

    Each bundle (bng-mac/bng-linux/bng-win) is created as an empty
    directory under tmp_path/bionetgen so the refresh has somewhere to
    write into. The PROTECTED_PATHS are populated as sentinels we then
    assert remained untouched.
    """
    spec = importlib.util.spec_from_file_location("_sync_bng_perl_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bundle_root = tmp_path / "bionetgen"
    bundle_root.mkdir()
    for name in module.BUNDLE_LINE_ENDINGS:
        bundle_dir = bundle_root / name
        bundle_dir.mkdir()
        (bundle_dir / "bin").mkdir()
        (bundle_dir / "bin" / "run_network").write_bytes(b"\xfeED\xfaCE")  # sentinel
        (bundle_dir / "VERSION").write_text("BioNetGen-2.9.3 unchanged\n")
    (bundle_root / "assets").mkdir()

    monkeypatch.setattr(module, "BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(module, "ASSETS_DIR", bundle_root / "assets")
    monkeypatch.setattr(
        module,
        "PERL_SOURCE_MARKER",
        bundle_root / "assets" / "BNG_PERL_SOURCE",
    )
    return module, bundle_root


@pytest.fixture
def fake_source(tmp_path):
    """Build a minimal upstream source tree with a Perl module and BNG2.pl."""
    src = tmp_path / "src"
    (src / "Perl2").mkdir(parents=True)
    (src / "BNG2.pl").write_text('#!/usr/bin/perl\nprint "hello\\n";\n')
    (src / "Perl2" / "Expression.pm").write_text("package Expression;\n# stub for tests\n1;\n")
    return src


def test_refresh_writes_perl_into_each_bundle(script_module, fake_source):
    module, bundle_root = script_module
    module.main(
        [
            "--source-dir",
            str(fake_source),
            "--commit-sha",
            "deadbeef",
            "--branch",
            "master",
        ]
    )

    for name, eol in module.BUNDLE_LINE_ENDINGS.items():
        assert (bundle_root / name / "BNG2.pl").is_file()
        assert (bundle_root / name / "Perl2" / "Expression.pm").is_file()
        # Binary sentinel must remain untouched
        assert (bundle_root / name / "bin" / "run_network").read_bytes() == b"\xfeED\xfaCE"
        # VERSION file must remain untouched (release-tag identity preserved)
        assert (bundle_root / name / "VERSION").read_text() == "BioNetGen-2.9.3 unchanged\n"

        # Line endings must match the bundle convention
        bng_pl_bytes = (bundle_root / name / "BNG2.pl").read_bytes()
        if eol == "crlf":
            assert b"\r\n" in bng_pl_bytes
        else:
            assert b"\r\n" not in bng_pl_bytes


def test_refresh_writes_marker(script_module, fake_source):
    module, bundle_root = script_module
    module.main(
        [
            "--source-dir",
            str(fake_source),
            "--commit-sha",
            "0123abc",
            "--branch",
            "feature/tfun-fix",
        ]
    )
    marker = bundle_root / "assets" / "BNG_PERL_SOURCE"
    assert marker.is_file()
    text = marker.read_text()
    assert "source_repo: RuleWorld/bionetgen" in text
    assert "source_sha: 0123abc" in text
    assert "source_branch: feature/tfun-fix" in text


def test_dry_run_does_not_modify(script_module, fake_source):
    module, bundle_root = script_module
    # Pre-populate the win bundle with a sentinel Perl file we can detect
    (bundle_root / "bng-win" / "BNG2.pl").write_text("ORIGINAL\n")
    module.main(
        [
            "--source-dir",
            str(fake_source),
            "--commit-sha",
            "deadbeef",
            "--dry-run",
        ]
    )
    assert (bundle_root / "bng-win" / "BNG2.pl").read_text() == "ORIGINAL\n"
    assert not (bundle_root / "assets" / "BNG_PERL_SOURCE").exists()


def test_missing_source_paths_raise(script_module, tmp_path):
    module, _ = script_module
    bad = tmp_path / "missing_src"
    bad.mkdir()
    # No BNG2.pl, no Perl2/
    with pytest.raises(FileNotFoundError):
        module.main(
            [
                "--source-dir",
                str(bad),
                "--commit-sha",
                "x",
            ]
        )
