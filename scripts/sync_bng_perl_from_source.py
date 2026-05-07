#!/usr/bin/env python3
"""Refresh only the Perl side of the vendored BNG2.pl bundles from a local source tree.

The companion script ``vendor_bionetgen_assets.py`` pulls full platform
archives from a tagged BNG release on GitHub. That works when upstream has
cut a release that contains the changes you need. When you need code that
has been merged to ``master`` upstream but not yet tagged (e.g. the inline
``tfun([..],[..],idx)`` parser that landed after BioNetGen-2.9.3), this
script does the strict subset that's actually safe: copy the Perl tree
from a local upstream checkout into the three platform bundles, preserve
the per-bundle line-ending convention, and leave the release-tagged
binaries (``bin/run_network``, ``bin/NFsim``, etc.) and version metadata
alone.

A small ``bionetgen/assets/BNG_PERL_SOURCE`` file is written so the source
of the post-release Perl is always discoverable. When upstream ships a
real release tag again, ``vendor_bionetgen_assets.py`` will overwrite the
Perl with the tagged release and this marker becomes stale — the test
suite will flag that the next time the marker is regenerated.

Usage::

    python scripts/sync_bng_perl_from_source.py \\
        --source-dir ~/Code/bionetgen/bng2 \\
        --commit-sha 62831910

The source directory must contain ``BNG2.pl`` and a ``Perl2/`` subtree.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "bionetgen"
ASSETS_DIR = BUNDLE_ROOT / "assets"
PERL_SOURCE_MARKER = ASSETS_DIR / "BNG_PERL_SOURCE"

# Per-bundle line-ending convention. bng-win has historically shipped
# with CRLF line endings; mac/linux ship with LF.
BUNDLE_LINE_ENDINGS = {
    "bng-mac": "lf",
    "bng-linux": "lf",
    "bng-win": "crlf",
}

# Files copied from <source>/ into each bundle. Keep this minimal — every
# entry here is also implicitly something we now own across platforms.
SYNCED_PATHS = ("BNG2.pl", "Perl2")

# Anything sitting alongside Perl in the bundles that we MUST preserve
# (not copied in, not deleted). Listed here defensively so a future
# refactor that broadens what gets clobbered is forced to think.
PROTECTED_PATHS = ("bin", "VERSION")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to a local bionetgen checkout's bng2/ directory (e.g. ~/Code/bionetgen/bng2)",
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=None,
        help="Source commit SHA recorded in BNG_PERL_SOURCE. If omitted, attempt git rev-parse on --source-dir.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Source branch name recorded in BNG_PERL_SOURCE (informational; default: 'master').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without modifying the bundle.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_sha(source_dir: Path, override: Optional[str]) -> str:
    if override:
        return override.strip()
    try:
        out = subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"--commit-sha was not provided and git rev-parse failed in {source_dir}: {exc}"
        ) from exc


def _validate_source(source_dir: Path) -> None:
    bng_pl = source_dir / "BNG2.pl"
    perl_dir = source_dir / "Perl2"
    if not bng_pl.is_file():
        raise FileNotFoundError(f"Missing BNG2.pl under {source_dir}")
    if not perl_dir.is_dir():
        raise FileNotFoundError(f"Missing Perl2/ under {source_dir}")


def _is_text(path: Path) -> bool:
    """Heuristically decide if a file should have its line endings normalized.

    We only rewrite endings on files that are unambiguously Perl source.
    Anything else (data files, README binaries, etc.) is copied verbatim.
    """
    return path.suffix.lower() in {".pl", ".pm", ".txt", ".md"}


def _normalize_line_endings(target_dir: Path, mode: str) -> None:
    """Rewrite every text file under *target_dir* to use the chosen ending.

    *mode* must be 'lf' or 'crlf'. Files identified as binary are left
    untouched.
    """
    if mode not in {"lf", "crlf"}:
        raise ValueError(f"Unsupported line-ending mode: {mode!r}")
    eol = b"\r\n" if mode == "crlf" else b"\n"
    for path in target_dir.rglob("*"):
        if not path.is_file() or not _is_text(path):
            continue
        data = path.read_bytes()
        if not data:
            continue
        # Normalize: collapse any existing \r\n, \r, or \n to \n, then
        # rewrite with the desired ending.
        canonical = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        rewritten = canonical.replace(b"\n", eol)
        if rewritten != data:
            path.write_bytes(rewritten)


def _refresh_one_bundle(source_dir: Path, bundle_name: str, dry_run: bool) -> None:
    bundle_dir = BUNDLE_ROOT / bundle_name
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Expected bundle directory not found: {bundle_dir}")

    eol = BUNDLE_LINE_ENDINGS[bundle_name]

    if dry_run:
        print(f"[dry-run] would refresh {bundle_dir} ({eol}) from {source_dir}")
        return

    for rel in SYNCED_PATHS:
        src = source_dir / rel
        dst = bundle_dir / rel
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    _normalize_line_endings(bundle_dir / "Perl2", eol)
    bng_pl = bundle_dir / "BNG2.pl"
    if bng_pl.is_file():
        # Treat BNG2.pl as a single text file so its endings match the bundle convention.
        data = bng_pl.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if eol == "crlf":
            data = data.replace(b"\n", b"\r\n")
        bng_pl.write_bytes(data)

    print(f"refreshed {bundle_dir} ({eol})")


def _write_marker(sha: str, branch: str, source_dir: Path, dry_run: bool) -> None:
    today = datetime.date.today().isoformat()
    body = (
        f"source_repo: RuleWorld/bionetgen\n"
        f"source_branch: {branch}\n"
        f"source_sha: {sha}\n"
        f"source_path: {source_dir}\n"
        f"refreshed_on: {today}\n"
    )
    if dry_run:
        print(f"[dry-run] would write {PERL_SOURCE_MARKER}:\n{body}")
        return
    PERL_SOURCE_MARKER.write_text(body, encoding="utf-8")
    print(f"wrote {PERL_SOURCE_MARKER}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.expanduser().resolve()
    _validate_source(source_dir)

    sha = _resolve_sha(source_dir, args.commit_sha)
    branch = args.branch or "master"

    for bundle in BUNDLE_LINE_ENDINGS:
        _refresh_one_bundle(source_dir, bundle, args.dry_run)

    _write_marker(sha=sha, branch=branch, source_dir=source_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
