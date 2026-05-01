#!/usr/bin/env python3
"""Vendor platform-specific BioNetGen bundles into the source tree.

This is an explicit release-maintenance step. It replaces the historical
``setup.py`` side effects that downloaded and unpacked BioNetGen assets during
installation and wheel building.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "bionetgen" / "assets"
DEFAULT_RELEASE_JSON = ASSETS_DIR / "ghapi.json"

TARGET_DIRS = {
    "linux": "bng-linux",
    "mac": "bng-mac",
    "win": "bng-win",
}

REQUIRED_PATHS = {
    "linux": (
        "BNG2.pl",
        "bin/NFsim",
        "bin/run_network",
        "bin/sundials-config",
        "Perl2",
        "VERSION",
    ),
    "mac": (
        "BNG2.pl",
        "bin/NFsim",
        "bin/run_network",
        "bin/sundials-config",
        "Perl2",
        "VERSION",
    ),
    "win": (
        "BNG2.pl",
        "bin/NFsim.exe",
        "bin/run_network.exe",
        "bin/sundials-config",
        "Perl2",
        "VERSION",
        "bin/cyggcc_s-seh-1.dll",
        "bin/cygstdc++-6.dll",
        "bin/cygwin1.dll",
        "bin/cygz.dll",
        "bin/cygzstd-1.dll",
    ),
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or reuse BioNetGen release archives from the tracked release "
            "metadata and vendor the required runtime files into bionetgen/bng-*."
        )
    )
    parser.add_argument(
        "--release-json",
        type=Path,
        default=DEFAULT_RELEASE_JSON,
        help="Path to the cached GitHub release JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root that contains the bionetgen package (default: %(default)s)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing pre-downloaded release archives. Matching "
            "archive filenames are reused from here before any download is attempted."
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to store downloaded archives. If omitted, a temporary "
            "directory is used."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Fail instead of downloading when a required archive is not already available.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_release_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_release_asset_urls(release_json: Mapping[str, object]) -> Dict[str, str]:
    assets = release_json.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("Release metadata does not contain an assets list")

    urls: Dict[str, str] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        browser_download_url = asset.get("browser_download_url")
        name = str(asset.get("name", ""))
        if not isinstance(browser_download_url, str):
            continue
        haystack = f"{name} {browser_download_url}".lower()
        if "linux" in haystack:
            urls["linux"] = browser_download_url
        elif "mac" in haystack:
            urls["mac"] = browser_download_url
        elif "win" in haystack:
            urls["win"] = browser_download_url

    missing = sorted(set(TARGET_DIRS) - set(urls))
    if missing:
        raise RuntimeError(f"Release metadata is missing assets for: {', '.join(missing)}")
    return urls


def archive_filename(url: str) -> str:
    name = Path(urlsplit(url).path).name
    if not name:
        raise RuntimeError(f"Unable to determine archive filename from URL: {url}")
    return name


def _copy_required_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _safe_extract_all(archive: tarfile.TarFile, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    for member in archive.getmembers():
        resolved_member = (destination / member.name).resolve()
        try:
            resolved_member.relative_to(resolved_destination)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to extract unsafe archive member: {member.name}") from exc
    try:
        archive.extractall(destination, filter="data")
    except TypeError:
        archive.extractall(destination)


def _find_archive_root(archive: tarfile.TarFile) -> str:
    for member in archive.getmembers():
        parts = Path(member.name).parts
        if not parts:
            continue
        first = parts[0]
        if first.startswith("._") or first == "__MACOSX":
            continue
        return first
    raise RuntimeError("Could not determine archive root directory")


def _ensure_archive(
    url: str,
    archive_dir: Optional[Path],
    download_dir: Path,
    skip_download: bool,
) -> Path:
    filename = archive_filename(url)

    if archive_dir is not None:
        cached = archive_dir / filename
        if cached.is_file():
            print(f"Reusing {cached}")
            return cached

    cached_download = download_dir / filename
    if cached_download.is_file():
        print(f"Reusing {cached_download}")
        return cached_download

    if skip_download:
        raise FileNotFoundError(
            f"Required archive {filename} was not found and --skip-download was requested"
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, cached_download)
    return cached_download


def vendor_archive(platform_name: str, archive_path: Path, project_root: Path) -> Path:
    target_dir = project_root / "bionetgen" / TARGET_DIRS[platform_name]
    required_paths = REQUIRED_PATHS[platform_name]

    if not tarfile.is_tarfile(archive_path):
        raise RuntimeError(f"Unsupported archive format for {archive_path}")

    with tempfile.TemporaryDirectory(prefix=f"pybng-vendor-{platform_name}-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        with tarfile.open(archive_path, "r:*") as archive:
            archive_root = _find_archive_root(archive)
            _safe_extract_all(archive, temp_dir)

        source_root = temp_dir / archive_root
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for relative_path in required_paths:
            source_path = source_root / relative_path
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Archive {archive_path} is missing required path {relative_path}"
                )
            destination_path = target_dir / relative_path
            _copy_required_path(source_path, destination_path)

    print(f"Vendored {platform_name} bundle into {target_dir}")
    return target_dir


def vendor_release(
    release_json_path: Path,
    project_root: Path,
    archive_dir: Optional[Path] = None,
    download_dir: Optional[Path] = None,
    skip_download: bool = False,
) -> Dict[str, Path]:
    project_root = project_root.resolve()
    release_json_path = release_json_path.resolve()
    release_json = load_release_json(release_json_path)
    release_assets = get_release_asset_urls(release_json)

    assets_dir = project_root / "bionetgen" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tag_name = str(release_json.get("tag_name", "UNKNOWN"))
    (assets_dir / "BNGVERSION").write_text(tag_name, encoding="utf-8")
    print(f"Wrote {assets_dir / 'BNGVERSION'} -> {tag_name}")

    if archive_dir is not None:
        archive_dir = archive_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="pybng-release-download-") as temp_download_dir_name:
        effective_download_dir = (
            download_dir.resolve()
            if download_dir is not None
            else Path(temp_download_dir_name)
        )
        targets: Dict[str, Path] = {}
        for platform_name, url in release_assets.items():
            archive_path = _ensure_archive(
                url=url,
                archive_dir=archive_dir,
                download_dir=effective_download_dir,
                skip_download=skip_download,
            )
            targets[platform_name] = vendor_archive(platform_name, archive_path, project_root)
        return targets


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    vendor_release(
        release_json_path=args.release_json,
        project_root=args.project_root,
        archive_dir=args.archive_dir,
        download_dir=args.download_dir,
        skip_download=args.skip_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
