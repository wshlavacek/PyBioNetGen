#!/usr/bin/env python3
"""Run the default developer checks with a provisioned BioNetGen runtime.

This launcher keeps local ``make test`` runs and CI on the same path:

- resolve a usable ``BNG2.pl`` directory for the current platform
- prefer a local editable ``bngsim`` checkout when available
- otherwise install published ``bngsim`` into the ephemeral uv env
- include the atomizer-only ``lxml`` and ``networkx`` deps so those tests run

The full model sweep remains opt-in through ``BNG_RUN_MODEL_SWEEPS=1``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_JSON = REPO_ROOT / "bionetgen" / "assets" / "ghapi.json"
DEFAULT_BNGSIM_CHECKOUT = Path.home() / "Code" / "PyBNF-Private" / "bngsim"
DEFAULT_BNG_RUNTIME = Path.home() / "Simulations" / "BioNetGen-2.9.3"
DEFAULT_CACHE_ROOT = Path(tempfile.gettempdir()) / "pybionetgen-dev-runtime"

PLATFORM_KEYS = {
    "Linux": "linux",
    "Darwin": "mac",
    "Windows": "win",
}

TARGET_DIRS = {
    "linux": "bng-linux",
    "mac": "bng-mac",
    "win": "bng-win",
}

REQUIRED_BNG_PATHS = {
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
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest and mypy in the standard uv dev-check environment.",
    )
    parser.add_argument(
        "--skip-bng-download",
        action="store_true",
        help="Fail instead of downloading the BioNetGen runtime when it is missing.",
    )
    parser.add_argument(
        "--no-bngsim",
        action="store_true",
        help="Run pytest/mypy WITHOUT bngsim. bngsim is an optional dependency "
        "and is not on public PyPI, so it can't be installed on a hosted CI "
        "runner; this skips it (bngsim-specific tests skip via conftest). Also "
        "enabled by setting PYBNG_DEV_NO_BNGSIM=1.",
    )
    args, pytest_args = parser.parse_known_args(argv)
    if os.environ.get("PYBNG_DEV_NO_BNGSIM"):
        args.no_bngsim = True
    args.pytest_args = pytest_args or ["tests/"]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run developer checks")

    bng_dir = resolve_bng_runtime(skip_download=args.skip_bng_download)
    uv_base = build_uv_command(uv, no_bngsim=args.no_bngsim)
    env = os.environ.copy()
    env["BNGPATH"] = str(bng_dir)

    print(f"Using BNGPATH={bng_dir}", flush=True)
    subprocess.run(
        [*uv_base, "python", "-m", "pytest", *args.pytest_args],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    subprocess.run(
        [*uv_base, "python", "-m", "mypy", "bionetgen", "tests"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return 0


def build_uv_command(uv_executable: str, *, no_bngsim: bool = False) -> list[str]:
    command = [
        uv_executable,
        "run",
        "--no-project",
        "--with-requirements",
        str(REPO_ROOT / "requirements-dev.txt"),
    ]
    bngsim_checkout = resolve_local_bngsim_checkout()
    if no_bngsim:
        # bngsim is optional and not on public PyPI, so it can't be installed
        # on a hosted CI runner. Run without it; conftest skips the
        # bngsim-specific tests. Local dev (with a checkout) still runs the
        # full suite by omitting --no-bngsim.
        print("Skipping bngsim (--no-bngsim); bngsim-specific tests will skip", flush=True)
    elif bngsim_checkout is not None:
        print(f"Using local editable bngsim checkout {bngsim_checkout}", flush=True)
        command.extend(["--with-editable", str(bngsim_checkout)])
    else:
        print("Using published bngsim package", flush=True)
        command.extend(["--with", "bngsim"])

    # bngsim supplies python-libsbml; add the remaining atomizer-only deps
    # so the default dev checks exercise those tests too.
    command.extend(["--with", "lxml", "--with", "networkx"])
    return command


def resolve_local_bngsim_checkout() -> Path | None:
    override = os.environ.get("PYBNG_DEV_BNGSIM_PATH")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(DEFAULT_BNGSIM_CHECKOUT)

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return None


def resolve_bng_runtime(*, skip_download: bool = False) -> Path:
    for candidate in bng_runtime_candidates():
        resolved = validate_bng_candidate(candidate)
        if resolved is not None:
            return resolved
    if skip_download:
        raise FileNotFoundError(
            "No usable BNG2.pl runtime found. "
            "Set BNGPATH/PYBNG_DEV_BNGPATH or allow runtime download."
        )
    return ensure_cached_bng_runtime()


def bng_runtime_candidates() -> list[Path]:
    candidates = []
    for env_name in ("PYBNG_DEV_BNGPATH", "BNGPATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    platform_key = current_platform_key()
    candidates.append(REPO_ROOT / "bionetgen" / TARGET_DIRS[platform_key])
    candidates.append(DEFAULT_BNG_RUNTIME)
    return candidates


def current_platform_key() -> str:
    system_name = platform.system()
    if system_name not in PLATFORM_KEYS:
        raise RuntimeError(f"Unsupported platform for developer checks: {system_name}")
    return PLATFORM_KEYS[system_name]


def validate_bng_candidate(candidate: Path) -> Path | None:
    if not candidate:
        return None
    candidate = candidate.expanduser()
    if candidate.name.lower() == "bng2.pl":
        bng_exec = candidate
        bng_dir = candidate.parent
    else:
        bng_dir = candidate
        bng_exec = candidate / "BNG2.pl"
    if not bng_exec.is_file():
        return None
    if not bng_exec_works(bng_exec):
        return None
    return bng_dir.resolve()


def bng_exec_works(bng_exec: Path) -> bool:
    perl = shutil.which("perl")
    if perl is None:
        return False
    proc = subprocess.run(
        [perl, str(bng_exec)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def ensure_cached_bng_runtime() -> Path:
    platform_key = current_platform_key()
    cache_root = Path(
        os.environ.get("PYBNG_DEV_RUNTIME_CACHE", str(DEFAULT_CACHE_ROOT))
    ).expanduser()
    install_root = cache_root / TARGET_DIRS[platform_key]
    required_paths = REQUIRED_BNG_PATHS[platform_key]

    if all((install_root / rel).exists() for rel in required_paths):
        resolved = validate_bng_candidate(install_root)
        if resolved is not None:
            print(f"Using cached BioNetGen runtime {resolved}", flush=True)
            return resolved

    archive_path = ensure_bng_archive(platform_key, cache_root / "archives")
    extract_archive_root(archive_path, install_root)
    resolved = validate_bng_candidate(install_root)
    if resolved is None:
        raise RuntimeError(f"Downloaded BioNetGen runtime is unusable: {install_root}")
    print(f"Provisioned BioNetGen runtime at {resolved}", flush=True)
    return resolved


def ensure_bng_archive(platform_key: str, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    url = release_asset_url(RELEASE_JSON, platform_key)
    filename = archive_filename(url)
    archive_path = archive_dir / filename
    if archive_path.is_file():
        print(f"Reusing cached archive {archive_path}", flush=True)
        return archive_path

    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def release_asset_url(release_json_path: Path, platform_key: str) -> str:
    release_data = json.loads(release_json_path.read_text(encoding="utf-8"))
    assets = release_data.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("Release metadata does not contain an asset list")

    for asset in assets:
        if not isinstance(asset, dict):
            continue
        url = asset.get("browser_download_url")
        name = str(asset.get("name", ""))
        if not isinstance(url, str):
            continue
        haystack = f"{name} {url}".lower()
        if platform_key in haystack:
            return url
    raise RuntimeError(f"Release metadata does not contain a {platform_key!r} archive")


def archive_filename(url: str) -> str:
    name = Path(urlsplit(url).path).name
    if not name:
        raise RuntimeError(f"Unable to determine archive filename from {url}")
    return name


def extract_archive_root(archive_path: Path, destination: Path) -> None:
    if not tarfile.is_tarfile(archive_path):
        raise RuntimeError(f"Unsupported archive format: {archive_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pybng-dev-extract-", dir=destination.parent) as tmp:
        temp_dir = Path(tmp)
        with tarfile.open(archive_path, "r:*") as archive:
            archive_root = find_archive_root(archive)
            safe_extract_all(archive, temp_dir)

        extracted_root = temp_dir / archive_root
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(extracted_root), str(destination))


def safe_extract_all(archive: tarfile.TarFile, destination: Path) -> None:
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


def find_archive_root(archive: tarfile.TarFile) -> str:
    for member in archive.getmembers():
        parts = Path(member.name).parts
        if not parts:
            continue
        root = parts[0]
        if root.startswith("._") or root == "__MACOSX":
            continue
        return root
    raise RuntimeError("Could not determine archive root directory")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
