import importlib.util
import json
import shutil
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "vendor_bionetgen_assets.py"

LINUX_PATHS = (
    "BNG2.pl",
    "bin/NFsim",
    "bin/run_network",
    "bin/sundials-config",
    "Perl2/README.txt",
    "VERSION",
)

MAC_PATHS = (
    "BNG2.pl",
    "bin/NFsim",
    "bin/run_network",
    "bin/sundials-config",
    "Perl2/README.txt",
    "VERSION",
)

WINDOWS_PATHS = (
    "BNG2.pl",
    "bin/NFsim.exe",
    "bin/run_network.exe",
    "bin/sundials-config",
    "Perl2/README.txt",
    "VERSION",
    "bin/cyggcc_s-seh-1.dll",
    "bin/cygstdc++-6.dll",
    "bin/cygwin1.dll",
    "bin/cygz.dll",
    "bin/cygzstd-1.dll",
)


def _load_vendor_release():
    spec = importlib.util.spec_from_file_location("vendor_bionetgen_assets", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "vendor_release")


def _write_archive(archive_path: Path, root_name: str, relative_paths) -> None:
    staging_root = archive_path.parent / root_name
    for relative_path in relative_paths:
        target = staging_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative_path, encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(staging_root, arcname=root_name)
    shutil.rmtree(staging_root)


def test_vendor_release_reuses_local_archives(tmp_path):
    vendor_release = _load_vendor_release()

    project_root = tmp_path / "project"
    assets_dir = project_root / "bionetgen" / "assets"
    assets_dir.mkdir(parents=True)

    release_json_path = assets_dir / "ghapi.json"
    release_json_path.write_text(
        json.dumps(
            {
                "tag_name": "BioNetGen-2.9.3",
                "assets": [
                    {
                        "name": "BioNetGen-2.9.3-linux.tar.gz",
                        "browser_download_url": "https://example.invalid/BioNetGen-2.9.3-linux.tar.gz",
                    },
                    {
                        "name": "BioNetGen-2.9.3-mac.tar.gz",
                        "browser_download_url": "https://example.invalid/BioNetGen-2.9.3-mac.tar.gz",
                    },
                    {
                        "name": "BioNetGen-2.9.3-win.tar.gz",
                        "browser_download_url": "https://example.invalid/BioNetGen-2.9.3-win.tar.gz",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    _write_archive(
        archive_dir / "BioNetGen-2.9.3-linux.tar.gz",
        "BioNetGen-2.9.3-linux",
        LINUX_PATHS,
    )
    _write_archive(
        archive_dir / "BioNetGen-2.9.3-mac.tar.gz",
        "BioNetGen-2.9.3-mac",
        MAC_PATHS,
    )
    _write_archive(
        archive_dir / "BioNetGen-2.9.3-win.tar.gz",
        "BioNetGen-2.9.3-win",
        WINDOWS_PATHS,
    )

    stale_target = project_root / "bionetgen" / "bng-linux"
    stale_target.mkdir(parents=True)
    (stale_target / "stale.txt").write_text("old", encoding="utf-8")

    vendor_release(
        release_json_path=release_json_path,
        project_root=project_root,
        archive_dir=archive_dir,
        skip_download=True,
    )

    assert (assets_dir / "BNGVERSION").read_text(encoding="utf-8") == "BioNetGen-2.9.3"
    assert not (project_root / "bionetgen" / "bng-linux" / "stale.txt").exists()
    assert (project_root / "bionetgen" / "bng-linux" / "bin" / "NFsim").is_file()
    assert (project_root / "bionetgen" / "bng-mac" / "Perl2" / "README.txt").is_file()
    assert (project_root / "bionetgen" / "bng-win" / "bin" / "cygzstd-1.dll").is_file()
