"""Tracked-corpus parity suite: integrity checks + the gated full run.

Two layers:

* ``test_manifest_*`` — fast, dependency-free integrity checks on
  ``tests/parity/manifest.json`` and the vendored model tree. These run in
  ordinary CI (no bngsim needed): they guarantee the reproducible corpus is
  self-consistent — every record points at a real file, ids are unique,
  expected buckets and overrides are well-formed.

* ``test_fast_tier_parity`` — the real BNGsim-vs-subprocess parity run on the
  fast tier. Heavy (runs every model on both simulators) and needs the pinned
  bngsim wheel, so it is OPT-IN: set ``RUN_PARITY_SUITE=1`` to enable. It is
  skipped otherwise (and whenever bngsim isn't the pinned version), so it never
  hangs a normal pytest/CI run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PARITY = REPO / "tests" / "parity"
MANIFEST = PARITY / "manifest.json"
MODELS_ROOT = PARITY / "models"

VALID_BUCKETS = {"PASS", "PASS_REF_BUG", "KNOWN_ARTIFACT"}
VALID_REGIMES = {"deterministic", "stochastic"}
VALID_OVERRIDE_KEYS = {"t_end", "n_scan_pts", "tol", "action_inject", "timeout"}
PINNED_BNGSIM = "0.9.7"


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.is_file(), f"missing {MANIFEST} — run build_parity_corpus.py"
    return json.loads(MANIFEST.read_text())


def test_manifest_schema(manifest):
    assert manifest["schema"] == "parity-corpus/1"
    assert manifest["n_models"] == len(manifest["models"]) > 0


def test_manifest_ids_unique(manifest):
    ids = [r["id"] for r in manifest["models"]]
    assert len(ids) == len(set(ids)), "duplicate manifest ids"


def test_manifest_files_exist(manifest):
    """Every record points at a vendored file under models/ (id == relpath)."""
    missing = []
    for r in manifest["models"]:
        p = MODELS_ROOT / r["id"]
        if not p.is_file():
            missing.append(r["id"])
        assert r["file"] == ("models/" + r["id"]), r["id"]
    assert not missing, f"{len(missing)} manifest models missing on disk: {missing[:5]}"


def test_manifest_fields_valid(manifest):
    for r in manifest["models"]:
        assert r["expected"] in VALID_BUCKETS, (r["id"], r["expected"])
        assert r["regime"] in VALID_REGIMES, (r["id"], r["regime"])
        assert r["source"] and r["license"]
        if "overrides" in r:
            bad = set(r["overrides"]) - VALID_OVERRIDE_KEYS
            assert not bad, (r["id"], bad)


def test_no_orphan_vendored_models(manifest):
    """Every vendored .bngl is accounted for in the manifest (no dead files)."""
    on_disk = {p.relative_to(MODELS_ROOT).as_posix() for p in MODELS_ROOT.rglob("*.bngl")}
    in_manifest = {r["id"] for r in manifest["models"]}
    orphans = on_disk - in_manifest
    assert not orphans, f"{len(orphans)} vendored .bngl not in manifest: {sorted(orphans)[:5]}"


def _bngsim_version():
    try:
        import bngsim

        return getattr(bngsim, "__version__", None)
    except Exception:
        return None


@pytest.mark.skipif(
    not os.environ.get("RUN_PARITY_SUITE"),
    reason="set RUN_PARITY_SUITE=1 to run the heavy parity suite",
)
@pytest.mark.skipif(
    _bngsim_version() != PINNED_BNGSIM, reason=f"needs pinned bngsim {PINNED_BNGSIM}"
)
def test_fast_tier_parity(tmp_path):
    """Run the fast tier on both simulators and assert verdicts match manifest."""
    rc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "parity_validate.py"),
            "--tier",
            "fast",
            "--out",
            str(tmp_path / "out"),
            "--strict-version",
            "--workers",
            "4",
        ],
    ).returncode
    assert rc == 0, "fast-tier parity verdicts diverged from the manifest"
