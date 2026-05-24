"""Tests for the RuleMonkey-oracle revalidation path in parity_diff:

  * stochastic_compare must not crash on a common file that has only a time
    column (no observables) — e.g. the time-only .cdat a network-free model
    writes. This surfaced when both compared sides are bngsim (NF vs RM), where
    that .cdat becomes a *common* file. It should pass vacuously.
  * the method nf->rm rewrite the revalidation applies.
  * the SUBPROCESS_NF_RULEMONKEY registry contract.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "parity_diff.py"
    spec = importlib.util.spec_from_file_location("parity_diff", p)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


pd = _load()

_HEADER_OBS = "#   time   Obs_A   Obs_B\n"
_HEADER_TIME_ONLY = "#   time\n"


def _write_gdat(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(header)
        for r in rows:
            f.write(" ".join(f"{v:.6e}" for v in r) + "\n")


class TestTimeOnlyFileGuard:
    def test_time_only_common_file_passes_vacuously(self, tmp_path):
        # two seed dirs per side, each with a model.gdat (real obs, identical)
        # AND a model.cdat that is time-only (the network-free case).
        sub_dirs, bng_dirs = [], []
        for side, dirs in (("sub", sub_dirs), ("bng", bng_dirs)):
            for s in (1, 2):
                d = tmp_path / side / f"seed{s}"
                _write_gdat(d / "m.gdat", _HEADER_OBS,
                            [[0.0, 10.0, 5.0], [1.0, 9.0, 6.0]])
                _write_gdat(d / "m.cdat", _HEADER_TIME_ONLY, [[0.0], [1.0]])
                dirs.append(str(d))
        # Must not raise (previously crashed on the time-only .cdat) and the
        # identical observable .gdat must pass.
        status, details = pd.stochastic_compare(sub_dirs, bng_dirs)
        assert status == "pass", details
        cdat = details["per_file"]["m.cdat"]
        assert cdat.get("pass") is True and cdat.get("n_cols") == 0

    def test_real_observable_diff_still_fails(self, tmp_path):
        sub_dirs, bng_dirs = [], []
        for s in (1, 2):
            d = tmp_path / "sub" / f"seed{s}"
            _write_gdat(d / "m.gdat", _HEADER_OBS, [[0.0, 10.0, 5.0], [1.0, 9.0, 6.0]])
            sub_dirs.append(str(d))
            d2 = tmp_path / "bng" / f"seed{s}"
            # large, zero-variance offset on Obs_A -> must DIFF
            _write_gdat(d2 / "m.gdat", _HEADER_OBS, [[0.0, 1000.0, 5.0], [1.0, 900.0, 6.0]])
            bng_dirs.append(str(d2))
        status, _ = pd.stochastic_compare(sub_dirs, bng_dirs)
        assert status == "diff"


class TestRuleMonkeyRewriteAndRegistry:
    def test_nf_to_rm_rewrite(self):
        assert pd._NF_METHOD_RE.sub('method=>"rm"',
                                    'simulate({method=>"nf",t_end=>10})') == \
            'simulate({method=>"rm",t_end=>10})'
        # single-quoted nf is matched too (rewrite normalizes to double quotes)
        assert pd._NF_METHOD_RE.subn('method=>"rm"',
                                     "simulate({method=>'nf'})")[1] == 1
        # leaves non-nf methods alone
        assert pd._NF_METHOD_RE.subn('method=>"rm"',
                                     'simulate({method=>"ssa"})')[1] == 0

    def test_registry_contract(self):
        for stem, entry in pd.SUBPROCESS_NF_RULEMONKEY.items():
            assert isinstance(stem, str) and not stem.endswith(".bngl")  # stems
            assert "reason" in entry and "issue" in entry
