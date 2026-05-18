from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sweep_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bngl_models_sweep.py"
    spec = importlib.util.spec_from_file_location("bngl_models_sweep", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_simulators_expands_both_mode():
    module = _load_sweep_module()

    assert module.normalize_simulators("both") == ["subprocess", "bngsim"]
    assert module.normalize_simulators("bngsim") == ["bngsim"]


def test_run_sweep_runs_both_modes_and_separates_outputs(tmp_path, monkeypatch):
    module = _load_sweep_module()
    bngl = tmp_path / "modelset" / "example.bngl"
    bngl.parent.mkdir(parents=True)
    bngl.write_text("begin model\nend model\n", encoding="utf-8")

    calls: list[tuple[str, str]] = []

    def fake_run_one(bngl_path, simulator, out_dir, timeout, abs_tol, rel_tol):
        calls.append((simulator, str(out_dir)))
        return module.ModelResult(
            model=bngl_path.parent.name,
            bngl=bngl_path.name,
            simulator=simulator,
            wall_seconds=0.1,
            ok=True,
        )

    monkeypatch.setattr(module, "run_one", fake_run_one)

    results = module.run_sweep(
        [bngl],
        module.normalize_simulators("both"),
        tmp_path / "out",
        timeout=5,
        abs_tol=1.0,
        rel_tol=1e-2,
        tolerances={},
    )

    assert [r.simulator for r in results] == ["subprocess", "bngsim"]
    assert calls == [
        ("subprocess", str(tmp_path / "out" / "subprocess" / "modelset__example")),
        ("bngsim", str(tmp_path / "out" / "bngsim" / "modelset__example")),
    ]
