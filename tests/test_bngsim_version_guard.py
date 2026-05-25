"""Tests for the bngsim runtime version guard in bngsim_bridge.

PyBioNetGen treats bngsim as an *optional* dependency by design, but
when bngsim is present it must be at MINIMUM_BNGSIM_VERSION or newer.
The guard sits in `bngsim_bridge` and downgrades `BNGSIM_AVAILABLE` to
False (with a descriptive reason) whenever the installed bngsim is too
old, so the existing fall-back paths (subprocess for the network path,
explicit-error for BNGsim-required formats) fire naturally.

These tests exercise the module-load logic by reimporting the bridge
with the bngsim module patched into something the version probe will
interpret as "wrong version." We don't touch the real bngsim install.
"""

import importlib
import sys
import types

import pytest

BRIDGE = "bionetgen.core.tools.bngsim_bridge"


def _reload_bridge_with_fake_bngsim(monkeypatch, fake_version, attrs=None):
    """Install a stub bngsim with __version__=fake_version and reload the
    bridge. Returns the freshly-reloaded module."""
    stub = types.ModuleType("bngsim")
    stub.__version__ = fake_version
    stub.HAS_NFSIM = True
    stub.HAS_RULEMONKEY = True
    if attrs:
        for k, v in attrs.items():
            setattr(stub, k, v)
    monkeypatch.setitem(sys.modules, "bngsim", stub)
    monkeypatch.delenv("BIONETGEN_NO_BNGSIM", raising=False)
    sys.modules.pop(BRIDGE, None)
    return importlib.import_module(BRIDGE)


def _reload_bridge_without_bngsim(monkeypatch):
    monkeypatch.setitem(sys.modules, "bngsim", None)  # forces ImportError
    sys.modules.pop(BRIDGE, None)
    return importlib.import_module(BRIDGE)


def _reload_bridge_with_env_disable(monkeypatch):
    monkeypatch.setenv("BIONETGEN_NO_BNGSIM", "1")
    sys.modules.pop(BRIDGE, None)
    return importlib.import_module(BRIDGE)


@pytest.fixture(autouse=True)
def _restore_bridge_after_test():
    """Make sure the real bngsim_bridge is restored so other tests see
    the real install state."""
    yield
    sys.modules.pop(BRIDGE, None)
    importlib.import_module(BRIDGE)


class TestVersionGuardStates:
    def test_too_old_marks_unavailable(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.3.0")
        assert not b.is_bngsim_available()
        reason = b.get_bngsim_unavailable_reason()
        assert reason is not None
        assert "0.3.0" in reason
        assert b.MINIMUM_BNGSIM_VERSION in reason

    def test_minimum_version_available(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.6.0")
        assert b.is_bngsim_available()
        assert b.get_bngsim_unavailable_reason() is None

    def test_newer_version_available(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "1.2.3")
        assert b.is_bngsim_available()

    def test_string_compare_does_not_misorder_0_10_vs_0_6(self, monkeypatch):
        """Naive string `<` says '0.10.0' < '0.6.0' (lexicographic);
        PEP 440 numeric compare says it's greater. The guard must use
        the latter. Catches a regression to plain string comparison.
        """
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.10.0")
        assert b.is_bngsim_available()

    def test_not_installed_reason_is_silent_path(self, monkeypatch):
        b = _reload_bridge_without_bngsim(monkeypatch)
        assert not b.is_bngsim_available()
        reason = b.get_bngsim_unavailable_reason()
        assert reason == "bngsim is not installed"

    def test_env_disable_marked_with_specific_reason(self, monkeypatch):
        b = _reload_bridge_with_env_disable(monkeypatch)
        assert not b.is_bngsim_available()
        assert "BIONETGEN_NO_BNGSIM" in b.get_bngsim_unavailable_reason()


class TestVersionGuardRouting:
    """The guard's effect on `classify_bngsim_route` for the two modes
    the user can pick."""

    def test_explicit_bngsim_with_too_old_errors_with_reason(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.3.0")
        decision = b.classify_bngsim_route(
            "model.bngl", "bngl", simulator="bngsim", bngl_actions=[]
        )
        assert decision.route == b.ROUTE_ERROR
        # The reason must surface in the error message so the user knows
        # whether to install vs. upgrade.
        assert "0.3.0" in decision.reason
        assert b.MINIMUM_BNGSIM_VERSION in decision.reason

    def test_auto_with_too_old_falls_back_silently(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.3.0")
        decision = b.classify_bngsim_route("model.bngl", "bngl", simulator="auto", bngl_actions=[])
        assert decision.route == b.ROUTE_SUBPROCESS

    def test_auto_with_too_old_warns_only_once(self, monkeypatch, caplog):
        import logging

        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.3.0")
        # Reset the warned flag so this test sees a fresh warning slot.
        b._VERSION_FALLBACK_WARNED = False
        with caplog.at_level(logging.WARNING, logger="bionetgen.bngsim_bridge"):
            b.classify_bngsim_route("a.bngl", "bngl", simulator="auto", bngl_actions=[])
            b.classify_bngsim_route("b.bngl", "bngl", simulator="auto", bngl_actions=[])
            b.classify_bngsim_route("c.bngl", "bngl", simulator="auto", bngl_actions=[])
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "0.3.0" in warnings[0].getMessage()

    def test_auto_when_not_installed_does_not_warn(self, monkeypatch, caplog):
        """The 'not installed' path is the documented optional contract —
        users on subprocess-only installs should not see a noisy warning."""
        import logging

        b = _reload_bridge_without_bngsim(monkeypatch)
        b._VERSION_FALLBACK_WARNED = False
        with caplog.at_level(logging.WARNING, logger="bionetgen.bngsim_bridge"):
            b.classify_bngsim_route("a.bngl", "bngl", simulator="auto", bngl_actions=[])
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []

    def test_required_format_with_too_old_errors_with_reason(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "0.3.0")
        # SBML requires bngsim — no subprocess fallback exists.
        decision = b.classify_bngsim_route("m.xml", "sbml", simulator="auto", bngl_actions=[])
        assert decision.route == b.ROUTE_ERROR
        assert "0.3.0" in decision.reason


class TestUnparseableVersionFallsOpen:
    """If bngsim ships a version string that `packaging` can't parse,
    err on the side of "available" — better to surface a downstream
    BNGsim-API error with its real cause than silently disable a
    possibly-fine install."""

    def test_unparseable_version_treated_as_available(self, monkeypatch):
        b = _reload_bridge_with_fake_bngsim(monkeypatch, "not-a-version")
        assert b.is_bngsim_available()
