"""Tests for the persistent BNGsim backend helper (serve mode).

The backend hook otherwise spawns a fresh Python process per atomic job,
paying ``import bngsim`` every time -- which dominates a parameter_scan (one
job per scan point). ``serve`` mode runs one long-lived helper for a whole
BNG2.pl run; ``BNGCLI`` starts it and advertises a Unix-domain socket, and
the hook falls back to the one-shot path if the socket is unavailable.
"""

import json
import os
import shutil
import socket
import tempfile
import threading
import time

import pytest

import bionetgen.core.tools.bngsim_backend_helper as helper


def _wait_for_socket(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.connect(path)
                return True
            except OSError:
                pass
        time.sleep(0.02)
    return False


def _request(path, message, read_reply=True):
    """Send one newline-terminated request; optionally read the reply line."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(5)
        sock.connect(path)
        sock.sendall((message + "\n").encode("utf-8"))
        if not read_reply:
            return None
        return sock.makefile("r", encoding="utf-8").readline()


@pytest.fixture
def serve_thread():
    """Run ``serve`` on a background thread; yields (socket_path, thread).

    The socket lives under a short temp dir -- AF_UNIX paths are length
    limited (~104 chars) and pytest's tmp_path is too long on macOS.
    """
    if os.name != "posix":
        # serve uses an AF_UNIX socket and BNGCLI only starts the persistent
        # helper on POSIX (it returns early on Windows, falling back to the
        # one-shot per-job path). Nothing to test off POSIX.
        pytest.skip("persistent helper (AF_UNIX serve) is POSIX-only")
    base = "/tmp" if os.path.isdir("/tmp") else None
    sock_dir = tempfile.mkdtemp(prefix="bngsh-test-", dir=base)
    sock_path = os.path.join(sock_dir, "h.sock")
    thread = threading.Thread(target=helper.serve, args=(sock_path,), daemon=True)
    thread.start()
    assert _wait_for_socket(sock_path), "serve socket never became ready"
    try:
        yield sock_path, thread
    finally:
        if thread.is_alive():
            try:
                _request(sock_path, helper.SHUTDOWN_REQUEST, read_reply=False)
            except OSError:
                pass
            thread.join(timeout=5)
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_serve_dispatches_multiple_jobs_in_one_process(serve_thread, monkeypatch):
    """One serve process handles every job -- the whole point of the mode."""
    sock_path, thread = serve_thread
    seen = []

    def fake_run(job_path):
        seen.append(job_path)
        return {"success": True, "job": job_path}

    monkeypatch.setattr(helper, "_run_job_file", fake_run)

    reply1 = _request(sock_path, "/jobs/point1.json")
    reply2 = _request(sock_path, "/jobs/point2.json")

    assert reply1.startswith("OK ")
    assert reply2.startswith("OK ")
    assert json.loads(reply1[3:])["job"] == "/jobs/point1.json"
    # Both jobs were served by the same (still-alive) process.
    assert seen == ["/jobs/point1.json", "/jobs/point2.json"]
    assert thread.is_alive()


def test_serve_reports_failed_job_as_err_without_dying(serve_thread, monkeypatch):
    """A job that returns success=False or raises is ERR, not fatal."""
    sock_path, thread = serve_thread

    def fake_run(job_path):
        if "boom" in job_path:
            raise RuntimeError("job blew up")
        return {"success": False, "error": "bngsim said no"}

    monkeypatch.setattr(helper, "_run_job_file", fake_run)

    raised = _request(sock_path, "/jobs/boom.json")
    assert raised.startswith("ERR ")
    assert "job blew up" in json.loads(raised[4:])["error"]

    # Server survived the failure and still serves the next job.
    declined = _request(sock_path, "/jobs/ok.json")
    assert declined.startswith("ERR ")
    assert json.loads(declined[4:])["error"] == "bngsim said no"
    assert thread.is_alive()


def test_serve_shutdown_request_stops_the_loop(serve_thread):
    sock_path, thread = serve_thread
    _request(sock_path, helper.SHUTDOWN_REQUEST, read_reply=False)
    thread.join(timeout=5)
    assert not thread.is_alive()
    # The socket file is cleaned up on exit.
    assert not os.path.exists(sock_path)


def test_run_job_file_executes_in_the_job_output_directory(tmp_path, monkeypatch):
    """serve dispatches through _run_job_file, which chdirs to the job's
    output dir so BNGsim writes alongside BNG2.pl's run directory (matching
    the one-shot helper, which inherited BNG2.pl's cwd)."""
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()
    job_path = tmp_path / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "artifact_path": str(tmp_path / "model.net"),
                "artifact_format": "net",
                "method": "ode",
                "simulation_options": {},
                "output_prefix": str(out_dir / "model"),
            }
        )
    )

    cwd_seen = {}

    def fake_exec(payload):
        cwd_seen["cwd"] = os.getcwd()
        return {"success": True}

    monkeypatch.setattr(helper, "execute_backend_payload", fake_exec)
    start_cwd = os.getcwd()
    status = helper._run_job_file(str(job_path))

    assert status == {"success": True}
    assert os.path.samefile(cwd_seen["cwd"], out_dir)
    # cwd is restored afterwards.
    assert os.path.samefile(os.getcwd(), start_cwd)
