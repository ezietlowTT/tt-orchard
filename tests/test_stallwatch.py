# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stall watcher: it kills on signs that nothing is happening, and a deadline is only its backstop."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from orchard.stallwatch import StallWatcher, run_watched, tree_cpu_ticks


def test_no_output_and_no_cpu_for_the_quiet_window_is_a_stall():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    assert w.observe(0.0, 100, 50) is None
    assert w.observe(9.0, 100, 50) is None
    assert "stalled" in w.observe(10.5, 100, 51)          # one tick of CPU is below min_cpu_ticks


def test_output_growth_counts_as_progress():
    w = StallWatcher(quiet_s=10)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 101, 50) is None                # a byte of output resets the window
    assert w.observe(18.0, 101, 50) is None
    assert "stalled" in w.observe(19.5, 101, 50)


def test_cpu_use_counts_as_progress_and_slow_cpu_adds_up():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 100, 52) is None                # two ticks since the mark: progress
    assert w.observe(18.0, 100, 52) is None               # the mark moved at 9 s, so 9 quiet seconds is still inside the window
    w2 = StallWatcher(quiet_s=10, min_cpu_ticks=3)
    w2.observe(0.0, 100, 50)
    w2.observe(4.0, 100, 51)
    w2.observe(8.0, 100, 52)
    assert w2.observe(9.0, 100, 53) is None               # one tick a sample still reaches 3 over the window
    assert w2.observe(15.0, 100, 53) is None              # that moved the mark to 9 s; without the CPU rule it stalls here


def test_a_repeating_line_is_reported():
    w = StallWatcher(repeat_lines=5)
    assert w.feed_lines(["compiling"] * 4) is None
    assert "repeating" in w.feed_lines(["compiling"])


def test_a_changed_line_resets_the_repeat_count():
    w = StallWatcher(repeat_lines=5)
    w.feed_lines(["a"] * 4)
    assert w.feed_lines(["b"]) is None
    assert w.feed_lines(["a"] * 4) is None


def test_tree_cpu_ticks_sums_a_process_and_its_descendants(tmp_path):
    def proc(pid, ppid, ut, st, comm="x"):
        d = tmp_path / str(pid)
        d.mkdir()
        # fields after the comm: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 0 0 0 0 0 0 {ut} {st} 0 0 20 0 1 0 0 0 0\n")
    proc(100, 1, 10, 5)
    proc(101, 100, 3, 2, comm="a (b) c")                  # a command name with spaces and parentheses
    proc(102, 101, 1, 1)
    proc(200, 1, 999, 999)                                # not a descendant
    assert tree_cpu_ticks(100, proc_root=str(tmp_path)) == 10 + 5 + 3 + 2 + 1 + 1


def test_a_silent_idle_child_is_killed_as_stalled(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    got = run_watched(["sleep", "60"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "stalled" in got.reason and got.elapsed_s < 10


def test_a_busy_silent_child_is_not_a_stall_and_the_backstop_ends_it(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    spin = [sys.executable, "-c", "while True: pass"]
    got = run_watched(spin, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=3, poll_s=0.2)
    assert got.killed and got.reason.startswith("backstop")


def test_a_child_that_keeps_printing_is_left_alone_until_it_exits(tmp_path):
    w = StallWatcher(quiet_s=1.0, repeat_lines=1000)
    cmd = [sys.executable, "-c", "import time\nfor i in range(8):\n print(i, flush=True); time.sleep(0.3)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert not got.killed and got.returncode == 0


def test_a_child_that_repeats_one_line_is_killed(tmp_path):
    w = StallWatcher(quiet_s=30, repeat_lines=10)
    cmd = [sys.executable, "-c", "import time\nwhile True:\n print('same', flush=True); time.sleep(0.05)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "repeating" in got.reason


def test_the_whole_process_group_is_killed_so_a_grandchild_does_not_survive(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    w = StallWatcher(quiet_s=1.0)
    cmd = ["sh", "-c", f"sleep 120 & echo $! > {pidfile}; wait"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed
    gone = False
    pid = int(pidfile.read_text())
    for _ in range(20):                      # the kernel needs a moment to reap; the loop checks the pid itself
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone


# ---------------------------------------------------------------------------
# Review fixes: stale log, dropped CPU ticks, cleanup on error, partial lines,
# a bounded wait after the kill, and a child that ends before the kill.
# ---------------------------------------------------------------------------
import subprocess as _sp

import orchard.stallwatch as stallwatch


class _ExitedProc:
    """A fake child that has already ended by itself, but whose wait() still times out once."""
    pid = 0
    returncode = 0

    def wait(self, timeout=None):
        raise _sp.TimeoutExpired("fake", timeout)

    def poll(self):
        return 0


class _StuckProc:
    """A fake child that never ends and ignores the kill. A wait with no timeout would block forever."""
    pid = 0
    returncode = None

    def wait(self, timeout=None):
        assert timeout is not None, "an unbounded wait on a child that will not end"
        raise _sp.TimeoutExpired("fake", timeout)

    def poll(self):
        return None


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_an_old_log_is_not_read_as_this_childs_output(tmp_path):
    log = tmp_path / "log"
    log.write_bytes(b"same\n" * 50)                       # left by an earlier attempt, appended to
    w = StallWatcher(quiet_s=30, repeat_lines=5)
    cmd = [sys.executable, "-c", "import time\nfor i in range(6):\n print('line', i, flush=True); time.sleep(0.3)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=log, watcher=w, backstop_s=30, poll_s=0.2)
    assert not got.killed and got.returncode == 0


def test_a_drop_in_tree_cpu_ticks_is_rebased_and_later_cpu_use_counts():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 500)
    assert w.observe(5.0, 100, 100) is None               # a busy descendant exited: its ticks left the total
    assert w.observe(11.0, 100, 103) is None              # 3 ticks since the rebase is progress


def test_a_drop_in_tree_cpu_ticks_does_not_reset_the_quiet_clock():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 500)
    assert w.observe(5.0, 100, 100) is None
    assert "stalled" in w.observe(10.5, 100, 100)         # no output and no new CPU since 0 s


def test_the_child_is_killed_when_the_watch_loop_raises(tmp_path):
    class Boom(StallWatcher):
        def observe(self, *a):
            raise RuntimeError("boom")
    made = []

    def popen(*a, **k):
        made.append(subprocess.Popen(*a, **k))
        return made[-1]
    try:
        with pytest.raises(RuntimeError):
            run_watched(["sleep", "60"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                        watcher=Boom(), backstop_s=30, poll_s=0.2, popen=popen)
        assert not _alive(made[0].pid)
    finally:
        for p in made:
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass


def test_a_line_split_across_two_reads_is_counted_once(tmp_path):
    w = StallWatcher(quiet_s=30, repeat_lines=3)
    code = ("import sys, time\n"
            "sys.stdout.write('same\\nsame\\nsa'); sys.stdout.flush(); time.sleep(0.8)\n"
            "sys.stdout.write('me\\n'); sys.stdout.flush(); time.sleep(5)")
    got = run_watched([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "repeating" in got.reason and got.elapsed_s < 4


def test_the_wait_after_the_kill_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(stallwatch, "_kill_group", lambda proc: None)
    monkeypatch.setattr(stallwatch, "WAIT_AFTER_KILL_S", 0.2)
    got = run_watched(["x"], cwd=tmp_path, env={}, log_path=tmp_path / "log", watcher=StallWatcher(),
                      backstop_s=0, poll_s=0.01, popen=lambda *a, **k: _StuckProc())
    assert got.killed and got.returncode is None and got.reason.startswith("backstop")


def test_a_child_that_ended_before_the_kill_is_not_reported_as_killed(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(stallwatch, "_kill_group", lambda proc: killed.append(proc))
    got = run_watched(["x"], cwd=tmp_path, env={}, log_path=tmp_path / "log", watcher=StallWatcher(),
                      backstop_s=0, poll_s=0.01, popen=lambda *a, **k: _ExitedProc())
    assert not got.killed and got.reason is None and got.returncode == 0 and not killed


def test_the_repeat_count_is_exact():
    w = StallWatcher(repeat_lines=3)
    assert w.feed_lines(["a", "b", "b"]) is None          # b has run twice
    assert "repeating" in w.feed_lines(["b"])             # the third b fires
