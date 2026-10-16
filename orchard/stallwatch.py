# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Watch a child process for the signs that it is stuck, so a long run does not wait on a timer alone.

A timer says "this has taken long". These signs say "nothing is happening": the output has stopped growing and the
process tree has used no CPU, or one output line keeps repeating. A hard deadline stays as the last backstop, because
a process can be busy and still wrong (a spin that prints nothing).

`run_watched` samples about once a second. The sampling interval is not the decision. The decision comes from what the
child did: bytes written, CPU ticks used by it and everything it started, and the lines it printed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


class StallWatcher:
    """Decides from samples whether a process has stopped making progress. No clock of its own: the caller passes `now`."""

    def __init__(self, *, quiet_s: float = 60.0, min_cpu_ticks: int = 2, repeat_lines: int = 20):
        self.quiet_s, self.min_cpu_ticks, self.repeat_lines = quiet_s, min_cpu_ticks, repeat_lines
        self._mark_at = self._mark_bytes = self._mark_ticks = None
        self._line, self._run = None, 0

    def observe(self, now: float, out_bytes: int, cpu_ticks: int) -> str | None:
        if self._mark_at is None:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        # Progress is measured against the last time progress was seen, so slow CPU use still adds up.
        if out_bytes > self._mark_bytes or cpu_ticks - self._mark_ticks >= self.min_cpu_ticks:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        quiet = now - self._mark_at
        if quiet >= self.quiet_s:
            return f"stalled: no output and no CPU use for {quiet:.0f} s"
        return None

    def feed_lines(self, lines) -> str | None:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            self._run = self._run + 1 if line == self._line else 1
            self._line = line
            if self._run >= self.repeat_lines:
                return f"repeating: {line[:80]!r} x {self._run}"
        return None


def tree_cpu_ticks(pid: int, proc_root: str = "/proc") -> int:
    """User plus system CPU ticks of `pid` and every descendant, read from /proc. 0 when the pid is gone."""
    stats: dict[int, tuple[int, int]] = {}            # pid -> (ppid, ticks)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return 0
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, name, "stat"), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # The command name sits in parentheses and may hold spaces and parentheses, so split after the last ")".
        rest = text[text.rfind(")") + 2:].split()
        try:
            stats[int(name)] = (int(rest[1]), int(rest[11]) + int(rest[12]))   # ppid; utime + stime
        except (IndexError, ValueError):
            continue
    children: dict[int, list[int]] = {}
    for p, (pp, _) in stats.items():
        children.setdefault(pp, []).append(p)
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        if p in stats:
            total += stats[p][1]
        todo.extend(children.get(p, []))
    return total


@dataclass
class Watched:
    returncode: int | None
    reason: str | None        # why it was killed, or None when it ended by itself
    killed: bool
    elapsed_s: float


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)       # the child leads its own session, so this reaches grandchildren
    except (ProcessLookupError, PermissionError):
        pass


def run_watched(argv, *, cwd, env, log_path, watcher: StallWatcher, backstop_s: float, poll_s: float = 1.0,
                proc_root: str = "/proc", popen=subprocess.Popen, clock=time.monotonic) -> Watched:
    """Run `argv` with stdout and stderr appended to `log_path`, and stop it when the watcher says it is stuck or
    when `backstop_s` has passed. The child gets its own session so the whole group can be killed."""
    start, read_pos = clock(), 0
    with open(log_path, "ab") as log:
        proc = popen(list(argv), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                return Watched(proc.wait(timeout=poll_s), None, False, clock() - start)
            except subprocess.TimeoutExpired:
                pass
            now = clock()
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as f:
                f.seek(read_pos)
                fresh = f.read(size - read_pos).decode("utf-8", errors="replace").splitlines()
            read_pos = size
            reason = watcher.feed_lines(fresh) or watcher.observe(now, size, tree_cpu_ticks(proc.pid, proc_root))
            if reason is None and now - start >= backstop_s:
                reason = f"backstop: still running after {backstop_s:.0f} s"
            if reason is not None:
                _kill_group(proc)
                return Watched(proc.wait(), reason, True, clock() - start)
