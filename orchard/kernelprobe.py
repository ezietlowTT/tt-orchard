# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard kernel-probe`: can the local coder write a tt-lang kernel? (spec section 5, Part B)

An attempt is one fresh agent conversation in `DIR/<op>/attempt-<n>/`. The directory holds copies of the kernel-loop
templates, the op's spec as `kernel_spec.json`, `check_config.json` (the simulator and the tt-lang facts) and
`reference/eltwise_add.py`, copied from the tt-lang checkout (it is not vendored into this repository). The agent
writes `kernel.py` and runs `python3 kernel_check.py` as often as it likes.

The verdict never comes from a file the agent can write. The agent's shell can change any file in its directory,
including its copy of the check and the spec. After the agent ends, `authoritative_check` copies kernel.py alone into
a fresh directory and runs the SHIPPED kernel_check.py (the template under orchard/skills/kernel-loop-templates/) on it, with the
SHIPPED spec, under the stall watcher. When the agent's own check.json says it passed and the authoritative check
says it did not, or its hashes do not match the shipped files, the attempt records `agent_check_untrusted: true`.

The hardware phase runs only on an attempt whose authoritative simulator check passed, and only after a smoke test
(upstream's example on one chip) passed. Every hardware command is a `gozer run --chips 1 --exact 1`; the probe never
resets a chip. A hardware run that the stall watcher kills is a hang: the probe stops with ProbeBlocked naming the
attempt. Before any gozer call the hardware path applies the kernel check's static gate again.

The probe keeps its own records outside the attempt directory, in `DIR/.verdict/<op>-<n>/`: the clean copy of
kernel.py, the authoritative check's result and output, the hardware result and output, and the probe's attempt.json.
A run's report is built from the records run_attempt returned in that run (and the hardware phase added to). Each run
gets a fresh probe id. load_attempts reads the probe-owned records back for a later resume and is not used to build a
report. kernel.py is copied only when it is a regular file (not a link, a
directory or a device) of at most MAX_KERNEL_BYTES; anything else ends the attempt with no check and no hardware run.
The sha256 of the copied bytes is recorded, and the hardware phase refuses a copy whose hash has changed. A
convenience copy of attempt.json is written into the attempt directory after whatever was at that path is removed,
so the probe never writes through a link the agent left.

These measures stop shortcuts and accidents: a model that edits its check, plants a link, or leaves a file in the
way. They do not stop an agent determined to attack the operator's account. Agent shells run as the operator. A
process the agent starts in the background carries ORCHARD_RUN_DIR=<attempt directory>; when the agent step ends
(also after a crash or an interrupt) and before any verdict, `sweep_strays` stops every process of the operator's
that carries that exact value, and the attempt record lists them under `swept`. A process whose /proc environ no
longer holds the value (an exec with a new environment, or a program that overwrote that memory) escapes the sweep
and can write anywhere the operator can, including the `.verdict` directory. The static gate and this boundary are
best effort.

The budget (BUDGET) and the watchdog's detectors bound each attempt. Attempts are independent draws, and the report
gives counts with no rates.

`main` is the command `tt-orchard kernel-probe`. It starts no model server: the coder must already answer at its
endpoint (starting it is `tt-orchard bringup`'s job), and the simulator environment must pass
`tt-orchard kernel-env --check`. Exit 0 means a report was written, whatever the attempts' results; 2 is a refusal
before anything ran; 3 means an attempt or the hardware phase had a crash; 4 means an interrupt (Ctrl-C or SIGTERM)
stopped the run; 5 means a hardware run hung. With 3, 4 and 5 the report covers what ran before the stop.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import stat
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from orchard import defaults, ttlang_env
from orchard.agent import AgentStep, Tools, agent_env, probe_model
from orchard.canary import post_json
from orchard.ledger import Ledger
from orchard.stallwatch import StallWatcher, Watched, run_watched
from orchard.watchdog import (IdenticalResponses, Ladder, NoFileWritten, RepeatedToolCall, RetryGuard,
                              ThinkingWithoutAction, TurnRepeat, Watchdog)

__all__ = ["Watched"]       # re-exported: tests and callers build fake run_watched results with it

TEMPLATES = Path(__file__).resolve().parent / "skills" / "kernel-loop-templates"
SKILL = Path(__file__).resolve().parent / "skills" / "kernel-probe.md"
OPS = defaults.KERNEL_PROBE_OPS
# Provisional: none of these has been measured (spec section 5.3).
BUDGET = {"max_turns": 40, "wall_s": 1800.0, "sim_timeout_s": 300.0, "hw_backstop_s": 600.0}
GOZER_WHO = "claude:orchard-kernel-probe"
# How long a `gozer run` gets after the watcher's SIGTERM before the SIGKILL. gozer forwards the SIGTERM to its child,
# waits for it, and then releases its lease in a `finally`; the release resets the chip, measured at 41.7 s on this
# box, and gozer also scans /proc and revalidates the lease. A grace shorter than that kills gozer during the reset and
# leaves the lease under a dead pid. The watcher stops waiting as soon as the group is gone, so the normal path does
# not wait out the whole grace.
GOZER_TERM_GRACE_S = 120.0
# The stall watcher's limits for the smoke test and each hardware run. Not measured: the report gives the longest
# quiet stretch and the longest repeat the watcher saw on each run, so these can be set from a measurement.
HW_QUIET_S = 60.0
HW_REPEAT_LINES = 20
AGENT = "kernel-probe"
# One shell command of the agent may run this long: one simulator check plus a margin. The wall budget is checked
# between turns, so an attempt can overrun it by up to this much.
AGENT_TOOL_TIMEOUT_S = BUDGET["sim_timeout_s"] + 120
# The probe's own directory, beside the ops' attempt directories: DIR/.verdict/<op>-<n>/.
VERDICT_DIR = ".verdict"
# Inside a verdict directory, the fresh directory where the authoritative check runs. It holds the copy of kernel.py
# and the files the check writes. The hardware phase runs this same copy.
CLEAN_DIR = "authoritative"
MAX_KERNEL_BYTES = 256 * 1024     # choice: the fixtures are under 4 KiB; a kernel this large is not a kernel
TEMPLATE_FILES = ("kernel_check.py", "kernel_harness.py", "references.py")
# Every category an attempt can end in. The check and the harness produce all but `hang`, which the probe sets when
# the stall watcher kills a hardware run.
CATEGORIES = ("no_kernel", "import_error", "tt_lang_error", "shape_mismatch", "numeric_mismatch", "timeout",
              "compile_error", "hang", "hw_numeric_mismatch", "harness_error", "forbidden_code")
SMOKE_TAIL_LINES = 20


class ProbeBlocked(Exception):
    """A hardware run hung. The probe stops and names the attempt; nothing is reset by hand.

    `hw` is the hardware block for the report header when the smoke test hung (None for a kernel hang, which is
    recorded in that attempt's own record)."""

    def __init__(self, message: str, *, hw: dict | None = None):
        super().__init__(message)
        self.hw = hw


def load_spec(op: str) -> dict:
    """The shipped spec of `op`. An op not in OPS is refused, so a name cannot reach outside the ops directory."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; the ops are {', '.join(OPS)}")
    return json.loads((TEMPLATES / "ops" / f"{op}.json").read_text(encoding="utf-8"))


def _template(name: str):
    """A template script loaded by path, for its functions (classify, the static gate, the bars)."""
    spec = importlib.util.spec_from_file_location(f"kernel_loop_{name}", TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def tt_lang_facts(env) -> dict:
    """What a report says about the tt-lang the kernels ran on, including whether the wheel carries the local patch."""
    target = ttlang_env.diagnostics_target(env)
    patched = bool(target and target.is_file() and ttlang_env.sha256_file(target) == defaults.TTLANG_DIAGNOSTICS_SHA256)
    return {"sim_version": ttlang_env.sim_version(env), "source_sha": ttlang_env.head_sha(env),
            "diagnostics_patched": patched, "patch_sha256": defaults.TTLANG_DIAGNOSTICS_SHA256}


def prepare_attempt(adir: Path, *, op: str, env, tt_lang: dict | None = None) -> None:
    """Fill an EMPTY attempt directory with what the agent reads and runs. `tt_lang` defaults to tt_lang_facts(env).

    Whatever is at the attempt's path is removed first, without following a link (a real directory is removed with its
    contents, a link or a file is unlinked), so nothing an earlier run left there (a kernel.py, a check.json) can be
    picked up."""
    load_spec(op)                                       # refuses an unknown op before anything is written
    adir = Path(adir)
    _real_dir(adir.parent)
    _remove(adir)
    adir.mkdir()
    for name in TEMPLATE_FILES:
        shutil.copy2(TEMPLATES / name, adir / name)
    shutil.copy2(TEMPLATES / "ops" / f"{op}.json", adir / "kernel_spec.json")
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    (adir / "check_config.json").write_text(json.dumps({"sim": str(env.sim), "timeout_s": BUDGET["sim_timeout_s"],
                                                        "tt_lang": facts}, indent=2) + "\n", encoding="utf-8")
    ref = Path(env.src_dir) / "examples" / "eltwise_add.py"
    if ref.is_file():
        (adir / "reference").mkdir(exist_ok=True)
        shutil.copy2(ref, adir / "reference" / "eltwise_add.py")


def attempt_tools(adir: Path) -> Tools:
    """The agent's tools. The attempt directory is both the run and the stage directory, so the agent reads and writes
    only there. The environment is agent_env's: the no-chip device mask, no credentials, HOME inside the attempt. Every
    shell command goes through the runner's rules (no pip, no gozer writes, no resets, no uploads). A shell command may
    run as long as one simulator check plus a margin."""
    adir = Path(adir)
    return Tools(adir, adir, agent_env(adir), timeout=AGENT_TOOL_TIMEOUT_S)


class ProbeControl:
    """The supervisor's actuator reduced to what a probe attempt needs: the three things the watchdog's Ladder calls
    (nudge, escalate, pause) and the two things AgentStep asks (take_nudges, stop_reason). The watchdog's findings are
    the main stop; the wall clock is the backstop.

    stop_reason() is short, because it becomes the attempt's status: "escalate", "pause" or "wall-clock".
    `detail` says why in words."""

    def __init__(self, *, wall_s: float, clock=time.monotonic):
        self.deadline, self.clock = clock() + wall_s, clock
        self.nudges: list[str] = []
        self.stop: str | None = None
        self.detail: str | None = None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop, self.detail = "escalate", "the watchdog saw the agent loop again after a nudge"

    def pause(self, reason, evidence):
        self.stop, self.detail = "pause", reason

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        if self.stop:
            return self.stop
        if self.clock() >= self.deadline:
            self.detail = self.detail or "the attempt's wall-clock budget is spent"
            return "wall-clock"
        return None


def verdict_dir(probe_dir, op: str, n) -> Path:
    """The probe-owned directory of attempt `n` of `op`: DIR/.verdict/<op>-<n>/."""
    return Path(probe_dir) / VERDICT_DIR / f"{op}-{n}"


def _verdict_dir_of(adir) -> Path:
    """The verdict directory of an attempt directory DIR/<op>/attempt-<n>."""
    adir = Path(adir)
    return verdict_dir(adir.parent.parent, adir.parent.name, adir.name.removeprefix("attempt-"))


def _remove(path: Path) -> None:
    """Remove whatever is at `path` without following a link: a link or a file is unlinked, a real directory is
    removed with its contents (shutil.rmtree unlinks links inside it and does not follow them)."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _real_dir(path: Path) -> None:
    """Make `path` a real directory: anything else at that path (a link, a file) is removed first."""
    try:
        st = os.lstat(path)
        if not stat.S_ISDIR(st.st_mode):
            _remove(path)
    except FileNotFoundError:
        pass
    path.mkdir(parents=True, exist_ok=True)


def _fresh_dir(path: Path) -> None:
    _remove(path)
    path.mkdir(parents=True)


def _write_new(path: Path, data: bytes) -> None:
    """Write a new file at `path`. Whatever was there is removed first, and the open refuses to follow a link or to
    reuse a file that appeared in between (O_CREAT | O_EXCL | O_NOFOLLOW)."""
    _remove(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _read_regular(path: Path, limit: int) -> bytes | None:
    """The bytes of `path` if it is a regular file (checked on the open file, which is opened without following a link
    and without blocking on a FIFO), at most `limit` + 1 of them. None otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None
        return f.read(limit + 1)


def copy_kernel(adir: Path, clean: Path) -> tuple[str | None, str | None, str]:
    """Copy the agent's kernel.py into `clean`. (category or None, sha256 of the copied bytes, first line).

    Only a regular file of at most MAX_KERNEL_BYTES is copied. A missing file is `no_kernel`. A link, a directory, a
    device or a larger file is `forbidden_code`, and nothing of its content is read into any record."""
    src = Path(adir) / "kernel.py"
    try:
        st = os.lstat(src)
    except FileNotFoundError:
        return "no_kernel", None, "kernel.py does not exist"
    if not stat.S_ISREG(st.st_mode):
        return "forbidden_code", None, "kernel.py is not a regular file (it is a link, a directory or a device)"
    if st.st_size > MAX_KERNEL_BYTES:
        return "forbidden_code", None, f"kernel.py is larger than 256 KiB ({st.st_size} bytes)"
    data = _read_regular(src, MAX_KERNEL_BYTES)
    if data is None:
        return "forbidden_code", None, "kernel.py is not a regular file (it changed while it was opened)"
    if len(data) > MAX_KERNEL_BYTES:
        return "forbidden_code", None, "kernel.py is larger than 256 KiB"
    _write_new(Path(clean) / "kernel.py", data)
    return None, hashlib.sha256(data).hexdigest(), ""


FILE_RULE_REMEDY = "write kernel.py with write_file as a plain file of at most 256 KiB"


def authoritative_check(adir: Path, *, op: str, env, run_watched=run_watched, tt_lang: dict | None = None,
                        vdir: Path | None = None) -> dict:
    """Run the SHIPPED check on a copy of the agent's kernel.py and return its result (check.json's fields plus
    `kernel_sha256`, the sha256 of the copied bytes).

    Everything happens in the probe-owned verdict directory, made fresh here: the clean copy of kernel.py
    (CLEAN_DIR), `check-authoritative.json` and `authoritative-output.txt`. Nothing is read back from the attempt
    directory except kernel.py, and that only when copy_kernel accepts it; otherwise no check runs. Every input the
    check would read from check_config.json is passed on the command line. A run the stall watcher killed is
    `timeout`, whatever file it left."""
    vdir = Path(vdir) if vdir is not None else _verdict_dir_of(adir)
    _real_dir(vdir.parent)
    _fresh_dir(vdir)
    clean = vdir / CLEAN_DIR
    clean.mkdir()
    category, sha, line = copy_kernel(adir, clean)
    if category is not None:
        rule = [{"rule": "file", "line": 0, "detail": line, "remedy": FILE_RULE_REMEDY}] \
            if category == "forbidden_code" else []
        return {"passed": False, "category": category, "first_line": line, "sets": [], "violations": rule,
                "kernel_sha256": None}
    out = vdir / "check-authoritative.json"
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    argv = [sys.executable, str(TEMPLATES / "kernel_check.py"), "--attempt-dir", str(clean),
            "--spec", str(TEMPLATES / "ops" / f"{op}.json"), "--sim", str(env.sim),
            "--timeout-s", str(BUDGET["sim_timeout_s"]), "--tt-lang-json", json.dumps(facts),
            "--out", str(out)]
    # env=None: the child inherits the probe's own environment. The shipped check needs PATH; it is not an agent shell.
    got = run_watched(argv, cwd=clean, env=None, log_path=vdir / "authoritative-output.txt",
                      watcher=StallWatcher(quiet_s=BUDGET["sim_timeout_s"] / 2),
                      backstop_s=BUDGET["sim_timeout_s"] + 120)
    if got.killed:
        return {"passed": False, "category": "timeout", "first_line": got.reason, "sets": [], "violations": [],
                "kernel_sha256": sha}
    raw = _read_regular(out, 64 * 1024 * 1024)
    if raw is None:
        return {"passed": False, "category": "harness_error", "sets": [], "violations": [], "kernel_sha256": sha,
                "first_line": f"kernel_check.py wrote no result (exit {got.returncode})"}
    try:
        result = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        result = None
    if not isinstance(result, dict):
        return {"passed": False, "category": "harness_error", "sets": [], "violations": [], "kernel_sha256": sha,
                "first_line": "kernel_check.py wrote a result that is not a JSON object"}
    result["kernel_sha256"] = sha
    return result


def set_measures(sets: list) -> dict:
    """The worst value of each measure over the input sets: the smallest PCC, the range of slopes, the largest relative
    L2 error. A null value (the harness could not compute it) is left out; a measure with no values is null."""
    def values(key):
        return [s[key] for s in sets if isinstance(s, dict) and isinstance(s.get(key), (int, float))
                and not isinstance(s.get(key), bool)]
    pcc, slope, rel = values("pcc"), values("slope"), values("rel_l2")
    return {"pcc_min": min(pcc) if pcc else None, "slope_range": [min(slope), max(slope)] if slope else None,
            "rel_l2_max": max(rel) if rel else None}


def provenance(*, kernel_sha256, tt_lang, coder_model, probe_id, simulator_pcc_min, hardware_pcc_min=None,
               sampling: dict | None = None) -> dict:
    """The provenance block of a probe attempt (spec section 6), with the sampling the coder was asked for."""
    return {"origin": "kernel", "kernel_sha256": kernel_sha256, "tt_lang": tt_lang, "coder_model": coder_model,
            "probe_id": probe_id, "simulator_pcc_min": simulator_pcc_min, "hardware_pcc_min": hardware_pcc_min,
            "sampling": sampling}


def _shipped_hashes(op: str) -> dict:
    """The sha256 values a check.json written by the shipped, unedited check records for its own files."""
    out = {name: _sha256(TEMPLATES / name) for name in TEMPLATE_FILES}
    out["spec"] = _sha256(TEMPLATES / "ops" / f"{op}.json")
    return out


def agent_check_untrusted(agent_check, authoritative: dict, op: str) -> bool:
    """Whether the agent's own check.json disagrees with the shipped check. `agent_check` is the parsed file, None when
    the agent left none (never untrusted), or "unreadable" when the file exists and is not a JSON object.

    Untrusted means: it says passed and the authoritative check does not, or any of its hashes of the check, the
    harness, the references or the spec differs from the shipped file (a missing hash counts as different: the
    shipped check always writes them)."""
    if agent_check is None:
        return False
    if not isinstance(agent_check, dict):
        return True
    if agent_check.get("passed") and not authoritative.get("passed"):
        return True
    recorded = agent_check.get("sha256") if isinstance(agent_check.get("sha256"), dict) else {}
    return any(recorded.get(name) != want for name, want in _shipped_hashes(op).items())


AGENT_CHECK_MAX_BYTES = 1024 * 1024     # choice: a real check.json is a few KiB


def _read_agent_check(path: Path):
    """The agent's own check.json: the parsed object, None when there is none, or "unreadable" when it is not a regular
    file (a link, a FIFO, a device), is larger than AGENT_CHECK_MAX_BYTES, or is not a JSON object. It is read through
    _read_regular, so a FIFO cannot block the probe and a link to /dev/zero is never read."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(st.st_mode):
        return "unreadable"
    raw = _read_regular(path, AGENT_CHECK_MAX_BYTES)
    if raw is None or len(raw) > AGENT_CHECK_MAX_BYTES:
        return "unreadable"
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return "unreadable"
    return data if isinstance(data, dict) else "unreadable"


def _write_json(path: Path, data) -> None:
    """Write a probe-owned JSON file (never through a link)."""
    _write_new(Path(path), (json.dumps(data, indent=2) + "\n").encode("utf-8"))


def _save_attempt(adir: Path, attempt: dict) -> None:
    """Write the attempt record. The verdict lives in the probe-owned directory; the copy in the attempt directory is a
    convenience for a person reading it, written after whatever was at that path is removed, and is never read back."""
    _write_json(_verdict_dir_of(adir) / "attempt.json", attempt)
    try:
        _write_json(Path(adir) / "attempt.json", attempt)
    except OSError:
        pass                                            # the convenience copy is optional


RECORD_MAX_BYTES = 64 * 1024 * 1024
ATTEMPT_NUMBER = re.compile(r"[1-9][0-9]*")         # a positive whole number in ASCII digits, no leading zero


def _load_record(entry: Path, ops, probe_id) -> dict | None:
    """One `.verdict/<op>-<n>/attempt.json`, or None when it does not belong to this probe (see load_attempts)."""
    op, sep, number = entry.name.rpartition("-")
    # ASCII digits only: str.isdigit also accepts superscripts and other scripts' digits ("²", "٥").
    if not sep or op not in ops or not ATTEMPT_NUMBER.fullmatch(number):
        return None
    try:
        if not stat.S_ISDIR(os.lstat(entry).st_mode) or not stat.S_ISREG(os.lstat(entry / "attempt.json").st_mode):
            return None
    except OSError:
        return None
    raw = _read_regular(entry / "attempt.json", RECORD_MAX_BYTES)
    if raw is None or len(raw) > RECORD_MAX_BYTES:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("op") != op:
        return None
    n = data.get("attempt")
    if not isinstance(n, int) or isinstance(n, bool) or n != int(number):
        return None
    prov = data.get("provenance")
    if not isinstance(prov, dict) or prov.get("probe_id") != probe_id:
        return None
    return data


def load_attempts(probe_dir, *, ops, probe_id) -> tuple[list[dict], int]:
    """(this probe's attempt records, the number of other entries ignored), for resuming a probe. A run's own report is
    built from the records run_attempt returned in that run.

    Only `.verdict/<op>-<n>/attempt.json` counts, where <op> is one of `ops`, <n> is a positive whole number without a
    leading zero, the directory and the file are real (no links), the file is a regular file of at most
    RECORD_MAX_BYTES, its `op` is <op>, its `attempt` is the int <n>, and its provenance names `probe_id`. Every other
    entry under `.verdict` (an earlier run's records, a planted one, a malformed one) is skipped and counted."""
    root = Path(probe_dir) / VERDICT_DIR
    try:
        if not stat.S_ISDIR(os.lstat(root).st_mode):
            return [], 0
        entries = sorted(root.iterdir())
    except OSError:
        return [], 0
    out, ignored = [], 0
    for entry in entries:
        rec = _load_record(entry, ops, probe_id)
        if rec is None:
            ignored += 1
        else:
            out.append(rec)
    order = {op: i for i, op in enumerate(ops)}
    return sorted(out, key=lambda a: (order[a["op"]], a["attempt"])), ignored


# ---- stray agent processes --------------------------------------------------------------------------------------
#
# An agent's shell command can start a process in the background and exit. spawn_checked stops a shell's session only
# on a timeout, so such a process would keep running as the operator through the verdict and the hardware phase. Every
# agent process carries ORCHARD_RUN_DIR=<attempt directory> (agent_env), so the probe finds them by that value and stops
# them right after the agent step ends, before any verdict.

SWEEP_GRACE_S = 2.0          # how long the strays get after SIGTERM before the SIGKILL
SWEEP_POLL_S = 0.05


def _run_dir_entry(run_dir) -> bytes:
    return b"ORCHARD_RUN_DIR=" + os.fsencode(str(Path(run_dir).resolve()))


def _environ_has(proc_root: str, pid: int, entry: bytes) -> bool:
    """Whether /proc/<pid>/environ holds exactly `entry` as one of its variables. False for a process that has ended,
    a zombie (its environ reads empty) or one whose environ cannot be read."""
    try:
        with open(os.path.join(proc_root, str(pid), "environ"), "rb") as f:
            return entry in f.read().split(b"\0")
    except OSError:
        return False


def scan_strays(run_dir, *, proc_root: str = "/proc", uid: int | None = None) -> list[int]:
    """The pids of the processes owned by `uid` (default: ours) whose environment has ORCHARD_RUN_DIR equal to the
    resolved `run_dir`. The comparison is exact, so attempt-1 does not match attempt-10. An entry that cannot be read
    (the process ended during the scan, or it belongs to someone else) is skipped."""
    uid = os.getuid() if uid is None else uid
    entry = _run_dir_entry(run_dir)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    found = []
    for name in names:
        if not name.isdigit():
            continue
        try:
            if os.stat(os.path.join(proc_root, name)).st_uid != uid:
                continue
        except OSError:
            continue
        if _environ_has(proc_root, int(name), entry):
            found.append(int(name))
    return sorted(found)


def _protected_pids(proc_root: str = "/proc") -> set[int]:
    """Our own pid, our parent's, every further ancestor that /proc shows, and 0 and 1. None of them is signalled."""
    out = {0, 1, os.getpid(), os.getppid()}
    pid = os.getppid()
    for _ in range(4096):                                # a bound in case /proc shows a loop
        try:
            with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8", errors="replace") as f:
                text = f.read()
            pid = int(text[text.rfind(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid in out:
            break
        out.add(pid)
    return out


def _cmdline(proc_root: str, pid: int) -> str:
    try:
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
            return f.read(4096).replace(b"\0", b" ").decode("utf-8", errors="replace").strip()[:120]
    except OSError:
        return ""


def _proc_state(proc_root: str, pid: int) -> tuple[str, int] | None:
    """(state letter, start time in clock ticks) of a process from /proc/<pid>/stat, or None when it does not exist."""
    try:
        with open(os.path.join(proc_root, str(pid), "stat"), encoding="utf-8", errors="replace") as f:
            text = f.read()
        rest = text[text.rfind(")") + 2:].split()
        return rest[0], int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def _gone(proc_root: str, pid: int, start: int | None) -> bool:
    """True when the process no longer runs: no /proc entry, a zombie, or a different process under the same pid
    (another start time)."""
    st = _proc_state(proc_root, pid)
    if st is None or st[0] == "Z":
        return True
    return start is not None and st[1] != start


def sweep_strays(run_dir, *, proc_root: str = "/proc", kill=os.kill, killpg=os.killpg, grace_s: float = SWEEP_GRACE_S,
                 clock=time.monotonic, sleep=time.sleep) -> list[dict]:
    """Stop every process that carries this attempt's ORCHARD_RUN_DIR: SIGTERM, up to `grace_s` for them to go, then a
    second scan and SIGKILL for everything it finds, so a process a stray started during the grace is caught too.
    Returns [{"pid", "cmdline", "stopped"}]. `stopped` is True only when the process is gone afterwards (no /proc
    entry, a zombie, or another process under the pid); a process that still runs, for example under a new program
    without the variable, is recorded with `stopped` False. Never raises an Exception. An interrupt at any point after
    the first scan (during the SIGTERM pass or the grace) still gets the SIGKILL pass before it goes on.

    Our own pid and our ancestors are never signalled. Each signal goes to the pid itself (os.kill), except for a
    process that leads its own group, whose group is signalled so its children go too; a group is never signalled when
    its id is 1 or less or is our own group (the stallwatch guards). Before each signal the process's environment is
    read again, so a pid that ended and was reused by another process in between is not signalled.

    What it cannot see: a process whose /proc/<pid>/environ no longer holds the value (an exec with a new environment,
    or a program that overwrote that memory; clearenv() alone does not change it), and a process whose /proc entry
    cannot be read (a non-dumpable process, for example one that ran a setuid program, has a root-owned entry)."""
    records: list[dict] = []
    starts: dict[int, int | None] = {}
    try:
        protected = _protected_pids(proc_root)
        entry = _run_dir_entry(run_dir)

        def found() -> list[int]:
            return [pid for pid in scan_strays(run_dir, proc_root=proc_root)
                    if isinstance(pid, int) and pid > 1 and pid not in protected]

        def note(pids) -> None:
            for pid in pids:
                if pid not in starts:
                    st = _proc_state(proc_root, pid)
                    starts[pid] = st[1] if st else None
                    records.append({"pid": pid, "cmdline": _cmdline(proc_root, pid), "stopped": False})

        def send(pid: int, sig) -> None:
            if not _environ_has(proc_root, pid, entry):
                return                                   # gone, a zombie, or no longer one of ours
            try:
                pgid = os.getpgid(pid)
                if pgid == pid and pgid > 1 and pgid != os.getpgrp() and pgid not in protected:
                    killpg(pgid, sig)
                else:
                    kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        def kill_pass() -> None:
            try:
                note(found())                            # a second scan: new processes started during the grace
            except Exception:
                pass                                     # the processes already known still get their SIGKILL
            for r in records:
                try:
                    send(r["pid"], signal.SIGKILL)
                except Exception:
                    pass

        targets = found()
        note(targets)
        try:
            for pid in targets:
                send(pid, signal.SIGTERM)
            deadline = clock() + grace_s
            while clock() < deadline and any(_environ_has(proc_root, pid, entry) for pid in targets):
                sleep(SWEEP_POLL_S)
        except BaseException:
            try:
                kill_pass()                              # an interrupt must not skip the SIGKILL
            except Exception:
                pass                                     # and an error in it must not replace the interrupt
            raise
        kill_pass()
        deadline = clock() + 1.0                         # a killed process leaves /proc within moments
        while clock() < deadline and not all(_gone(proc_root, r["pid"], starts.get(r["pid"])) for r in records):
            sleep(SWEEP_POLL_S)
        for r in records:
            r["stopped"] = _gone(proc_root, r["pid"], starts.get(r["pid"]))
    except Exception:
        pass
    return records


def build_watchdog(control: ProbeControl, ledger) -> Watchdog:
    """The supervisor's loop detectors with their default thresholds, answering through the probe's control object.
    NoNewEvidence is left out: an attempt has no evidence directory the agent is asked to fill, and the wall-clock
    budget covers a slow agent."""
    return Watchdog([IdenticalResponses(), RepeatedToolCall(), TurnRepeat(), NoFileWritten(), ThinkingWithoutAction()],
                    Ladder(control, ledger, {AGENT}))


def run_attempt(*, op, n, probe_dir, env, endpoint, model, coder_model, probe_id, max_turns=BUDGET["max_turns"],
                wall_s=BUDGET["wall_s"], http=post_json, clock=time.time, watchdog=None, run_watched=run_watched,
                tt_lang: dict | None = None, temperature: float = 0.0, top_p: float | None = None,
                seed: int | None = None) -> dict:
    """One attempt: prepare the directory, run the agent, run the authoritative check, write and return attempt.json.
    The sampling (temperature, top_p, seed) goes to the coder and into the record as `sampling`."""
    sampling = {"temperature": temperature, "top_p": top_p, "seed": seed}
    adir = Path(probe_dir) / op / f"attempt-{n}"
    facts = tt_lang if tt_lang is not None else tt_lang_facts(env)
    prepare_attempt(adir, op=op, env=env, tt_lang=facts)
    control = ProbeControl(wall_s=wall_s)
    ledger = Ledger(adir / "ledger.jsonl")
    started = time.monotonic()
    try:
        dog = watchdog if watchdog is not None else build_watchdog(control, ledger)
        step = AgentStep(agent=AGENT, endpoint=endpoint, model=model, tools=attempt_tools(adir), ledger=ledger,
                         stage=0, phase="probe", feed=dog.feed, control=control, run_dir=adir,
                         evidence_dir=adir / "evidence", log_path=adir / "log" / "probe-00001.jsonl", http=http,
                         clock=clock, guard=RetryGuard(), max_turns=max_turns, temperature=temperature, top_p=top_p,
                         seed=seed)
        user = (f"Write kernel.py for the op `{op}` in kernel_spec.json. "
                "Start by reading kernel_spec.json and reference/eltwise_add.py.")
        outcome = step.run(SKILL.read_text(encoding="utf-8"), user)
    finally:
        ledger.close()
        # Before any verdict, and on a crash or an interrupt too: no process the agent started may keep running.
        swept = sweep_strays(adir)
    agent_wall_s = round(time.monotonic() - started, 2)
    check = authoritative_check(adir, op=op, env=env, run_watched=run_watched, tt_lang=facts)
    sets = check.get("sets") if isinstance(check.get("sets"), list) else []
    measures = set_measures(sets)
    passed = check.get("passed") is True
    attempt = {
        "op": op, "attempt": n, "status": outcome.status, "status_detail": outcome.detail or control.detail,
        "turns": outcome.turns, "agent_wall_s": agent_wall_s, "passed_sim": passed,
        "category": None if passed else check.get("category"), "first_line": check.get("first_line") or "",
        **measures, "violations": check.get("violations") or [], "sets": sets, "hw": None,
        "agent_check_untrusted": agent_check_untrusted(_read_agent_check(adir / "check.json"), check, op),
        "swept": swept, "sampling": dict(sampling),
        "provenance": provenance(kernel_sha256=check.get("kernel_sha256"), tt_lang=facts,
                                 coder_model=coder_model, probe_id=probe_id, simulator_pcc_min=measures["pcc_min"],
                                 sampling=dict(sampling)),
    }
    _save_attempt(adir, attempt)
    return attempt


# ---- the hardware phase -----------------------------------------------------------------------------------------

def _gozer(reason: str, shared_python: str, *args: str) -> list[str]:
    """A one-chip gozer lease around a command run by the shared venv's python. The lease's release resets the chip."""
    return ["gozer", "run", "--chips", "1", "--exact", "1", "--who", GOZER_WHO, "--reason", reason, "--",
            shared_python, *args]


def _new_text(path: Path, offset: int) -> str:
    """What was appended to `path` after byte `offset` (run_watched appends to its log)."""
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", errors="replace")


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _hw_watcher() -> StallWatcher:
    return StallWatcher(quiet_s=HW_QUIET_S, repeat_lines=HW_REPEAT_LINES)


def _watch_facts(got) -> dict:
    """What the stall watcher saw on one hardware run, beside the limits it applied."""
    return {"max_quiet_s": round(float(getattr(got, "max_quiet_s", 0.0) or 0.0), 1),
            "max_repeat": int(getattr(got, "max_repeat", 0) or 0),
            "quiet_limit_s": HW_QUIET_S, "repeat_limit": HW_REPEAT_LINES}


def _watch_text(w: dict) -> str:
    return (f"longest quiet stretch {w['max_quiet_s']:.1f} s, longest repeat {w['max_repeat']} lines; "
            f"watcher limits {w['quiet_limit_s']:.0f} s and {w['repeat_limit']} lines")


def smoke_test(*, env, shared_python, run_watched=run_watched, record: dict | None = None) -> tuple[bool, list[str]]:
    """Upstream's examples/eltwise_add.py with the shared venv's tt-lang and ttnn, on one leased chip (spec 5.4).
    (ok, the last lines of its output). ok needs exit 0 and `PASSED!` in the output. A run the stall watcher
    killed raises ProbeBlocked. `record`, when given, receives `watch`: what the stall watcher saw (_watch_facts)."""
    Path(env.scratch_dir).mkdir(parents=True, exist_ok=True)
    log = Path(env.scratch_dir) / "smoke.txt"
    start = _size(log)
    argv = _gozer("kernel-probe smoke test", str(shared_python), str(Path(env.src_dir) / "examples" / "eltwise_add.py"))
    got = run_watched(argv, cwd=env.scratch_dir, env=None, log_path=log, watcher=_hw_watcher(),
                      backstop_s=BUDGET["hw_backstop_s"], term_grace_s=GOZER_TERM_GRACE_S)
    text = _new_text(log, start)
    tail = text.splitlines()[-SMOKE_TAIL_LINES:]
    if record is not None:
        record["watch"] = _watch_facts(got)
    if got.killed:
        # A hang blocks the probe. The watcher's SIGTERM reaches gozer too, and gozer normally releases its lease
        # (with the chip reset) within GOZER_TERM_GRACE_S. Only a gozer that had to be killed after the grace can
        # leave its lease under a dead process.
        hw = {"available": False, "hung": True, "reason": got.reason, "smoke_tail": tail, "watch": _watch_facts(got)}
        last = " / ".join(tail[-5:]) or "(no output)"
        raise ProbeBlocked(f"the smoke test hung: {got.reason}. Its last lines: {last}. The watcher stopped it, and "
                           "gozer normally releases the lease when it stops. If gozer itself had to be killed (after "
                           f"the {GOZER_TERM_GRACE_S:.0f} s grace, or at once by a second interrupt during the grace), "
                           "the lease may still be held by a dead process: look with `gozer status` and use the "
                           "gozer-gatekeeper skill. Never reset a chip by hand", hw=hw)
    return (got.returncode == 0 and "PASSED!" in text), tail


def _record_hw(adir: Path, attempt: dict, hw: dict) -> dict:
    attempt["hw"] = hw
    if isinstance(attempt.get("provenance"), dict):
        attempt["provenance"]["hardware_pcc_min"] = hw.get("pcc_min")
    _save_attempt(adir, attempt)
    return hw


def hardware_phase(adir: Path, attempt: dict, *, env, shared_python, run_watched=run_watched) -> dict:
    """Run one attempt's checked kernel on one chip and judge it with the hardware bars. Call it only for an attempt
    whose authoritative simulator check passed, after a passing smoke test (run_hardware does both).

    It runs the copy of kernel.py the authoritative check judged (the verdict directory's CLEAN_DIR), with the
    shipped harness and spec. The static gate runs again first; a kernel that breaks a rule ends `forbidden_code` and
    no gozer command runs. Then the copy is hashed again: when it differs from the sha256 recorded at copy time, the
    attempt ends `harness_error` and no gozer command runs. A run the stall watcher kills is a hang: it is recorded,
    and ProbeBlocked names the attempt. Returns the `hw` block, which is also written into attempt.json."""
    adir = Path(adir)
    vdir = _verdict_dir_of(adir)
    op, n = attempt["op"], attempt["attempt"]
    spec = load_spec(op)
    check = _template("kernel_check")
    kernel = vdir / CLEAN_DIR / "kernel.py"
    if not kernel.is_file() or kernel.is_symlink():
        return _record_hw(adir, attempt, {"passed": False, "category": "no_kernel", "pcc_min": None, "sets": [],
                                          "violations": [], "first_line": "the checked copy of kernel.py is missing"})
    violations = check.kernel_source_violations(kernel)
    ok, reason = check.check_kernel_source(kernel)
    if violations or not ok:
        category = "no_kernel" if reason.startswith("no-operation") else "forbidden_code"
        return _record_hw(adir, attempt, {"passed": False, "category": category, "pcc_min": None, "sets": [],
                                          "violations": violations, "first_line": reason})
    recorded = (attempt.get("provenance") or {}).get("kernel_sha256")
    data = _read_regular(kernel, MAX_KERNEL_BYTES)
    if data is None or recorded is None or hashlib.sha256(data).hexdigest() != recorded:
        return _record_hw(adir, attempt, {"passed": False, "category": "harness_error", "pcc_min": None, "sets": [],
                                          "violations": [], "first_line": "the clean kernel copy changed since the "
                                                                          "simulator check; it was not run"})
    out = vdir / "hw-result.json"
    _remove(out)
    log = vdir / "hw-output.txt"
    start = _size(log)
    argv = _gozer(f"kernel-probe {op} attempt {n}", str(shared_python), str(TEMPLATES / "kernel_harness.py"),
                  "--spec", str(TEMPLATES / "ops" / f"{op}.json"), "--kernel", str(kernel), "--out", str(out),
                  "--phase", "hw")
    got = run_watched(argv, cwd=vdir / CLEAN_DIR, env=None, log_path=log, watcher=_hw_watcher(),
                      backstop_s=BUDGET["hw_backstop_s"], term_grace_s=GOZER_TERM_GRACE_S)
    if got.killed:
        # The first line is the watcher's own reason ("backstop: ..." or "stalled: ..." or "repeating: ..."), so a
        # watcher stop is not read as a device hang without saying which rule stopped it.
        _record_hw(adir, attempt, {"passed": False, "category": "hang", "pcc_min": None, "sets": [], "violations": [],
                                   "first_line": got.reason, "elapsed_s": round(got.elapsed_s, 1),
                                   "watch": _watch_facts(got)})
        raise ProbeBlocked(f"{op} attempt {n}: {got.reason}")
    try:
        raw = _read_regular(out, 64 * 1024 * 1024)
        result = json.loads(raw.decode("utf-8", errors="replace")) if raw is not None else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        result = None                                   # classify reads a missing or broken result file the same way
    category, line = check.classify(kernel_present=True, timed_out=False, returncode=got.returncode,
                                    text=_new_text(log, start), result=result,
                                    bar=spec["pcc_bar_hw"], phase="hw", bars=check.bars_from_spec(spec, "hw"),
                                    diagnostics_present=None)
    sets = result.get("sets") if result and isinstance(result.get("sets"), list) else []
    return _record_hw(adir, attempt, {"passed": category is None, "category": category, "first_line": line,
                                      **set_measures(sets), "sets": sets, "violations": [],
                                      "elapsed_s": round(got.elapsed_s, 1), "watch": _watch_facts(got)})


def run_hardware(attempts: list[dict], *, probe_dir, env, shared_python, run_watched=run_watched,
                 state: dict | None = None) -> dict:
    """The hardware phase of a probe: one smoke test, then hardware_phase for each attempt that passed the simulator.

    A failed smoke test ends the phase with no further gozer call, and the report says `hw: unavailable` with the
    smoke test's last lines. A hung smoke test or kernel run raises ProbeBlocked, so no later attempt runs. Returns
    {"available", "smoke_tail"}.

    Results are kept as they happen, so an exception part way through loses nothing that finished: hardware_phase
    writes each attempt's `hw` block into its record in place, and the smoke test's result is stored in `state["smoke"]`
    before any kernel runs."""
    seen: dict = {}
    ok, tail = smoke_test(env=env, shared_python=shared_python, run_watched=run_watched, record=seen)
    if state is not None:
        state["smoke"] = {"available": ok, "smoke_tail": tail, "watch": seen.get("watch")}
    if not ok:
        return {"available": False, "smoke_tail": tail, "watch": seen.get("watch")}
    for attempt in attempts:
        if attempt.get("passed_sim") is True:
            adir = Path(probe_dir) / attempt["op"] / f"attempt-{attempt['attempt']}"
            hardware_phase(adir, attempt, env=env, shared_python=shared_python, run_watched=run_watched)
    return {"available": True, "smoke_tail": tail, "watch": seen.get("watch")}


# ---- the report -------------------------------------------------------------------------------------------------

def report_header(*, env, coder_model, probe_id, ops, hw=None, tt_lang: dict | None = None,
                  ignored_records: int = 0) -> dict:
    """The report's header: the tt-lang facts, the coder's model id, the probe id, the provisional bars of each op,
    the budget, and the hardware phase's result (None when it was not run)."""
    check = _template("kernel_check")
    bars = {}
    for op in ops:
        spec = load_spec(op)
        bars[op] = {phase: check.bars_from_spec(spec, phase) for phase in ("sim", "hw")}
    return {"tt_lang": tt_lang if tt_lang is not None else tt_lang_facts(env), "coder_model": coder_model,
            "probe_id": probe_id, "ops": list(ops), "bars": bars, "provisional": True, "budget": dict(BUDGET),
            "inputs": "synthetic", "hw": hw, "ignored_records": ignored_records,
            "budget_note": (f"The wall budget ({BUDGET['wall_s']:.0f} s) is checked between turns, so an attempt can "
                            f"overrun it by up to one tool timeout ({AGENT_TOOL_TIMEOUT_S:.0f} s at most, for the "
                            "kernel check).")}


def _first_pass(rows: list[dict], passed) -> int | None:
    for a in sorted(rows, key=lambda a: a["attempt"]):
        if passed(a):
            return a["attempt"]
    return None


def _hw_passed(a: dict) -> bool:
    return isinstance(a.get("hw"), dict) and a["hw"].get("passed") is True


def _cell(value) -> str:
    """A table cell: no pipe and no line break may leak into the Markdown table."""
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        value = f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")[:160]


def _pcc_cell(value) -> str:
    """A PCC cell. When six significant digits would round a value near 1 to `1`, the distance from 1 is printed with
    its sign (`1 - 6.0e-08`, or `1 + 1.0e-07` for a value a little above 1 from floating-point error), so the table
    never shows a perfect PCC for an imperfect one."""
    if isinstance(value, float) and value != 1.0 and f"{value:.6g}" == "1":
        return f"1 - {1.0 - value:.1e}" if value < 1.0 else f"1 + {value - 1.0:.1e}"
    return _cell(value)


def _status_cell(a: dict) -> str:
    """How the agent step ended. A step that did not end `done` also gets the first line of the reason."""
    status = a.get("status")
    detail = str(a.get("status_detail") or "").strip().splitlines()
    if status and status != "done" and detail:
        return _cell(f"{status}: {detail[0]}")
    return _cell(status)


def _agent_cell(a: dict) -> str:
    """The agent column: whether its own check.json was untrusted, and how many of its processes the sweep stopped."""
    parts = ["untrusted"] if a.get("agent_check_untrusted") else []
    swept = [r for r in a.get("swept") or [] if isinstance(r, dict)]
    stopped = [r for r in swept if r.get("stopped") is not False]
    left = [r for r in swept if r.get("stopped") is False]
    if stopped:
        parts.append(f"{len(stopped)} stray agent process{'es' if len(stopped) != 1 else ''} stopped")
    if left:
        pids = ", ".join(str(r.get("pid")) for r in left)
        parts.append(f"{len(left)} stray agent process{'es' if len(left) != 1 else ''} could not be confirmed stopped "
                     f"(pid{'s' if len(left) != 1 else ''} {pids})")
    return _cell("; ".join(parts))


def _bars_text(b: dict) -> str:
    lo, hi = b["slope"]
    return (f"pcc >= {b['pcc']}, rel_l2 <= {b['rel_l2']}, slope in [{lo}, {hi}], "
            f"max_abs_err <= {b['max_abs_atol']} + {b['max_abs_rtol']} * max|expected|")


def write_report(probe_dir, attempts: list[dict], header: dict) -> dict:
    """Write kernel-probe-report.json (header, attempts, attempts to first pass per op) and kernel-probe-report.md."""
    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    ops = list(header.get("ops") or [])
    for a in attempts:
        if a["op"] not in ops:
            ops.append(a["op"])
    by_op = {op: [a for a in attempts if a["op"] == op] for op in ops}
    first = {op: {"sim": _first_pass(rows, lambda a: a.get("passed_sim") is True), "hw": _first_pass(rows, _hw_passed)}
             for op, rows in by_op.items()}
    data = {"header": header, "attempts": attempts, "first_pass": first}
    _write_json(probe_dir / "kernel-probe-report.json", data)

    t = header.get("tt_lang") or {}
    lines = ["# Kernel probe report", "",
             f"Probe id: {header.get('probe_id')}. Coder model: {header.get('coder_model')}.", "",
             "Attempts are independent draws from a model server that may not accept a seed. The report gives counts "
             "and no rates. The inputs are synthetic: seeded random values plus a heavy-tailed set with a few scaled "
             "outliers. A kernel that passes them can still fail on real activations.", "",
             "The simulator promotes bf16 to f32 (tt-lang-sim's default) and the references compute in f32, so on a "
             "simple op a correct kernel shows a max_abs_err of 0.0 and a PCC of 1.0. Rounding on hardware is "
             "coarser (bf16) than the simulator's f32.", "",
             *(["`add` is a calibration op: its working kernel is upstream's eltwise_add.py, the reference example "
                "the agent reads. A pass of `add` shows the loop works; it is not evidence that the coder can write a "
                "kernel.", ""] if "add" in ops else []),
             *([header["budget_note"], ""] if header.get("budget_note") else []),
             *([f"{header['ignored_records']} unrelated or unreadable records ignored under {VERDICT_DIR}.", ""]
               if header.get("ignored_records") else []),
             "## tt-lang", "",
             f"- simulator version: {t.get('sim_version')}",
             f"- source SHA: {t.get('source_sha')}",
             f"- diagnostics patch applied: {'yes' if t.get('diagnostics_patched') else 'no'} "
             f"(a local patch of the tt-lang-sim wheel; patch sha256 {t.get('patch_sha256')})", "",
             "## Bars (provisional: none has been measured)", ""]
    for op, b in (header.get("bars") or {}).items():
        lines.append(f"- {op}, simulator: {_bars_text(b['sim'])}")
        lines.append(f"- {op}, hardware: {_bars_text(b['hw'])}")
    lines += ["", "## Hardware", ""]
    hw = header.get("hw")
    if header.get("stopped") or header.get("blocked"):
        lines += [f"The probe stopped: {header.get('stopped') or header['blocked']}", ""]
    if isinstance(hw, dict) and isinstance(hw.get("watch"), dict):
        lines += [f"smoke test: {_watch_text(hw['watch'])}", ""]
    if hw is None:
        lines.append("hw: not run")
    elif hw.get("available"):
        lines.append("hw: available (the smoke test passed)")
    elif hw.get("hung"):
        lines += [f"hw: blocked. The probe stopped: the smoke test hung ({hw.get('reason')}). Its last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    else:
        lines += ["hw: unavailable. The smoke test (upstream's eltwise_add.py on one chip) failed. Its last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    for op, rows in by_op.items():
        lines += ["", f"## {op}", "",
                  "| attempt | simulator | hardware | status | PCC min (sim) | PCC min (hw) | turns | wall s | category | "
                  "first line | agent check |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a in sorted(rows, key=lambda a: a["attempt"]):
            h = a.get("hw") if isinstance(a.get("hw"), dict) else None
            category = a.get("category") or (h or {}).get("category")
            first_line = a.get("first_line") if a.get("category") else (h or {}).get("first_line")
            lines.append("| " + " | ".join([
                _cell(a["attempt"]), "yes" if a.get("passed_sim") else "no",
                "-" if h is None else ("yes" if h.get("passed") else "no"), _status_cell(a),
                _pcc_cell(a.get("pcc_min")), _pcc_cell((h or {}).get("pcc_min")), _cell(a.get("turns")),
                _cell(a.get("agent_wall_s")), _cell(category), _cell(first_line), _agent_cell(a)]) + " |")
        f = first[op]
        lines.append("")
        lines.append(f"attempts to first pass in the simulator: {f['sim']}" if f["sim"] is not None
                     else f"none of {len(rows)} attempts passed in the simulator")
        ran = [a for a in rows if isinstance(a.get("hw"), dict)]
        if ran:
            # Count only the attempts that ran on hardware. When the probe stopped early, simulator passes that never
            # reached a chip are said to be left out.
            left = [a for a in rows if a.get("passed_sim") is True and not isinstance(a.get("hw"), dict)]
            count = f"{sum(1 for a in ran if _hw_passed(a))} of {len(ran)} attempts that ran on hardware passed"
            if left and (header.get("stopped") or header.get("blocked")):
                count += "; the probe stopped before the rest"
            lines.append(count)
            if f["hw"] is not None:
                lines.append(f"attempts to first pass on hardware: {f['hw']}")
            for a in sorted(ran, key=lambda a: a["attempt"]):
                if isinstance(a["hw"].get("watch"), dict):
                    lines.append(f"attempt {a['attempt']} on hardware: {_watch_text(a['hw']['watch'])}")
    (probe_dir / "kernel-probe-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


# ---- the command ------------------------------------------------------------------------------------------------

# The same codes as orchard/cli.py (a test checks they agree): 3 a crash, 4 an interrupt, 5 a hardware hang.
EXIT_OK, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED, EXIT_BLOCKED = 0, 2, 3, 4, 5
CODER_STAGE = 2              # the probe's agent is the coder that writes stage 2's code, so it uses stage 2's tier


class _Refused(Exception):
    """The command stops before anything runs. The message says what to do."""


def default_env(args) -> ttlang_env.KEnv:
    """The tt-lang environment the flags or the ORCHARD_TTLANG_* variables name, as `tt-orchard kernel-env` sees it."""
    # Like ttlang_env.main, this reads os.environ and Path.home() directly.
    home = Path.home()
    return ttlang_env.KEnv(
        src_dir=Path(getattr(args, "src_dir", None) or os.environ.get("ORCHARD_TTLANG_SRC")
                     or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(getattr(args, "venv_dir", None) or os.environ.get("ORCHARD_TTLANG_VENV")
                      or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        environ=dict(os.environ))


def coder_endpoint(args, tier_config, probe) -> tuple[str, str]:
    """(endpoint, model) of the coder the attempts talk to.

    --endpoint and --model always win. A value that neither flag gives comes from the tier stage 2 runs on, in the tier
    config. With both flags missing, the tier's endpoint is found as the supervisor finds it (resolve_endpoint: another
    chip tier serving the same model may stand in). With no tier config, the missing flag is refused by name."""
    from orchard.stages import TierUnavailable, resolve_endpoint, tier_for
    endpoint, model = getattr(args, "endpoint", None), getattr(args, "model", None)
    if endpoint and model:
        return endpoint, model
    if tier_config is None:
        raise _Refused("no coder endpoint is known: pass --endpoint URL and --model NAME, or keep a bringup.toml "
                       "whose tier config names the coder")
    tier = tier_for(tier_config, CODER_STAGE, phase="main", escalated=False)
    if endpoint:
        return endpoint, tier_config.tiers[tier]["model"]
    if model:
        return tier_config.tiers[tier]["endpoint"], model
    try:
        used, endpoint, _note = resolve_endpoint(tier_config, tier, probe)
    except TierUnavailable:
        t = tier_config.tiers[tier]
        raise _Refused(f"tier {tier} ({t['model']} at {safe_url(t['endpoint'])}) is not serving. The probe starts no server: start the coder with `tt-orchard bringup`, or pass "
                       "--endpoint and --model for a server that runs") from None
    return endpoint, tier_config.tiers[used]["model"]


def _attempt_line(a: dict) -> str:
    """One line for an attempt that has ended: op, number, result, first line, turns, wall time."""
    result = "passed in the simulator" if a.get("passed_sim") else (a.get("category") or "failed")
    first = f" ({a['first_line']})" if a.get("first_line") and not a.get("passed_sim") else ""
    return f"{a['op']} {a['attempt']}: {result}{first}, {a.get('turns')} turns, {a.get('agent_wall_s')} s"


def _hw_line(a: dict) -> str:
    h = a["hw"]
    result = "passed on hardware" if h.get("passed") else (h.get("category") or "failed")
    first = f" ({h['first_line']})" if h.get("first_line") and not h.get("passed") else ""
    return f"{a['op']} {a['attempt']}: {result}{first}"


def new_probe_id() -> str:
    """A token no earlier run used: the UTC time and 8 random hex digits."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(4)


def safe_url(url) -> str:
    """`url` with any `user:password@` removed, for printing. The request itself still uses the URL as given."""
    try:
        parts = urlsplit(str(url))
    except ValueError:
        return "<an unparseable URL>"
    if "@" not in parts.netloc:
        return str(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


LOCK_NAME = "probe.lock"


class ProbeLock:
    """The held lock: the open file (whose flock is the lock) and its path."""

    def __init__(self, fd: int, path: Path):
        self.fd, self.path = fd, path


def _holder_note(fd: int) -> str:
    """` (pid N)` from the holder's lock file, or "" when the file is empty, garbage or names no sane pid. The text is
    information for the refusal message only; the flock decides who holds the lock."""
    try:
        rec = json.loads(os.pread(fd, 4096, 0).decode("utf-8"))
        pid = rec.get("pid") if isinstance(rec, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and 0 < pid < (1 << 32):
            return f" (pid {pid})"
    except Exception:
        pass
    return ""


def acquire_lock(probe_dir: Path) -> ProbeLock:
    """Take an exclusive flock on `probe_dir/probe.lock` for this process, or raise _Refused with a plain message.

    Two probes on one directory would write the same attempt directories, and each one's sweep would stop the other's
    agent processes. The kernel releases a flock when its holder dies (SIGKILL included), so a lock left by an ended
    probe needs no staleness check: the next probe simply gets it. The file is opened with O_NOFOLLOW (a link is
    refused) and must be a regular file. After the flock, the file at the path must be the file we locked; if another
    probe removed it and made a new one in between, the lock is taken again once. The pid written into the file is
    information for a refused probe's message."""
    path = Path(probe_dir) / LOCK_NAME
    try:
        Path(probe_dir).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _Refused(f"cannot create the output directory {probe_dir}: {exc.strerror or exc}") from None
    for _ in range(2):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise _Refused(f"{path} is a link; remove it if no probe is using {probe_dir}") from None
            raise _Refused(f"cannot open the lock {path}: {exc.strerror or exc}") from None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise _Refused(f"{path} is not a regular file; remove it if no probe is using {probe_dir}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise _Refused(f"another probe is using {probe_dir}{_holder_note(fd)}; wait for it to end or "
                                   "pass another --out") from None
                raise _Refused(f"cannot lock {path}: {exc.strerror or exc}") from None
            try:
                st = os.stat(path, follow_symlinks=False)
                same = (st.st_ino, st.st_dev) == (os.fstat(fd).st_ino, os.fstat(fd).st_dev)
            except OSError:
                same = False
            if not same:
                os.close(fd)                             # we locked a file nobody else can reach: take it again
                continue
            try:
                os.ftruncate(fd, 0)
                os.pwrite(fd, json.dumps({"pid": os.getpid()}).encode("utf-8"), 0)
            except OSError:
                pass                                     # the pid is information only; the flock is the lock
            return ProbeLock(fd, path)
        except BaseException as exc:
            try:
                os.close(fd)
            except OSError:
                pass
            if isinstance(exc, OSError):                 # fstat, stat or another call on the lock file failed
                raise _Refused(f"cannot use the lock {path}: {exc.strerror or exc}") from None
            raise
    raise _Refused(f"another probe is using {probe_dir}: its lock file kept changing while this probe took it")


def release_lock(lock: ProbeLock) -> None:
    """Remove the lock file if it is still the file we hold, then close it (which releases the flock). Never raises.
    After a SIGKILL nothing runs here: the kernel releases the flock, and the file that stays is harmless."""
    try:
        st = os.stat(lock.path, follow_symlinks=False)
        if (st.st_ino, st.st_dev) == (os.fstat(lock.fd).st_ino, os.fstat(lock.fd).st_dev):
            os.unlink(lock.path)
    except OSError:
        pass
    try:
        os.close(lock.fd)
    except OSError:
        pass


def _raise_interrupt(signum, frame):
    """SIGTERM handler while the probe runs: stop the way Ctrl-C does, so the report is still written."""
    raise KeyboardInterrupt


def _crash(where: str, exc: BaseException) -> str:
    return f"{where} crashed: {type(exc).__name__}: {str(exc)[:200]}"


def main(args, *, env=None, out=None, http=post_json, probe=probe_model, run_watched_fn=run_watched,
         tier_config=None) -> int:
    """`tt-orchard kernel-probe --op NAME [--attempts N] [--phase sim|hw] [--out DIR] [--endpoint URL] [--model NAME]
    [--src-dir D] [--venv-dir D] [--shared-python P]`.

    Order: the op is checked; the simulator environment must pass `ttlang_env.ensure(check_only=True)`; the coder's
    endpoint and model are resolved and must answer `probe`. Each refusal returns 2 before any attempt directory
    exists. Then N attempts run one after another, each printing one line as it ends. With --phase hw, run_hardware
    runs one smoke test and then each attempt that passed the simulator, each on one leased chip. The report is
    written from the records this run's attempts returned. `tier_config` is the TierConfig the coder comes from when a
    flag is missing (the CLI loads it from bringup.toml).

    The output directory is locked (an flock on `probe.lock`) from the first write to the end. A second probe on the
    same directory is refused with exit 2. The kernel releases the lock when this process dies, however it dies; the
    file may remain after a SIGKILL and is harmless. A clean exit removes it.

    A run that stops early still writes its report from what finished, with "The probe stopped: <reason>" in it:
    a hung hardware run (ProbeBlocked) exits 5, an exception in an attempt or in the hardware phase exits 3, and Ctrl-C
    or SIGTERM exits 4. No hardware runs after a crash or an interrupt. If the report itself cannot be written, the
    records are printed to stderr as JSON and the exit is 3. URLs are printed without any user:password part."""
    say = (lambda line: print(line, file=out)) if out is not None else print

    def refuse(message: str) -> int:
        print(f"refused: {message}", file=sys.stderr)
        return EXIT_REFUSED

    if args.op not in OPS:
        return refuse(f"unknown op {args.op!r}; the ops are {', '.join(OPS)}")
    attempts = int(getattr(args, "attempts", 5) or 0)
    if attempts < 1:
        return refuse("--attempts must be at least 1")
    env = env or default_env(args)
    results = ttlang_env.ensure(env, check_only=True, say=say)
    if not results or any(r.status != ttlang_env.OK for r in results):
        return refuse("the tt-lang simulator environment is not ready: run `tt-orchard kernel-env`")
    try:
        endpoint, model = coder_endpoint(args, tier_config, probe)
    except _Refused as exc:
        return refuse(str(exc))
    if not probe(endpoint, model):
        return refuse(f"the coder at {safe_url(endpoint)} does not answer with the model {model!r}. The probe starts "
                      "no server: start the coder with `tt-orchard bringup`, or pass --endpoint and --model for a "
                      "server that runs")

    probe_dir = Path(getattr(args, "out", None) or "kernel-probe")
    try:
        lock = acquire_lock(probe_dir)
    except _Refused as exc:
        return refuse(str(exc))
    try:
        probe_id = new_probe_id()
        facts = tt_lang_facts(env)
        say(f"kernel probe {probe_id}: op {args.op}, {attempts} attempts, coder {model} at {safe_url(endpoint)}")
        records: list[dict] = []
        hw, stopped, blocked, code = None, None, None, EXIT_OK
        state: dict = {}
        hw_started = False
        try:
            previous = signal.signal(signal.SIGTERM, _raise_interrupt)
        except ValueError:                                  # not the main thread: no handler can be installed
            previous = None
        try:
            where = "attempt 1"
            try:
                for n in range(1, attempts + 1):
                    where = f"attempt {n}"
                    record = run_attempt(op=args.op, n=n, probe_dir=probe_dir, env=env, endpoint=endpoint, model=model,
                                         coder_model=model, probe_id=probe_id, http=http, run_watched=run_watched_fn,
                                         tt_lang=facts)
                    records.append(record)
                    say(_attempt_line(record))
                if getattr(args, "phase", "sim") == "hw":
                    where, hw_started = "the hardware phase", True
                    shared = (getattr(args, "shared_python", None)
                              or str(Path.home() / ".tenstorrent-venv" / "bin" / "python"))
                    try:
                        hw = run_hardware(records, probe_dir=probe_dir, env=env, shared_python=shared,
                                          run_watched=run_watched_fn, state=state)
                    except ProbeBlocked as exc:
                        blocked = stopped = str(exc)
                        code = EXIT_BLOCKED
                        # A hung smoke test carries its own hardware block. A kernel hang came after a passing
                        # smoke test.
                        hw = exc.hw if exc.hw is not None else state.get("smoke")
            except KeyboardInterrupt:
                stopped, code = f"interrupted during {where}", EXIT_ABORTED
            except Exception as exc:
                stopped, code = _crash(where, exc), EXIT_ERROR
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        if hw_started and hw is None:
            hw = state.get("smoke")                         # a crash or an interrupt after the smoke test ran
        try:
            for a in records:
                if isinstance(a.get("hw"), dict):
                    say(_hw_line(a))
            if hw and not hw.get("available") and not hw.get("hung"):
                say("hardware unavailable: the smoke test failed (the report has its last lines)")
            header = report_header(env=env, coder_model=model, probe_id=probe_id, ops=[args.op], hw=hw, tt_lang=facts,
                                   ignored_records=0)
            if stopped:
                header["stopped"] = stopped
            if blocked:
                header["blocked"] = blocked
            write_report(probe_dir, records, header)
            say(f"report: {probe_dir / 'kernel-probe-report.md'}")
            if stopped:
                say(f"stopped: {stopped}")
        except Exception as exc:                            # never let the last step lose the records
            try:
                print(f"the report could not be written ({type(exc).__name__}: {exc}); the attempt records follow",
                      file=sys.stderr)
                print(json.dumps({"stopped": stopped, "attempts": records}, default=str), file=sys.stderr)
            except Exception:
                pass
            return EXIT_ERROR
        return code
    finally:
        release_lock(lock)
