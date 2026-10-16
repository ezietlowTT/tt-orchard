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
The report is built from these (load_attempts). kernel.py is copied only when it is a regular file (not a link, a
directory or a device) of at most MAX_KERNEL_BYTES; anything else ends the attempt with no check and no hardware run.
The sha256 of the copied bytes is recorded, and the hardware phase refuses a copy whose hash has changed. A
convenience copy of attempt.json is written into the attempt directory after whatever was at that path is removed,
so the probe never writes through a link the agent left.

These measures stop shortcuts and accidents: a model that edits its check, plants a link, or leaves a file in the
way. They do not stop an agent determined to attack the operator's account. Agent shells run as the operator, so a
background process the agent started can outlive its step and write anywhere the operator can, including the
`.verdict` directory. The static gate and this boundary are best effort.

The budget (BUDGET) and the watchdog's detectors bound each attempt. Attempts are independent draws, and the report
gives counts with no rates.

`main` (the command) is wired in by a later task.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

from orchard import defaults, ttlang_env
from orchard.agent import AgentStep, Tools, agent_env
from orchard.canary import post_json
from orchard.ledger import Ledger
from orchard.stallwatch import StallWatcher, Watched, run_watched
from orchard.watchdog import (IdenticalResponses, Ladder, NoFileWritten, RepeatedToolCall, RetryGuard,
                              ThinkingWithoutAction, TurnRepeat, Watchdog)

__all__ = ["Watched"]       # re-exported: tests and callers build fake run_watched results with it

TEMPLATES = Path(__file__).resolve().parent / "skills" / "kernel-loop-templates"
SKILL = Path(__file__).resolve().parent / "skills" / "kernel-probe.md"
OPS = ("add", "silu_mul", "rmsnorm", "linear_scan")
# Provisional: none of these has been measured (spec section 5.3).
BUDGET = {"max_turns": 40, "wall_s": 1800.0, "sim_timeout_s": 300.0, "hw_backstop_s": 600.0}
GOZER_WHO = "claude:orchard-kernel-probe"
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
    """A hardware run hung. The probe stops and names the attempt; nothing is reset by hand."""


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
    """Fill an attempt directory with what the agent reads and runs. `tt_lang` defaults to tt_lang_facts(env)."""
    load_spec(op)                                       # refuses an unknown op before anything is written
    adir = Path(adir)
    adir.mkdir(parents=True, exist_ok=True)
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


def provenance(*, kernel_sha256, tt_lang, coder_model, probe_id, simulator_pcc_min, hardware_pcc_min=None) -> dict:
    """The provenance block of a probe attempt (spec section 6)."""
    return {"origin": "kernel", "kernel_sha256": kernel_sha256, "tt_lang": tt_lang, "coder_model": coder_model,
            "probe_id": probe_id, "simulator_pcc_min": simulator_pcc_min, "hardware_pcc_min": hardware_pcc_min}


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


def _read_agent_check(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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


def load_attempts(probe_dir) -> list[dict]:
    """Every attempt record the probe wrote, read from the probe-owned directory (never from an attempt directory),
    ordered by op (OPS order) and attempt number."""
    out = []
    root = Path(probe_dir) / VERDICT_DIR
    for path in sorted(root.glob("*/attempt.json")) if root.is_dir() else []:
        raw = _read_regular(path, 64 * 1024 * 1024)
        try:
            data = json.loads(raw.decode("utf-8")) if raw is not None else None
        except ValueError:
            data = None
        if isinstance(data, dict) and "op" in data and "attempt" in data:
            out.append(data)
    order = {op: i for i, op in enumerate(OPS)}
    return sorted(out, key=lambda a: (order.get(a["op"], len(OPS)), a["op"], a["attempt"]))


def build_watchdog(control: ProbeControl, ledger) -> Watchdog:
    """The supervisor's loop detectors with their default thresholds, answering through the probe's control object.
    NoNewEvidence is left out: an attempt has no evidence directory the agent is asked to fill, and the wall-clock
    budget covers a slow agent."""
    return Watchdog([IdenticalResponses(), RepeatedToolCall(), TurnRepeat(), NoFileWritten(), ThinkingWithoutAction()],
                    Ladder(control, ledger, {AGENT}))


def run_attempt(*, op, n, probe_dir, env, endpoint, model, coder_model, probe_id, max_turns=BUDGET["max_turns"],
                wall_s=BUDGET["wall_s"], http=post_json, clock=time.time, watchdog=None, run_watched=run_watched,
                tt_lang: dict | None = None) -> dict:
    """One attempt: prepare the directory, run the agent, run the authoritative check, write and return attempt.json."""
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
                         clock=clock, guard=RetryGuard(), max_turns=max_turns)
        user = (f"Write kernel.py for the op `{op}` in kernel_spec.json. "
                "Start by reading kernel_spec.json and reference/eltwise_add.py.")
        outcome = step.run(SKILL.read_text(encoding="utf-8"), user)
    finally:
        ledger.close()
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
        "provenance": provenance(kernel_sha256=check.get("kernel_sha256"), tt_lang=facts,
                                 coder_model=coder_model, probe_id=probe_id, simulator_pcc_min=measures["pcc_min"]),
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


def smoke_test(*, env, shared_python, run_watched=run_watched) -> tuple[bool, list[str]]:
    """Upstream's examples/eltwise_add.py with the shared venv's tt-lang and ttnn, on one leased chip (spec 5.4).
    (ok, the last lines of its output). ok needs exit 0 and `PASSED!` in the output. A run the stall watcher
    killed raises ProbeBlocked."""
    Path(env.scratch_dir).mkdir(parents=True, exist_ok=True)
    log = Path(env.scratch_dir) / "smoke.txt"
    start = _size(log)
    argv = _gozer("kernel-probe smoke test", str(shared_python), str(Path(env.src_dir) / "examples" / "eltwise_add.py"))
    got = run_watched(argv, cwd=env.scratch_dir, env=None, log_path=log, watcher=StallWatcher(quiet_s=60),
                      backstop_s=BUDGET["hw_backstop_s"])
    if got.killed:
        # A hang blocks the probe. gozer was killed with the example, so its lease may outlive it.
        raise ProbeBlocked(f"the smoke test hung: {got.reason}. Its lease may be held by a dead process: use "
                           "gozer-gatekeeper or `gozer reconcile`, never reset by hand")
    text = _new_text(log, start)
    tail = text.splitlines()[-SMOKE_TAIL_LINES:]
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
    got = run_watched(argv, cwd=vdir / CLEAN_DIR, env=None, log_path=log, watcher=StallWatcher(quiet_s=60),
                      backstop_s=BUDGET["hw_backstop_s"])
    if got.killed:
        _record_hw(adir, attempt, {"passed": False, "category": "hang", "pcc_min": None, "sets": [], "violations": [],
                                   "first_line": got.reason, "elapsed_s": round(got.elapsed_s, 1)})
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
                                      "elapsed_s": round(got.elapsed_s, 1)})


def run_hardware(attempts: list[dict], *, probe_dir, env, shared_python, run_watched=run_watched) -> dict:
    """The hardware phase of a probe: one smoke test, then hardware_phase for each attempt that passed the simulator.

    A failed smoke test ends the phase with no further gozer call, and the report says `hw: unavailable` with the
    smoke test's last lines. A hung smoke test or kernel run raises ProbeBlocked, so no later attempt runs. Returns
    {"available", "smoke_tail"}."""
    ok, tail = smoke_test(env=env, shared_python=shared_python, run_watched=run_watched)
    if not ok:
        return {"available": False, "smoke_tail": tail}
    for attempt in attempts:
        if attempt.get("passed_sim") is True:
            adir = Path(probe_dir) / attempt["op"] / f"attempt-{attempt['attempt']}"
            hardware_phase(adir, attempt, env=env, shared_python=shared_python, run_watched=run_watched)
    return {"available": True, "smoke_tail": tail}


# ---- the report -------------------------------------------------------------------------------------------------

def report_header(*, env, coder_model, probe_id, ops, hw=None, tt_lang: dict | None = None) -> dict:
    """The report's header: the tt-lang facts, the coder's model id, the probe id, the provisional bars of each op,
    the budget, and the hardware phase's result (None when it was not run)."""
    check = _template("kernel_check")
    bars = {}
    for op in ops:
        spec = load_spec(op)
        bars[op] = {phase: check.bars_from_spec(spec, phase) for phase in ("sim", "hw")}
    return {"tt_lang": tt_lang if tt_lang is not None else tt_lang_facts(env), "coder_model": coder_model,
            "probe_id": probe_id, "ops": list(ops), "bars": bars, "provisional": True, "budget": dict(BUDGET),
            "inputs": "synthetic", "hw": hw,
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
             *([header["budget_note"], ""] if header.get("budget_note") else []),
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
    if hw is None:
        lines.append("hw: not run")
    elif hw.get("available"):
        lines.append("hw: available (the smoke test passed)")
    else:
        lines += ["hw: unavailable. The smoke test (upstream's eltwise_add.py on one chip) failed. Its last lines:", "",
                  "```", *[str(x) for x in hw.get("smoke_tail") or []], "```"]
    for op, rows in by_op.items():
        lines += ["", f"## {op}", "",
                  "| attempt | simulator | hardware | PCC min (sim) | PCC min (hw) | turns | wall s | category | "
                  "first line | agent check |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a in sorted(rows, key=lambda a: a["attempt"]):
            h = a.get("hw") if isinstance(a.get("hw"), dict) else None
            category = a.get("category") or (h or {}).get("category")
            first_line = a.get("first_line") if a.get("category") else (h or {}).get("first_line")
            lines.append("| " + " | ".join([
                _cell(a["attempt"]), "yes" if a.get("passed_sim") else "no",
                "-" if h is None else ("yes" if h.get("passed") else "no"),
                _cell(a.get("pcc_min")), _cell((h or {}).get("pcc_min")), _cell(a.get("turns")),
                _cell(a.get("agent_wall_s")), _cell(category), _cell(first_line),
                "untrusted" if a.get("agent_check_untrusted") else "-"]) + " |")
        f = first[op]
        lines.append("")
        lines.append(f"attempts to first pass in the simulator: {f['sim']}" if f["sim"] is not None
                     else f"none of {len(rows)} attempts passed in the simulator")
        if any(isinstance(a.get("hw"), dict) for a in rows):
            lines.append(f"attempts to first pass on hardware: {f['hw']}" if f["hw"] is not None
                         else f"none of {len(rows)} attempts passed on hardware")
    (probe_dir / "kernel-probe-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data
