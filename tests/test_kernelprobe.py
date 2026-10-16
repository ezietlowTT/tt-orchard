# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel probe (orchard/kernelprobe.py): attempts, the authoritative re-check, the hardware phase and the report.

Most tests use a fake `run_watched` and a FakeModel, so they need no simulator and no chip. The `needs_sim` tests run the
shipped kernel_check.py on the real tt-lang simulator and skip, saying so, when the ttlang venv or the tt-lang checkout is
absent. No test runs gozer: an autouse tripwire puts a stub gozer first on PATH and fails the test if anything ran it.
Every hardware call goes through a fake `run_watched`, which records the argv and never executes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import kernelprobe as kp, ttlang_env
from orchard.agent import NO_CHIP

REPO = Path(__file__).resolve().parent.parent
KERNELS = Path(__file__).parent / "fixtures" / "kernel-loop" / "kernels"
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SRC = Path(os.environ.get("ORCHARD_TTLANG_SRC", Path.home() / "orchard-src" / "tt-lang"))
needs_sim = pytest.mark.skipif(not (VENV / "bin" / "tt-lang-sim").exists() or not (SRC / "examples").is_dir(),
                               reason="SKIPPED: no tt-lang-sim venv or tt-lang checkout (run `tt-orchard kernel-env`)")


@pytest.fixture(autouse=True)
def gozer_tripwire(stub_tools):
    """Every test runs with stub tools first on PATH. A real `gozer` call from anywhere (an agent shell, a hardware
    phase that forgot its fake) lands in the stub's log, and the test fails here."""
    yield stub_tools
    assert not [c for c in stub_tools.calls() if c.startswith("gozer")], stub_tools.calls()


@pytest.fixture
def kenv(tmp_path):
    return ttlang_env.KEnv(src_dir=SRC, venv_dir=VENV, scratch_dir=tmp_path / "scratch")


FACTS = {"sim_version": "1.1.10+gtest", "source_sha": "a" * 40, "diagnostics_patched": True, "patch_sha256": "b" * 64}


@pytest.fixture
def fake_kenv(tmp_path):
    """A KEnv whose venv and checkout are made up, so tests that do not run the simulator need neither."""
    src = tmp_path / "src"
    (src / "examples").mkdir(parents=True)
    (src / "examples" / "eltwise_add.py").write_text("# stand-in for upstream's example\nprint('PASSED!')\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "tt-lang-sim").write_text("#!/bin/sh\nexit 1\n")
    lib = tmp_path / "purelib"
    lib.mkdir()

    def run(argv, timeout, env=None):
        if argv[-1] == "--version":
            return 0, "tt-lang-sim 1.1.10+gfake\n", ""
        if "rev-parse" in argv:
            return 0, "c" * 40 + "\n", ""
        if "sysconfig" in " ".join(argv):
            return 0, f"{lib}\n", ""
        return 1, "", "not faked"

    (src / ".git").mkdir()
    return ttlang_env.KEnv(src_dir=src, venv_dir=venv, scratch_dir=tmp_path / "scratch", run=run)


def scripted(*messages):
    it = iter(messages)
    return lambda request: next(it, final("done"))


def check_writer(result: dict, calls: list | None = None):
    """A fake run_watched that writes `result` where the authoritative check would, and records the argv."""
    def fake(argv, **kw):
        if calls is not None:
            calls.append(list(argv))
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(result))
        return kp.Watched(1 if not result.get("passed") else 0, None, False, 0.1)
    return fake


def run_attempt(tmp_path, env, server, **kw):
    args = dict(op="add", n=1, probe_dir=tmp_path / "probe", env=env, endpoint=server.endpoint, model="fake",
                coder_model="fake-coder", probe_id="t", max_turns=10, wall_s=120, tt_lang=FACTS)
    args.update(kw)
    return kp.run_attempt(**args)


# ---- the simulator phase, against the real simulator -----------------------------------------------------------


@needs_sim
def test_an_agent_that_writes_a_good_kernel_passes_and_the_report_records_it(tmp_path, kenv):
    good = (KERNELS / "add_good.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=good),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is True and attempt["category"] is None and attempt["agent_check_untrusted"] is False
    assert (tmp_path / "add" / "attempt-1" / "attempt.json").is_file()
    assert attempt["pcc_min"] >= 0.999 and attempt["provenance"]["simulator_pcc_min"] == attempt["pcc_min"]
    # the agent's own check ran too (it wrote check.json), and it agreed
    assert json.loads((tmp_path / "add" / "attempt-1" / "check.json").read_text())["passed"] is True
    kp.write_report(tmp_path, [attempt], {"tt_lang": attempt["provenance"]["tt_lang"], "coder_model": "fake",
                                          "probe_id": "t", "bars": {}, "hw": None})
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "| 1 | yes | - |" in md and "attempts to first pass in the simulator: 1" in md


@needs_sim
def test_an_agent_that_fakes_its_own_check_is_not_believed(tmp_path, kenv):
    """Review focus 3. The agent replaces kernel_check.py with a script that prints PASS and writes a passing check.json."""
    liar = ("import json,sys\n"
            "json.dump({'passed': True, 'category': None, 'sha256': {}}, open('check.json','w'))\n"
            "print('PASS')\n")
    bad = (KERNELS / "add_wrong_op.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=bad),
                            call("write_file", path="kernel_check.py", content=liar),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is False and attempt["category"] == "numeric_mismatch"
    assert attempt["agent_check_untrusted"] is True


# ---- the authoritative check ------------------------------------------------------------------------------------


def test_the_authoritative_check_uses_the_shipped_spec_and_not_the_attempts_copy(tmp_path, fake_kenv):
    """The agent can edit kernel_spec.json in its directory (lower the bar, change the reference). The verdict must not read it."""
    seen = {}

    def fake_run_watched(argv, **kw):
        seen["argv"], seen["kw"] = list(argv), kw
        (Path(argv[argv.index("--out") + 1])).write_text(json.dumps({"passed": False, "category": "numeric_mismatch", "first_line": "x", "sets": []}))
        return kp.Watched(1, None, False, 0.1)

    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel_spec.json").write_text("{}")                       # tampered
    (adir / "kernel.py").write_text("# the agent's kernel\n")
    (adir / "ttnn.py").write_text("# a planted module\n")
    kp.authoritative_check(adir, op="add", env=fake_kenv, run_watched=fake_run_watched, tt_lang=FACTS)
    argv = seen["argv"]
    assert Path(argv[1]) == kp.TEMPLATES / "kernel_check.py"
    assert Path(argv[argv.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"
    assert argv[argv.index("--sim") + 1] == str(fake_kenv.sim)
    assert json.loads(argv[argv.index("--tt-lang-json") + 1]) == FACTS
    # The check runs in a fresh directory that holds only a copy of kernel.py, so nothing else the agent wrote is near it.
    clean = Path(argv[argv.index("--attempt-dir") + 1])
    assert clean != adir and sorted(p.name for p in clean.iterdir()) == ["kernel.py"]
    assert (clean / "kernel.py").read_text() == "# the agent's kernel\n"


def test_a_check_killed_for_a_stall_is_a_timeout_or_hang_never_a_pass(tmp_path, fake_kenv):
    def stalled(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": True, "category": None, "sets": []}))
        return kp.Watched(-9, "stalled: no output and no CPU use for 120 s", True, 130.0)
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / "add_spin.py").read_text())
    got = kp.authoritative_check(adir, op="add", env=fake_kenv, run_watched=stalled, tt_lang=FACTS)
    assert got["passed"] is False and got["category"] == "timeout" and "stalled" in got["first_line"]


def test_an_agent_check_from_a_changed_template_is_untrusted_even_when_it_failed(tmp_path, fake_kenv):
    """The agent's check.json says FAIL, as the authoritative one does, but its hashes say it came from an edited check."""
    def agent_writes_check(request):
        if turn(request) == 0:
            return call("write_file", path="check.json",
                        content=json.dumps({"passed": False, "category": "numeric_mismatch",
                                            "sha256": {"kernel_check.py": "0" * 64}}))
        return final("done")
    authoritative = {"passed": False, "category": "numeric_mismatch", "first_line": "set seed0: pcc 0.1", "sets": []}
    with FakeModel(agent_writes_check) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    assert attempt["agent_check_untrusted"] is True


def test_no_agent_check_is_not_untrusted(tmp_path, fake_kenv):
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    assert attempt["agent_check_untrusted"] is False and attempt["category"] == "no_kernel"


def test_attempt_json_records_violations_and_the_range_of_each_measure(tmp_path, fake_kenv):
    sets = [{"name": "seed0", "pcc": 0.9995, "rel_l2": 0.004, "slope": 0.999, "max_abs_err": 0.01, "exp_abs_max": 3.0,
             "finite": True},
            {"name": "seed0-tail", "pcc": 0.9991, "rel_l2": 0.008, "slope": 1.004, "max_abs_err": 0.02,
             "exp_abs_max": 40.0, "finite": True},
            {"name": "seed1", "pcc": None, "rel_l2": None, "slope": None, "max_abs_err": 0.0, "exp_abs_max": 3.0,
             "finite": False}]
    violations = [{"rule": "import", "line": 1, "detail": "import os", "remedy": "import only ttl"}]
    authoritative = {"passed": False, "category": "forbidden_code", "first_line": "1 violation: import at line 1",
                     "sets": sets, "violations": violations}
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kernel = (KERNELS / "forbidden_import.py").read_text()
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text(kernel))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(authoritative))
    on_disk = json.loads((tmp_path / "probe" / ".verdict" / "add-1" / "attempt.json").read_text())
    assert on_disk == attempt
    assert json.loads((tmp_path / "probe" / "add" / "attempt-1" / "attempt.json").read_text()) == attempt
    assert attempt["category"] == "forbidden_code" and attempt["violations"] == violations
    assert attempt["pcc_min"] == 0.9991 and attempt["slope_range"] == [0.999, 1.004] and attempt["rel_l2_max"] == 0.008
    assert set(attempt) >= {"op", "attempt", "status", "turns", "agent_wall_s", "passed_sim", "category", "first_line",
                            "pcc_min", "sets", "hw", "agent_check_untrusted", "provenance"}
    assert attempt["hw"] is None and attempt["status"] == "done"
    prov = attempt["provenance"]
    assert prov["origin"] == "kernel" and prov["tt_lang"] == FACTS and prov["coder_model"] == "fake-coder"
    assert prov["probe_id"] == "t" and prov["hardware_pcc_min"] is None
    assert prov["kernel_sha256"] == hashlib.sha256(kernel.encode()).hexdigest()


# ---- the agent, its budget and its watchers ---------------------------------------------------------------------


def test_the_budget_ends_the_agent_through_the_control_object():
    now = [0.0]
    ctl = kp.ProbeControl(wall_s=100, clock=lambda: now[0])
    assert ctl.stop_reason() is None and ctl.take_nudges("a") == []
    now[0] = 101.0
    assert "wall" in ctl.stop_reason()


def test_run_attempt_stops_the_agent_when_its_wall_budget_is_spent(tmp_path, fake_kenv):
    """The budget is wired into the step: with no time left the agent gets no turn at all."""
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(lambda r: call("shell", command="ls")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, wall_s=0, run_watched=check_writer(authoritative))
        assert server.requests == []
    assert "wall" in attempt["status"] and attempt["turns"] == 0


def test_the_agent_environment_has_no_device_and_no_network_tools(tmp_path):
    adir = tmp_path / "add" / "attempt-1"
    adir.mkdir(parents=True)
    tools = kp.attempt_tools(adir)
    assert tools.env["TT_VISIBLE_DEVICES"] == NO_CHIP and tools.env["TT_METAL_VISIBLE_DEVICES"] == NO_CHIP
    assert tools.run_dir == adir and tools.stage_dir == adir
    assert tools.call("shell", json.dumps({"command": "pip install x"})).startswith("refused:")
    assert tools.call("shell", json.dumps({"command": "gozer run --chips 1 -- ls"})).startswith("refused:")


def test_a_model_that_repeats_the_same_call_is_stopped_by_the_watchdog(tmp_path, fake_kenv):
    def same(request):
        msg = call("shell", command="ls")
        msg["content"] = "let me look again"
        return msg
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(same) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, max_turns=30, run_watched=check_writer(authoritative))
    assert attempt["turns"] < 10 and attempt["status"] == "escalate"
    from orchard.ledger import read_entries
    entries = read_entries(tmp_path / "probe" / "add" / "attempt-1" / "ledger.jsonl")
    assert [e["event"] for e in entries if e["data"].get("watchdog")] == ["retry", "escalate"]


def test_a_model_that_never_writes_kernel_py_is_nudged(tmp_path, fake_kenv):
    def wanders(request):
        n = turn(request)
        return call("shell", command=f"echo word{n}") if n < 24 else final("gave up")
    authoritative = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}
    with FakeModel(wanders) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, max_turns=30, run_watched=check_writer(authoritative))
        later = [m["content"] for r in server.requests[20:] for m in r["messages"] if m["role"] == "user"]
    assert any("no file was written" in t for t in later)
    from orchard.ledger import read_entries
    entries = read_entries(tmp_path / "probe" / "add" / "attempt-1" / "ledger.jsonl")
    nudges = [e for e in entries if e["data"].get("watchdog") and e["event"] == "retry"]
    assert nudges and nudges[0]["data"]["findings"][0]["detector"] == "no_file_written"
    assert attempt["status"] == "done"


# ---- preparing an attempt ---------------------------------------------------------------------------------------


def test_prepare_attempt_copies_the_reference_example_from_the_checkout_and_vendors_nothing(tmp_path, fake_kenv):
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    assert (adir / "reference" / "eltwise_add.py").read_text() == \
        (fake_kenv.src_dir / "examples" / "eltwise_add.py").read_text()
    for name in ("kernel_check.py", "kernel_harness.py", "references.py"):
        assert (adir / name).read_bytes() == (kp.TEMPLATES / name).read_bytes()
    assert (adir / "kernel_spec.json").read_bytes() == (kp.TEMPLATES / "ops" / "add.json").read_bytes()
    assert not (adir / "kernel.py").exists()
    if (SRC / "examples" / "eltwise_add.py").is_file():
        upstream = (SRC / "examples" / "eltwise_add.py").read_bytes()
        # One test fixture is upstream's file byte for byte on purpose: the static gate's tests feed it to the gate
        # to show that the example's host code is refused (test_kernel_loop_templates.py, UPSTREAM). It is never given
        # to an agent. Nothing else may be a copy.
        allowed = {KERNELS / "upstream_eltwise_add.py"}
        for root in (REPO / "orchard", REPO / "tests"):
            for p in root.rglob("*.py"):
                if p not in allowed:
                    assert p.read_bytes() != upstream, f"{p} is a copy of upstream's eltwise_add.py"


def test_the_skill_names_the_gate_rules_and_the_thread_functions():
    text = kp.SKILL.read_text()
    flat = " ".join(text.split())
    for needle in ("kernel.py", "kernel_check.py", "reference/eltwise_add.py", "OPERATION", "@ttl.operation",
                   "import torch", "from_torch", "to_torch", "Do not open a device", "main()", "@ttl.compute()",
                   "@ttl.datamovement()", "import only", "forbidden_code", "ttl.dfb_reset"):
        assert needle in flat, needle
    assert "gozer" not in text.lower()


# ---- the hardware phase -----------------------------------------------------------------------------------------


def _passed_attempt(tmp_path, env, kernel="add_good.py", n=1):
    """An attempt whose simulator check passed, built without the simulator: the authoritative copy is in place."""
    adir = tmp_path / "add" / f"attempt-{n}"
    kp.prepare_attempt(adir, op="add", env=env, tt_lang=FACTS)
    (adir / "kernel.py").write_text((KERNELS / kernel).read_text())
    clean = kp.verdict_dir(tmp_path, "add", n) / kp.CLEAN_DIR
    clean.mkdir(parents=True)
    (clean / "kernel.py").write_text((KERNELS / kernel).read_text())
    attempt = {"op": "add", "attempt": n, "passed_sim": True, "category": None, "hw": None,
               "provenance": kp.provenance(kernel_sha256=hashlib.sha256((clean / "kernel.py").read_bytes()).hexdigest(),
                                           tt_lang=FACTS, coder_model="c", probe_id="t", simulator_pcc_min=0.9999)}
    (adir / "attempt.json").write_text(json.dumps(attempt))
    return adir, attempt


def hw_writer(sets, calls, *, passed_log="PASSED!\n"):
    """A fake run_watched for the hardware: the smoke test logs PASSED!, a kernel run writes hw-result.json."""
    def fake(argv, **kw):
        calls.append(list(argv))
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"phase": "hw", "sets": sets, "error": None}))
        else:
            with open(kw["log_path"], "a") as f:
                f.write(passed_log)
        return kp.Watched(0, None, False, 1.0)
    return fake


GOOD_HW_SET = {"name": "seed0", "pcc": 0.995, "rel_l2": 0.03, "slope": 1.03, "max_abs_err": 0.1, "exp_abs_max": 3.0,
               "finite": True}


def test_hardware_commands_run_only_through_gozer_and_only_after_a_passing_smoke_test(tmp_path, fake_kenv):
    calls = []

    def run_watched_fake(argv, **kw):
        calls.append(list(argv))
        return kp.Watched(0, None, False, 1.0)

    ok, _ = kp.smoke_test(env=fake_kenv, shared_python="/shared/python", run_watched=run_watched_fake)
    assert calls[0][:2] == ["gozer", "run"] and "--chips" in calls[0] and calls[0][calls[0].index("--chips") + 1] == "1"
    assert calls[0][calls[0].index("--") + 1] == "/shared/python"
    assert calls[0][calls[0].index("--exact") + 1] == "1" and kp.GOZER_WHO in calls[0]
    assert ok is False                                     # exit 0 without PASSED! in the log is not a pass

    # With a passing smoke test, every kernel run is a gozer run of the SHIPPED harness on the checked copy.
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    calls.clear()
    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/shared/python",
                         run_watched=hw_writer([GOOD_HW_SET], calls))
    assert hw["available"] is True and len(calls) == 2
    smoke, run = calls
    assert smoke[-1] == str(fake_kenv.src_dir / "examples" / "eltwise_add.py")
    assert run[:2] == ["gozer", "run"] and run[run.index("--chips") + 1] == "1" and run[run.index("--exact") + 1] == "1"
    tail = run[run.index("--") + 1:]
    assert tail[0] == "/shared/python" and Path(tail[1]) == kp.TEMPLATES / "kernel_harness.py"
    assert Path(tail[tail.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"
    assert Path(tail[tail.index("--kernel") + 1]) == kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py"
    assert Path(tail[tail.index("--out") + 1]).parent == kp.verdict_dir(tmp_path, "add", 1)
    assert tail[tail.index("--phase") + 1] == "hw"


def test_the_hardware_result_is_judged_with_the_hardware_bars(tmp_path, fake_kenv):
    """pcc 0.995 and slope 1.03 fail the simulator bars and pass the hardware ones."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], []))
    assert hw["passed"] is True and hw["category"] is None and hw["pcc_min"] == 0.995
    assert json.loads((kp.verdict_dir(tmp_path, "add", 1) / "attempt.json").read_text())["hw"]["passed"] is True
    assert attempt["provenance"]["hardware_pcc_min"] == 0.995
    adir2, attempt2 = _passed_attempt(tmp_path, fake_kenv, n=2)
    low = dict(GOOD_HW_SET, pcc=0.98)
    hw = kp.hardware_phase(adir2, attempt2, env=fake_kenv, shared_python="/p", run_watched=hw_writer([low], []))
    assert hw["passed"] is False and hw["category"] == "hw_numeric_mismatch"


def test_a_hardware_hang_blocks_the_probe_and_names_the_attempt(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, n=3)
    adir4, attempt4 = _passed_attempt(tmp_path, fake_kenv, n=4)
    calls = []

    def hangs(argv, **kw):
        calls.append(list(argv))
        if "--out" not in argv:
            with open(kw["log_path"], "a") as f:
                f.write("PASSED!\n")
            return kp.Watched(0, None, False, 1.0)
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)

    with pytest.raises(kp.ProbeBlocked) as info:
        kp.run_hardware([attempt, attempt4], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=hangs)
    assert "add attempt 3" in str(info.value) and "stalled" in str(info.value)
    assert len(calls) == 2                                 # the smoke test and the first kernel run, which hung
    on_disk = json.loads((kp.verdict_dir(tmp_path, "add", 3) / "attempt.json").read_text())
    assert on_disk["hw"]["category"] == "hang" and on_disk["hw"]["passed"] is False


def test_a_failed_smoke_test_makes_the_report_say_hw_unavailable_and_no_gozer_call_for_kernels_follows(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    calls = []
    hw = kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                         run_watched=hw_writer([GOOD_HW_SET], calls, passed_log="TT_FATAL: no device\n"))
    assert len(calls) == 1 and hw["available"] is False and "TT_FATAL: no device" in hw["smoke_tail"]
    assert attempt["hw"] is None
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "hw: unavailable" in md and "TT_FATAL: no device" in md


def test_a_forbidden_kernel_never_reaches_gozer_in_the_hardware_phase(tmp_path, fake_kenv):
    """The simulator check applies the static gate, and the hardware path applies it again before any gozer call."""
    adir, attempt = _passed_attempt(tmp_path, fake_kenv, kernel="forbidden_import.py")
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert calls == []
    assert hw["passed"] is False and hw["category"] == "forbidden_code"
    assert hw["violations"] and hw["violations"][0]["rule"] == "import"


def test_an_attempt_that_failed_the_simulator_gets_no_hardware_run(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    attempt["passed_sim"] = False
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    assert len(calls) == 1 and attempt["hw"] is None     # only the smoke test ran


# ---- the report -------------------------------------------------------------------------------------------------


def test_the_report_header_lists_versions_patch_hashes_coder_and_provisional_bars(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=fake_kenv)
    header = kp.report_header(env=fake_kenv, coder_model="qwen3-coder-next", probe_id="p1", ops=["add", "rmsnorm"])
    facts = header["tt_lang"]
    assert facts["sim_version"] == "1.1.10+gfake" and facts["source_sha"] == "c" * 40
    assert facts["diagnostics_patched"] is False and facts["patch_sha256"] == kp.defaults.TTLANG_DIAGNOSTICS_SHA256
    # the agent's own check sees the same facts the report names
    assert json.loads((adir / "check_config.json").read_text())["tt_lang"] == facts
    assert header["bars"]["add"]["sim"]["pcc"] == 0.999 and header["bars"]["add"]["hw"]["pcc"] == 0.99
    kp.write_report(tmp_path / "probe", [], header)
    md = (tmp_path / "probe" / "kernel-probe-report.md").read_text()
    for needle in ("1.1.10+gfake", "c" * 40, kp.defaults.TTLANG_DIAGNOSTICS_SHA256, "qwen3-coder-next", "provisional",
                   "pcc >= 0.999", "pcc >= 0.99", "independent draws", "synthetic", "diagnostics patch applied: no"):
        assert needle in md, needle
    data = json.loads((tmp_path / "probe" / "kernel-probe-report.json").read_text())
    assert data["header"] == header and data["attempts"] == []


def _row(op, n, passed, hw=None, category=None):
    return {"op": op, "attempt": n, "status": "done", "turns": 5, "agent_wall_s": 12.5, "passed_sim": passed,
            "category": category, "first_line": "" if passed else "set seed0: pcc 0.2", "pcc_min": 0.9999 if passed else 0.2,
            "sets": [], "hw": hw, "agent_check_untrusted": False, "provenance": {}}


def test_write_report_counts_attempts_to_first_pass_per_op(tmp_path):
    attempts = [_row("silu_mul", 1, False, category="numeric_mismatch"), _row("silu_mul", 2, False, category="tt_lang_error"),
                _row("silu_mul", 3, True, hw={"passed": True, "category": None, "pcc_min": 0.995}),
                _row("rmsnorm", 1, False, category="forbidden_code"), _row("rmsnorm", 2, False, category="timeout")]
    attempts[1]["agent_check_untrusted"] = True
    kp.write_report(tmp_path, attempts, {"tt_lang": FACTS, "coder_model": "c", "probe_id": "t", "bars": {}, "hw": None})
    md = (tmp_path / "kernel-probe-report.md").read_text()
    silu, rms = md.split("## silu_mul")[1].split("## rmsnorm")
    assert "attempts to first pass in the simulator: 3" in silu and "attempts to first pass on hardware: 3" in silu
    assert "none of 2 attempts passed in the simulator" in rms
    assert "| 3 | yes | yes |" in silu and "| 1 | no | - |" in rms
    assert "forbidden_code" in rms and "untrusted" in silu.split("| 2 |")[1].splitlines()[0]
    data = json.loads((tmp_path / "kernel-probe-report.json").read_text())
    assert data["first_pass"] == {"silu_mul": {"sim": 3, "hw": 3}, "rmsnorm": {"sim": None, "hw": None}}


def test_the_category_list_includes_forbidden_code_and_hang():
    assert {"no_kernel", "import_error", "tt_lang_error", "shape_mismatch", "numeric_mismatch", "timeout",
            "compile_error", "hang", "hw_numeric_mismatch", "harness_error", "forbidden_code"} == set(kp.CATEGORIES)


# ---- the probe-owned verdict directory (fix round 1) ------------------------------------------------------------


NO_KERNEL = {"passed": False, "category": "no_kernel", "first_line": "kernel.py does not exist", "sets": []}


def agent_does(action):
    """A scripted agent whose first turn runs `action()` in the test process (standing in for a shell command the
    agent ran), then ends."""
    def script(request):
        if turn(request) == 0:
            action()
            return call("shell", command="ls")
        return final("done")
    return script


def test_a_symlinked_attempt_json_is_replaced_and_its_target_left_alone(tmp_path, fake_kenv):
    operator_file = tmp_path / "operator.txt"
    operator_file.write_text("the operator's file\n")
    adir = tmp_path / "probe" / "add" / "attempt-1"
    with FakeModel(agent_does(lambda: (adir / "attempt.json").symlink_to(operator_file))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL))
    assert operator_file.read_text() == "the operator's file\n"
    assert json.loads((kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text()) == attempt
    assert not (adir / "attempt.json").is_symlink() and json.loads((adir / "attempt.json").read_text()) == attempt


def test_a_symlinked_kernel_py_is_forbidden_and_nothing_runs(tmp_path, fake_kenv):
    secret = tmp_path / "secret.txt"
    secret.write_text("import ttl\nTOKEN = 'hf_secretvalue'\n")
    adir = tmp_path / "probe" / "add" / "attempt-1"
    calls = []
    with FakeModel(agent_does(lambda: (adir / "kernel.py").symlink_to(secret))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["passed_sim"] is False and attempt["category"] == "forbidden_code"
    assert "regular file" in attempt["first_line"]
    assert "hf_secretvalue" not in (kp.verdict_dir(tmp_path / "probe", "add", 1) / "attempt.json").read_text()
    assert not (kp.verdict_dir(tmp_path / "probe", "add", 1) / kp.CLEAN_DIR / "kernel.py").exists()


def test_an_oversized_kernel_py_is_forbidden_and_nothing_runs(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    calls = []
    big = "# padding\n" * (kp.MAX_KERNEL_BYTES // 10 + 10)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text(big))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["category"] == "forbidden_code" and "256 KiB" in attempt["first_line"]


def test_a_missing_kernel_py_is_no_kernel_and_nothing_runs(tmp_path, fake_kenv):
    calls = []
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    assert calls == [] and attempt["category"] == "no_kernel"


def test_a_planted_authoritative_symlink_does_not_crash_and_the_check_uses_the_probe_owned_copy(tmp_path, fake_kenv):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    adir = tmp_path / "probe" / "add" / "attempt-1"

    def plant():
        (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text())
        (adir / "authoritative").symlink_to(elsewhere, target_is_directory=True)
    calls = []
    with FakeModel(agent_does(plant)) as server:
        run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL, calls))
    clean = Path(calls[0][calls[0].index("--attempt-dir") + 1])
    assert clean == kp.verdict_dir(tmp_path / "probe", "add", 1) / kp.CLEAN_DIR
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"]


def test_a_check_result_rewritten_in_the_attempt_directory_is_not_read(tmp_path, fake_kenv):
    """A background process the agent left behind writes a passing check-authoritative.json into its directory after
    the shipped check wrote the real one in the probe-owned directory."""
    adir = tmp_path / "probe" / "add" / "attempt-1"
    real = {"passed": False, "category": "numeric_mismatch", "first_line": "set seed0: pcc 0.1", "sets": []}

    def fake(argv, **kw):
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(real))
        (adir / "check-authoritative.json").write_text(json.dumps({"passed": True, "category": None, "sets": []}))
        return kp.Watched(1, None, False, 0.1)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text()))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=fake)
    assert attempt["passed_sim"] is False and attempt["category"] == "numeric_mismatch"


def test_the_hardware_phase_refuses_a_clean_copy_that_changed(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    (kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py").write_text(
        (KERNELS / "add_good.py").read_text() + "# changed\n")
    calls = []
    hw = kp.hardware_phase(adir, attempt, env=fake_kenv, shared_python="/p", run_watched=hw_writer([GOOD_HW_SET], calls))
    assert calls == [] and hw["passed"] is False and hw["category"] == "harness_error"
    assert "the clean kernel copy changed" in hw["first_line"]


def test_load_attempts_reads_the_probe_owned_records(tmp_path, fake_kenv):
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=check_writer(NO_KERNEL))
    (tmp_path / "probe" / "add" / "attempt-1" / "attempt.json").write_text(json.dumps(dict(attempt, passed_sim=True)))
    assert kp.load_attempts(tmp_path / "probe", ops=["add"], probe_id="t") == ([attempt], 0)


def test_a_smoke_test_killed_by_the_watcher_blocks_the_probe(tmp_path, fake_kenv):
    def hangs(argv, **kw):
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    with pytest.raises(kp.ProbeBlocked) as info:
        kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=hangs)
    text = str(info.value)
    assert "the smoke test hung" in text and "stalled" in text and "gozer-gatekeeper" in text


def test_a_smoke_test_that_exits_non_zero_is_unavailable_without_blocking(tmp_path, fake_kenv):
    def fails(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write("PASSED!\nthen a crash\n")
        return kp.Watched(1, None, False, 3.0)
    ok, tail = kp.smoke_test(env=fake_kenv, shared_python="/p", run_watched=fails)
    assert ok is False and "then a crash" in tail


def test_the_skill_states_every_rule_the_gate_enforces():
    """The skill's rules are compared with the gate's own tables, so a rule added to the gate fails this test until the
    skill names it. Each REMEDIES rule needs its word; every name the gate refuses or allows by name must appear."""
    gate = kp._template("kernel_check")
    words = {"import": "import only", "name": "getattr", "dunder": "double underscores",
             "readback": "to_torch", "operation": "OPERATION"}
    assert set(words) == set(gate.REMEDIES)
    flat = " ".join(kp.SKILL.read_text().split())
    for rule, word in words.items():
        assert word in flat, rule
    named = (gate.FORBIDDEN_NAMES | gate.CALL_ONLY_NAMES | gate.READBACK_NAMES | gate.ALLOWED_IMPORTS
             | gate.HARMLESS_FROM_MODULES | gate.HARMLESS_DUNDER_NAMES)
    for name in sorted(named):
        if gate._is_dunder(name) and name not in gate.HARMLESS_DUNDER_NAMES:
            continue                                      # covered by the double-underscore rule
        assert re.search(rf"`[^`]*\b{re.escape(name)}\b[^`]*`", flat), name
    # the star-import refusal (its remedy text lives in the gate's import branch)
    assert "import *" in gate.Path(gate.__file__).read_text() and "`from X import *`" in flat
    assert "`__name__` and `__all__`" in flat and "from __future__" in flat
    # the file rule belongs to the probe, which applies it when it takes kernel.py
    assert "The probe also refuses" in flat
    assert len(kp.SKILL.read_text().splitlines()) <= 55


def test_the_report_header_says_the_wall_budget_can_overrun_by_one_tool_timeout(tmp_path, fake_kenv):
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], tt_lang=FACTS)
    kp.write_report(tmp_path, [], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "checked between turns" in md and f"{int(kp.AGENT_TOOL_TIMEOUT_S)} s" in md
    assert kp.AGENT_TOOL_TIMEOUT_S == 420


def test_a_stale_verdict_from_an_earlier_run_is_not_read(tmp_path, fake_kenv):
    """The verdict directory is made fresh for each attempt: a passing result left by an earlier run is gone before
    the check runs, so a check that writes nothing is a harness_error."""
    vdir = kp.verdict_dir(tmp_path / "probe", "add", 1)
    vdir.mkdir(parents=True)
    (vdir / "check-authoritative.json").write_text(json.dumps({"passed": True, "category": None, "sets": []}))
    adir = tmp_path / "probe" / "add" / "attempt-1"

    def writes_nothing(argv, **kw):
        return kp.Watched(1, None, False, 0.1)
    with FakeModel(agent_does(lambda: (adir / "kernel.py").write_text((KERNELS / "add_good.py").read_text()))) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=writes_nothing)
    assert attempt["passed_sim"] is False and attempt["category"] == "harness_error"


# ---- fix round 2: records of this probe only, capped regular-file reads, fresh attempt directories ---------------

import stat as _stat
import threading


def _record(probe_dir, op, n, probe_id, **extra):
    vdir = kp.verdict_dir(probe_dir, op, n)
    vdir.mkdir(parents=True, exist_ok=True)
    rec = {"op": op, "attempt": n, "passed_sim": True, "category": None, "hw": None, "turns": 1, "agent_wall_s": 1.0,
           "pcc_min": 0.9999, "first_line": "", "agent_check_untrusted": False,
           "provenance": {"probe_id": probe_id}, **extra}
    (vdir / "attempt.json").write_text(json.dumps(rec))
    return rec


def test_a_stale_record_from_a_longer_earlier_run_is_not_in_the_new_report(tmp_path, fake_kenv):
    probe = tmp_path / "probe"
    _record(probe, "add", 5, "old-run")                    # an earlier 5-attempt run left add-5 behind
    with FakeModel(lambda r: final("nothing")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, probe_id="new-run", run_watched=check_writer(NO_KERNEL))
    got, ignored = kp.load_attempts(probe, ops=["add"], probe_id="new-run")
    assert got == [attempt] and ignored == 1
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="new-run", ops=["add"], tt_lang=FACTS,
                              ignored_records=ignored)
    data = kp.write_report(probe, got, header)
    assert data["first_pass"]["add"] == {"sim": None, "hw": None}
    md = (probe / "kernel-probe-report.md").read_text()
    assert "1 unrelated or unreadable records ignored" in md and "| 5 |" not in md


def test_planted_and_malformed_records_are_ignored(tmp_path):
    probe = tmp_path / "probe"
    good = _record(probe, "add", 1, "p")
    _record(probe, "add", 99, "someone-else")             # another probe id
    _record(probe, "add", 7, "p", attempt=8)              # the number does not match its directory
    _record(probe, "add", 3, "p", attempt="3")            # not an int: must not make the sort raise
    _record(probe, "add", 4, "p", attempt=True)           # a bool is not an attempt number
    _record(probe, "rmsnorm", 1, "p")                     # an op this probe did not run
    (probe / ".verdict" / "add-x").mkdir()
    (probe / ".verdict" / "add-0").mkdir()
    (probe / ".verdict" / "notes").mkdir()
    # links whose targets hold records that would otherwise match their directory names
    outside = tmp_path / "outside"
    _record(outside, "add", 6, "p")
    _record(outside, "add", 2, "p")
    linked = probe / ".verdict" / "add-6"
    linked.mkdir()
    (linked / "attempt.json").symlink_to(kp.verdict_dir(outside, "add", 6) / "attempt.json")
    (probe / ".verdict" / "add-2").symlink_to(kp.verdict_dir(outside, "add", 2), target_is_directory=True)
    got, ignored = kp.load_attempts(probe, ops=["add"], probe_id="p")
    assert got == [good] and ignored == 10                  # every entry above but add-1


def test_a_bool_attempt_number_is_ignored(tmp_path):
    """True == 1 in Python, so the number check alone would accept it in add-1."""
    _record(tmp_path / "probe", "add", 1, "p", attempt=True)
    assert kp.load_attempts(tmp_path / "probe", ops=["add"], probe_id="p") == ([], 1)


def test_a_reused_attempt_directory_starts_empty(tmp_path, fake_kenv):
    adir = tmp_path / "add" / "attempt-1"
    adir.mkdir(parents=True)
    (adir / "kernel.py").write_text("# left by an earlier run\n")
    (adir / "check.json").write_text(json.dumps({"passed": True}))
    kp.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=FACTS)
    assert not (adir / "kernel.py").exists() and not (adir / "check.json").exists()
    assert (adir / "kernel_spec.json").is_file()


def test_an_attempt_directory_that_is_a_link_is_replaced_and_its_target_left_alone(tmp_path, fake_kenv):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    (tmp_path / "add").mkdir()
    (tmp_path / "add" / "attempt-1").symlink_to(elsewhere, target_is_directory=True)
    kp.prepare_attempt(tmp_path / "add" / "attempt-1", op="add", env=fake_kenv, tt_lang=FACTS)
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"]
    assert not (tmp_path / "add" / "attempt-1").is_symlink()


def _in_thread(fn, timeout=20):
    """Run fn in a thread; fail (without hanging the suite) if it has not returned within `timeout` seconds."""
    box = {}

    def target():
        box["value"] = fn()
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    return th.is_alive(), box.get("value")


def _unblock(fifo):
    """Open the FIFO's write end once, so a reader blocked on it gets EOF (cleanup after a failing run)."""
    try:
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    except OSError:
        pass


def test_read_regular_refuses_links_and_fifos_and_caps_its_read(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"x" * 10)
    assert kp._read_regular(f, 100) == b"x" * 10
    assert kp._read_regular(f, 4) == b"xxxxx"             # limit + 1 bytes: the caller sees it is too large
    link = tmp_path / "link"
    link.symlink_to(f)
    assert kp._read_regular(link, 100) is None
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    try:
        hung, value = _in_thread(lambda: kp._read_regular(fifo, 100), timeout=10)
    finally:
        _unblock(fifo)
    assert not hung and value is None


def test_agent_check_json_that_is_a_fifo_or_a_link_is_untrusted_and_never_blocks(tmp_path):
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"passed": False, "pad": "x" * (2 * 1024 * 1024)}))
    cases = {}
    fifo = tmp_path / "fifo-check.json"
    os.mkfifo(fifo)
    try:
        cases["fifo"] = _in_thread(lambda: kp._read_agent_check(fifo), timeout=10)
    finally:
        _unblock(fifo)
    for name, target in (("big", big),):
        link = tmp_path / f"{name}-check.json"
        link.symlink_to(target)
        cases[name] = _in_thread(lambda link=link: kp._read_agent_check(link), timeout=10)
    oversized = tmp_path / "oversized-check.json"
    oversized.write_bytes(big.read_bytes())
    cases["oversized"] = _in_thread(lambda: kp._read_agent_check(oversized), timeout=10)
    for name, (hung, value) in cases.items():
        assert not hung, name
        assert value == "unreadable", (name, value)
        assert kp.agent_check_untrusted(value, {"passed": False}, "add") is True
    assert kp._read_agent_check(tmp_path / "missing.json") is None


def test_agent_check_json_linked_to_dev_zero_is_refused_without_reading_it(tmp_path):
    """Run in a child process with a 2 GiB address-space limit and a timeout: a reader that follows the link and reads
    to the end would exhaust memory, and the limit turns that into a MemoryError in the child alone."""
    import subprocess
    import sys
    link = tmp_path / "check.json"
    link.symlink_to("/dev/zero")
    code = ("import resource, sys; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30)); "
            f"sys.path.insert(0, {str(REPO)!r}); from pathlib import Path; from orchard import kernelprobe as kp; "
            f"print(kp._read_agent_check(Path({str(link)!r})))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "unreadable"


def test_a_fifo_check_json_left_by_the_agent_does_not_block_the_attempt(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "add" / "attempt-1"
    with FakeModel(agent_does(lambda: os.mkfifo(adir / "check.json"))) as server:
        try:
            hung, attempt = _in_thread(lambda: run_attempt(tmp_path, fake_kenv, server,
                                                           run_watched=check_writer(NO_KERNEL)), timeout=30)
        finally:
            _unblock(adir / "check.json")
    assert not hung and attempt["agent_check_untrusted"] is True


def _lstat_says_regular(monkeypatch, path):
    """os.lstat reports a regular file for `path`, as if it changed between the lstat and the open."""
    real = os.lstat

    def fake(p, *a, **kw):
        st = real(p, *a, **kw)
        if Path(p) == path:
            return os.stat_result((_stat.S_IFREG | 0o644,) + tuple(st)[1:6] + (10,) + tuple(st)[7:])
        return st
    monkeypatch.setattr(kp.os, "lstat", fake)


def test_copy_kernel_refuses_a_link_even_when_lstat_was_fooled(tmp_path, monkeypatch):
    """The read-side layer alone: the open does not follow the link."""
    secret = tmp_path / "secret.py"
    secret.write_text("import ttl\n")
    adir = tmp_path / "a"
    adir.mkdir()
    (adir / "kernel.py").symlink_to(secret)
    clean = tmp_path / "clean"
    clean.mkdir()
    _lstat_says_regular(monkeypatch, adir / "kernel.py")
    category, sha, line = kp.copy_kernel(adir, clean)
    assert category == "forbidden_code" and sha is None and not (clean / "kernel.py").exists()


def test_copy_kernel_refuses_a_fifo_even_when_lstat_was_fooled(tmp_path, monkeypatch):
    """The read-side layer alone: fstat on the open file says it is not a regular file."""
    adir = tmp_path / "a"
    adir.mkdir()
    os.mkfifo(adir / "kernel.py")
    clean = tmp_path / "clean"
    clean.mkdir()
    _lstat_says_regular(monkeypatch, adir / "kernel.py")
    try:
        hung, got = _in_thread(lambda: kp.copy_kernel(adir, clean), timeout=10)
    finally:
        _unblock(adir / "kernel.py")
    assert not hung and got[0] == "forbidden_code" and not (clean / "kernel.py").exists()


def test_a_hung_smoke_test_names_its_last_lines_and_the_report_records_the_hang(tmp_path, fake_kenv):
    def hangs(argv, **kw):
        with open(kw["log_path"], "a") as f:
            f.write("opening device 0\nwaiting for ARC\n")
        return kp.Watched(-9, "stalled: no output and no CPU use for 60 s", True, 70.0)
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    with pytest.raises(kp.ProbeBlocked) as info:
        kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p", run_watched=hangs)
    assert "waiting for ARC" in str(info.value)
    hw = info.value.hw
    assert hw["available"] is False and hw["hung"] is True and "waiting for ARC" in hw["smoke_tail"]
    header = kp.report_header(env=fake_kenv, coder_model="c", probe_id="t", ops=["add"], hw=hw, tt_lang=FACTS)
    kp.write_report(tmp_path, [attempt], header)
    md = (tmp_path / "kernel-probe-report.md").read_text()
    assert "hw: blocked" in md and "the smoke test hung" in md and "waiting for ARC" in md
