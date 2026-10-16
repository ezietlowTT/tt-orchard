# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard kernel-env: the tt-lang simulator environment. A fake runner stands in for git, uv and the venv's
python, and the files live in tmp_path, so nothing here touches the network, the shared venv or a device."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchard import defaults, ttlang_env as ke

DIAG = b"# fake diagnostics\n"        # the sha256 of this stands in for the pinned file


class FakeRun:
    """Records every call. A rule is (argv prefix, (rc, out, err), effect); the first matching rule answers.
    An argv with no rule answers 127, so a command the test did not expect is visible."""

    def __init__(self):
        self.calls, self.rules = [], []

    def on(self, prefix, result=(0, "", ""), effect=None):
        self.rules.append((tuple(prefix), result, effect))

    def __call__(self, argv, timeout, env=None):
        self.calls.append((list(argv), dict(env or {})))
        for prefix, result, effect in self.rules:
            if tuple(argv[:len(prefix)]) == prefix:
                if effect:
                    effect(argv)
                return result
        return 127, "", f"no rule for {argv}"

    def argvs(self):
        return [a for a, _ in self.calls]

    def changed_something(self):
        """True when any call could change the machine: git init/remote/fetch/checkout, uv venv/build/pip."""
        for a in self.argvs():
            if a[0] == "git" and any(w in a for w in ("init", "remote", "fetch", "checkout")):
                return True
            if a[0] == "uv":
                return True
        return False


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", ke.hashlib.sha256(DIAG).hexdigest())
    return ke.KEnv(src_dir=tmp_path / "src" / "tt-lang", venv_dir=tmp_path / "venv", scratch_dir=tmp_path / "scratch",
                   run=FakeRun(), environ={"PATH": "/usr/bin"})


def test_check_only_never_changes_the_machine(env):
    results = ke.ensure(env, check_only=True, say=lambda s: None)
    assert not env.run.changed_something()
    assert results[0].name == "clone" and results[0].status == ke.TODO


def test_a_clone_at_another_commit_fails_and_is_left_alone(env):
    (env.src_dir / ".git").mkdir(parents=True)
    env.run.on(["git", "-C", str(env.src_dir), "rev-parse", "HEAD"], (0, "deadbeef" * 5 + "\n", ""))
    results = ke.ensure(env, say=lambda s: None)
    assert results[-1].status == ke.FAIL and "is not reset" in results[-1].detail
    assert not env.run.changed_something()


def test_a_directory_that_is_not_a_checkout_fails_and_is_left_alone(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / "notes.txt").write_text("mine")
    results = ke.ensure(env, say=lambda s: None)
    assert results[-1].status == ke.FAIL and "not a git checkout" in results[-1].detail
    assert not env.run.changed_something()
    assert (env.src_dir / "notes.txt").read_text() == "mine"


def test_verify_needs_the_named_text_and_not_just_a_failing_exit(env):
    """The wheel bug ends upstream's negative examples with a non-zero exit and FileNotFoundError. That must not pass."""
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / "bin").mkdir()
    env.sim.write_text("")
    env.run.on([str(env.sim)], (1, "", "FileNotFoundError: [Errno 2] ... ttl/ttl/diagnostics.py"))
    got = ke.verify(env)
    assert got.status == ke.FAIL


NEEDLES = [n for _, ns in defaults.TTLANG_NEGATIVE_EXAMPLES for n in ns]


def _sim_env(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / "bin").mkdir()
    env.sim.write_text("")


def test_verify_passes_when_every_named_text_is_present_and_the_exit_is_non_zero(env):
    _sim_env(env)
    env.run.on([str(env.sim)], (1, "\n".join(NEEDLES), ""))
    assert ke.verify(env).status == ke.OK


def test_verify_fails_when_the_text_is_present_but_filenotfounderror_is_too(env):
    _sim_env(env)
    env.run.on([str(env.sim)], (1, "\n".join(NEEDLES), "FileNotFoundError: diagnostics.py"))
    got = ke.verify(env)
    assert got.status == ke.FAIL and "FileNotFoundError" in got.detail


def _diag_setup(env, tmp_path, content):
    _sim_env(env)
    site = tmp_path / "site"
    site.mkdir()
    src = env.src_dir / "python" / "ttl"
    src.mkdir(parents=True)
    (src / "diagnostics.py").write_bytes(content)
    env.run.on([str(env.py), "-c"], (0, f"{site}\n", ""))
    return site


def test_the_diagnostics_copy_is_refused_when_the_pinned_file_changed(env, tmp_path):
    site = _diag_setup(env, tmp_path, b"# someone edited this\n")
    got = ke.step_diagnostics(env, False)
    assert got.status == ke.FAIL
    assert not (site / "ttl" / "ttl" / "diagnostics.py").exists()


def test_the_diagnostics_copy_lands_at_site_packages_ttl_ttl(env, tmp_path):
    site = _diag_setup(env, tmp_path, DIAG)
    got = ke.step_diagnostics(env, False)
    assert got.status == ke.OK
    assert (site / "ttl" / "ttl" / "diagnostics.py").read_bytes() == DIAG


def _at_pin(env):
    (env.src_dir / ".git").mkdir(parents=True)
    env.run.on(["git", "-C", str(env.src_dir), "rev-parse", "HEAD"], (0, defaults.TTLANG_PIN + "\n", ""))


def test_the_wheel_is_built_with_the_version_override_in_its_environment(env):
    _at_pin(env)
    want = ke.expected_version()

    def build(argv):
        wheels = env.scratch_dir / "wheels"
        wheels.mkdir(parents=True, exist_ok=True)
        (wheels / f"tt_lang_sim-{want}-py3-none-any.whl").write_text("")

    env.run.on(["uv", "build"], effect=build)
    env.run.on(["uv", "pip", "install"])
    ke.step_wheel(env, False)
    builds = [e for a, e in env.run.calls if a[:2] == ["uv", "build"]]
    assert builds and builds[0]["TTLANG_VERSION_OVERRIDE"] == "1.1.10+g9b7ad24" == want


def test_torch_comes_from_the_cpu_index_before_the_wheel(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / "bin").mkdir()
    env.py.write_text("")
    env.run.on(["uv"])
    ke.step_torch(env, False)
    ke.step_wheel(env, False)
    argvs = env.run.argvs()
    torch_i = next(i for i, a in enumerate(argvs) if a[:3] == ["uv", "pip", "install"] and "torch" in a)
    t = argvs[torch_i]
    assert "--index-url" in t and ke.TORCH_CPU_INDEX in t
    wheel_i = [i for i, a in enumerate(argvs) if a[:2] == ["uv", "build"]]
    assert all(torch_i < i for i in wheel_i)


def test_ensure_stops_at_the_first_failure(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / "notes.txt").write_text("mine")
    results = ke.ensure(env, say=lambda s: None)
    assert results[-1].status == ke.FAIL
    assert not any(a[0] == "uv" for a in env.run.argvs())


def test_a_second_run_does_nothing_when_everything_is_in_place(env, tmp_path):
    site = _diag_setup(env, tmp_path, DIAG)
    (site / "ttl" / "ttl").mkdir(parents=True)
    (site / "ttl" / "ttl" / "diagnostics.py").write_bytes(DIAG)
    (env.src_dir / ".git").mkdir()
    env.py.write_text("")
    env.run.on(["git", "-C", str(env.src_dir), "rev-parse", "HEAD"], (0, defaults.TTLANG_PIN + "\n", ""))
    env.run.on([str(env.py), "-c", "import torch"])
    env.run.on([str(env.sim), "--version"], (0, "tt-lang-sim " + ke.expected_version() + "\n", ""))
    env.run.on([str(env.sim)], (1, "\n".join(NEEDLES), ""))
    results = ke.ensure(env, say=lambda s: None)
    assert [r.status for r in results] == [ke.OK] * 6
    assert not env.run.changed_something()
