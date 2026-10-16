from __future__ import annotations

import hashlib
import json

import pytest

from orchard import ttnn_dump as td


def test_the_dump_runs_under_a_one_chip_lease_and_never_directly():
    argv = td.dump_command("/py/bin/python")
    assert argv[:2] == ["gozer", "run"] and "--chips" in argv and argv[argv.index("--chips") + 1] == "1"
    assert "--who" in argv and argv[argv.index("--who") + 1].startswith("claude:")
    assert argv[argv.index("--") + 1] == "/py/bin/python"


def test_parse_reads_the_marked_line_and_ignores_the_rest():
    out = "noise\nORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.65.1", "names": ["add", "matmul"]}) + "\nmore\n"
    assert td.parse(out) == {"ttnn_version": "0.65.1", "names": ["add", "matmul"]}


def test_parse_without_the_marker_raises():
    with pytest.raises(ValueError):
        td.parse("Traceback ...")


def test_write_dump_names_the_file_by_version_and_records_a_sha256_of_the_names(tmp_path):
    p = td.write_dump({"ttnn_version": "0.65.1", "names": ["matmul", "add"]}, tmp_path)
    data = json.loads(p.read_text())
    assert p.name == "ttnn_ops-0.65.1.json" and data["names"] == ["add", "matmul"]      # sorted, no duplicates
    assert data["sha256"] == hashlib.sha256(json.dumps(["add", "matmul"]).encode()).hexdigest()


def test_write_dump_refuses_a_version_that_is_a_path(tmp_path):
    with pytest.raises(ValueError):
        td.write_dump({"ttnn_version": "../evil", "names": ["x"]}, tmp_path)


def test_main_prints_the_command_before_running_it_and_writes_the_file(tmp_path):
    said, calls = [], []

    def run(argv, timeout):
        calls.append(argv)
        return 0, "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.65.1", "names": ["add"]}) + "\n", ""

    code = td.main("/py/bin/python", run=run, data_dir=tmp_path, say=said.append)
    assert code == 0 and (tmp_path / "ttnn_ops-0.65.1.json").is_file()
    assert any("gozer run" in line for line in said[:1]), "the command is shown before it runs"


def test_main_reports_a_failed_dump_and_writes_nothing(tmp_path):
    code = td.main("/py/bin/python", run=lambda a, t: (1, "", "RuntimeError: no device"), data_dir=tmp_path, say=lambda s: None)
    assert code == 2 and not list(tmp_path.glob("ttnn_ops-*"))


def test_a_failed_exit_writes_nothing_even_when_the_output_holds_a_dump(tmp_path):
    """A lease timeout or a crash after the print must not leave a file behind."""
    marked = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.65.1", "names": ["add"]}) + "\n"
    code = td.main("/py/bin/python", run=lambda a, t: (1, marked, "crashed on exit"), data_dir=tmp_path, say=lambda s: None)
    assert code == 2 and not list(tmp_path.glob("ttnn_ops-*"))


# ---------------------------------------------------------------- fix round 1

import importlib.metadata
import os
import sys
import types


def test_parse_finds_a_marker_after_banner_text_on_the_same_line():
    out = "ttnn banner no newline" + "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "1", "names": ["a"]}) + "\n"
    assert td.parse(out) == {"ttnn_version": "1", "names": ["a"]}


def test_parse_takes_the_last_marker_line():
    first = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "1", "names": ["old"]})
    last = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "2", "names": ["new"]})
    assert td.parse(first + "\n" + last + "\n")["names"] == ["new"]


def _run_script(monkeypatch, capsys):
    """Exec the dump script against a fake ttnn. The real ttnn is never imported."""
    class Fake(types.ModuleType):
        @property
        def lazy_broken(self):
            raise RuntimeError("binding error")

        def __dir__(self):                            # a module's dir() lists only its dict; show the property too
            return list(self.__dict__) + ["lazy_broken"]

    def make_ns(name, *fns):
        ns = types.ModuleType(name)
        for f in fns:
            setattr(ns, f, lambda *a: None)
        ns.SomeClass = type("SomeClass", (), {})
        return ns

    fake = Fake("ttnn")
    fake.matmul = lambda *a: None
    fake.add = len                                   # a builtin is a callable too
    fake.Tensor = type("Tensor", (), {})              # a class: not an op
    fake.bfloat16 = object()                          # dtype-like and uncallable
    fake._private = lambda: None
    fake.experimental = make_ns("experimental", "rotary", "_hidden")
    fake.transformer = make_ns("transformer", "sdpa")
    fake.other = make_ns("other", "x")                # a namespace that is not walked
    monkeypatch.setitem(sys.modules, "ttnn", fake)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    exec(compile(td.SCRIPT, "<dump>", "exec"), {"__name__": "dump"})
    return capsys.readouterr().out


def test_the_script_lists_ops_namespaced_sorted_and_skips_what_raises(monkeypatch, capsys):
    out = _run_script(monkeypatch, capsys)
    assert "ttnn" in sys.modules or True
    dump = td.parse(out)
    assert dump["ttnn_version"] == "9.9.9"
    assert dump["names"] == ["add", "experimental.rotary", "matmul", "transformer.sdpa"]
    assert dump["skipped"] == ["lazy_broken"]


def test_the_script_leaves_classes_out(monkeypatch, capsys):
    names = td.parse(_run_script(monkeypatch, capsys))["names"]
    assert "Tensor" not in names and "experimental.SomeClass" not in names and "transformer.SomeClass" not in names


def test_the_script_prints_its_marker_on_a_line_of_its_own(monkeypatch, capsys):
    out = _run_script(monkeypatch, capsys)
    assert out.startswith("\n") and out.splitlines()[-1].startswith("ORCHARD_TTNN_DUMP ")


def test_write_dump_leaves_no_final_file_when_the_replace_fails(tmp_path, monkeypatch):
    def boom(a, b):
        raise OSError("disk gone")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        td.write_dump({"ttnn_version": "1.0", "names": ["a"]}, tmp_path)
    assert not (tmp_path / "ttnn_ops-1.0.json").exists()


def test_write_dump_creates_a_missing_directory(tmp_path):
    p = td.write_dump({"ttnn_version": "1.0", "names": ["a"]}, tmp_path / "new" / "dir")
    assert p.is_file()


def test_main_returns_2_when_the_directory_cannot_be_written(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")                           # a file where the directory should be
    said = []
    out = "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "1.0", "names": ["a"]}) + "\n"
    code = td.main("/py/bin/python", run=lambda a, t: (0, out, ""), data_dir=blocker / "sub", say=said.append)
    assert code == 2 and any("could not" in s for s in said)


def test_the_printed_command_shows_the_interpreter_and_not_the_script(tmp_path):
    said = []
    td.main("/py/bin/python", run=lambda a, t: (1, "", "x"), data_dir=tmp_path, say=said.append)
    assert "/py/bin/python" in said[0] and "import ttnn" not in said[0] and said[0].endswith("-c <script>")


def test_a_queued_lease_is_reported_as_queued(tmp_path):
    said = []
    code = td.main("/py/bin/python", run=lambda a, t: (10, "", "queued ticket 7"), data_dir=tmp_path, say=said.append)
    assert code == 2 and any("queued" in s and "gozer wait" in s for s in said) and not any("dump failed" in s for s in said)
