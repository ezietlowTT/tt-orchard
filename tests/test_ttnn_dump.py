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
