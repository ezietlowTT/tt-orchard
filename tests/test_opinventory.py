# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import opinventory as oi

FIXTURES = Path(__file__).parent / "fixtures" / "ops-inventory"
MAPPING = {"aten.mm.default": {"ttnn": "matmul"}, "aten.embedding.default": {"ttnn": "embedding"},
           "aten.add.Tensor": {"ttnn": "add"}}
DECOMPS = {"aten.addmm.default": ["aten.mm.default", "aten.add.Tensor"],
           "aten.bad.default": ["aten.mm.default", "aten.weird_scan.default"]}
DUMP = {"ttnn_version": "0.65.1", "names": ["matmul", "embedding", "add"]}


def op(name, count, in_layers, modules=("model.layers.0.x",)):
    return {"op": name, "count": count, "count_in_layers": in_layers, "sample": None, "modules": list(modules)}


def raw(**over):
    base = {"schema": 1, "model_dir": "/m", "architectures": ["XForCausalLM"], "traced_class": "XForCausalLM",
            "layers_in_config": 24, "layers_traced": 2, "seq_len": 8, "torch": "2", "transformers": "5", "failures": [],
            "ops": [op("aten.mm.default", 10, 8), op("aten.embedding.default", 1, 0, ("model.embed",)),
                    op("aten.weird_scan.default", 3, 3), op("aten.addmm.default", 2, 2), op("aten.bad.default", 1, 1)]}
    base.update(over)
    return base


def rows(inv):
    return {r["op"]: r for r in inv["ops"]}


def test_a_name_in_the_matching_dump_is_mapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["status"] == "mapped" and rows(inv)["aten.mm.default"]["ttnn"] == "matmul"


@pytest.mark.parametrize("dump,installed", [(None, "0.65.1"), (DUMP, "0.70.0"), (DUMP, None)],
                         ids=["no dump", "dump for another version", "installed version unknown"])
def test_nothing_is_claimed_without_a_dump_for_the_installed_version(dump, installed):
    """Review focus 2."""
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn=installed)
    assert {r["status"] for r in inv["ops"]} == {"unchecked"}
    assert inv["ttnn_dump"] is None and inv["ttnn_dump_note"]


def test_an_op_with_no_mapping_and_no_decomposition_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.weird_scan.default"]["status"] == "unmapped"


def test_a_decomposition_with_every_leaf_mapped_is_composable():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    r = rows(inv)["aten.addmm.default"]
    assert r["status"] == "composable" and r["leaves"] == ["aten.mm.default", "aten.add.Tensor"]


def test_a_decomposition_with_an_unmapped_leaf_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.bad.default"]["status"] == "unmapped"


def test_a_decomposition_cycle_does_not_loop():
    cyc = {"aten.a.default": ["aten.b.default"], "aten.b.default": ["aten.a.default"]}
    assert oi.status_of("aten.a.default", {}, cyc, {"matmul"})[0] == "unmapped"


def test_counts_in_layers_are_scaled_to_the_real_depth():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 8 * 12 + 2      # 8 in layers x (24 / 2) + 2 outside
    assert rows(inv)["aten.embedding.default"]["scaled_count"] == 1


def test_without_a_layer_count_nothing_is_scaled_and_the_report_says_so():
    """Review focus 5."""
    inv = oi.build(raw(layers_in_config=None, layers_traced=None), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                   installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 10 and "not scaled" in inv["scaling"]


def test_a_skipped_trace_gives_an_inventory_with_the_reason_and_no_ops():
    inv = oi.build({"schema": 1, "model_dir": "/m", "architectures": [], "skipped": "needs remote code"},
                   mapping=MAPPING, decomps=DECOMPS, dump=DUMP)
    assert inv["skipped"] == "needs remote code" and inv["ops"] == []


def test_against_marks_ops_the_base_model_also_uses():
    base = oi.build(raw(ops=[op("aten.mm.default", 1, 1)]), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                    installed_ttnn="0.65.1")
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1", against=base)
    assert rows(inv)["aten.mm.default"]["in_base"] is True and rows(inv)["aten.weird_scan.default"]["in_base"] is False


def test_the_summary_counts_ops_by_status():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert inv["summary"]["distinct"] == {"mapped": 2, "composable": 1, "unmapped": 2, "unchecked": 0}


def test_a_dump_version_that_is_a_path_is_refused(tmp_path):
    assert oi.load_dump("../x", data_dir=tmp_path) is None and oi.load_dump("", data_dir=tmp_path) is None


def test_a_version_cannot_reach_a_file_outside_the_data_directory(tmp_path):
    """The path resolves to a real, well-formed dump outside data_dir; only the version check stops it."""
    data = tmp_path / "data"
    (data / "ttnn_ops-x").mkdir(parents=True)
    evil = "x/../../evil"
    (tmp_path / "evil.json").write_text(json.dumps({"ttnn_version": evil, "names": ["add"]}))
    assert oi.load_dump(evil, data_dir=data) is None


def test_load_dump_needs_the_version_inside_the_file_to_match(tmp_path):
    (tmp_path / "ttnn_ops-0.65.1.json").write_text(json.dumps({"ttnn_version": "0.9", "names": ["add"]}))
    assert oi.load_dump("0.65.1", data_dir=tmp_path) is None


def test_a_table_entry_without_a_ttnn_name_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"aten.x.default": {"nope": 1}}))
    with pytest.raises(ValueError):
        oi.load_table(p)


def test_every_table_entry_names_an_op_that_a_real_trace_contains():
    """A typo in a table would never match and would quietly leave an op unmapped."""
    seen = set()
    for f in FIXTURES.glob("raw-*.json"):
        seen |= {o["op"] for o in json.loads(f.read_text()).get("ops", [])}
    table = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    unknown = sorted(k for k in table if k not in seen)
    assert not unknown, f"not in any fixture trace: {unknown}"


def test_the_markdown_says_what_mapped_means_and_names_the_dump():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    md = oi.render_md(inv)
    assert "name exists in the dump of ttnn 0.65.1" in md and "does not say" in md


# ---- run_inventory, with a fake command runner ----

def fake_run(template_rc, version="0.65.1", fixture="raw-qwen3-tiny.json"):
    calls = []

    def run(argv, timeout):
        calls.append(argv)
        if "--out" in argv:                       # the template call
            if template_rc in (0, 3):
                Path(argv[argv.index("--out") + 1]).write_text((FIXTURES / fixture).read_text())
            return template_rc, "", "boom" if template_rc not in (0, 3) else ""
        return 0, version + "\n", ""              # the metadata call
    run.calls = calls
    return run


def data_dir_without_dump(tmp_path):
    """A data directory with the shipped tables and no ttnn dump."""
    d = tmp_path / "data"
    d.mkdir()
    for name in ("aten_to_ttnn.json", "decompositions.json"):
        (d / name).write_text((oi.DATA_DIR / name).read_text())
    return d


def test_run_inventory_writes_both_files(tmp_path):
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=fake_run(0),
                          data_dir=data_dir_without_dump(tmp_path), say=said.append)
    assert rc == 0
    assert (tmp_path / "o" / "ops-inventory.json").is_file() and (tmp_path / "o" / "ops-inventory.md").is_file()
    inv = json.loads((tmp_path / "o" / "ops-inventory.json").read_text())
    assert {r["status"] for r in inv["ops"]} == {"unchecked"}      # no dump in data_dir
    assert any("distinct ops by status" in s for s in said)


def test_run_inventory_returns_2_when_the_template_exits_4(tmp_path):
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=fake_run(4),
                          data_dir=data_dir_without_dump(tmp_path), say=said.append)
    assert rc == 2 and not (tmp_path / "o" / "ops-inventory.json").exists()
    assert any("did not run" in s and "boom" in s for s in said)


def test_run_inventory_passes_the_class_and_sizes_to_the_template(tmp_path):
    run = fake_run(0)
    oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path, layers=3, seq=16, cls="XCls", run=run,
                     data_dir=data_dir_without_dump(tmp_path), say=lambda s: None)
    argv = run.calls[0]
    assert argv[argv.index("--class") + 1] == "XCls" and argv[argv.index("--layers") + 1] == "3" \
        and argv[argv.index("--seq") + 1] == "16"
