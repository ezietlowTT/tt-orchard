# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ops_inventory.py, the template `tt-orchard inventory` runs in the reference interpreter. Run end to end on a tiny
config; nothing is downloaded and no device is opened (the model lives on the meta device)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "ops-inventory-templates" / "ops_inventory.py"
QWEN3_TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 128, "hidden_size": 32,
              "intermediate_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
              "head_dim": 16, "max_position_embeddings": 64}


def stack_python():
    """An interpreter with torch and transformers: $ORCHARD_REFERENCE_PYTHON, else this one, else None."""
    for cand in (os.environ.get("ORCHARD_REFERENCE_PYTHON"), sys.executable):
        if cand and subprocess.run([cand, "-c", "import torch, transformers"], capture_output=True).returncode == 0:
            return cand
    return None


PY = stack_python()
needs_stack = pytest.mark.skipif(PY is None, reason=(
    "SKIPPED: no interpreter with torch and transformers (set ORCHARD_REFERENCE_PYTHON), so the ops-inventory "
    "template was not run end to end"))


def run_template(tmp_path, config: dict, *extra):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    out = tmp_path / "raw.json"
    done = subprocess.run([PY or sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(out), *extra],
                          capture_output=True, text=True)
    return done, (json.loads(out.read_text()) if out.exists() else None)


def test_a_config_with_auto_map_is_skipped_and_no_code_runs(tmp_path):
    """Review focus 1. A repo's own modeling code is never imported. This test needs no torch."""
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "auto_map": {"AutoModelForCausalLM": "modeling_x.X"}})
    assert done.returncode == 0 and "remote code" in raw["skipped"] and "ops" not in raw


def test_a_missing_config_exits_2(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    done = subprocess.run([sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(tmp_path / "o.json")],
                          capture_output=True, text=True)
    assert done.returncode == 2 and "config.json" in done.stderr


@needs_stack
def test_a_class_the_installed_transformers_lacks_is_skipped(tmp_path):
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "architectures": ["NoSuchForCausalLM"]})
    assert done.returncode == 0 and "NoSuchForCausalLM" in raw["skipped"]


@needs_stack
def test_a_tiny_qwen3_traces_and_reports_layers_and_ops(tmp_path):
    done, raw = run_template(tmp_path, QWEN3_TINY, "--layers", "2")
    assert done.returncode == 0, done.stderr
    assert raw["layers_in_config"] == 4 and raw["layers_traced"] == 2 and raw["traced_class"] == "Qwen3ForCausalLM"
    ops = {o["op"]: o for o in raw["ops"]}
    assert len(ops) >= 10 and not raw["failures"]
    assert any(o["count_in_layers"] > 0 for o in ops.values())
    assert any(o["count_in_layers"] == 0 for o in ops.values()), "embedding and the final norm are outside the layers"
    assert all(0 <= o["count_in_layers"] <= o["count"] for o in ops.values())


def test_reduce_layers_reports_none_when_the_config_has_no_layer_count():
    """Review focus 5. The template must not invent a layer count. No torch needed."""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("ops_inventory_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.reduce_layers(types.SimpleNamespace(hidden_size=8), 2) == (None, None)
    cfg = types.SimpleNamespace(num_hidden_layers=24)
    assert mod.reduce_layers(cfg, 2) == (24, 2) and cfg.num_hidden_layers == 2
    assert mod.reduce_layers(types.SimpleNamespace(num_hidden_layers=1), 2) == (1, 1)


@needs_stack
def test_an_export_failure_is_recorded_and_exits_3(tmp_path):
    """--seq 0 makes the export fail on a zero-length input. The template must record the failure and exit 3 without a traceback."""
    done, raw = run_template(tmp_path, QWEN3_TINY, "--seq", "0")
    if done.returncode == 0:
        pytest.skip("this transformers traced a zero-length input; pick another way to force a failure")
    assert done.returncode == 3 and raw["failures"] and raw["failures"][0]["stage"] in ("export", "build")


def test_the_template_never_trusts_remote_code():
    """Source-level guard. A trust_remote_code=True cannot be seen at run time, so look for it in the text."""
    assert "trust_remote_code=True" not in TEMPLATE.read_text()


FIXTURES = REPO / "tests" / "fixtures" / "ops-inventory"


@pytest.mark.parametrize("name", ["qwen3-tiny", "rwkv", "mamba2", "moshi-lm"])
def test_each_recorded_fixture_has_the_schema_keys(name):
    raw = json.loads((FIXTURES / f"raw-{name}.json").read_text())
    assert raw["schema"] == 1 and "traced_class" in raw and "failures" in raw
    assert "ops" in raw or "skipped" in raw
    assert raw["model_dir"] == f"<fixture {name}>"
    for op in raw.get("ops", []):
        assert {"op", "count", "count_in_layers", "sample", "modules"} <= set(op)
