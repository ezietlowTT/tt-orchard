# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research package reuses the stage 0 template's rules instead of copying them."""
import pytest

pytest.importorskip("tokenizers", reason="delta_triage.py is imported by the template tests the same way")

from orchard.research import triage_rules  # noqa: E402

CFG = {"model_type": "qwen3", "num_hidden_layers": 4, "hidden_size": 64, "num_attention_heads": 4,
       "num_key_value_heads": 2, "vocab_size": 100}


def tensors(**over):
    base = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [100, 64], "shard": "m.safetensors"},
            "model.layers.0.q.weight": {"dtype": "BF16", "shape": [64, 64], "shard": "m.safetensors"}}
    base.update(over)
    return base


def estimate(mcfg, bcfg, mt, bt):
    rules = triage_rules.load()
    c = rules.compare_config_dicts(mcfg, bcfg, [], "new/config.json", "base/config.json")
    t = rules.compare_tensor_maps(mt, bt, mshards=["m.safetensors"], bshards=["m.safetensors"], merr=[], berr=[],
                                  model_path="new", base_path="base")
    reasons = rules.decide_path(c, t)
    return rules.decide_class(c, t, reasons, {"problems": [], "sidecars": []}), reasons


def test_the_loader_returns_one_cached_module_with_the_rules():
    rules = triage_rules.load()
    assert rules is triage_rules.load()
    for name in ("DECISIVE", "effective", "text_part", "decide_path", "decide_class", "compare_tensor_maps",
                 "compare_config_dicts", "BACKBONE_SHARD", "MAX_HEADER"):
        assert hasattr(rules, name), name


def test_equal_configs_and_tensors_are_weights_only():
    cls, reasons = estimate(CFG, dict(CFG), tensors(), tensors())
    assert cls == "weights-only" and reasons == []


def test_a_changed_shape_is_full_port():
    bad = tensors(**{"model.layers.0.q.weight": {"dtype": "BF16", "shape": [64, 32], "shard": "m.safetensors"}})
    cls, reasons = estimate(CFG, dict(CFG), bad, tensors())
    assert cls == "full-port" and any("shape" in r for r in reasons)


def test_a_changed_decisive_config_value_is_full_port():
    cls, reasons = estimate({**CFG, "num_hidden_layers": 8}, dict(CFG), tensors(), tensors())
    assert cls == "full-port" and any("num_hidden_layers" in r for r in reasons)


def test_no_shared_tensor_is_unknown():
    cls, _ = estimate(CFG, dict(CFG), tensors(), {"other.weight": {"dtype": "BF16", "shape": [1], "shard": "x"}})
    assert cls == "unknown"


def test_a_header_that_could_not_be_read_is_unknown():
    rules = triage_rules.load()
    c = rules.compare_config_dicts(CFG, dict(CFG), [], "n", "b")
    t = rules.compare_tensor_maps(tensors(), tensors(), mshards=["m"], bshards=["m"], merr=["m: cut short"],
                                  berr=[], model_path="n", base_path="b")
    assert rules.decide_class(c, t, rules.decide_path(c, t), {"problems": [], "sidecars": []}) == "unknown"
