"""merge_lora.py: W' = W + strength * (alpha / rank) * B @ A, written into a copy of the model directory.

Each test builds a tiny two-shard model and an adapter, runs the script as a subprocess (as the stage
skill does) and reads the output back. The expected numbers are computed here with plain loops, not
with the script's own matmul.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

SCRIPT = Path(__file__).resolve().parent.parent / "orchard/skills/lora-merge-templates/merge_lora.py"
OUT_F, IN_F, RANK = 4, 3, 2


def run(*argv):
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True)


def build_base(root):
    comp = root / "language_model"
    comp.mkdir(parents=True)
    w0 = torch.arange(OUT_F * IN_F, dtype=torch.float32).reshape(OUT_F, IN_F).to(torch.bfloat16)
    w1 = torch.ones(OUT_F, IN_F).to(torch.bfloat16)
    st.save_file({"model.layers.0.mlp.up_proj.weight": w0, "model.norm.weight": torch.ones(2)},
                 str(comp / "a.safetensors"))
    st.save_file({"model.layers.1.mlp.up_proj.weight": w1}, str(comp / "b.safetensors"))
    (comp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.mlp.up_proj.weight": "a.safetensors", "model.norm.weight": "a.safetensors",
        "model.layers.1.mlp.up_proj.weight": "b.safetensors"}}))
    (root / "config.json").write_text("{}")
    return w0, w1


def build_adapter(path, layers=(0,), prefix="language_model."):
    a = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.0]])
    state = {}
    for n in layers:
        stem = f"{prefix}model.layers.{n}.mlp.up_proj"
        state[stem + ".lora_A.weight"] = a.clone()      # safetensors refuses tensors that share memory
        state[stem + ".lora_B.weight"] = b.clone()
    st.save_file(state, str(path))
    return a, b


def flat(rows):
    return [x for row in rows for x in row]


def expected(w, a, b, scale):
    out = [[0.0] * IN_F for _ in range(OUT_F)]
    for i in range(OUT_F):
        for j in range(IN_F):
            out[i][j] = float(w[i][j]) + scale * sum(float(b[i][r]) * float(a[r][j]) for r in range(RANK))
    return out


@pytest.fixture
def setup(tmp_path):
    base = tmp_path / "base"
    w0, w1 = build_base(base)
    a, b = build_adapter(tmp_path / "lora.safetensors")
    return tmp_path, base, w0, w1, a, b


def merge(tmp, *extra):
    return run("--base", str(tmp / "base"), "--lora", str(tmp / "lora.safetensors"), "--out", str(tmp / "out"), *extra)


def read(tmp, shard):
    return st.load_file(str(tmp / "out/language_model" / shard))


def test_merged_matrix_equals_w_plus_scaled_ba(setup):
    tmp, base, w0, _, a, b = setup
    r = merge(tmp, "--alpha", "4", "--strength", "0.5")      # scale = 0.5 * 4 / 2 = 1.0
    assert r.returncode == 0, r.stderr
    got = read(tmp, "a.safetensors")["model.layers.0.mlp.up_proj.weight"]
    want = expected(w0.tolist(), a.tolist(), b.tolist(), 1.0)
    assert flat(got.tolist()) == pytest.approx(flat(want), abs=0.1)       # bf16 storage
    assert got.dtype == torch.bfloat16


def test_alpha_defaults_to_rank_and_strength_scales_the_change(setup):
    tmp, base, w0, _, a, b = setup
    assert merge(tmp, "--strength", "2").returncode == 0      # alpha = rank = 2, scale = 2.0
    got = read(tmp, "a.safetensors")["model.layers.0.mlp.up_proj.weight"]
    assert flat(got.tolist()) == pytest.approx(flat(expected(w0.tolist(), a.tolist(), b.tolist(), 2.0)), abs=0.1)


def test_untouched_tensors_and_shards_are_unchanged_and_linked(setup):
    tmp, base, w0, w1, _, _ = setup
    assert merge(tmp).returncode == 0
    assert read(tmp, "a.safetensors")["model.norm.weight"].tolist() == [1.0, 1.0]
    b_out = tmp / "out/language_model/b.safetensors"           # holds no adapted tensor
    assert b_out.is_symlink() and os.path.realpath(b_out) == os.path.realpath(base / "language_model/b.safetensors")
    assert (tmp / "out/config.json").is_symlink()
    assert not (tmp / "out/language_model/a.safetensors").is_symlink()   # rewritten, so a real file


def test_report_records_counts(setup):
    tmp, *_ = setup
    assert merge(tmp).returncode == 0
    rep = json.loads((tmp / "out/MERGE.json").read_text())
    assert rep["modules_merged"] == 1 and rep["shards_rewritten"] == ["a.safetensors"]
    assert rep["max_relative_change"] > 0


def test_module_with_no_base_tensor_is_refused(setup):
    tmp, *_ = setup
    build_adapter(tmp / "lora.safetensors", layers=(0, 7))     # layer 7 does not exist
    r = merge(tmp)
    assert r.returncode == 2 and "layers.7.mlp.up_proj" in r.stderr
    assert not (tmp / "out").exists() or not any((tmp / "out").iterdir())


def test_shape_mismatch_is_refused(setup):
    tmp, base, *_ = setup
    comp = base / "language_model"
    st.save_file({"model.layers.1.mlp.up_proj.weight": torch.ones(5, IN_F).to(torch.bfloat16)}, str(comp / "b.safetensors"))
    build_adapter(tmp / "lora.safetensors", layers=(1,))
    r = merge(tmp)
    assert r.returncode == 2 and "does not fit" in r.stderr


def test_wrong_prefix_is_refused(setup):
    tmp, *_ = setup
    build_adapter(tmp / "lora.safetensors", prefix="transformer.")
    r = merge(tmp)
    assert r.returncode == 2 and "--prefix" in r.stderr


def test_lone_lora_a_is_refused(setup):
    tmp, *_ = setup
    st.save_file({"language_model.model.layers.0.mlp.up_proj.lora_A.weight": torch.ones(RANK, IN_F)},
                 str(tmp / "lora.safetensors"))
    r = merge(tmp)
    assert r.returncode == 2 and "only one of" in r.stderr


def test_non_empty_output_is_refused(setup):
    tmp, *_ = setup
    (tmp / "out").mkdir()
    (tmp / "out/keep.txt").write_text("x")
    r = merge(tmp)
    assert r.returncode == 2 and "not empty" in r.stderr
    assert (tmp / "out/keep.txt").read_text() == "x"
