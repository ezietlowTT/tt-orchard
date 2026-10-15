#!/usr/bin/env python3
"""Merge a LoRA adapter into one component of a Hugging Face model directory.

Orchard's bring-up stages serve full weights, so a LoRA must become a full model first. This script
does that one job: for every adapted matrix W it writes W' = W + strength * (alpha / rank) * (B @ A),
computed in float32 and stored back in the base dtype. Only the shards that hold an adapted tensor are
rewritten. Every other file in the output is a symlink to the base snapshot, so the output is small.

It refuses (exit 2) when the merge would be a guess:
  * an adapter module has no matching base tensor;
  * a shape does not fit (A is rank x in, B is out x rank, W is out x in);
  * an A without a B, or the adapter holds a tensor the script does not understand;
  * the output directory exists and is not empty;
  * --component leaves the base, or the weight index names a shard outside the component;
  * a shard's destination is a link (it would be the base model's own file).

Usage:
  merge_lora.py --base SNAPSHOT_DIR --lora FILE.safetensors --out DIR \
      [--component language_model] [--prefix language_model.] [--alpha 64] [--strength 1.0]

--component is a path inside the base ("." for weights at the root) that holds the adapted weights and its
model.safetensors.index.json (or a single model.safetensors). --prefix is what the adapter's keys
start with before the base tensor name; --alpha defaults to the rank (scale 1.0), and
the adapter's author states it, so pass it when known.
Writes MERGE.json in the output (what was merged, counts and the largest relative change).
"""
import argparse
import json
import os
import sys
from pathlib import Path

A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def fail(msg):
    print(f"merge_lora: {msg}", file=sys.stderr)
    sys.exit(2)


def read_adapter(lora_path, prefix):
    """Return {base tensor name (without .weight): (A, B)} from an adapter file."""
    from safetensors.torch import load_file
    state = load_file(str(lora_path))
    mods = {}
    for key, value in state.items():
        if key.endswith(A_SUFFIX):
            side, stem = 0, key[:-len(A_SUFFIX)]
        elif key.endswith(B_SUFFIX):
            side, stem = 1, key[:-len(B_SUFFIX)]
        else:
            fail(f"adapter tensor {key!r} is neither lora_A nor lora_B")
        if not stem.startswith(prefix):
            fail(f"adapter tensor {key!r} does not start with --prefix {prefix!r}")
        slot = mods.setdefault(stem[len(prefix):], [None, None])
        slot[side] = value
    for name, (a, b) in mods.items():
        if a is None or b is None:
            fail(f"adapter module {name!r} has only one of lora_A and lora_B")
        if a.shape[0] != b.shape[1]:
            fail(f"adapter module {name!r}: A {tuple(a.shape)} and B {tuple(b.shape)} disagree on the rank")
    return {n: (ab[0], ab[1]) for n, ab in mods.items()}


def shard_map(component_dir):
    """{tensor name: shard file name} for a component directory."""
    index = component_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = component_dir / "model.safetensors"
    if not single.exists():
        fail(f"{component_dir} has neither model.safetensors.index.json nor model.safetensors")
    from safetensors import safe_open
    with safe_open(str(single), "pt") as f:
        return {k: single.name for k in f.keys()}


def link_tree(base, out, skip):
    """Symlink every file under base into out, except the paths in skip (relative strings)."""
    for root, dirs, files in os.walk(base, followlinks=True):
        rel = Path(root).relative_to(base)
        (out / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            if (rel / name).as_posix() in skip:          # skip holds normalized paths (see main)
                continue
            os.symlink(os.path.realpath(Path(root) / name), out / rel / name)


def normalize_component(raw: str) -> str:
    """The component as a clean path inside the base ("." for root-level weights). Refuses a path that leaves the base."""
    if os.path.isabs(raw):
        fail(f"--component {raw!r} must be a path inside the base, not an absolute path")
    rel = os.path.normpath(raw or ".")
    if rel == ".." or rel.startswith(".." + os.sep):
        fail(f"--component {raw!r} must be a path inside the base")
    return rel


def check_shard_name(shard: str, component: str) -> None:
    """A shard named by the weight index must be a plain relative path inside the component."""
    norm = os.path.normpath(shard)
    if os.path.isabs(shard) or norm == ".." or norm.startswith(".." + os.sep):
        fail(f"the weight index names {shard!r}, which is outside {component}")


def write_shard(dest: Path, tensors: dict) -> None:
    """Write one shard. Never through a link: a link here would be the base model's own file."""
    from safetensors.torch import save_file
    if dest.is_symlink():
        fail(f"{dest} is a link; refusing to write through it to the base weights")
    save_file(tensors, str(dest), metadata={"format": "pt"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--component", default="language_model")
    ap.add_argument("--prefix", default="language_model.")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args(argv)

    import torch
    from safetensors.torch import load_file

    base, out = Path(args.base), Path(args.out)
    component = normalize_component(args.component)
    comp = base / component
    if not comp.is_dir():
        fail(f"{comp} is not a directory")
    if out.exists() and any(out.iterdir()):
        fail(f"{out} exists and is not empty")

    adapter = read_adapter(args.lora, args.prefix)
    wmap = shard_map(comp)
    by_shard, targets = {}, {}
    for mod in adapter:
        name = mod + ".weight"
        if name not in wmap:
            fail(f"adapter module {mod!r} has no base tensor {name!r} in {component}")
        check_shard_name(wmap[name], component)
        targets[mod] = name
        by_shard.setdefault(wmap[name], []).append(mod)

    rewritten = {os.path.normpath(os.path.join(component, s)) for s in by_shard}
    link_tree(base, out, skip=rewritten)
    # the component directory is real in out (link_tree made it); shards below are written, not linked
    worst = 0.0
    for shard, mods in sorted(by_shard.items()):
        tensors = load_file(str(comp / shard))
        for mod in mods:
            a, b = adapter[mod]
            name = targets[mod]
            w = tensors[name]
            rank = a.shape[0]
            if w.shape != (b.shape[0], a.shape[1]):
                fail(f"{name}: base {tuple(w.shape)} does not fit B {tuple(b.shape)} @ A {tuple(a.shape)}")
            alpha = args.alpha if args.alpha is not None else float(rank)
            delta = (b.float() @ a.float()) * (args.strength * alpha / rank)
            merged = w.float() + delta
            worst = max(worst, float(delta.norm() / (w.float().norm() + 1e-12)))
            tensors[name] = merged.to(w.dtype)
        write_shard(out / component / shard, tensors)
        print(f"merged {len(mods)} matrices into {os.path.normpath(os.path.join(component, shard))}", flush=True)

    (out / "MERGE.json").write_text(json.dumps({
        "base": str(base), "lora": str(args.lora), "component": component, "strength": args.strength,
        "alpha": args.alpha, "modules_merged": len(adapter), "shards_rewritten": sorted(by_shard),
        "max_relative_change": worst,
    }, indent=2))
    print(f"done: {len(adapter)} matrices, largest ||delta|| / ||W|| = {worst:.4f}")


if __name__ == "__main__":
    main()
