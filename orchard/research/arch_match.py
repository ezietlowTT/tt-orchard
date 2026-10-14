# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Does a candidate's architecture match the new model's? Compares two config.json files with the stage 0
template's decisive keys, so the answer follows the same rules the later stages use."""
from __future__ import annotations

from orchard.research import triage_rules
from orchard.research.types import INFERRED, WEAK

CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"     # read by report._config


def family(cfg: dict) -> str:
    """`gemma4_unified` and `gemma4` are one family: the model_type up to its first underscore."""
    return str(cfg.get("model_type") or "").split("_")[0]


def compare(new_cfg: dict, cand_cfg: dict) -> dict:
    rules = triage_rules.load()
    same = bool(family(new_cfg)) and family(new_cfg) == family(cand_cfg)
    differing = {}
    for key in rules.DECISIVE:
        a, b = rules.effective(new_cfg, key), rules.effective(cand_cfg, key)
        if a != b:
            differing[key] = {"new": a, "candidate": b}
    if not same:
        confidence = None
    else:
        confidence = INFERRED if not differing else WEAK
    return {"confidence": confidence, "differing": differing, "same_family": same}
