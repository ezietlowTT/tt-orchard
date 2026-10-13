# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 0 template's rules, loaded by path.

The class estimate must follow the same rules stage 0 applies (equal decisive config values, equal shapes
and dtypes for shared tensors, the explained-extra tensors, the mtp allowance). The template is a script
agents copy, so it cannot import from this package; this package loads it instead of copying its rules."""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "skills" / "delta-triage-templates" / "delta_triage.py"


@lru_cache(maxsize=1)
def load():
    spec = importlib.util.spec_from_file_location("orchard_delta_triage_rules", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
