# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Research for a bring-up: which models could a new model be based on, and what already exists for them.

Each module reads one kind of source and returns evidence or an explicit "unavailable". `report.py` ranks
the candidates. Nothing here decides with a model, and nothing here writes outside the run directory.
Design: docs/superpowers/specs/2026-10-12-base-research-design.md."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Options:
    """How the research runs. `cache_dir` None keeps no cache (a dry run)."""
    cache_dir: Path | None = None
    offline: bool = False
    no_github: bool = False
    refresh: bool = False
