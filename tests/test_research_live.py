# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The humanizer case against the real hub and tt-model (GitHub searches are off: no_github=True). Opt in with ORCHARD_LIVE_RESEARCH=1.
It makes read-only requests and writes nothing outside pytest's tmp_path."""
import os

import pytest

from orchard import nearest, preflight as pf
from orchard.research import report
from orchard.research.http import Fetcher

pytestmark = pytest.mark.skipif(os.environ.get("ORCHARD_LIVE_RESEARCH") != "1",
                                reason="set ORCHARD_LIVE_RESEARCH=1 to query the real hub and GitHub")


def hub_info(repo):
    import json
    import urllib.request
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true", headers={"User-Agent": "tt-orchard"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return pf.parse_hub(json.load(resp)), None
    except OSError as exc:
        return None, str(exc)


def test_the_humanizer_gets_a_ranked_answer_with_the_gemma_bundles(tmp_path):
    model = "jialinyyzz/humanizer"
    hub, err = hub_info(model)
    assert hub is not None, err
    r = report.build(model_id=model, hub=hub, base_override=None, installed=[], search=nearest.search,
                     hub_info=hub_info, fetch=Fetcher(tmp_path / "cache"), no_github=True)
    names = [c["name"] for c in r["bundle_choices"]]
    # Structure: these hold whatever the hub currently lists.
    for key in ("model", "stated_bases", "candidates", "bundle_choices", "selected", "ports", "evidence", "unavailable"):
        assert key in r, key
    assert r["bundle_choices"], r["unavailable"]
    groups = [c["group"] for c in r["bundle_choices"]]
    assert groups == sorted(groups), groups
    assert all(c["confidence"] in ("stated", "inferred", "weak") for c in r["bundle_choices"])
    assert r["selected"] is None            # nothing is installed in this test
    # Canary: the humanizer case must still find a Gemma bundle.
    assert any("gemma" in n.lower() for n in names), (names, r["unavailable"])
    # Pinned to the hub as of 2026-10-12. It may drift when bundles are published or renamed;
    # when it does, check that the new first result is right and update this line.
    assert "gemma-4-12b" in names[0].lower(), names
