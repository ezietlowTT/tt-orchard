# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The humanizer case against the real hub, GitHub and tt-model. Opt in with ORCHARD_LIVE_RESEARCH=1.
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
    assert any("gemma" in n.lower() for n in names), (names, r["unavailable"])
    assert "gemma-4-12b" in names[0].lower(), names             # the -it bundle for the same 12B architecture ranks first
    assert r["stated_bases"] and "gemma" in r["stated_bases"][0].lower()
    assert r["selected"] is None            # nothing is installed in this test
