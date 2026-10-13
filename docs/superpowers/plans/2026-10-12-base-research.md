# Base Model Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before a bring-up downloads anything large, find and rank the models it could be based on, estimate the class of work, and find existing ports, from the Hugging Face hub, the tt-model catalog and GitHub.

**Architecture:** A new `orchard/research/` package of small standard-library collectors. Each returns evidence or an explicit "unavailable". A scorer in `report.py` ranks the candidates and writes `base-candidates.json`. `preflight.check_base` uses the report, and stage 0 reads it as an input. No model takes part in the research.

**Tech Stack:** Python 3.12 standard library (`urllib`, `json`, `struct`, `dataclasses`), pytest. The existing `orchard.scrub`, `orchard.nearest` and `delta_triage.py` template.

**Spec:** `docs/superpowers/specs/2026-10-12-base-research-design.md` (amended in Task 0 below).

## Global Constraints

- Standard library only. No new dependency.
- Hub calls carry no token: no `Authorization` header, ever (the existing rule that `bringup` never uses the operator's HF token).
- Hosts: `huggingface.co`, `api.github.com`, `raw.githubusercontent.com`, and CDN redirects whose host ends in `.hf.co` or `.huggingface.co`. Any other host, and any `http://` URL, is refused.
- Fetched text is untrusted: one line, control characters removed, at most 300 characters, passed through `orchard/scrub.py` (a hit replaces the text with `[removed: ...]`). It never reaches a command line or a prompt as an instruction.
- Ids taken from a model card, a search result or a file listing must match `valid_id` or `SAFE_FILE` before they enter a URL or a command line.
- Header reads cap at 100 MB per file (`delta_triage.MAX_HEADER`). A range request must be answered with status 206.
- Every source is optional: a failure yields `Unavailable(source, reason)`, never an exception out of `report.build` and never an empty result in its place.
- `--offline` makes no network call. A dry run makes the read-only research calls but keeps no cache and writes no file.
- The collectors write only inside the run directory (`<run_dir>/research/`). They never publish, push, pull or install.
- Every guard gets a mutation: remove it, watch a test go red, restore. Check the baseline is green before reading a mutation table. Clear `__pycache__` between mutation runs.
- Bump `orchard.__version__` and `pyproject.toml` together (0.4.0 to 0.5.0) in the last task. Log the work in `CLAUDE.md`.
- Prose for people (README, docs, the log, commit messages) follows the plain-English rules in the global CLAUDE.md.
- Branch `research-base-discovery`, in a worktree. One commit per task.

## Review Focus

- A card whose `base_model` is a hostile string (`x/../../y?z`, spaces, a newline): skipped with an `Unavailable`, never placed in a URL. (Task 2)
- A search term built from a model's `architectures` containing `repo:other/secret`: refused, so the GitHub query cannot be redirected. (Task 5)
- The hub returns the base model's own id as its base, or two models name each other: no loop, no duplicate rows. (Task 2)
- A shard name in a file listing that contains `/` or `..`: not fetched. (Task 4)
- A gated or private base (HTTP 401 or 403): shown as unavailable with the status, and the run still reaches a clear answer. (Tasks 1, 6)
- A resume with a cache: the report is the same as the first run's. (Tasks 1, 6)

---

## File Structure

| File | Responsibility |
|---|---|
| `orchard/research/__init__.py` | package docstring and the `Options` dataclass |
| `orchard/research/types.py` | `Evidence`, `Unavailable`, `Result`, confidence names, `clean`, `valid_id` |
| `orchard/research/http.py` | `Fetcher`: host allow-list, no token, size cap, ranges, cache, offline |
| `orchard/research/triage_rules.py` | loads the `delta_triage.py` template by path so its rules are not copied |
| `orchard/research/lineage.py` | the chain of base models and sibling fine-tunes from the hub |
| `orchard/research/arch_match.py` | compare a candidate's `config.json` with the new model's |
| `orchard/research/headers.py` | safetensors headers by range request, and the class estimate |
| `orchard/research/ports.py` | existing ports: catalog, tt-metal listings, GitHub issues and code, bundle code |
| `orchard/research/report.py` | rank candidates, write `base-candidates.json` and `.md` |
| `orchard/skills/delta-triage-templates/delta_triage.py` | split two functions so the rules take dictionaries; record `base_research` |
| `orchard/preflight.py` | `Signals.research`, `check_base(report=...)`, `default_signals(research_options=...)` |
| `orchard/cli.py` | `--offline`, `--no-github`, `--refresh-research`; write the report; pass it as an input |
| `orchard/skills/delta-triage.md` | tell the stage 0 agent about the report |
| `tests/test_research_*.py` | one test file per module |

---

### Task 0: Amend the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-12-base-research-design.md`

Four things learned while planning (checked on 2026-10-12) change the spec:

1. Hugging Face serves safetensors files through a signed redirect to `us.aws.cdn.hf.co`. "Refuse a redirect to another host" would make every header read fail. The allow-list gains `*.hf.co` and `*.huggingface.co`.
2. Published tt-model bundles carry `tt_kernel_manifest.json` at their root (`weights.repo_id`, `schema_version`) and tags (`thin` for v6, `tt-model-container` for 5.1). Some container bundles ship `code/models/autoports/<model>/`, which is direct evidence that a port exists (`stisiTT/gemma-4-12b-it-p150`, `tt-hous/gemma-4-26B-A4B-it_p150_p150x2_p150x4`).
3. A dry run is how the operator reads the ranked list, so it makes the read-only research calls. It keeps no cache and writes no file.
4. The class estimate can only say `weights-only`, `full-port` or `unknown`, plus a `sidecar_possible` flag when the new repo has safetensors files outside its index.

- [ ] **Step 1: Apply the edits**

In section 4 "Rules", replace the bullet beginning "Hosts are fixed" with:

```
- Hosts are fixed: `huggingface.co`, `api.github.com`, `raw.githubusercontent.com`. Hugging Face answers a
  file request with a signed redirect to a CDN host, so a redirect is followed only when its host ends in
  `.hf.co` or `.huggingface.co`. Any other host, and any `http://` URL, is refused.
```

In section 3.4 "Flags", replace the dry-run sentence with:

```
A dry run makes the same read-only research calls as a real run, because it is how the operator reads the
ranked list. It keeps no cache and writes no file. `--offline` makes no network call.
```

In section 3, after the `ports.py` row, add:

```
Published bundles carry `tt_kernel_manifest.json` at the root (its `weights.repo_id` says which model they
serve) and tags (`thin`: v6; `tt-model-container`: 5.1). A container bundle that lists
`code/models/autoports/<model>/` ships a port; `ports.py` reports it as evidence.
```

In section 6, add the bullet: `- The class estimate says weights-only, full-port or unknown, and flags sidecar_possible. It cannot see a sidecar's tensors without downloading them.`

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-10-12-base-research-design.md docs/superpowers/plans/2026-10-12-base-research.md
git commit -m "Plan: base model research; spec amended (CDN redirects, bundle manifests, dry run)"
```

---

### Task 1: Types and the guarded fetcher

**Files:**
- Create: `orchard/research/__init__.py`, `orchard/research/types.py`, `orchard/research/http.py`
- Test: `tests/test_research_http.py`

**Interfaces:**
- Produces (`types.py`): `STATED`, `INFERRED`, `WEAK` (strings), `CONFIDENCE_RANK: dict[str,int]`, `Evidence(source, claim, url, confidence, fetched_at)` with `.as_dict()`, `Unavailable(source, reason)` with `.as_dict()`, `Result(evidence, unavailable, data)` with `.extend(other)`, `clean(text, limit=300) -> str`, `valid_id(value) -> bool`.
- Produces (`http.py`): `FetchError(reason)`, `host_allowed(host) -> bool`, `Fetcher(cache_dir, *, opener=None, offline=False, refresh=False, now=None, timeout=20)` with `.get(url, *, byte_range=None, max_bytes=MAX_BODY) -> bytes`, `.json(url, **kw)`, `.now() -> str`, and `.offline`, `.requests` attributes.
- Produces (`__init__.py`): `Options(cache_dir, offline, no_github, refresh)` dataclass.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_http.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research fetcher (orchard/research/http.py): the one place the research package touches the network.

Every guard here is a rule from the design: no token, fixed hosts, a size cap, byte ranges answered with
206, a cache a resume reads, and no network at all when offline."""
import urllib.error
import urllib.request
from email.message import Message

import pytest

from orchard.research import types
from orchard.research.http import FetchError, Fetcher, _Redirects, host_allowed

HF = "https://huggingface.co/api/models/a/b"


class Resp:
    def __init__(self, body=b"{}", status=200):
        self.body, self.status, self.headers = body, status, Message()

    def read(self, n=-1):
        self.asked = n
        return self.body if n is None or n < 0 else self.body[:n]

    def close(self):
        pass


class FakeOpener:
    def __init__(self, routes):
        self.routes, self.requests = routes, []

    def open(self, req, timeout=None):
        self.requests.append(req)
        got = self.routes[req.full_url]
        if isinstance(got, Exception):
            raise got
        return got


def http_error(code, **headers):
    msg = Message()
    for key, value in headers.items():
        msg[key.replace("_", "-")] = value
    return urllib.error.HTTPError(HF, code, "x", msg, None)


def fetcher(tmp_path, routes, **kw):
    return Fetcher(tmp_path / "cache", opener=FakeOpener(routes), now=lambda: "2026-10-12T00:00:00Z", **kw)


def test_no_authorization_header_is_ever_sent_even_with_tokens_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 34)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "b" * 36)
    f = fetcher(tmp_path, {HF: Resp(b'{"id": "a/b"}')})
    assert f.json(HF) == {"id": "a/b"}
    sent = {k.lower() for k in f.opener.requests[0].headers}
    assert "authorization" not in sent and "x-api-key" not in sent


@pytest.mark.parametrize("url", ["https://evil.example/x", "http://huggingface.co/api/models/a/b",
                                 "https://huggingface.co.evil.example/x", "ftp://huggingface.co/x"])
def test_only_https_on_the_allowed_hosts_is_fetched(tmp_path, url):
    f = fetcher(tmp_path, {})
    with pytest.raises(FetchError, match="not allowed"):
        f.get(url)
    assert f.opener.requests == []


def test_host_allow_list():
    assert host_allowed("huggingface.co") and host_allowed("api.github.com")
    assert host_allowed("raw.githubusercontent.com") and host_allowed("us.aws.cdn.hf.co")
    assert not host_allowed("evilhf.co") and not host_allowed("hf.co.evil.example") and not host_allowed(None)


def test_a_redirect_to_another_host_is_refused_and_a_hub_cdn_redirect_is_followed():
    req = urllib.request.Request(HF)
    with pytest.raises(FetchError, match="redirect"):
        _Redirects().redirect_request(req, None, 302, "Found", Message(), "https://evil.example/x")
    ok = _Redirects().redirect_request(req, None, 302, "Found", Message(), "https://us.aws.cdn.hf.co/x?sig=1")
    assert ok.full_url.startswith("https://us.aws.cdn.hf.co/")


def test_a_range_must_be_answered_with_206(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"12345678", status=200)})
    with pytest.raises(FetchError, match="range"):
        f.get(HF, byte_range=(0, 7))


def test_a_range_is_sent_and_a_206_body_is_returned(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"12345678", status=206)})
    assert f.get(HF, byte_range=(0, 7)) == b"12345678"
    assert f.opener.requests[0].get_header("Range") == "bytes=0-7"


def test_a_body_over_the_cap_is_refused(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"x" * 100)})
    with pytest.raises(FetchError, match="larger"):
        f.get(HF, max_bytes=50)


def test_the_read_itself_is_bounded_by_the_cap(tmp_path):
    resp = Resp(b"x" * 10)
    f = fetcher(tmp_path, {HF: resp})
    f.get(HF, max_bytes=50)
    assert resp.asked == 51              # one byte over the cap is enough to know it was exceeded


def test_a_range_body_longer_than_the_range_is_refused(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"x" * 100, status=206)})
    with pytest.raises(FetchError, match="larger"):
        f.get(HF, byte_range=(0, 7))


def test_the_second_get_comes_from_the_cache_and_a_new_fetcher_reads_it_too(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b'{"n": 1}')})
    f.json(HF), f.json(HF)
    assert len(f.opener.requests) == 1
    g = fetcher(tmp_path, {HF: Resp(b'{"n": 2}')})
    assert g.json(HF) == {"n": 1} and g.opener.requests == []


def test_refresh_ignores_the_cache(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b'{"n": 1}')})
    f.json(HF)
    g = fetcher(tmp_path, {HF: Resp(b'{"n": 2}')}, refresh=True)
    assert g.json(HF) == {"n": 2}


def test_without_a_cache_directory_nothing_is_written(tmp_path):
    f = Fetcher(None, opener=FakeOpener({HF: Resp(b"{}")}))
    f.json(HF)
    assert list(tmp_path.iterdir()) == []


def test_offline_makes_no_request_even_when_the_cache_has_the_answer(tmp_path):
    fetcher(tmp_path, {HF: Resp(b"{}")}).json(HF)
    g = fetcher(tmp_path, {HF: Resp(b"{}")}, offline=True)
    with pytest.raises(FetchError, match="offline"):
        g.json(HF)
    assert g.opener.requests == []


@pytest.mark.parametrize("err,reason", [
    (http_error(401), "HTTP 401"), (http_error(404), "HTTP 404"), (http_error(429), "rate limited"),
    (http_error(403, x_ratelimit_remaining="0"), "rate limited"), (http_error(403), "HTTP 403")])
def test_http_errors_become_reasons(tmp_path, err, reason):
    f = fetcher(tmp_path, {HF: err})
    with pytest.raises(FetchError) as info:
        f.get(HF)
    assert reason in info.value.reason


def test_a_failure_is_not_cached(tmp_path):
    f = fetcher(tmp_path, {HF: http_error(500)})
    with pytest.raises(FetchError):
        f.get(HF)
    g = fetcher(tmp_path, {HF: Resp(b"{}")})
    assert g.json(HF) == {}


def test_a_body_that_is_not_json_is_a_reason_not_a_crash(tmp_path):
    f = fetcher(tmp_path, {HF: Resp(b"<html>")})
    with pytest.raises(FetchError, match="JSON"):
        f.json(HF)


def test_a_network_error_is_a_reason(tmp_path):
    f = fetcher(tmp_path, {HF: OSError("timed out")})
    with pytest.raises(FetchError, match="timed out"):
        f.get(HF)


# ---- types ---------------------------------------------------------------------------------------

def test_clean_makes_one_short_line_and_removes_a_home_path_or_token():
    assert types.clean("a\x00b\n  c") == "a b c"
    assert len(types.clean("x" * 1000)) == 300
    assert types.clean("see /home/someone/.cache/x").startswith("[removed:")
    assert types.clean("tok " + "hf_" + "a" * 34).startswith("[removed:")


@pytest.mark.parametrize("value,ok", [
    ("google/gemma-4-12B", True), ("Qwen/Qwen3.8-27B", True), ("a/b", True),
    ("x/../../y?z", False), ("a/b c", False), ("a/b\n", False), ("nogroup", False), ("a/b/c", False),
    ("", False), (None, False), (7, False), ("a/" + "b" * 200, False)])
def test_valid_id(value, ok):
    assert types.valid_id(value) is ok


def test_result_extend_merges_lists():
    a = types.Result(evidence=[1], unavailable=[2])
    a.extend(types.Result(evidence=[3], unavailable=[4]))
    assert a.evidence == [1, 3] and a.unavailable == [2, 4]
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_http.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'orchard.research'`.

- [ ] **Step 3: Write the code**

`orchard/research/__init__.py`:

```python
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
```

`orchard/research/types.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Shared shapes for the research collectors, and the two functions that make untrusted text safe to keep."""
from __future__ import annotations

import re
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchard.scrub import scrub_text

STATED, INFERRED, WEAK = "stated", "inferred", "weak"
CONFIDENCE_RANK = {STATED: 0, INFERRED: 1, WEAK: 2}
MAX_CLAIM = 300

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


@dataclass(frozen=True)
class Evidence:
    source: str          # "hub", "catalog", "tt-metal", "github", "bundle", "config", "headers"
    claim: str           # one cleaned line
    url: str
    confidence: str      # STATED, INFERRED or WEAK
    fetched_at: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Unavailable:
    source: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    evidence: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def extend(self, other: "Result") -> "Result":
        self.evidence += other.evidence
        self.unavailable += other.unavailable
        return self


def clean(text, limit: int = MAX_CLAIM) -> str:
    """Fetched text is untrusted data: one line, no control characters, a length limit, and nothing the
    scrub would refuse (a home path, a host name, a token). A hit replaces the whole text."""
    one = " ".join(_CONTROL.sub(" ", str(text)).split())[:limit]
    hits = scrub_text(one, hostname=socket.gethostname(), home=str(Path.home()))
    return f"[removed: {', '.join(hits)}]" if hits else one


def valid_id(value) -> bool:
    """A Hugging Face `org/name`. Ids from a card, a search or a listing are checked before they enter a URL."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None
```

`orchard/research/http.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research package's only network access.

Rules (design section 4): https only; fixed hosts, with CDN redirects allowed on `.hf.co` and
`.huggingface.co` because that is where Hugging Face serves file bytes; no Authorization header, so the
operator's token is never sent; a size cap on every body; a byte range must be answered with 206; successes
are cached by URL and range so a resume reads the same answers; `offline` makes no request at all."""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

ALLOWED_HOSTS = ("huggingface.co", "api.github.com", "raw.githubusercontent.com")
ALLOWED_SUFFIXES = (".hf.co", ".huggingface.co")
MAX_BODY = 8 * 1024 * 1024
TIMEOUT_S = 20


class FetchError(Exception):
    """A request that could not be answered. `reason` is the text the report shows."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def host_allowed(host) -> bool:
    return isinstance(host, str) and (host in ALLOWED_HOSTS or host.endswith(ALLOWED_SUFFIXES))


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an allowed host over https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != "https" or not host_allowed(parts.hostname):
            raise FetchError(f"redirect to {parts.hostname} refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_reason(exc: urllib.error.HTTPError) -> str:
    remaining = exc.headers.get("x-ratelimit-remaining") if exc.headers else None
    if exc.code == 429 or (exc.code == 403 and remaining == "0"):
        return "rate limited"
    if exc.code == 401:
        return "HTTP 401 (needs a login)"
    return f"HTTP {exc.code}"


class Fetcher:
    def __init__(self, cache_dir, *, opener=None, offline: bool = False, refresh: bool = False,
                 now=None, timeout: int = TIMEOUT_S):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.opener = opener or urllib.request.build_opener(_Redirects)
        self.offline, self.refresh, self.timeout = offline, refresh, timeout
        self._now = now or (lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def now(self) -> str:
        return self._now()

    # ---- cache -----------------------------------------------------------------------------------

    def _path(self, url: str, byte_range) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(f"{url}\n{byte_range}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str, byte_range) -> bytes | None:
        path = self._path(url, byte_range)
        if path is None or self.refresh or not path.is_file():
            return None
        try:
            return base64.b64decode(json.loads(path.read_text(encoding="utf-8"))["body"])
        except (OSError, ValueError, KeyError):
            return None

    def _write_cache(self, url: str, byte_range, body: bytes) -> None:
        path = self._path(url, byte_range)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "range": byte_range, "fetched_at": self.now(),
                                        "body": base64.b64encode(body).decode()}), encoding="utf-8")
        except OSError:
            pass                                    # a cache that cannot be written is only a missed cache

    # ---- requests --------------------------------------------------------------------------------

    def get(self, url: str, *, byte_range=None, max_bytes: int = MAX_BODY) -> bytes:
        parts = urlsplit(url)
        if parts.scheme != "https" or not host_allowed(parts.hostname):
            raise FetchError(f"{parts.scheme}://{parts.hostname} is not allowed")
        if self.offline:
            raise FetchError("offline")
        cached = self._read_cache(url, byte_range)
        if cached is not None:
            return cached
        headers = {"User-Agent": "tt-orchard", "Accept": "*/*"}      # no Authorization, ever
        limit = max_bytes
        if byte_range is not None:
            start, end = byte_range
            headers["Range"] = f"bytes={start}-{end}"
            limit = min(max_bytes, end - start + 1)
        try:
            resp = self.opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise FetchError(_http_reason(exc)) from None
        except FetchError:
            raise
        except (OSError, ValueError) as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from None
        with closing(resp):
            if byte_range is not None and getattr(resp, "status", None) != 206:
                raise FetchError("the server ignored the byte range")
            body = resp.read(limit + 1)
        if len(body) > limit:
            raise FetchError(f"response larger than {limit} bytes")
        self._write_cache(url, byte_range, body)
        return body

    def json(self, url: str, **kw):
        try:
            return json.loads(self.get(url, **kw).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise FetchError("the answer is not JSON") from None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_http.py -q`
Expected: all pass.

- [ ] **Step 5: Mutations** (each must turn a named test red; restore after each)

| Mutation | Red test |
|---|---|
| add `headers["Authorization"] = "Bearer x"` | `test_no_authorization_header...` |
| `host_allowed` returns True always | `test_only_https_on_the_allowed_hosts...`, `test_host_allow_list` |
| `_Redirects.redirect_request` skips the host check | `test_a_redirect_to_another_host...` |
| drop the `!= 206` check | `test_a_range_must_be_answered_with_206` |
| `resp.read(limit + 1)` becomes `resp.read()` | `test_the_read_itself_is_bounded_by_the_cap` (the size check after the read still catches the body, so only the argument to `read` shows it) |
| `self.offline` check moved after the cache read | `test_offline_makes_no_request...` |
| `_write_cache` also called on errors (write in the `except`) | `test_a_failure_is_not_cached` |
| `clean` returns the text unscrubbed | `test_clean_makes_one_short_line...` |
| `_ID` allows `/` more than once | `test_valid_id` |

- [ ] **Step 6: Commit**

```bash
git add orchard/research tests/test_research_http.py
git commit -m "research: types and a guarded fetcher (fixed hosts, no token, size cap, cache, offline)"
```

---

### Task 2: Lineage from the hub

**Files:**
- Create: `orchard/research/lineage.py`
- Test: `tests/test_research_lineage.py`

**Interfaces:**
- Consumes: `Fetcher` (`.json`, `.now`), `FetchError`, `Evidence`, `Unavailable`, `Result`, `STATED`, `clean`, `valid_id`.
- Produces: `card_bases(payload: dict) -> list[tuple[str, str]]` (id, relation), `chain(fetch, model_id, *, max_depth=5) -> Result` with `data = {"ancestors": [ids, nearest first], "edges": [{"child","parent","relation"}]}`, `siblings(fetch, base_id, *, exclude=(), limit=20) -> Result` with `data = {"siblings": [ids]}`.

The tag shapes below were seen on `episod/clef-p300` on 2026-10-12: `base_model:Cloudflare/clef` and `base_model:finetune:Cloudflare/clef`.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_lineage.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Lineage from the hub: the chain of base models and the sibling fine-tunes (orchard/research/lineage.py)."""
import pytest

from orchard.research import lineage
from orchard.research.http import FetchError


class FakeFetch:
    """Answers `.json(url)` from a dict; a missing url is a 404 FetchError."""

    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def json(self, url, **kw):
        self.urls.append(url)
        if url not in self.routes:
            raise FetchError("HTTP 404")
        got = self.routes[url]
        if isinstance(got, Exception):
            raise got
        return got

    def now(self):
        return "2026-10-12T00:00:00Z"


def model(id, *tags, **card):
    return {"id": id, "tags": list(tags), "cardData": card, "config": {"model_type": "gemma4"}}


def url(id):
    return f"https://huggingface.co/api/models/{id}"


def test_the_tags_give_the_base_and_the_relation():
    p = model("a/b", "base_model:x/y", "base_model:finetune:x/y", "license:mit")
    assert lineage.card_bases(p) == [("x/y", "finetune")]


def test_a_plain_tag_alone_has_an_unknown_relation_and_the_card_field_is_a_fallback():
    assert lineage.card_bases(model("a/b", "base_model:x/y")) == [("x/y", "unknown")]
    assert lineage.card_bases(model("a/b", base_model="x/y", base_model_relation="quantized")) == [("x/y", "quantized")]
    assert lineage.card_bases(model("a/b", base_model=["x/y", "p/q"])) == [("x/y", "unknown"), ("p/q", "unknown")]
    assert lineage.card_bases(model("a/b", base_model=7)) == []


def test_the_chain_follows_bases_upward_with_one_evidence_row_per_edge():
    f = FakeFetch({url("h/model"): model("h/model", "base_model:finetune:g/it"),
                   url("g/it"): model("g/it", "base_model:finetune:g/base"),
                   url("g/base"): model("g/base")})
    r = lineage.chain(f, "h/model")
    assert r.data["ancestors"] == ["g/it", "g/base"]
    assert [(e["child"], e["parent"], e["relation"]) for e in r.data["edges"]] == [
        ("h/model", "g/it", "finetune"), ("g/it", "g/base", "finetune")]
    assert [e.confidence for e in r.evidence] == ["stated", "stated"]
    assert r.unavailable == []


def test_a_model_that_names_itself_or_a_loop_does_not_repeat():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:a/b", "base_model:finetune:c/d"),
                   url("c/d"): model("c/d", "base_model:finetune:a/b")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == ["c/d"]
    assert len(f.urls) == 2


def test_the_depth_is_limited():
    routes = {url(f"o/m{i}"): model(f"o/m{i}", f"base_model:finetune:o/m{i + 1}") for i in range(20)}
    r = lineage.chain(FakeFetch(routes), "o/m0", max_depth=3)
    assert r.data["ancestors"] == ["o/m1", "o/m2", "o/m3"]


@pytest.mark.parametrize("hostile", ["x/../../y?z", "a/b c", "a/b\nc", "nogroup", "a/b/../c"])
def test_a_hostile_base_name_never_reaches_a_url(hostile):
    f = FakeFetch({url("a/b"): model("a/b", f"base_model:finetune:{hostile}")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == [] and f.urls == [url("a/b")]
    assert any("not a model id" in u.reason for u in r.unavailable)


def test_a_gated_or_missing_base_is_an_unavailable_row_and_the_rest_still_works():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:g/gated"), url("g/gated"): FetchError("HTTP 401")})
    r = lineage.chain(f, "a/b")
    assert r.data["ancestors"] == ["g/gated"]
    assert [(u.source, u.reason) for u in r.unavailable] == [("lineage", "g/gated: HTTP 401")]


def test_the_new_models_hub_failure_is_unavailable_and_the_data_is_empty():
    r = lineage.chain(FakeFetch({}), "a/b")
    assert r.data["ancestors"] == [] and r.unavailable[0].reason == "a/b: HTTP 404"


def test_the_card_text_in_a_claim_is_cleaned():
    f = FakeFetch({url("a/b"): model("a/b", "base_model:finetune:g/base")})
    claim = lineage.chain(f, "a/b").evidence[0].claim
    assert claim == "a/b lists g/base as its base model (relation: finetune)"


LIST = "https://huggingface.co/api/models?filter=base_model:finetune:g/base&sort=downloads&direction=-1&limit=20"


def test_siblings_are_the_other_fine_tunes_of_the_base():
    f = FakeFetch({LIST: [{"id": "a/b"}, {"id": "s/one"}, {"id": "s/two"}, {"id": "bad id"}, {"x": 1}]})
    r = lineage.siblings(f, "g/base", exclude={"a/b"})
    assert r.data["siblings"] == ["s/one", "s/two"]
    assert all("another fine-tune of g/base" in e.claim for e in r.evidence)


def test_a_failed_sibling_query_is_unavailable():
    r = lineage.siblings(FakeFetch({LIST: FetchError("rate limited")}), "g/base")
    assert r.data["siblings"] == [] and r.unavailable[0].reason == "g/base: rate limited"


def test_a_sibling_answer_that_is_not_a_list_is_unavailable():
    r = lineage.siblings(FakeFetch({LIST: {"error": "x"}}), "g/base")
    assert r.unavailable and r.data["siblings"] == []
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_lineage.py -q`
Expected: `ModuleNotFoundError` for `orchard.research.lineage`.

- [ ] **Step 3: Write the code**

`orchard/research/lineage.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where a model came from, from the Hugging Face hub (no token).

A model card names its base in `cardData.base_model` and in tags: `base_model:<id>` and
`base_model:<relation>:<id>` (relation: finetune, quantized, adapter or merge). `chain` follows those upward.
`siblings` lists the other fine-tunes of a base, which is how a bundle built for a sibling is found."""
from __future__ import annotations

from orchard.research.http import FetchError
from orchard.research.types import STATED, Evidence, Result, Unavailable, clean, valid_id

HUB_MODEL = "https://huggingface.co/api/models/{id}"
HUB_LIST = ("https://huggingface.co/api/models?filter=base_model:finetune:{id}"
            "&sort=downloads&direction=-1&limit={limit}")
RELATIONS = ("finetune", "quantized", "adapter", "merge")
MAX_DEPTH = 5
SIBLING_LIMIT = 20


def card_bases(payload: dict) -> list[tuple[str, str]]:
    """[(base id, relation)] from a hub model payload. Tags first (they carry the relation), then the card."""
    found: dict[str, str] = {}
    for tag in payload.get("tags") or []:
        if not isinstance(tag, str) or not tag.startswith("base_model:"):
            continue
        rest = tag[len("base_model:"):]
        relation, _, ident = rest.partition(":")
        if relation in RELATIONS and ident:
            found[ident] = relation
        else:
            found.setdefault(rest, "unknown")
    card = payload.get("cardData") or {}
    raw = card.get("base_model")
    names = [raw] if isinstance(raw, str) else [b for b in raw if isinstance(b, str)] if isinstance(raw, list) else []
    relation = card.get("base_model_relation")
    for name in names:
        found.setdefault(name, relation if relation in RELATIONS else "unknown")
    return list(found.items())


def chain(fetch, model_id: str, *, max_depth: int = MAX_DEPTH) -> Result:
    """The bases of `model_id`, nearest first. Every base id is checked before it is put in a URL."""
    res = Result(data={"ancestors": [], "edges": []})
    seen = {model_id}
    frontier = [model_id]
    for _ in range(max_depth):
        following = []
        for child in frontier:
            try:
                payload = fetch.json(HUB_MODEL.format(id=child))
            except FetchError as exc:
                res.unavailable.append(Unavailable("lineage", f"{child}: {exc.reason}"))
                continue
            if not isinstance(payload, dict):
                res.unavailable.append(Unavailable("lineage", f"{child}: the answer is not a model"))
                continue
            for parent, relation in card_bases(payload):
                if not valid_id(parent):
                    res.unavailable.append(Unavailable("lineage", f"{child}: {clean(parent, 60)!r} is not a model id"))
                    continue
                if parent == child:
                    continue
                res.evidence.append(Evidence(
                    "hub", clean(f"{child} lists {parent} as its base model (relation: {relation})"),
                    f"https://huggingface.co/{child}", STATED, fetch.now()))
                res.data["edges"].append({"child": child, "parent": parent, "relation": relation})
                if parent not in seen:
                    seen.add(parent)
                    res.data["ancestors"].append(parent)
                    following.append(parent)
        frontier = following
        if not frontier:
            break
    return res


def siblings(fetch, base_id: str, *, exclude=(), limit: int = SIBLING_LIMIT) -> Result:
    """Other fine-tunes of `base_id`, most downloaded first."""
    res = Result(data={"siblings": []})
    try:
        items = fetch.json(HUB_LIST.format(id=base_id, limit=limit))
    except FetchError as exc:
        res.unavailable.append(Unavailable("siblings", f"{base_id}: {exc.reason}"))
        return res
    if not isinstance(items, list):
        res.unavailable.append(Unavailable("siblings", f"{base_id}: the answer is not a list"))
        return res
    for item in items:
        ident = item.get("id") if isinstance(item, dict) else None
        if not valid_id(ident) or ident in exclude:
            continue
        res.data["siblings"].append(ident)
        res.evidence.append(Evidence("hub", clean(f"{ident} is another fine-tune of {base_id}"),
                                     f"https://huggingface.co/{ident}", STATED, fetch.now()))
    return res
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_lineage.py -q`
Expected: all pass.

- [ ] **Step 5: Mutations**

| Mutation | Red test |
|---|---|
| remove the `valid_id(parent)` check | `test_a_hostile_base_name_never_reaches_a_url` |
| remove the `seen` guard | `test_a_model_that_names_itself_or_a_loop_does_not_repeat` |
| `max_depth` ignored | `test_the_depth_is_limited` |
| tags parsed with the relation lost (always `"unknown"`) | `test_the_tags_give_the_base...` |
| drop the `isinstance(items, list)` check | `test_a_sibling_answer_that_is_not_a_list...` |
| drop `exclude` | `test_siblings_are_the_other_fine_tunes...` |

- [ ] **Step 6: Commit**

```bash
git add orchard/research/lineage.py tests/test_research_lineage.py
git commit -m "research: lineage from the hub (base chain, relations, sibling fine-tunes)"
```

---

### Task 3: Split the triage rules so they take dictionaries, and load them

**Files:**
- Modify: `orchard/skills/delta-triage-templates/delta_triage.py` (the functions `compare_configs` at line 287 and `compare_tensors` at line 412)
- Create: `orchard/research/triage_rules.py`
- Test: `tests/test_research_triage_rules.py`

**Interfaces:**
- Produces (template): `compare_config_dicts(mcfg, bcfg, errors, model_path, base_path) -> dict` and `compare_tensor_maps(mt, bt, *, mshards, bshards, merr, berr, model_path, base_path) -> dict`. `compare_configs(model_dir, base_dir)` and `compare_tensors(model_dir, base_dir)` keep their signatures and results, as thin wrappers.
- Produces (`triage_rules.py`): `load()` returns the template module (cached).

This is a refactor with a guard: the existing `tests/test_delta_triage_template.py` must stay green and byte-identical in output. The template stays one self-contained file, because agents copy it.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_triage_rules.py`:

```python
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
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_triage_rules.py -q`
Expected: `ModuleNotFoundError: orchard.research.triage_rules`.

- [ ] **Step 3: Refactor the template**

In `delta_triage.py`, replace the first four lines of `compare_configs` (lines 287 to 294, from `def compare_configs` to the `return out` of the missing-config branch) so the function reads:

```python
def compare_configs(model_dir: Path, base_dir: Path) -> dict:
    mcfg, merr = read_json(model_dir / "config.json")
    bcfg, berr = read_json(base_dir / "config.json")
    return compare_config_dicts(mcfg, bcfg, [e for e in (merr, berr) if e],
                                str(model_dir / "config.json"), str(base_dir / "config.json"))


def compare_config_dicts(mcfg, bcfg, errors: list, model_path: str, base_path: str) -> dict:
    """The config comparison on two already-read configs. `compare_configs` reads them from snapshot
    directories; the research package reads them from the hub."""
    out = {"model_config_path": model_path, "base_config_path": base_path, "errors": list(errors)}
    if not isinstance(mcfg, dict) or not isinstance(bcfg, dict):
        out["errors"].append("config.json is missing or not an object in "
                             + ("the new model" if not isinstance(mcfg, dict) else "the nearest model"))
        out["decisive_keys_equal"] = False
        return out
```

The rest of the old body (from `mt, bt = text_part(mcfg), text_part(bcfg)` to `return out`) stays unchanged under `compare_config_dicts`.

Replace the head of `compare_tensors` (lines 412 to 418, through `m_raw, b_raw = ...`) so it reads:

```python
def compare_tensors(model_dir: Path, base_dir: Path) -> dict:
    mshards, mt, merr = read_tensors(model_dir)
    bshards, bt, berr = read_tensors(base_dir)
    return compare_tensor_maps(mt, bt, mshards=mshards, bshards=bshards, merr=merr, berr=berr,
                               model_path=str(model_dir), base_path=str(base_dir))


def compare_tensor_maps(mt: dict, bt: dict, *, mshards: list, bshards: list, merr: list, berr: list,
                        model_path: str, base_path: str) -> dict:
    """The tensor comparison on two already-read {name: {"dtype", "shape", "shard"}} maps. The research
    package builds the maps from safetensors headers fetched by range request."""
    m_raw, b_raw = set(mt), set(bt)
```

In the returned dict of the old `compare_tensors`, change `"path": str(model_dir)` to `"path": model_path` and `"path": str(base_dir)` to `"path": base_path`. Everything else in the body is unchanged.

`orchard/research/triage_rules.py`:

```python
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
```

- [ ] **Step 4: Run the new and the existing template tests**

Run: `python -m pytest tests/test_research_triage_rules.py tests/test_delta_triage_template.py tests/test_skills.py -q`
Expected: all pass, same count as before for the template tests (run them once before the refactor and compare).

- [ ] **Step 5: Mutations**

| Mutation | Red test |
|---|---|
| `compare_tensor_maps` stops reporting `unexplained_only_in_model` | `test_a_changed_shape...` is not red; an extra-tensor case in `test_delta_triage_template.py::test_an_unexplained_extra_text_tensor_makes_the_path_full_port` is |
| `compare_config_dicts` skips the `decisive_keys_differing` list | `test_a_changed_decisive_config_value_is_full_port` |
| `compare_tensors` passes `merr`/`berr` swapped for `[]` | `test_a_header_that_could_not_be_read_is_unknown` is not affected (it calls the map function), so add this check to the template tests: a truncated shard in the nearest model must still give `unknown`/errors via the wrapper. If no existing test covers it, add one (`test_a_cut_short_shard_is_reported_by_the_directory_wrapper`) before the mutation. |
| `triage_rules.load` not cached | `test_the_loader_returns_one_cached_module...` |

- [ ] **Step 6: Commit**

```bash
git add orchard/skills/delta-triage-templates/delta_triage.py orchard/research/triage_rules.py tests/test_research_triage_rules.py tests/test_delta_triage_template.py
git commit -m "research: the triage rules take dictionaries, and the research package loads them"
```

---

### Task 4: Architecture match and header-based class estimate

**Files:**
- Create: `orchard/research/arch_match.py`, `orchard/research/headers.py`
- Test: `tests/test_research_arch_headers.py`

**Interfaces:**
- Consumes: `Fetcher`, `FetchError`, `triage_rules.load()`, `Result`, `Unavailable`, `Evidence`, `INFERRED`, `WEAK`, `clean`, `valid_id`.
- Produces (`arch_match.py`): `fetch_config(fetch, repo) -> dict | None` (raises nothing; None on any failure), `compare(new_cfg, cand_cfg) -> dict` with keys `confidence` (`"inferred"`, `"weak"` or `None`), `differing` (`{key: {"new", "candidate"}}`), `same_family` (bool).
- Produces (`headers.py`): `read_model(fetch, repo, files) -> Result` with `data = {"tensors", "shards", "extra_safetensors", "errors"}` and `estimate_class(new_cfg, new, cand_cfg, cand) -> {"class", "reasons", "sidecar_possible", "shared_tensors"}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_arch_headers.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Architecture matching and the header-only class estimate (arch_match.py and headers.py)."""
import json
import struct

import pytest

pytest.importorskip("tokenizers", reason="the class rules live in the delta-triage template")

from orchard.research import arch_match, headers  # noqa: E402
from orchard.research.http import FetchError  # noqa: E402

CFG = {"model_type": "gemma4", "architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 30,
       "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 262144}


def test_the_same_family_and_decisive_values_are_inferred():
    r = arch_match.compare({**CFG, "model_type": "gemma4_unified"}, CFG)
    assert r["confidence"] == "inferred" and r["differing"] == {} and r["same_family"] is True


def test_the_same_family_with_a_different_size_is_weak_and_names_the_difference():
    r = arch_match.compare(CFG, {**CFG, "num_hidden_layers": 48})
    assert r["confidence"] == "weak"
    assert r["differing"] == {"num_hidden_layers": {"new": 30, "candidate": 48}}


def test_another_family_is_not_a_candidate():
    assert arch_match.compare(CFG, {**CFG, "model_type": "qwen3"})["confidence"] is None


def test_a_nested_text_config_is_read_through_the_templates_rules():
    nested = {"model_type": "gemma4", "text_config": {k: v for k, v in CFG.items() if k != "model_type"}}
    assert arch_match.compare(nested, CFG)["confidence"] == "inferred"


class FakeFetch:
    def __init__(self, routes):
        self.routes = routes

    def json(self, url, **kw):
        got = self.routes.get(url)
        if got is None or isinstance(got, Exception):
            raise got or FetchError("HTTP 404")
        return got

    def now(self):
        return "2026-10-12T00:00:00Z"


def test_fetch_config_returns_the_config_or_none():
    url = "https://huggingface.co/g/it/resolve/main/config.json"
    assert arch_match.fetch_config(FakeFetch({url: CFG}), "g/it") == CFG
    assert arch_match.fetch_config(FakeFetch({}), "g/it") is None
    assert arch_match.fetch_config(FakeFetch({url: [1]}), "g/it") is None
    assert arch_match.fetch_config(FakeFetch({}), "bad id") is None


# ---- headers ---------------------------------------------------------------------------------------

def header_bytes(tensors: dict) -> bytes:
    blob = json.dumps({k: {"dtype": v[0], "shape": v[1], "data_offsets": [0, 0]} for k, v in tensors.items()}
                      | {"__metadata__": {}}).encode()
    return struct.pack("<Q", len(blob)) + blob


class RangeFetch:
    """Serves a file's first 8 bytes, then its header, by byte range, and a JSON index."""

    def __init__(self, files: dict, indexes=None):
        self.files, self.indexes, self.ranges = files, indexes or {}, []

    def get(self, url, *, byte_range=None, max_bytes=0):
        name = url.rsplit("/", 1)[1]
        self.ranges.append((name, byte_range))
        if name not in self.files:
            raise FetchError("HTTP 404")
        data = self.files[name]
        start, end = byte_range
        return data[start:end + 1]

    def json(self, url, **kw):
        name = url.rsplit("/", 1)[1]
        if name not in self.indexes:
            raise FetchError("HTTP 404")
        return self.indexes[name]

    def now(self):
        return "2026-10-12T00:00:00Z"


T = {"model.embed_tokens.weight": ("BF16", [100, 64]), "model.layers.0.q.weight": ("BF16", [64, 64])}


def test_a_header_is_read_with_two_range_requests_and_no_weights():
    f = RangeFetch({"model.safetensors": header_bytes(T) + b"\0" * 1000})
    r = headers.read_model(f, "g/it", ["config.json", "model.safetensors"])
    assert r.data["tensors"]["model.layers.0.q.weight"] == {"dtype": "BF16", "shape": [64, 64],
                                                           "shard": "model.safetensors"}
    assert [rng for _, rng in f.ranges] == [(0, 7), (8, len(header_bytes(T)) - 1)]
    assert r.data["errors"] == []


def test_the_index_decides_the_backbone_and_other_safetensors_are_possible_sidecars():
    f = RangeFetch({"model-00001-of-00001.safetensors": header_bytes(T), "joint_head.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {"a": "model-00001-of-00001.safetensors"}}})
    files = ["model.safetensors.index.json", "model-00001-of-00001.safetensors", "joint_head.safetensors"]
    r = headers.read_model(f, "g/it", files)
    assert r.data["shards"] == ["model-00001-of-00001.safetensors"]
    assert r.data["extra_safetensors"] == ["joint_head.safetensors"]
    assert all(name != "joint_head.safetensors" for name, _ in f.ranges)


def test_the_index_wins_over_the_file_name_pattern():
    f = RangeFetch({"weights-a.safetensors": header_bytes(T), "model.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {"a": "weights-a.safetensors"}}})
    r = headers.read_model(f, "g/it", ["model.safetensors.index.json", "weights-a.safetensors", "model.safetensors"])
    assert r.data["shards"] == ["weights-a.safetensors"] and r.data["extra_safetensors"] == ["model.safetensors"]


@pytest.mark.parametrize("name", ["../evil.safetensors", "a/b.safetensors", "x y.safetensors", ".safetensors"])
def test_an_unsafe_shard_name_is_never_fetched(name):
    f = RangeFetch({})
    r = headers.read_model(f, "g/it", [name])
    assert f.ranges == [] and r.unavailable and r.data["tensors"] == {}


def test_an_implausible_header_length_is_an_error_row_not_a_crash():
    f = RangeFetch({"model.safetensors": struct.pack("<Q", 1 << 60) + b"x" * 100})
    r = headers.read_model(f, "g/it", ["model.safetensors"])
    assert r.data["errors"] and "not plausible" in r.data["errors"][0]


def test_no_safetensors_is_unavailable_and_too_many_shards_is_refused():
    assert headers.read_model(RangeFetch({}), "g/it", ["config.json"]).unavailable[0].reason.endswith("no safetensors files")
    many = [f"model-{i:05d}-of-00100.safetensors" for i in range(100)]
    r = headers.read_model(RangeFetch({}), "g/it", many)
    assert "more than" in r.unavailable[0].reason


def test_a_failed_range_is_recorded_per_shard():
    f = RangeFetch({"model-00001-of-00002.safetensors": header_bytes(T)})
    r = headers.read_model(f, "g/it", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    assert len(r.data["errors"]) == 1 and "00002" in r.data["errors"][0]


def model_data(tensors, extra=()):
    return {"tensors": {k: {"dtype": v[0], "shape": v[1], "shard": "m"} for k, v in tensors.items()},
            "shards": ["m"], "extra_safetensors": list(extra), "errors": []}


def test_the_estimate_follows_the_template_rules():
    same = headers.estimate_class(CFG, model_data(T), CFG, model_data(T))
    assert same["class"] == "weights-only" and same["shared_tensors"] == 2 and same["sidecar_possible"] is False
    other = dict(T, **{"model.layers.0.q.weight": ("BF16", [64, 32])})
    assert headers.estimate_class(CFG, model_data(other), CFG, model_data(T))["class"] == "full-port"


def test_a_changed_dtype_is_full_port():
    other = dict(T, **{"model.layers.0.q.weight": ("F32", [64, 64])})
    r = headers.estimate_class(CFG, model_data(other), CFG, model_data(T))
    assert r["class"] == "full-port" and any("dtype" in x for x in r["reasons"])


def test_a_repo_with_extra_safetensors_files_flags_a_possible_sidecar():
    r = headers.estimate_class(CFG, model_data(T, extra=["joint_head.safetensors"]), CFG, model_data(T))
    assert r["class"] == "weights-only" and r["sidecar_possible"] is True


def test_reasons_are_cleaned_and_limited():
    other = {f"model.layers.{i}.q.weight": ("BF16", [1, 1]) for i in range(20)}
    r = headers.estimate_class(CFG, model_data(other), CFG, model_data(
        {f"model.layers.{i}.q.weight": ("BF16", [2, 2]) for i in range(20)}))
    assert len(r["reasons"]) <= 5
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_arch_headers.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write the code**

`orchard/research/arch_match.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Does a candidate's architecture match the new model's? Compares two config.json files with the stage 0
template's decisive keys, so the answer follows the same rules the later stages use."""
from __future__ import annotations

from orchard.research import triage_rules
from orchard.research.http import FetchError
from orchard.research.types import INFERRED, WEAK, valid_id

CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"


def fetch_config(fetch, repo: str) -> dict | None:
    """A repo's config.json, or None when it cannot be read for any reason."""
    if not valid_id(repo):
        return None
    try:
        cfg = fetch.json(CONFIG_URL.format(repo=repo))
    except FetchError:
        return None
    return cfg if isinstance(cfg, dict) else None


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
```

`orchard/research/headers.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Safetensors headers by HTTP range request, and a class estimate from them.

A safetensors file starts with an 8-byte little-endian length N, then N bytes of JSON that name every
tensor with its dtype and shape. Two range requests read that without fetching any weights. The estimate
applies the stage 0 template's rules (triage_rules) to those headers, so a model can be judged weights-only
or full-port before the download. A sidecar's tensors would need its own download, so the estimate only
flags that one may exist."""
from __future__ import annotations

import json
import re
import struct

from orchard.research import triage_rules
from orchard.research.http import FetchError
from orchard.research.types import Result, Unavailable, clean, valid_id

RAW_URL = "https://huggingface.co/{repo}/resolve/main/{name}"
INDEX_NAME = "model.safetensors.index.json"
SAFE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}")
MAX_SHARDS = 64
MAX_REASONS = 5


def shard_header(fetch, repo: str, name: str) -> dict:
    """{tensor: {"dtype", "shape"}} for one file, from two range requests. Raises FetchError or ValueError."""
    url = RAW_URL.format(repo=repo, name=name)
    first = fetch.get(url, byte_range=(0, 7), max_bytes=8)
    if len(first) != 8:
        raise ValueError("shorter than 8 bytes")
    (n,) = struct.unpack("<Q", first)
    if n <= 0 or n > triage_rules.load().MAX_HEADER:
        raise ValueError(f"header length {n} is not plausible")
    blob = fetch.get(url, byte_range=(8, 8 + n - 1), max_bytes=n)
    if len(blob) != n:
        raise ValueError(f"header is cut short ({len(blob)} of {n} bytes)")
    header = json.loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return {k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
            for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}


def read_model(fetch, repo: str, files: list[str]) -> Result:
    """The backbone's tensors from the headers of its shards. `files` is the repo's file list from the hub.
    The backbone is the shards the index lists, or `model*.safetensors` when there is no index."""
    res = Result(data={"tensors": {}, "shards": [], "extra_safetensors": [], "errors": []})
    if not valid_id(repo):
        res.unavailable.append(Unavailable("headers", f"{clean(repo, 60)!r} is not a model id"))
        return res
    names = sorted(f for f in files if isinstance(f, str) and f.endswith(".safetensors"))
    safe = [n for n in names if SAFE_FILE.fullmatch(n)]
    if not safe:
        res.unavailable.append(Unavailable("headers", f"{repo}: no safetensors files"))
        return res
    shards = None
    if INDEX_NAME in files:
        try:
            index = fetch.json(RAW_URL.format(repo=repo, name=INDEX_NAME))
            listed = set((index.get("weight_map") or {}).values()) if isinstance(index, dict) else set()
            shards = [n for n in safe if n in listed] or None
        except FetchError:
            shards = None
    if shards is None:
        pattern = triage_rules.load().BACKBONE_SHARD
        shards = [n for n in safe if pattern.fullmatch(n)] or safe
    if len(shards) > MAX_SHARDS:
        res.unavailable.append(Unavailable("headers", f"{repo}: more than {MAX_SHARDS} shards"))
        return res
    res.data["shards"] = shards
    res.data["extra_safetensors"] = [n for n in safe if n not in shards]
    for shard in shards:
        try:
            for name, info in shard_header(fetch, repo, shard).items():
                res.data["tensors"][name] = {**info, "shard": shard}
        except (FetchError, ValueError, UnicodeDecodeError) as exc:
            res.data["errors"].append(f"{shard}: {getattr(exc, 'reason', exc)}")
    return res


def estimate_class(new_cfg: dict, new: dict, cand_cfg: dict, cand: dict) -> dict:
    """weights-only, full-port or unknown, by the stage 0 rules, from configs and headers alone."""
    rules = triage_rules.load()
    c = rules.compare_config_dicts(new_cfg, cand_cfg, [], "new/config.json", "candidate/config.json")
    t = rules.compare_tensor_maps(new["tensors"], cand["tensors"], mshards=new["shards"], bshards=cand["shards"],
                                  merr=new["errors"], berr=cand["errors"], model_path="new", base_path="candidate")
    reasons = rules.decide_path(c, t)
    cls = rules.decide_class(c, t, reasons, {"problems": [], "sidecars": []})
    return {"class": cls, "reasons": [clean(r) for r in reasons[:MAX_REASONS]],
            "sidecar_possible": bool(new["extra_safetensors"]),
            "shared_tensors": t["normalized_name_sets"]["common"]}
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_arch_headers.py -q`
Expected: all pass.

- [ ] **Step 5: Mutations**

| Mutation | Red test |
|---|---|
| `family()` compares the whole `model_type` | `test_the_same_family_and_decisive_values_are_inferred` |
| `compare` returns INFERRED whatever the differences | `test_the_same_family_with_a_different_size_is_weak...` |
| drop `SAFE_FILE` filtering | `test_an_unsafe_shard_name_is_never_fetched` |
| drop the plausibility check on `n` | `test_an_implausible_header_length...` |
| `MAX_SHARDS` check removed | `test_no_safetensors_is_unavailable_and_too_many_shards_is_refused` |
| ignore the index (always `model*` pattern) | `test_the_index_wins_over_the_file_name_pattern` (the plain index test passes without the index, because its names also match the pattern) |
| `sidecar_possible` always False | `test_a_repo_with_extra_safetensors_files_flags_a_possible_sidecar` |
| the template's `decide_path` stops reading `dtype_diffs_normalized` (stands for a copy of the rules that drifted) | `test_a_changed_dtype_is_full_port` |

- [ ] **Step 6: Commit**

```bash
git add orchard/research/arch_match.py orchard/research/headers.py tests/test_research_arch_headers.py
git commit -m "research: architecture match and a header-only class estimate"
```

---

### Task 5: Existing ports

**Files:**
- Create: `orchard/research/ports.py`
- Test: `tests/test_research_ports.py`, fixtures `tests/fixtures/research/`

**Interfaces:**
- Consumes: `Fetcher` (`.json`, `.get`, `.now`), `FetchError`, `Evidence`, `Unavailable`, `Result`, `INFERRED`, `WEAK`, `clean`, `nearest.parse_search`.
- Produces: `terms(config) -> {"specific": [..], "family": [..], "classes": [..]}`; `catalog(family, *, run, which, now) -> Result`; `metal(fetch, terms) -> Result`; `issues(fetch, terms, *, limit=5) -> Result`; `github_code(terms, *, run, which, now) -> Result`; `bundle_code(siblings: list, bundle: str, now: str) -> list[Evidence]`; `collect(fetch, config, *, family, run, which, no_github) -> Result`.

**Before writing the tests, record real data once** (read-only: `tt model search` and public GitHub and
Hugging Face reads; save under `tests/fixtures/research/`). Read each file in full before the tests use it.
These are the commands that produced the fixtures the tests below were checked against on 2026-10-12:

```bash
mkdir -p tests/fixtures/research && cd tests/fixtures/research
tt model search gemma --catalog --json > catalog-gemma.json
curl -s "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/demos" | python3 -c "
import json, sys
json.dump([{k: i[k] for k in ('name', 'type', 'html_url')} for i in json.load(sys.stdin)], open('metal-demos.json', 'w'), indent=1)"
curl -s "https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md" | head -30 | cut -c1-200 > metal-readme-head.md
curl -s "https://huggingface.co/api/models/stisiTT/gemma-4-12b-it-p150" | python3 -c "
import json, sys
d = json.load(sys.stdin)
seen, keep = {}, []
for s in d['siblings']:
    n = s['rfilename']
    if not n.startswith('code/'):
        keep.append({'rfilename': n}); continue
    key = '/'.join(n.split('/')[:4])           # keep two files per code directory
    if seen.get(key, 0) < 2:
        seen[key] = seen.get(key, 0) + 1; keep.append({'rfilename': n})
json.dump({'id': d['id'], 'tags': d['tags'], 'siblings': keep}, open('bundle-gemma-4-12b-p150.json', 'w'), indent=1)"
cd -
```

What the recordings showed on that day, which the tests pin: `models/demos` lists 36 entries, including
`gemma4`, `gemma4_31b_qb2` and `gemma4_d_p`; the README table has 22 model rows, the first
`Qwen 3 32B`; the catalog lists one Gemma bundle, `stisiTT/gemma-4-12b-it-p150`, and that bundle ships
`code/models/demos/gemma4`. If a later recording differs, change the pinned values to the new recording and say so
in the commit message. A GitHub issue search for `gemma4%20repo:tenstorrent%2Ftt-metal` returned 444 results.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_ports.py` (the fixture-driven tests read the recorded files; names of expected hits are taken from what the recording shows, so fill the `EXPECT_*` constants from the files when writing):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Existing ports: the catalog, tt-metal listings, GitHub issues and code, and code shipped in bundles."""
import json
import subprocess
from pathlib import Path

import pytest

from orchard.research import ports
from orchard.research.http import FetchError

FIX = Path(__file__).parent / "fixtures" / "research"
NOW = "2026-10-12T00:00:00Z"
CFG = {"model_type": "gemma4_unified", "architectures": ["Gemma4UnifiedForConditionalGeneration"]}


def test_terms_come_from_the_model_type_and_architecture():
    t = ports.terms(CFG)
    assert t["specific"] == ["gemma4"] and t["family"] == ["gemma"]
    assert t["classes"] == ["Gemma4UnifiedForConditionalGeneration"]


@pytest.mark.parametrize("cfg", [{}, {"model_type": None}, {"model_type": 5}, {"architectures": "x"},
                                 {"architectures": ["bad class name", "x" * 200, "a/b", "repo:o/p"]}])
def test_odd_configs_give_no_terms_not_a_crash_and_no_unsafe_class_name(cfg):
    t = ports.terms(cfg)
    assert t["classes"] == [] and all(ports.SAFE_TERM.fullmatch(x) for x in t["specific"] + t["family"])


# ---- name matching ---------------------------------------------------------------------------------

def test_a_specific_match_is_inferred_and_a_family_only_match_is_weak():
    hits = ports.match_names(["gemma3", "gemma4_it", "llama3", "gemma"], ports.terms(CFG))
    assert hits == [("gemma4_it", "inferred"), ("gemma3", "weak"), ("gemma", "weak")]


# ---- tt-metal ---------------------------------------------------------------------------------------

class FakeFetch:
    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def json(self, url, **kw):
        self.urls.append(url)
        got = self.routes.get(url, FetchError("HTTP 404"))
        if isinstance(got, Exception):
            raise got
        return got

    def get(self, url, **kw):
        self.urls.append(url)
        got = self.routes.get(url, FetchError("HTTP 404"))
        if isinstance(got, Exception):
            raise got
        return got if isinstance(got, bytes) else json.dumps(got).encode()

    def now(self):
        return NOW


DEMOS = "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/demos"
EXPER = "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/experimental"
README = "https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md"


def test_metal_lists_matching_demo_directories_and_readme_rows():
    f = FakeFetch({DEMOS: [{"name": "gemma4", "type": "dir", "html_url": "https://github.com/tenstorrent/tt-metal/tree/main/models/demos/gemma4"},
                           {"name": "llama3", "type": "dir", "html_url": "https://github.com/x/y"}],
                   EXPER: [],
                   README: b"| [Gemma 4 12B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/gemma4) | 32 |\n"
                           b"| [Llama 3.1 (TP=4)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) | 32 |\n"})
    r = ports.metal(f, ports.terms(CFG))
    claims = [e.claim for e in r.evidence]
    assert any("models/demos/gemma4" in c for c in claims) and any("Gemma 4 12B" in c for c in claims)
    assert not any("llama" in c.lower() for c in claims)
    assert {e.confidence for e in r.evidence} == {"inferred"}


def test_the_recorded_tt_metal_readme_still_parses_into_rows():
    rows = ports.readme_rows((FIX / "metal-readme-head.md").read_text())
    assert len(rows) == 22
    assert rows[0] == ("Qwen 3 32B", "https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers")


def test_the_recorded_tt_metal_demos_listing_finds_the_gemma4_ports():
    demos = json.loads((FIX / "metal-demos.json").read_text())
    f = FakeFetch({DEMOS: demos, EXPER: [], README: (FIX / "metal-readme-head.md").read_bytes()})
    r = ports.metal(f, ports.terms(CFG))
    assert [(e.claim, e.confidence) for e in r.evidence] == [
        ("tt-metal has models/demos/gemma4", "inferred"), ("tt-metal has models/demos/gemma4_31b_qb2", "inferred"),
        ("tt-metal has models/demos/gemma4_d_p", "inferred")]
    assert r.unavailable == []


def test_a_tt_metal_listing_that_fails_is_unavailable_per_listing():
    r = ports.metal(FakeFetch({DEMOS: FetchError("rate limited")}), ports.terms(CFG))
    assert any(u.source == "tt-metal" and "rate limited" in u.reason for u in r.unavailable)


# ---- GitHub issues -----------------------------------------------------------------------------------

def issues_url(term):
    # The query as GitHub accepts it (checked 2026-10-12: gemma4 gave 444 results).
    return f"https://api.github.com/search/issues?q={term}%20repo:tenstorrent%2Ftt-metal&per_page=5"


def test_issues_become_weak_evidence_with_cleaned_titles_and_checked_links():
    f = FakeFetch({issues_url("gemma4"): {"items": [
        {"number": 12, "title": "Gemma 4 bring-up", "state": "open", "html_url": "https://github.com/tenstorrent/tt-metal/pull/12", "pull_request": {}},
        {"number": 13, "title": "see /home/someone/x", "state": "closed", "html_url": "https://github.com/tenstorrent/tt-metal/issues/13"},
        {"number": 14, "title": "evil", "state": "open", "html_url": "https://evil.example/x"}]}})
    r = ports.issues(f, ports.terms(CFG))
    assert [e.confidence for e in r.evidence] == ["weak", "weak"]
    assert r.evidence[0].claim == "PR #12 (open): Gemma 4 bring-up"
    assert r.evidence[1].claim == "[removed: an absolute home path]"


def test_a_rate_limited_issue_search_is_unavailable():
    r = ports.issues(FakeFetch({issues_url("gemma4"): FetchError("rate limited")}), ports.terms(CFG))
    assert r.evidence == [] and r.unavailable[0].source == "github issues"


# ---- gh code search ------------------------------------------------------------------------------------

class Run:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.out = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.argv = []

    def __call__(self, argv, **kw):
        self.argv.append(argv)
        return self.out


def which_gh(name):
    return "/usr/bin/gh" if name == "gh" else None


def test_code_search_runs_gh_with_a_checked_term_as_one_argument():
    run = Run(json.dumps({"items": [{"path": "models/demos/gemma4/tt/model.py",
                                     "html_url": "https://github.com/tenstorrent/tt-metal/blob/main/models/demos/gemma4/tt/model.py"}]}))
    r = ports.github_code(ports.terms(CFG), run=run, which=which_gh, now=NOW)
    assert r.evidence[0].claim == "tt-metal code mentions Gemma4UnifiedForConditionalGeneration in models/demos/gemma4/tt/model.py"
    assert run.argv[0][:4] == ["/usr/bin/gh", "api", "-X", "GET"]
    assert "q=Gemma4UnifiedForConditionalGeneration repo:tenstorrent/tt-metal" in run.argv[0]


def test_code_search_is_skipped_without_gh_or_when_gh_fails():
    assert "not installed" in ports.github_code(ports.terms(CFG), run=Run(), which=lambda n: None, now=NOW).unavailable[0].reason
    r = ports.github_code(ports.terms(CFG), run=Run(returncode=1, stderr="To get started with GitHub CLI, run: gh auth login"),
                          which=which_gh, now=NOW)
    assert r.evidence == [] and "gh failed" in r.unavailable[0].reason


def test_a_term_that_could_change_the_query_is_never_passed_to_gh():
    run = Run("{}")
    t = ports.terms({"architectures": ["x repo:other/secret"], "model_type": "gemma4"})
    ports.github_code(t, run=run, which=which_gh, now=NOW)
    assert all("other/secret" not in " ".join(a) for a in run.argv)


# ---- the catalog ----------------------------------------------------------------------------------------

def test_the_catalog_search_uses_tt_and_the_catalog_flag():
    run = Run((FIX / "catalog-gemma.json").read_text())
    r = ports.catalog("gemma", run=run, which=lambda n: "/usr/bin/tt" if n == "tt" else None, now=NOW)
    assert run.argv[0] == ["/usr/bin/tt", "model", "search", "gemma", "--catalog", "--json"]
    assert [b["name"] for b in r.data["bundles"]] == ["stisiTT/gemma-4-12b-it-p150"]
    assert r.evidence[0].claim == "the community catalog lists stisiTT/gemma-4-12b-it-p150"


def test_a_missing_tt_or_a_failed_search_is_unavailable():
    assert ports.catalog("gemma", run=Run(), which=lambda n: None, now=NOW).unavailable
    assert ports.catalog("gemma", run=Run(returncode=2), which=lambda n: "/x/tt", now=NOW).unavailable
    assert ports.catalog("gemma", run=Run("not json"), which=lambda n: "/x/tt", now=NOW).unavailable


# ---- code shipped in a bundle -----------------------------------------------------------------------------

def test_a_bundle_that_ships_autoport_code_is_evidence_of_a_port():
    sib = [{"rfilename": "code/models/autoports/google_gemma_4_26b_a4b_it/tt/model.py"},
           {"rfilename": "code/models/common/README.md"},
           {"rfilename": "code/models/demos/gemma4/x.py"},
           {"rfilename": "README.md"}, {"x": 1}]
    ev = ports.bundle_code(sib, "tt-hous/gemma-4-26B-A4B-it_p150", NOW)
    assert [e.claim for e in ev] == [
        "bundle tt-hous/gemma-4-26B-A4B-it_p150 ships code for google_gemma_4_26b_a4b_it",
        "bundle tt-hous/gemma-4-26B-A4B-it_p150 ships code for gemma4"]
    assert {e.confidence for e in ev} == {"inferred"}


def test_the_recorded_gemma_bundle_listing_yields_one_port_row_and_ignores_shared_code():
    payload = json.loads((FIX / "bundle-gemma-4-12b-p150.json").read_text())
    ev = ports.bundle_code(payload["siblings"], "stisiTT/gemma-4-12b-it-p150", NOW)
    assert [e.claim for e in ev] == ["bundle stisiTT/gemma-4-12b-it-p150 ships code for gemma4"]


# ---- collect ------------------------------------------------------------------------------------------------

def test_collect_skips_github_when_asked_and_runs_the_rest():
    f = FakeFetch({DEMOS: [], EXPER: [], README: b""})
    run = Run(json.dumps({"bundles": []}))
    r = ports.collect(f, CFG, family="gemma", run=run, which=which_gh, no_github=True)
    assert not any("search/issues" in u for u in f.urls)
    assert run.argv == []                                     # no gh, and `tt` is not found by which_gh
    assert [u.source for u in r.unavailable] == ["catalog"]   # `tt` is not installed in this test


def test_collect_runs_the_github_searches_by_default():
    f = FakeFetch({DEMOS: [], EXPER: [], README: b""})
    run = Run(json.dumps({"items": []}))
    ports.collect(f, CFG, family="gemma", run=run, which=which_gh)
    assert any("search/issues" in u for u in f.urls)
    assert any("search/code" in argv for argv in run.argv)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_ports.py -q`
Expected: `ModuleNotFoundError: orchard.research.ports`.

- [ ] **Step 3: Write the code**

`orchard/research/ports.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Has anyone already ported this architecture?

Sources: the tt-model community catalog (`tt model search --catalog`), tt-metal's `models/` listings and
models README, GitHub issues and pull requests for the architecture name, GitHub code search through `gh`
(code search needs a login; the other calls do not), and the code a published bundle ships under
`code/models/`. A search term comes from the new model's own config, so it is checked against SAFE_TERM
before it reaches a URL or a command line."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from urllib.parse import quote

from orchard import nearest
from orchard.research.http import FetchError
from orchard.research.types import INFERRED, WEAK, Evidence, Result, Unavailable, clean

SAFE_TERM = re.compile(r"[A-Za-z0-9_.-]{3,80}")
METAL = "tenstorrent/tt-metal"
CONTENTS = "https://api.github.com/repos/" + METAL + "/contents/models/{sub}"
README_URL = "https://raw.githubusercontent.com/" + METAL + "/main/models/README.md"
ISSUES_URL = "https://api.github.com/search/issues?q={q}&per_page={n}"
GITHUB_PREFIX = "https://github.com/"
README_ROW = re.compile(r"^\|\s*\[([^\]]+)\]\((https://[^)\s]+)\)")
NOT_A_MODEL = frozenset({"common", "utils", "perf"})      # shared code in a bundle, not a port
GH_TIMEOUT_S = 30
MAX_ITEMS = 5


def terms(config) -> dict:
    """Search terms from a model's config: `specific` (gemma4), `family` (gemma) and `classes` (the exact
    architecture class names). Anything that does not match SAFE_TERM is left out."""
    config = config if isinstance(config, dict) else {}
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else ""
    first = model_type.lower().split("_")[0]
    letters = re.match(r"[a-z]+", first)
    specific = [first] if SAFE_TERM.fullmatch(first) and letters and first != letters.group(0) else []
    family = [letters.group(0)] if letters and SAFE_TERM.fullmatch(letters.group(0)) else []
    arch = config.get("architectures")
    classes = [a for a in arch if isinstance(a, str) and SAFE_TERM.fullmatch(a)] if isinstance(arch, list) else []
    return {"specific": specific, "family": family, "classes": classes[:2]}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_names(names, t: dict) -> list[tuple[str, str]]:
    """[(name, confidence)]: inferred when a name holds the specific term (gemma4), weak when it holds only
    the family (gemma)."""
    out = []
    for name in names:
        norm = _norm(str(name))
        if any(s in norm for s in t["specific"]):
            out.append((name, INFERRED))
        elif any(f in norm for f in t["family"]):
            out.append((name, WEAK))
    return sorted(out, key=lambda p: p[1] != INFERRED)


def readme_rows(text: str) -> list[tuple[str, str]]:
    """[(model name, link)] from the models README's performance table."""
    rows = []
    for line in text.splitlines():
        m = README_ROW.match(line)
        if m:
            rows.append((re.sub(r"\s*\(.*?\)\s*$", "", m.group(1)).strip(), m.group(2)))
    return rows


def metal(fetch, t: dict) -> Result:
    res = Result()
    for sub in ("demos", "experimental"):
        url = CONTENTS.format(sub=sub)
        try:
            items = fetch.json(url)
        except FetchError as exc:
            res.unavailable.append(Unavailable("tt-metal", f"models/{sub}: {exc.reason}"))
            continue
        if not isinstance(items, list):
            res.unavailable.append(Unavailable("tt-metal", f"models/{sub}: the answer is not a listing"))
            continue
        by_name = {i["name"]: i for i in items if isinstance(i, dict) and isinstance(i.get("name"), str)}
        for name, conf in match_names(list(by_name), t):
            link = by_name[name].get("html_url")
            res.evidence.append(Evidence(
                "tt-metal", clean(f"tt-metal has models/{sub}/{name}"),
                link if isinstance(link, str) and link.startswith(GITHUB_PREFIX) else GITHUB_PREFIX + METAL,
                conf, fetch.now()))
    try:
        text = fetch.get(README_URL, max_bytes=2 * 1024 * 1024).decode("utf-8", "replace")
    except FetchError as exc:
        res.unavailable.append(Unavailable("tt-metal", f"models/README.md: {exc.reason}"))
        return res
    rows = readme_rows(text)
    by_name = {n: u for n, u in rows}
    for name, conf in match_names(list(by_name), t):
        res.evidence.append(Evidence("tt-metal", clean(f"tt-metal's models README lists {name}"),
                                     by_name[name] if by_name[name].startswith(GITHUB_PREFIX) else README_URL,
                                     conf, fetch.now()))
    return res


def issues(fetch, t: dict, *, limit: int = MAX_ITEMS) -> Result:
    res = Result()
    for term in (t["specific"] or t["family"])[:1] + t["classes"][:1]:
        url = ISSUES_URL.format(q=quote(f"{term} repo:{METAL}", safe="+:"), n=limit)
        try:
            data = fetch.json(url)
        except FetchError as exc:
            res.unavailable.append(Unavailable("github issues", f"{term}: {exc.reason}"))
            continue
        for item in (data.get("items") if isinstance(data, dict) else None) or []:
            if not isinstance(item, dict) or not str(item.get("html_url", "")).startswith(GITHUB_PREFIX):
                continue
            kind = "PR" if "pull_request" in item else "issue"
            res.evidence.append(Evidence(
                "github", clean(f"{kind} #{item.get('number')} ({item.get('state')}): {item.get('title')}"),
                item["html_url"], WEAK, fetch.now()))
    return res


def github_code(t: dict, *, run=subprocess.run, which=shutil.which, now: str = "") -> Result:
    """Code search needs a login, so it goes through `gh` (read only). Without gh, or when gh fails, it is
    reported as unavailable."""
    res = Result()
    exe = which("gh")
    if not exe:
        res.unavailable.append(Unavailable("github code search", "gh is not installed"))
        return res
    for term in t["classes"][:1]:
        argv = [exe, "api", "-X", "GET", "search/code", "-f", f"q={term} repo:{METAL}", "-f", f"per_page={MAX_ITEMS}"]
        try:
            done = run(argv, capture_output=True, text=True, timeout=GH_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            res.unavailable.append(Unavailable("github code search", f"gh failed: {type(exc).__name__}"))
            continue
        if done.returncode != 0:
            first = (done.stderr or "").strip().splitlines()[:1]
            res.unavailable.append(Unavailable("github code search", clean("gh failed: " + (first[0] if first else f"exit {done.returncode}"))))
            continue
        try:
            items = json.loads(done.stdout).get("items") or []
        except (ValueError, AttributeError):
            res.unavailable.append(Unavailable("github code search", "gh gave no JSON"))
            continue
        for item in items[:MAX_ITEMS]:
            if isinstance(item, dict) and str(item.get("html_url", "")).startswith(GITHUB_PREFIX):
                res.evidence.append(Evidence("github", clean(f"tt-metal code mentions {term} in {item.get('path')}"),
                                             item["html_url"], INFERRED, now))
    return res


def catalog(family: str, *, run=subprocess.run, which=shutil.which, now: str = "") -> Result:
    """Bundles in the community catalog for this family (`tt model search <family> --catalog --json`)."""
    res = Result(data={"bundles": []})
    exe = which("tt")
    if not exe:
        res.unavailable.append(Unavailable("catalog", "the `tt` command is not installed"))
        return res
    try:
        done = run([exe, "model", "search", family, "--catalog", "--json"], capture_output=True, text=True,
                   timeout=nearest.SEARCH_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        res.unavailable.append(Unavailable("catalog", f"{type(exc).__name__}"))
        return res
    got = nearest.parse_search(done.stdout) if done.returncode == 0 else None
    if got is None:
        res.unavailable.append(Unavailable("catalog", f"`tt model search --catalog` gave no answer (exit {done.returncode})"))
        return res
    res.data["bundles"] = got
    for b in got:
        res.evidence.append(Evidence("catalog", clean(f"the community catalog lists {b['name']}"),
                                     f"https://huggingface.co/{b['name']}", INFERRED, now))
    return res


def bundle_code(siblings, bundle: str, now: str) -> list[Evidence]:
    """A bundle's file list names the code it ships: code/models/autoports/<model>/ or code/models/demos/<model>/."""
    seen: list[str] = []
    for item in siblings if isinstance(siblings, list) else []:
        name = item.get("rfilename") if isinstance(item, dict) else None
        m = re.match(r"code/models/(?:autoports|demos)/([A-Za-z0-9_.-]+)/", name or "")
        if m and m.group(1) not in seen and m.group(1) not in NOT_A_MODEL:
            seen.append(m.group(1))
    return [Evidence("bundle", clean(f"bundle {bundle} ships code for {model}"), f"https://huggingface.co/{bundle}",
                     INFERRED, now) for model in seen]


def collect(fetch, config, *, family: str, run=subprocess.run, which=shutil.which, no_github: bool = False) -> Result:
    t = terms(config)
    res = Result()
    res.extend(metal(fetch, t))
    cat = catalog(family, run=run, which=which, now=fetch.now())
    res.extend(cat)
    res.data["catalog"] = cat.data["bundles"]
    if not no_github:
        res.extend(issues(fetch, t))
        res.extend(github_code(t, run=run, which=which, now=fetch.now()))
    return res
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_ports.py -q`
Expected: all pass. If a recorded-fixture test fails because the real file differs from the draft assertion, change the assertion to what the recording shows (and say so in the commit message).

- [ ] **Step 5: Mutations**

| Mutation | Red test |
|---|---|
| drop the `SAFE_TERM` filter on classes | `test_odd_configs_give_no_terms...` and `test_a_term_that_could_change_the_query_is_never_passed_to_gh` |
| issue links not checked for the GitHub prefix | `test_issues_become_weak_evidence...` |
| issue titles not cleaned | same test |
| the `catalog` call drops `--catalog` | `test_the_catalog_search_uses_tt_and_the_catalog_flag` |
| `match_names` marks every hit inferred | `test_a_specific_match_is_inferred_and_a_family_only_match_is_weak` |
| `collect` ignores `no_github` | `test_collect_skips_github_when_asked...` |
| `github_code` passes the query through a shell string | `test_code_search_runs_gh_with_a_checked_term_as_one_argument` |
| `bundle_code` also reports `common`, `utils` and `perf` (drop `NOT_A_MODEL`) | `test_the_recorded_gemma_bundle_listing_yields_one_port_row_and_ignores_shared_code` |
| `terms` stops checking class names against `SAFE_TERM` | `test_odd_configs_give_no_terms_not_a_crash_and_no_unsafe_class_name` (and the `gh` test, which reads the same terms) |

- [ ] **Step 6: Commit**

```bash
git add orchard/research/ports.py tests/test_research_ports.py tests/fixtures/research
git commit -m "research: existing ports (catalog, tt-metal, GitHub issues and code, bundle code)"
```

---

### Task 6: The report

**Files:**
- Create: `orchard/research/report.py`
- Test: `tests/test_research_report.py`

**Interfaces:**
- Consumes: everything above, plus `nearest.Installed` (`name`, `weights_repo`, `schema`, `chips`), `nearest.family_query`, and a `hub_info(repo) -> (HubInfo | None, error)` callable (`HubInfo.files`, `.total_bytes`).
- Produces: `build(*, model_id, hub, base_override, installed, search, hub_info, fetch, run=subprocess.run, which=shutil.which, no_github=False) -> dict`, `write(run_dir, report) -> Path` (the JSON path), `markdown(report) -> str`.

Report shape (all keys always present):

```
{"model", "generated_at", "offline": bool, "stated_bases": [ids],
 "candidates": [{"repo", "confidence", "why", "differing", "group", "bundles": [{"name","installed","schema"}],
                 "class_estimate": {...} | None, "download_gb": float | None}],
 "bundle_choices": [{"name","installed","repo","schema","class","confidence","group"}],
 "selected": {"repo","bundle","confidence","class"} | None,
 "ports": [Evidence dicts], "evidence": [Evidence dicts], "unavailable": [Unavailable dicts]}
```

`group`: 0 an installed v6 bundle serves it; 1 a published v6 bundle; 2 a container (5.1) bundle, installed or published; 3 no bundle. Sort key: `(group, CONFIDENCE_RANK, len(differing), CLASS_RANK)`. `selected`: the first candidate with confidence `stated` or `inferred` in group 0; otherwise `None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_report.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research report: ranking, the offline case, and the files it writes (orchard/research/report.py)."""
import json
import struct

import pytest

pytest.importorskip("tokenizers", reason="the class estimate uses the delta-triage template")

from orchard import nearest, preflight as pf  # noqa: E402
from orchard.nearest import Installed  # noqa: E402
from orchard.research import report  # noqa: E402
from orchard.research.http import FetchError  # noqa: E402

NOW = "2026-10-12T00:00:00Z"
CFG = {"model_type": "gemma4", "architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 30,
       "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 262144}
T = {"model.embed_tokens.weight": ("BF16", [100, 64]), "model.layers.0.q.weight": ("BF16", [64, 64])}


def header(tensors):
    blob = json.dumps({k: {"dtype": v[0], "shape": v[1], "data_offsets": [0, 0]} for k, v in tensors.items()}).encode()
    return struct.pack("<Q", len(blob)) + blob


class World:
    """A fake hub, tt-model and GitHub. `json` and `get` answer from tables; a missing url is a 404."""

    def __init__(self, **tables):
        self.json_routes = tables.get("json", {})
        self.files = tables.get("files", {})
        self.offline = tables.get("offline", False)
        self.urls = []

    def json(self, url, **kw):
        self.urls.append(url)
        if self.offline:
            raise FetchError("offline")
        got = self.json_routes.get(url)
        if got is None:
            raise FetchError("HTTP 404")
        if isinstance(got, Exception):
            raise got
        return got

    def get(self, url, *, byte_range=None, **kw):
        self.urls.append(url)
        name = url.rsplit("/", 1)[1]
        if self.offline or name not in self.files:
            raise FetchError("HTTP 404")
        data = self.files[name]
        return data[byte_range[0]:byte_range[1] + 1] if byte_range else data

    def now(self):
        return NOW


def hub_model(id, *tags, config=CFG):
    return {"id": id, "tags": list(tags), "cardData": {}, "config": config, "siblings": []}


def cfg_url(repo):
    return f"https://huggingface.co/{repo}/resolve/main/config.json"


def manifest_url(bundle):
    return f"https://huggingface.co/{bundle}/resolve/main/tt_kernel_manifest.json"


def api(id):
    return f"https://huggingface.co/api/models/{id}"


class Hub:
    def __init__(self, table):
        self.table = table

    def __call__(self, repo):
        files = self.table.get(repo)
        if files is None:
            return None, "HTTP 404"
        return pf.HubInfo(id=repo, sha="abc", private=False, gated=False, license="mit", total_bytes=2_000_000_000,
                          files=files, code_files=[], has_license_file=False), None


def world(extra_json=None, files=None):
    routes = {api("h/new"): hub_model("h/new", "base_model:finetune:g/base"),
              api("g/base"): hub_model("g/base"),
              cfg_url("h/new"): CFG, cfg_url("g/base"): CFG,
              "https://huggingface.co/api/models?filter=base_model:finetune:g/base&sort=downloads&direction=-1&limit=20": []}
    routes.update(extra_json or {})
    f = {"model.safetensors": header(T) + b"\0" * 64}
    f.update(files or {})
    return World(json=routes, files=f)


FILES = ["config.json", "model.safetensors"]
HUBINFO = Hub({"h/new": FILES, "g/base": FILES, "g/pub": FILES, "g/cont": FILES, "g/none": FILES, "g/big": FILES})


def new_hub(*bases):
    card = {"base_model": list(bases)} if bases else {}
    return pf.parse_hub({"id": "h/new", "cardData": card, "siblings": [{"rfilename": n, "size": 1} for n in FILES]})
NO_SEARCH = lambda q: []  # noqa: E731
RUN = lambda *a, **k: (_ for _ in ()).throw(OSError("no subprocess in this test"))  # noqa: E731


def build(w, **over):
    args = dict(model_id="h/new", hub=new_hub("g/base"), base_override=None, installed=[Installed("tt/base-p150", "g/base", "6", 1)], search=NO_SEARCH,
                hub_info=HUBINFO, fetch=w, run=RUN, which=lambda n: None, no_github=True)
    args.update(over)
    return report.build(**args)


def test_the_card_base_with_an_installed_v6_bundle_is_selected_with_its_class():
    r = build(world())
    top = r["candidates"][0]
    assert top["repo"] == "g/base" and top["confidence"] == "stated" and top["group"] == 0
    assert top["class_estimate"]["class"] == "weights-only" and top["download_gb"] == 2.0
    assert r["selected"] == {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "stated", "class": "weights-only"}
    assert r["bundle_choices"][0]["name"] == "tt/base-p150" and r["offline"] is False


def test_every_key_is_always_present():
    r = build(world())
    for key in ("model", "generated_at", "offline", "stated_bases", "candidates", "bundle_choices", "selected",
                "ports", "evidence", "unavailable"):
        assert key in r


def test_a_card_with_no_base_infers_one_from_the_architecture_of_an_installed_bundle():
    w = world({api("h/new"): hub_model("h/new")})
    r = build(w, hub=new_hub())
    top = r["candidates"][0]
    assert top["repo"] == "g/base" and top["confidence"] == "inferred"
    assert r["selected"]["confidence"] == "inferred"


def test_a_candidate_of_another_family_is_left_out():
    other = {**CFG, "model_type": "qwen3"}
    w = world({cfg_url("q/other"): other})
    r = build(w, installed=[Installed("tt/q", "q/other", "6", 1)], hub=new_hub())
    assert all(c["repo"] != "q/other" for c in r["candidates"])


def test_a_weak_candidate_is_listed_but_never_selected():
    w = world({api("h/new"): hub_model("h/new"), cfg_url("g/big"): {**CFG, "num_hidden_layers": 60}})
    r = build(w, hub=new_hub(), installed=[Installed("tt/big", "g/big", "6", 1)])
    big = [c for c in r["candidates"] if c["repo"] == "g/big"][0]
    assert big["confidence"] == "weak" and "num_hidden_layers" in big["differing"]
    assert r["selected"] is None
    assert r["bundle_choices"][0]["confidence"] == "weak"


def test_ranking_puts_installed_v6_before_published_v6_before_container_before_no_bundle():
    w = world({cfg_url("g/pub"): CFG, cfg_url("g/cont"): CFG, cfg_url("g/none"): CFG,
               manifest_url("o/pub"): {"weights": {"repo_id": "g/pub"}, "schema_version": "6"},
               manifest_url("o/cont"): {"weights": {"repo_id": "g/cont"}, "schema_version": "5.1"},
               api("o/pub"): {"siblings": []}, api("o/cont"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/cont", "installed": False}, {"name": "o/pub", "installed": False}],
              base_override="g/none")
    assert [(c["repo"], c["group"], c["confidence"]) for c in r["candidates"]] == [
        ("g/base", 0, "stated"), ("g/pub", 1, "inferred"), ("g/cont", 2, "inferred"), ("g/none", 3, "stated")]
    assert r["selected"]["repo"] == "g/base"
    assert [c["name"] for c in r["bundle_choices"]] == ["tt/base-p150", "o/pub", "o/cont"]


def test_a_published_only_candidate_is_never_selected():
    w = world({api("h/new"): hub_model("h/new"), cfg_url("g/pub"): CFG,
               manifest_url("o/pub"): {"weights": {"repo_id": "g/pub"}, "schema_version": "6"}, api("o/pub"): {"siblings": []}})
    r = build(w, hub=new_hub(), installed=[], search=lambda q: [{"name": "o/pub", "installed": False}])
    assert r["candidates"][0]["group"] == 1 and r["selected"] is None


def test_a_stale_installed_container_bundle_is_group_2():
    r = build(world(), installed=[Installed("tt/base-container", "g/base", "5.1", 1)])
    assert r["candidates"][0]["group"] == 2 and r["selected"] is None


def test_published_bundle_manifests_are_read_and_a_missing_manifest_is_unavailable():
    w = world({api("o/gone"): {"siblings": []}})
    r = build(w, search=lambda q: [{"name": "o/gone", "installed": False}])
    assert any(u["source"] == "bundle manifest" and "o/gone" in u["reason"] for u in r["unavailable"])


def test_a_hostile_bundle_name_from_the_search_is_never_fetched():
    w = world()
    build(w, search=lambda q: [{"name": "../../x", "installed": False}, {"name": "a b/c", "installed": False}])
    assert not any("x/resolve" in u or "a b" in u for u in w.urls)


def test_headers_that_cannot_be_read_leave_no_estimate_and_say_why():
    w = world()
    w.files = {}                                  # no safetensors served at all
    r = build(w)
    assert r["candidates"][0]["class_estimate"] is None
    assert [u["reason"] for u in r["unavailable"] if u["source"] == "headers"] == [
        "h/new: model.safetensors: HTTP 404", "g/base: model.safetensors: HTTP 404"]


def test_offline_makes_no_request_and_says_so():
    w = world()
    w.offline = True
    r = report.build(model_id="h/new", hub=None, base_override=None, installed=[], search=NO_SEARCH,
                     hub_info=HUBINFO, fetch=w, run=RUN, which=lambda n: None, no_github=True)
    assert r["offline"] is True and r["candidates"] == [] and r["selected"] is None
    assert r["unavailable"] == [{"source": "research", "reason": "--offline"}]
    assert w.urls == []


def test_search_that_raises_is_unavailable_and_the_rest_of_the_report_stands():
    def boom(q):
        raise RuntimeError("tt is broken")
    r = build(world(), search=boom)
    assert r["candidates"] and any(u["source"] == "bundle search" for u in r["unavailable"])


def test_the_report_is_json_and_contains_no_home_path():
    r = build(world())
    json.dumps(r)
    assert "/home/" not in json.dumps(r)


def test_write_puts_json_and_markdown_in_the_run_directory_only(tmp_path):
    path = report.write(tmp_path, build(world()))
    assert path == tmp_path / "research" / "base-candidates.json"
    assert json.loads(path.read_text())["model"] == "h/new"
    md = (tmp_path / "research" / "base-candidates.md").read_text()
    assert "**g/base** [stated]" in md and "expected class weights-only" in md and "2.0 GB" in md
    assert sorted(p.name for p in (tmp_path / "research").iterdir()) == ["base-candidates.json", "base-candidates.md"]


def test_the_markdown_lists_unavailable_sources_so_a_gap_is_visible():
    r = build(world(), search=lambda q: [{"name": "o/gone", "installed": False}])
    assert "- bundle manifest: o/gone: HTTP 404" in report.markdown(r)


def test_only_the_top_candidates_get_a_class_estimate():
    repos = [f"g/c{i}" for i in range(8)]
    w = world({cfg_url(r): CFG for r in repos})
    table = Hub({**{r: FILES for r in repos}, "g/base": FILES, "h/new": FILES})
    calls = []

    def counting(repo):
        calls.append(repo)
        return table(repo)
    r = build(w, installed=[Installed(f"tt/c{i}", repo, "6", 1) for i, repo in enumerate(repos)], hub_info=counting)
    assert len(calls) == report.MAX_ESTIMATES
    assert sum(1 for c in r["candidates"] if c["class_estimate"]) == report.MAX_ESTIMATES
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_report.py -q`
Expected: `ModuleNotFoundError: orchard.research.report`.

- [ ] **Step 3: Write the code**

`orchard/research/report.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Merge the collectors into a ranked list of base candidates and write it to the run directory.

A candidate is a model stage 0 could compare against and a bundle could serve. Ranking (design 3.2): an
installed v6 bundle serves it (group 0), a published v6 bundle (1), a container bundle (2), no bundle (3);
then confidence (stated, inferred, weak), fewer differing config keys, and the cheaper class. Only a stated
or inferred candidate in group 0 is `selected`. Nothing here asks a model; every row says why it is there."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from orchard import nearest
from orchard.research import arch_match, headers, lineage, ports
from orchard.research.http import FetchError
from orchard.research.types import (CONFIDENCE_RANK, INFERRED, STATED, WEAK, Result, Unavailable, clean,
                                    valid_id)

MANIFEST_URL = "https://huggingface.co/{name}/resolve/main/tt_kernel_manifest.json"
MAX_PUBLISHED = 12          # published bundles whose manifests are read
MAX_CANDIDATES = 30         # candidate repos whose config.json is read
MAX_ESTIMATES = 5           # candidates whose headers are read for a class estimate
CLASS_RANK = {"weights-only": 0, "weights+sidecar": 1, "full-port": 2, "unknown": 3}
V6 = "6"
RESEARCH_DIR = "research"


def _unique(items) -> list[str]:
    out: list[str] = []
    for item in items:
        if valid_id(item) and item not in out:
            out.append(item)
    return out


def _empty(model_id: str, now: str, offline: bool, reason: str) -> dict:
    return {"model": model_id, "generated_at": now, "offline": offline, "stated_bases": [], "candidates": [],
            "bundle_choices": [], "selected": None, "ports": [], "evidence": [],
            "unavailable": [Unavailable("research", reason).as_dict()]}


def _bundles(installed, search, family: str, fetch, res: Result) -> dict[str, dict]:
    """name -> {"name", "installed", "schema", "repo", "code"} for installed and published bundles."""
    rows = {b.name: {"name": b.name, "installed": True, "schema": str(b.schema), "repo": b.weights_repo, "code": []}
            for b in installed}
    try:
        found = search(family) if search is not None else []
    except Exception as exc:                                  # a failing search is a gap, not a crash
        res.unavailable.append(Unavailable("bundle search", f"{type(exc).__name__}"))
        found = []
    for item in (found or [])[:MAX_PUBLISHED]:
        name = item.get("name") if isinstance(item, dict) else None
        if not valid_id(name) or name in rows:
            continue
        try:
            manifest = fetch.json(MANIFEST_URL.format(name=name))
            repo = manifest["weights"]["repo_id"]
            schema = str(manifest.get("schema_version", ""))
        except (FetchError, KeyError, TypeError, AttributeError) as exc:
            res.unavailable.append(Unavailable("bundle manifest", f"{name}: {getattr(exc, 'reason', 'unreadable')}"))
            continue
        if not valid_id(repo):
            continue
        code = []
        try:
            payload = fetch.json(lineage.HUB_MODEL.format(id=name))
            code = ports.bundle_code(payload.get("siblings"), name, fetch.now()) if isinstance(payload, dict) else []
        except FetchError:
            pass
        rows[name] = {"name": name, "installed": False, "schema": schema, "repo": repo, "code": code}
    return rows


def _note_header_errors(res: Result, repo: str, read: Result) -> None:
    """A header that could not be read leaves the estimate empty; say why, so the gap is visible."""
    errors = read.data.get("errors") or []
    if errors:
        more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
        res.unavailable.append(Unavailable("headers", clean(f"{repo}: {errors[0]}{more}")))


def _group(bundles: list[dict]) -> int:
    if any(b["installed"] and b["schema"] == V6 for b in bundles):
        return 0
    if any(b["schema"] == V6 for b in bundles):
        return 1
    return 2 if bundles else 3


def build(*, model_id, hub, base_override, installed, search, hub_info, fetch, run=subprocess.run,
          which=shutil.which, no_github: bool = False) -> dict:
    now = fetch.now()
    if getattr(fetch, "offline", False):
        return _empty(model_id, now, True, "--offline")
    res = Result()
    lin = lineage.chain(fetch, model_id)
    res.extend(lin)
    stated = _unique([base_override] + list(hub.base_models if hub else []) + lin.data["ancestors"])
    new_cfg = arch_match.fetch_config(fetch, model_id)
    if new_cfg is None:
        res.unavailable.append(Unavailable("config", f"{model_id}: config.json could not be read"))
        new_cfg = {}
    letters = re.match(r"[a-z]+", str(new_cfg.get("model_type") or "").lower())
    family = nearest.family_query(stated[0]) if stated else (letters.group(0) if letters else "")
    sib = lineage.siblings(fetch, stated[0], exclude={model_id}) if stated else Result(data={"siblings": []})
    res.extend(sib)
    sibling_ids = set(sib.data["siblings"])
    rows = _bundles(installed, search, family, fetch, res) if family else {}
    by_repo: dict[str, list[dict]] = {}
    for row in rows.values():
        by_repo.setdefault(row["repo"], []).append(row)
        res.evidence += row["code"]

    candidates: dict[str, dict] = {}
    cfgs: dict[str, dict] = {}
    for repo in _unique(stated + list(by_repo))[:MAX_CANDIDATES]:
        cfg = arch_match.fetch_config(fetch, repo)
        cmp = arch_match.compare(new_cfg, cfg) if cfg is not None and new_cfg else {"confidence": None, "differing": {}}
        if repo in stated:
            confidence, why = STATED, ("named with --base" if repo == base_override else "named as the base model")
        elif repo in sibling_ids:
            confidence, why = INFERRED, f"another fine-tune of {stated[0]}"
        elif cmp["confidence"]:
            confidence = cmp["confidence"]
            why = ("same architecture family and the same decisive config values" if confidence == INFERRED
                   else "same architecture family, other decisive config values: " + ", ".join(cmp["differing"]))
        else:
            continue
        if cfg is not None:
            cfgs[repo] = cfg
        bundles = [{"name": b["name"], "installed": b["installed"], "schema": b["schema"]}
                   for b in sorted(by_repo.get(repo, []), key=lambda b: (not b["installed"], b["schema"] != V6, b["name"]))]
        candidates[repo] = {"repo": repo, "confidence": confidence, "why": clean(why), "differing": cmp["differing"],
                            "group": _group(bundles), "bundles": bundles, "class_estimate": None, "download_gb": None}

    ordered = sorted(candidates.values(), key=lambda c: (c["group"], CONFIDENCE_RANK[c["confidence"]],
                                                         len(c["differing"]), c["repo"]))
    new_model = None
    for cand in [c for c in ordered if c["confidence"] != WEAK][:MAX_ESTIMATES]:
        info, err = hub_info(cand["repo"])
        if info is None:
            res.unavailable.append(Unavailable("class estimate", f"{cand['repo']}: {err}"))
            continue
        cand["download_gb"] = round(info.total_bytes / 1e9, 1)
        if new_model is None:
            new_model = (headers.read_model(fetch, model_id, list(hub.files)) if hub is not None
                         else Result(unavailable=[Unavailable("headers", f"{model_id}: the hub did not answer")]))
            res.extend(Result(unavailable=new_model.unavailable))
            _note_header_errors(res, model_id, new_model)
        other = headers.read_model(fetch, cand["repo"], list(info.files))
        res.extend(Result(unavailable=other.unavailable))
        _note_header_errors(res, cand["repo"], other)
        if new_model.data.get("tensors") and other.data.get("tensors") and cand["repo"] in cfgs:
            cand["class_estimate"] = headers.estimate_class(new_cfg, new_model.data, cfgs[cand["repo"]], other.data)
    ordered.sort(key=lambda c: (c["group"], CONFIDENCE_RANK[c["confidence"]], len(c["differing"]),
                                CLASS_RANK.get((c["class_estimate"] or {}).get("class"), 3), c["repo"]))

    choices = [{"name": b["name"], "installed": b["installed"], "repo": c["repo"], "schema": b["schema"],
                "class": (c["class_estimate"] or {}).get("class"), "confidence": c["confidence"], "group": c["group"]}
               for c in ordered for b in c["bundles"]]
    selected = None
    for c in ordered:
        if c["group"] == 0 and c["confidence"] in (STATED, INFERRED):
            b = next(b for b in c["bundles"] if b["installed"] and b["schema"] == V6)
            selected = {"repo": c["repo"], "bundle": b["name"], "confidence": c["confidence"],
                        "class": (c["class_estimate"] or {}).get("class")}
            break
    port_res = ports.collect(fetch, new_cfg, family=family or "", run=run, which=which, no_github=no_github) \
        if family else Result()
    res.extend(port_res)
    return {"model": model_id, "generated_at": now, "offline": False, "stated_bases": stated,
            "candidates": ordered, "bundle_choices": choices, "selected": selected,
            "ports": [e.as_dict() for e in port_res.evidence],
            "evidence": [e.as_dict() for e in res.evidence if e not in port_res.evidence],
            "unavailable": [u.as_dict() for u in res.unavailable]}


def markdown(report: dict) -> str:
    lines = [f"# Base candidates for {report['model']}", "",
             f"Generated {report['generated_at']}. Every row says where it came from; a source that could not be "
             "read is listed under Unavailable."]
    if report["offline"]:
        lines += ["", "Research was skipped (`--offline`)."]
    lines += ["", "## Candidates", ""]
    for c in report["candidates"]:
        est = c["class_estimate"]
        cls = f"expected class {est['class']}" + (" (a sidecar is possible)" if est["sidecar_possible"] else "") if est \
            else "no class estimate"
        names = ", ".join(f"{b['name']} ({'installed' if b['installed'] else 'published'}, schema {b['schema']})"
                          for b in c["bundles"]) or "no bundle"
        size = f", {c['download_gb']} GB" if c["download_gb"] is not None else ""
        lines.append(f"- **{c['repo']}** [{c['confidence']}] {c['why']}. {cls}{size}. Bundles: {names}.")
    if not report["candidates"]:
        lines.append("- none found")
    lines += ["", "## Existing ports", ""]
    lines += [f"- [{p['confidence']}] {p['claim']} ({p['url']})" for p in report["ports"]] or ["- none found"]
    lines += ["", "## Unavailable", ""]
    lines += [f"- {u['source']}: {u['reason']}" for u in report["unavailable"]] or ["- nothing"]
    return "\n".join(lines) + "\n"


def write(run_dir, report: dict) -> Path:
    """`<run_dir>/research/base-candidates.json` and `.md`. Nothing else is written."""
    out = Path(run_dir) / RESEARCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / "base-candidates.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "base-candidates.md").write_text(markdown(report), encoding="utf-8")
    return path
```

Note for `write`: the Fetcher cache lives in `<run_dir>/research/cache/`, so the "only two files" test uses a Fetcher with `cache_dir=None` (the fake world has no cache). A separate preflight-level test (Task 7) covers the real layout.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_report.py -q`
Expected: all pass.

- [ ] **Step 5: Mutations**

| Mutation | Red test |
|---|---|
| group computed as `0 if bundles else 3` (no schema check) | `test_a_stale_installed_container_bundle_is_group_2` |
| `selected` allows `weak` | `test_a_weak_candidate_is_listed_but_never_selected` |
| `selected` allows group 1 | `test_a_published_only_candidate_is_never_selected` |
| skip `valid_id` for search names | `test_a_hostile_bundle_name_from_the_search_is_never_fetched` |
| search exception propagates | `test_search_that_raises_is_unavailable...` |
| offline still calls the network | `test_offline_makes_no_request_and_says_so` |
| `write` also writes a third file | `test_write_puts_json_and_markdown_in_the_run_directory_only` |
| the markdown prints `- nothing` instead of the unavailable rows | `test_the_markdown_lists_unavailable_sources_so_a_gap_is_visible` |
| a failed header read adds no `headers` row (drop `_note_header_errors`) | `test_headers_that_cannot_be_read_leave_no_estimate_and_say_why` |
| class estimate ignores `MAX_ESTIMATES` | `test_only_the_top_candidates_get_a_class_estimate` |

- [ ] **Step 6: Commit**

```bash
git add orchard/research/report.py tests/test_research_report.py
git commit -m "research: rank base candidates and write base-candidates.json and .md"
```

---

### Task 7: Preflight uses the report

**Files:**
- Modify: `orchard/preflight.py` (`Signals` at line 266, `check_base` at line 186, `run_preflight` at line 311, `default_signals` at line 380)
- Test: `tests/test_research_preflight.py`

**Interfaces:**
- Consumes: `report.build`, `Options`, `Fetcher`.
- Produces: `Signals.research: Callable | None = None` taking `(model_id, hub, base_override, installed) -> dict`; `check_base(..., report: dict | None = None)`; `default_signals(cfg, research_options=None)`. With `report=None` every existing behaviour and message is unchanged (`tests/test_nearest.py`, `tests/test_preflight.py` and `tests/test_cli.py` stay green without edits).

Behaviour of `check_base` with a report:
- The base is the override, else the card's base, else `report["selected"]["repo"]`. An inferred base gives status WARN with the text "inferred from its architecture; the card names none".
- When a bundle serves the base the text gains `; research expects class <class>` if the report has an estimate.
- With no base and no selection, the WARN text gains `Closest bundles found: <name> (<class or "no estimate">, <confidence>), ...` (at most 3).
- On the block, `data["candidates"]` is `bundle_choices` when it is non-empty (each entry has `name` and `installed`, plus `class`, `confidence`, `repo`), the listing reads `ranked bundles: ...`, and `Existing work: ...` names the first two `ports`. Otherwise the earlier listing is used.
- Every return carries `data["report"] = report`.
- `Check` has `.detail` for its text (not `.message`).

- [ ] **Step 1: Write the failing tests**

`tests/test_research_preflight.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""check_base with a research report, and the preflight wiring (orchard/preflight.py)."""
from orchard import preflight as pf
from orchard.nearest import Installed

GEMMA = "google/gemma-4-12B"


def hub(**card):
    return pf.parse_hub({"id": "x/humanizer", "private": False, "gated": False, "sha": "c0",
                         "cardData": {"license": "apache-2.0", **card}, "config": {},
                         "siblings": [{"rfilename": "LICENSE", "size": 1},
                                      {"rfilename": "model.safetensors", "size": 24_000_000_000}]})


def report(**over):
    base = {"model": "x/humanizer", "selected": None, "candidates": [], "bundle_choices": [], "ports": [],
            "evidence": [], "unavailable": [], "offline": False, "stated_bases": [], "generated_at": "t"}
    base.update(over)
    return base


def choice(name, repo, installed, cls="full-port", conf="inferred", group=2, schema="5.1"):
    return {"name": name, "installed": installed, "repo": repo, "schema": schema, "class": cls,
            "confidence": conf, "group": group}


def port(claim, url="https://huggingface.co/x/y"):
    return {"claim": claim, "url": url, "confidence": "inferred", "source": "bundle", "fetched_at": "t"}


def check(rep, *, h=None, override=None, installed=(), snapshot=None):
    return pf.check_base(model_id="x/humanizer", hub=h if h is not None else hub(), base_override=override,
                         local_snapshot=lambda m: snapshot, installed=list(installed), search=lambda q: [], report=rep)


def test_without_a_report_the_messages_are_the_earlier_ones():
    block = check(None, h=hub(base_model=GEMMA))
    assert block.status == pf.BLOCK and block.data["report"] is None
    assert block.detail.endswith("found no bundle. Pick one with --base <bundle>, or install a bundle yourself "
                                  "and run bringup again")
    assert check(None).detail.endswith("Pass --base <bundle or model id> to name the model to start from")


def test_a_card_with_no_base_and_an_inferred_installed_bundle_warns_and_names_the_inference():
    sel = {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "inferred", "class": "weights-only"}
    rep = report(selected=sel, candidates=[{"repo": "g/base", "class_estimate": {"class": "weights-only"}}])
    c = check(rep, installed=[Installed("tt/base-p150", "g/base", "6", 1)])
    assert c.status == pf.WARN
    assert "inferred from its architecture" in c.detail and "g/base" in c.detail
    assert "research expects class weights-only" in c.detail
    assert c.data["base"] == "g/base" and c.data["bundle"] == "tt/base-p150" and c.data["needs_fetch"] is True


def test_a_card_base_wins_over_the_selection():
    sel = {"repo": "g/other", "bundle": "tt/other", "confidence": "inferred", "class": None}
    c = check(report(selected=sel), h=hub(base_model=GEMMA),
              installed=[Installed("tt/g", GEMMA, "6", 1), Installed("tt/other", "g/other", "6", 1)])
    assert c.status == pf.OK and c.data["base"] == GEMMA


def test_a_card_with_no_base_and_no_selection_lists_the_closest_bundles():
    rep = report(bundle_choices=[choice("o/a", "g/a", False, "weights-only", "weak"),
                                 choice("o/b", "g/b", True, None, "inferred")])
    c = check(rep)
    assert c.status == pf.WARN and c.data["base"] is None
    assert "Closest bundles found: o/a (weights-only, weak), o/b (no estimate, inferred)." in c.detail


def test_the_block_lists_the_ranked_bundles_with_class_and_the_first_two_ports():
    rep = report(bundle_choices=[choice("stisiTT/gemma-4-12b-it-p150", "google/gemma-4-12B-it", False,
                                        "weights-only", "inferred"),
                                 choice("tt-hous/gemma-4-26B", "google/gemma-4-26B", False, "full-port", "weak", 3)],
                 ports=[port("bundle stisiTT/x ships code for gemma4"), port("tt-metal has models/demos/gemma3"),
                        port("third")])
    c = check(rep, h=hub(base_model=GEMMA))
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"
    assert [x["name"] for x in c.data["candidates"]] == ["stisiTT/gemma-4-12b-it-p150", "tt-hous/gemma-4-26B"]
    assert ("ranked bundles: stisiTT/gemma-4-12b-it-p150 (weights-only, inferred), "
            "tt-hous/gemma-4-26B (full-port, weak).") in c.detail
    assert "Existing work: bundle stisiTT/x ships code for gemma4 (https://huggingface.co/x/y); " \
           "tt-metal has models/demos/gemma3 (https://huggingface.co/x/y)." in c.detail
    assert "third" not in c.detail


def test_a_report_with_no_choices_keeps_the_old_listing():
    c = check(report(), h=hub(base_model=GEMMA))
    assert c.status == pf.BLOCK and c.data["candidates"] == [] and "found no bundle" in c.detail


def test_an_installed_bundle_that_serves_the_base_gets_the_class_note():
    rep = report(candidates=[{"repo": GEMMA, "class_estimate": {"class": "weights-only"}}])
    c = check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "6", 1)])
    assert c.status == pf.OK and c.detail.endswith("its snapshot will be downloaded; research expects class weights-only")


def test_a_container_bundle_that_serves_the_base_still_warns_with_the_class_note():
    rep = report(candidates=[{"repo": GEMMA, "class_estimate": {"class": "full-port"}}])
    c = check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "5.1", 1)])
    assert c.status == pf.WARN and "5.1 container package" in c.detail and "research expects class full-port" in c.detail


def test_the_report_is_carried_in_every_result():
    rep = report()
    assert check(rep).data["report"] is rep
    assert check(rep, h=hub(base_model=GEMMA)).data["report"] is rep
    assert check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "6", 1)]).data["report"] is rep


# ---- run_preflight wiring ------------------------------------------------------------------------------

def test_run_preflight_gives_research_the_installed_bundles_and_uses_its_report(tmp_path):
    from test_preflight import cfg, signals
    seen = []
    rep = report(bundle_choices=[choice("o/a", "g/a", False, "weights-only", "weak")])

    def research(model_id, h, override, installed):
        seen.append((model_id, h.id, override, [b.name for b in installed]))
        return rep
    s = signals(installed_bundles=lambda: [Installed("tt/q", "Qwen/Qwen3.8-27B", "6", 1)], research=research)
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s, base_override=None)
    base = next(c for c in out if c.name == "base")
    assert seen == [("Cloudflare/clef", "Cloudflare/clef", None, ["tt/q"])]
    assert base.data["report"] is rep and "o/a (weights-only, weak)" in base.detail


def test_a_failing_research_leaves_the_earlier_behaviour_and_no_crash(tmp_path):
    from test_preflight import cfg, signals

    def research(*a):
        raise RuntimeError("hub down")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(research=research))
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.WARN and base.data["report"] is None


def test_without_research_run_preflight_is_unchanged(tmp_path):
    from test_preflight import cfg, signals
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert next(c for c in out if c.name == "base").data["report"] is None


def test_default_signals_builds_research_only_when_options_are_given(tmp_path):
    from test_preflight import cfg
    from orchard.research import Options
    assert pf.default_signals(cfg(tmp_path)).research is None
    assert callable(pf.default_signals(cfg(tmp_path), research_options=Options(offline=True)).research)


def test_default_signals_research_with_offline_makes_no_request(tmp_path):
    from test_preflight import cfg
    from orchard.research import Options
    sig = pf.default_signals(cfg(tmp_path), research_options=Options(offline=True))
    rep = sig.research("a/b", None, None, [])
    assert rep["offline"] is True and rep["unavailable"] == [{"source": "research", "reason": "--offline"}]
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_preflight.py -q`
Expected: failures, `TypeError: check_base() got an unexpected keyword argument 'report'`.

- [ ] **Step 3: Write the code**

In `orchard/preflight.py`:

1. Add to `Signals`, after `lab_status`:

```python
    research: Callable | None = None            # (model_id, hub, base_override, installed) -> report dict
```

2. Replace `check_base` (from `def check_base(` up to, not including, `def check_gozer(`) with the following. Keep `from orchard import nearest` inside it as before.

```python
def _class_note(report, repo: str) -> str:
    for c in (report or {}).get("candidates", []):
        est = c.get("class_estimate")
        if c.get("repo") == repo and est:
            return f"; research expects class {est['class']}"
    return ""


def _closest(report) -> str:
    choices = (report or {}).get("bundle_choices") or []
    if not choices:
        return ""
    return " Closest bundles found: " + ", ".join(
        f"{c['name']} ({c.get('class') or 'no estimate'}, {c['confidence']})" for c in choices[:3]) + "."


def _port_note(report) -> str:
    found = (report or {}).get("ports") or []
    if not found:
        return ""
    return " Existing work: " + "; ".join(f"{p['claim']} ({p['url']})" for p in found[:2]) + "."


def check_base(*, model_id: str, hub: HubInfo | None, base_override: str | None, local_snapshot: Callable,
               installed: list, search: Callable | None, report: dict | None = None) -> Check:
    """The model this run is compared with and served through. It needs an installed bundle (stages 2 and
    4 serve the new weights with it) and a snapshot on disk (stage 0 compares tensors and configs). A missing
    snapshot is downloaded by the command. A missing bundle is a block that lists what `tt model search`
    found, so the operator can pick one with --base. `report` is the research report
    (orchard/research/report.py): it supplies an inferred base when the card names none, the expected
    class, ranked bundle choices and existing ports. Without a report the behaviour is the earlier one."""
    from orchard import nearest
    base = base_override or (hub.base_models[0] if hub and hub.base_models else None)
    inferred = False
    if not base and report and report.get("selected"):
        base, inferred = report["selected"]["repo"], True
    if not base:
        return Check("base", WARN, "the model card names no base_model, so stage 0 has nothing to compare it "
                     "with. Pass --base <bundle or model id> to name the model to start from"
                     + _closest(report) + _port_note(report),
                     data={"base": None, "snapshot": None, "needs_fetch": False, "candidates": [], "report": report})
    serving = [b for b in installed if b.weights_repo == base]
    if serving:
        snap = local_snapshot(base)
        data = {"base": base, "snapshot": snap, "bundle": serving[0].name, "needs_fetch": snap is None,
                "candidates": [], "report": report}
        where = "its snapshot is local" if snap else "its snapshot will be downloaded"
        note = _class_note(report, base)
        if inferred:
            return Check("base", WARN, f"base {base}: inferred from its architecture; the card names none. Bundle "
                         f"{serving[0].name} is installed; {where}{note}. Pass --base to choose another", data=data)
        if not any(b.schema == "6" for b in serving):
            return Check("base", WARN, f"base {base}: bundle {serving[0].name} is installed, but it is a "
                         f"{serving[0].schema} container package and stage 2's swap check expects a v6 thin "
                         f"bundle, so stage 2 will probably not run; {where}{note}", data=data)
        return Check("base", OK, f"base {base}: bundle {serving[0].name} is installed; {where}{note}", data=data)
    family = nearest.family_query(base)
    found = None
    if search is not None:
        try:
            found = search(family)
        except Exception:
            found = None
    choices = (report or {}).get("bundle_choices") or []
    if choices:
        ordered = [dict(c) for c in choices]
        listing = "ranked bundles: " + ", ".join(
            f"{c['name']} ({c.get('class') or 'no estimate'}, {c['confidence']})" for c in ordered[:5])
    else:
        cands: dict[str, dict] = {}
        for b in installed:
            if nearest.family_query(b.weights_repo) == family and b.name != model_id:
                cands[b.name] = {"name": b.name, "installed": True}
        for c in found or []:
            if c["name"] != model_id and c["name"] not in cands:
                cands[c["name"]] = {"name": c["name"], "installed": c.get("installed")}
        ordered = sorted(cands.values(), key=lambda c: (not c["installed"], c["name"]))
        if found is None:
            listing = f"the search for '{family}' bundles could not be made"
        elif ordered:
            listing = f"`tt model search {family}` found: " + ", ".join(c["name"] for c in ordered)
        else:
            listing = f"`tt model search {family}` found no bundle"
    return Check("base", BLOCK, f"no installed bundle serves {base}, the model this one is based on, so stages "
                 f"0, 2 and 4 have nothing to compare it with or serve it through. {listing}.{_port_note(report)} "
                 "Pick one with --base <bundle>, or install a bundle yourself and run bringup again",
                 "nearest-model-missing", data={"base": base, "snapshot": None, "bundle": None,
                                                "needs_fetch": False, "candidates": ordered, "report": report})
```

3. In `run_preflight`, replace the body of `base()`:

```python
    def base() -> Check:
        inst = list(s.installed_bundles()) if s.installed_bundles else []
        report = None
        if s.research is not None:
            try:
                report = s.research(model_id, hub, base_override, inst)
            except Exception:                         # research is an aid; its failure never blocks a run
                report = None
        return check_base(model_id=model_id, hub=hub, base_override=base_override,
                          local_snapshot=s.local_snapshot, installed=inst, search=s.search_bundles, report=report)
```

4. `default_signals(cfg, research_options=None)`: change the signature, and replace `    from orchard import nearest\n    return Signals(` with:

```python
    from orchard import nearest
    research = None
    if research_options is not None:
        from orchard.research import report as research_report
        from orchard.research.http import Fetcher
        fetcher = Fetcher(research_options.cache_dir, offline=research_options.offline,
                          refresh=research_options.refresh)
        research = lambda model_id, hub, override, inst: research_report.build(     # noqa: E731
            model_id=model_id, hub=hub, base_override=override, installed=inst, search=nearest.search,
            hub_info=hub_info, fetch=fetcher, no_github=research_options.no_github)
    return Signals(
```

and add `research=research` as the last argument of that `Signals(...)` call.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_research_preflight.py tests/test_nearest.py tests/test_preflight.py tests/test_cli.py -q`
Expected: all pass, and the three older files unchanged.

- [ ] **Step 5: Mutations** (each seen red when the plan was checked on 2026-10-12)

| Mutation | Red test |
|---|---|
| inferred base reported as OK | `test_a_card_with_no_base_and_an_inferred_installed_bundle_warns_and_names_the_inference` |
| the selection used even when the card names a base | `test_a_card_base_wins_over_the_selection` |
| research exception propagates | `test_a_failing_research_leaves_the_earlier_behaviour_and_no_crash` |
| a space added before `_port_note` in the block text | `test_without_a_report_the_messages_are_the_earlier_ones` |
| `data["report"]` dropped on the serving branch | `test_the_report_is_carried_in_every_result` |
| the class note dropped | `test_a_card_with_no_base_and_an_inferred_installed_bundle_warns_and_names_the_inference` and `test_an_installed_bundle_that_serves_the_base_gets_the_class_note` |
| ports not limited to two | `test_the_block_lists_the_ranked_bundles_with_class_and_the_first_two_ports` |
| research built when no options are given | `test_default_signals_builds_research_only_when_options_are_given` |

- [ ] **Step 6: Commit**

```bash
git add orchard/preflight.py tests/test_research_preflight.py
git commit -m "preflight: the base check uses the research report (inferred base, ranked bundles, expected class, ports)"
```

---

### Task 8: The `bringup` command

**Files:**
- Modify: `orchard/cli.py` (the `bringup` parser after `--quiet`; the `bringup` body around `sig = signals or preflight.default_signals(cfg)` and `inputs = {...}`)
- Modify: `README.md` (the command block in 4.0, the `base` row of the checks table, a new subsection before 4.1)
- Test: `tests/test_research_cli.py`

**Interfaces:**
- Consumes: `Options`, `report.write`, `preflight.default_signals(cfg, research_options=...)`.
- Produces: flags `--offline`, `--no-github`, `--refresh-research` on `bringup`; the file `<run_dir>/research/base-candidates.json` and the supervisor input `base_candidates=<that path>`.
- Test helpers: `tests/test_cli.py` already has `conf` (a fixture), `run(conf, args, rec=None, sig=None, ...)`, `Recorder`, `Pipe` and `signals(**over)`. Tests import them as `from test_cli import ...` (pytest puts `tests/` on `sys.path`; `from tests.x import ...` would find an unrelated package in the shared venv). `tests/test_readme.py` already fails when a `tt-orchard` flag is missing from the README, which is why the README step is part of this task.

- [ ] **Step 1: Write the failing tests**

`tests/test_research_cli.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` and the research report: flags, files, and the input handed to the supervisor."""
import json

from test_cli import Pipe, Recorder, run, signals  # noqa: F401  (the existing helpers)
from test_cli import conf  # noqa: F401  (a pytest fixture)

from orchard import cli, preflight

REPORT = {"model": "Cloudflare/clef", "generated_at": "2026-10-12T00:00:00Z", "offline": False, "stated_bases": [],
          "candidates": [], "bundle_choices": [], "selected": None, "ports": [], "evidence": [], "unavailable": []}


def with_report(**over):
    return signals(research=lambda model_id, hub, override, inst: REPORT, **over)


def run_dir_of(tmp_path):
    return tmp_path / "runs" / "cloudflare--clef"


def test_the_new_flags_exist_and_default_off():
    ns = cli._parser().parse_args(["bringup", "a/b"])
    assert (ns.offline, ns.no_github, ns.refresh_research) == (False, False, False)
    ns = cli._parser().parse_args(["bringup", "a/b", "--offline", "--no-github", "--refresh-research"])
    assert (ns.offline, ns.no_github, ns.refresh_research) == (True, True, True)


def spied_main(conf, args, monkeypatch):
    """cli.main with no injected signals, so cli builds them; returns the options it passed on."""
    seen = {}

    def fake_default(cfg, research_options=None):
        seen["options"] = research_options
        return signals()
    monkeypatch.setattr(preflight, "default_signals", fake_default)
    rec = Recorder()
    code = cli.main(["--config", str(conf), *args], env={}, stdout=Pipe(), supervisor_main=rec.supervisor,
                    fetcher=rec.fetcher)
    return seen["options"], code, rec


def test_a_dry_run_builds_signals_with_no_cache_directory(conf, monkeypatch):
    options, code, rec = spied_main(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--offline"], monkeypatch)
    assert options.cache_dir is None and options.offline is True and rec.argv is None


def test_a_real_start_caches_under_the_run_directory_and_passes_the_flags_on(conf, tmp_path, monkeypatch):
    options, code, rec = spied_main(conf, ["bringup", "Cloudflare/clef", "--no-github", "--refresh-research"],
                                    monkeypatch)
    assert options.cache_dir == run_dir_of(tmp_path) / "research" / "cache"
    assert (options.no_github, options.refresh, options.offline) == (True, True, False)


def test_the_report_is_written_and_handed_to_the_supervisor_as_an_input(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=with_report())
    path = run_dir_of(tmp_path) / "research" / "base-candidates.json"
    assert code == 0 and json.loads(path.read_text())["model"] == "Cloudflare/clef"
    i = rec.argv.index(f"base_candidates={path}")
    assert rec.argv[i - 1] == "--input"


def test_a_dry_run_never_writes_the_report_or_names_it_in_the_command(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=with_report())
    assert code == 0 and rec.argv is None
    assert not (run_dir_of(tmp_path) / "research").exists() and "base_candidates" not in out


def test_a_blocked_preflight_writes_no_report_and_starts_nothing(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=with_report(gozer_status=lambda: ""))
    assert code == 2 and rec.argv is None and not (run_dir_of(tmp_path) / "research").exists()


def test_without_a_report_no_input_is_added(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=signals())
    assert code == 0 and not any("base_candidates" in a for a in rec.argv)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_research_cli.py -q`
Expected: failures (`unrecognized arguments: --offline`, `AttributeError` on `ns.offline`).

- [ ] **Step 3: Write the code**

In `orchard/cli.py`, after the `--quiet` argument of the `bringup` parser add:

```python
    b.add_argument("--offline", action="store_true",
                   help="do not look anything up online: the base-model research is skipped (the preflight's "
                        "own hub lookup still happens)")
    b.add_argument("--no-github", action="store_true",
                   help="skip the GitHub part of the base-model research")
    b.add_argument("--refresh-research", action="store_true",
                   help="ignore the cached answers from an earlier attempt of this run")
```

In the `bringup` body, replace `sig = signals or preflight.default_signals(cfg)` with:

```python
    from orchard.research import Options
    research_options = Options(cache_dir=None if args.dry_run else run_dir / "research" / "cache",
                               offline=args.offline, no_github=args.no_github, refresh=args.refresh_research)
    sig = signals or preflight.default_signals(cfg, research_options=research_options)
```

and replace the line `inputs = {"model": str(snapshot), **({"base": str(base_snapshot)} if base_snapshot else {})}` with:

```python
    candidates_path = None
    if base_data.get("report") and not args.dry_run:
        from orchard.research import report as research_report
        candidates_path = research_report.write(run_dir, base_data["report"])
    inputs = {"model": str(snapshot), **({"base": str(base_snapshot)} if base_snapshot else {}),
              **({"base_candidates": str(candidates_path)} if candidates_path else {})}
```

A dry run leaves `candidates_path` as `None`, so the command it prints names no file that was not written. The report is written after the preflight passed, so a blocked run writes nothing.

In `README.md`:

(a) In the command block of section 4.0, after the `--base` line, add:

```
tt-orchard bringup org/name --offline                         # skip the base-model research; no lookups online
tt-orchard bringup org/name --no-github                       # research without the GitHub searches
tt-orchard bringup org/name --refresh-research                # ignore answers cached by an earlier attempt
```

(b) In the `base` row of the checks table, replace `A card with no `base_model` is a warning |` with:

```
A card with no `base_model` is a warning. If an installed v6 bundle has the model's architecture, it becomes an inferred base, also a warning (see "How the base model is found") |
```

(c) Before the heading `### 4.1 Before you start`, add this section:

```markdown
#### How the base model is found

Stage 0 compares the new model with a model that already runs on the chips, and later stages serve the new
weights through that model's bundle. `bringup` researches which model that should be before it downloads
anything large. It reads, with no Hugging Face token:

- the Hugging Face hub: the card's `base_model` and its relation (fine-tune, quantized, adapter, merge), the
  chain of bases above it, and other fine-tunes of the same base;
- each candidate's `config.json`, to compare the architecture (model type and the layer, width, head and
  vocabulary sizes);
- the safetensors headers of the new model and a candidate, by two byte-range requests per file, to
  estimate the class (`weights-only`, `full-port` or `unknown`) and the download size. A sidecar's
  tensors would need a download, so the estimate only says one is possible;
- the tt-model community catalog, published bundles (their manifest says which model they serve, and a
  container bundle may ship its own port under `code/models/`), tt-metal's `models/` directories and
  models README, and GitHub issues and pull requests for the architecture name. GitHub code search needs
  `gh auth login`; without it that one search is skipped and the report says so.

The result is `<run dir>/research/base-candidates.md` and `.json`. Each row says where it came from and how
sure it is: `stated` (the card or `--base` names it), `inferred` (same architecture family and the same
decisive config values) or `weak` (same family only). Only a stated or inferred candidate with an installed v6
bundle is chosen on its own. Stage 0 gets the file as the input `base_candidates` and records in
`delta.json` whether its measured class agreed with the estimate. The measurement decides; the estimate is
a hint.

A source that cannot be reached is listed as unavailable in the report. `--offline` skips all of it.
`--dry-run` does the same lookups and prints what it finds, but keeps no cache and writes no file.
```

- [ ] **Step 4: The supervisor accepts a run directory that already holds `research/`**

Checked by reading `orchard/supervisor.py`: every directory it creates uses `mkdir(..., exist_ok=True)`, and it
never tests that the run directory is empty. Nothing to change. Confirm once more by running
`python -m pytest tests/test_run_e2e.py -q` after creating `research/` in that test's run directory if the
e2e helper makes that easy; if it does not, say in the commit message that the check was a read of the code.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_research_cli.py tests/test_cli.py tests/test_readme.py -q`
Expected: pass.

- [ ] **Step 6: Mutations** (each seen red when the plan was checked on 2026-10-12)

| Mutation | Red test |
|---|---|
| a dry run gets a cache directory | `test_a_dry_run_builds_signals_with_no_cache_directory` |
| a dry run writes the report | `test_a_dry_run_never_writes_the_report_or_names_it_in_the_command` |
| `--offline` not passed to the options | `test_a_dry_run_builds_signals_with_no_cache_directory` |
| the input key renamed | `test_the_report_is_written_and_handed_to_the_supervisor_as_an_input` |
| `--no-github` not passed on | `test_a_real_start_caches_under_the_run_directory_and_passes_the_flags_on` |
| the report written before the block check | `test_a_blocked_preflight_writes_no_report_and_starts_nothing` once the mutation also skips the dry-run condition (the dry-run test goes red first) |

- [ ] **Step 7: Commit**

```bash
git add orchard/cli.py README.md tests/test_research_cli.py
git commit -m "bringup: research flags, the report in the run directory, and base_candidates as a supervisor input"
```

---

### Task 9: Stage 0 reads the report

**Files:**
- Modify: `orchard/skills/delta-triage-templates/delta_triage.py` (`CONFIG_KEYS` at line 82, a new function after `with_revision`, `main`)
- Modify: `orchard/skills/delta-triage.md` (steps 1 and 2)
- Modify: `tests/test_skills.py` (`test_the_delta_triage_skill_runs_the_template_that_exists_in_this_repo`)
- Test: `tests/test_delta_triage_template.py` (append)

**Interfaces:**
- Consumes: the file `base-candidates.json` (`candidates[].repo`, `.confidence`, `.class_estimate.class`).
- Produces: an optional `base_candidates` key in `triage_config.json` (a path; the template lists it in a new `OPTIONAL_KEYS`), and, when the key is present, `delta.json["base_research"] = {"candidate": {"repo", "confidence"} | None, "estimated_class": str | None, "measured_class": str, "agreed": bool | None}`. The class and the path come from the measurement alone. `gate_delta` accepts the extra key (checked by a test below).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_delta_triage_template.py` (it already has `text_case`, `run_triage`, `delta`, `gate_delta`):

```python
# ---- the research report as an optional input -------------------------------------------------------

def write_candidates(stage, rows):
    path = stage.parent.parent / "research" / "base-candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": rows}))
    cfg = json.loads((stage / "triage_config.json").read_text())
    cfg["base_candidates"] = str(path)
    (stage / "triage_config.json").write_text(json.dumps(cfg))
    return path


ROW = {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated", "class_estimate": {"class": "weights-only"}}


def test_the_draft_records_how_the_research_estimate_compared_with_the_measurement(text_case):
    stage = text_case()
    write_candidates(stage, [ROW])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    assert br["candidate"] == {"repo": "Qwen/Qwen3.8-27B", "confidence": "stated"}
    assert (br["estimated_class"], br["measured_class"], br["agreed"]) == ("weights-only", "weights-only", True)


def test_a_disagreement_is_recorded_and_changes_nothing_else(text_case):
    stage = text_case()
    write_candidates(stage, [{**ROW, "class_estimate": {"class": "full-port"}}])
    run_triage(stage)
    d = delta(stage)
    assert d["base_research"]["agreed"] is False and d["path"] == "weights-only" and d["class"] == "weights-only"


@pytest.mark.parametrize("content", ["not json", "[]", '{"candidates": "x"}', '{"candidates": []}',
                                     '{"candidates": [7, null]}'])
def test_an_unreadable_or_empty_report_gives_a_null_candidate_and_no_failure(text_case, content):
    stage = text_case()
    write_candidates(stage, []).write_text(content)
    r = run_triage(stage)
    assert r.returncode == 0, r.stdout + r.stderr
    br = delta(stage)["base_research"]
    assert br["candidate"] is None and br["agreed"] is None and br["estimated_class"] is None


@pytest.mark.parametrize("estimate", [None, "weights-only", ["weights-only"], {"class": None}, {}])
def test_a_missing_or_malformed_estimate_is_ignored(text_case, estimate):
    stage = text_case()
    write_candidates(stage, [{"repo": "Qwen/Qwen3.8-27B", "confidence": "stated", "class_estimate": estimate}])
    assert run_triage(stage).returncode == 0
    br = delta(stage)["base_research"]
    assert br["candidate"]["repo"] == "Qwen/Qwen3.8-27B" and br["estimated_class"] is None and br["agreed"] is None


def test_without_the_key_the_draft_has_no_base_research(text_case):
    stage = text_case()
    run_triage(stage)
    assert "base_research" not in delta(stage)


def test_a_candidate_for_another_repo_is_not_matched(text_case):
    stage = text_case()
    write_candidates(stage, [{**ROW, "repo": "other/model"}])
    run_triage(stage)
    assert delta(stage)["base_research"]["candidate"] is None


def test_the_draft_with_base_research_still_passes_the_stage_0_gate(text_case):
    stage = text_case()
    write_candidates(stage, [ROW])
    run_triage(stage)
    assert "base_research" in delta(stage)
    assert gate_delta(stage, stage.parent.parent).ok
```

- [ ] **Step 2: Run to confirm they fail**

Run: `python -m pytest tests/test_delta_triage_template.py -q -k "research or base_research or candidate or estimate"`
Expected: `KeyError: 'base_research'` and similar.

- [ ] **Step 3: Write the code**

In `delta_triage.py`, after `CONFIG_KEYS = (...)` add:

```python
OPTIONAL_KEYS = ("base_candidates",)      # read when present: the research report (research/base-candidates.json)
```

Add this function before `read_json`:

```python
def base_research(cfg: dict, nearest_id: str, measured_class: str):
    """How the research report's estimate for the nearest model compared with this script's measurement.
    Only recorded; the class and the path come from the measurement. None when `triage_config.json` names no
    report. A report that cannot be read, or has no row for the nearest model, gives a null candidate."""
    path = cfg.get("base_candidates")
    if not isinstance(path, str) or not path.strip():
        return None
    data, _ = read_json(Path(path))
    rows = data.get("candidates") if isinstance(data, dict) else None
    repo = nearest_id.split("@", 1)[0]
    row = (next((r for r in rows if isinstance(r, dict) and r.get("repo") == repo), None)
           if isinstance(rows, list) else None)
    estimate = row.get("class_estimate") if row else None
    est = estimate.get("class") if isinstance(estimate, dict) else None
    return {"candidate": ({"repo": row["repo"], "confidence": row.get("confidence")} if row else None),
            "estimated_class": est, "measured_class": measured_class,
            "agreed": (est == measured_class) if est else None}
```

In `main()`, immediately before the line that writes `delta.json` (the one beginning `(HERE_DIR / "delta.json").write_text(`), add:

```python
    research = base_research(cfg, nearest_id, cls)
    if research is not None:
        delta["base_research"] = research
```

In `orchard/skills/delta-triage.md`, step 1: after the `nearest_snapshot` bullet add

```
   - The run facts may also list an input `base_candidates`: a ranked list of base models that the run's
     research wrote (`research/base-candidates.json`). Use it only to choose `nearest_model_id` when the
     run facts name no `input base` and the README names no base: take the first candidate whose
     `confidence` is `stated` or `inferred`. Do not search the web. The script measures; the list is an
     estimate, and `delta.json` records whether the two agreed.
```

and in step 2 change the JSON example so its last two lines read

```
        "nearest_snapshot": "<nearest model snapshot>",
        "base_candidates": "<the input base_candidates path, when the run facts list one>"}

   Leave the `base_candidates` key out when there is no such input.
```

In `tests/test_skills.py`, `test_the_delta_triage_skill_runs_the_template_that_exists_in_this_repo`: the skill's config example now has one optional key, so change

```python
    assert set(_config_block(text, "run_dir")) == set(mod.CONFIG_KEYS)
```

to

```python
    assert set(_config_block(text, "run_dir")) == set(mod.CONFIG_KEYS) | set(mod.OPTIONAL_KEYS)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_delta_triage_template.py tests/test_skills.py tests/test_stages.py -q`
Expected: pass.

- [ ] **Step 5: Mutations** (each seen red when the plan was checked on 2026-10-12)

| Mutation | Red test |
|---|---|
| `agreed` always True when there is an estimate | `test_a_disagreement_is_recorded_and_changes_nothing_else` |
| the estimate overwrites `delta["class"]` | `test_a_disagreement_is_recorded_and_changes_nothing_else` |
| the repo match keeps the `@revision` | `test_the_draft_records_how_the_research_estimate_compared_with_the_measurement` |
| a missing or unreadable report raises | `test_an_unreadable_or_empty_report_gives_a_null_candidate_and_no_failure` |
| `base_research` written when the key is absent | `test_without_the_key_the_draft_has_no_base_research` |
| any repo matches | `test_a_candidate_for_another_repo_is_not_matched` |
| the estimate not type-checked | `test_a_missing_or_malformed_estimate_is_ignored` |

- [ ] **Step 6: Commit**

```bash
git add orchard/skills/delta-triage-templates/delta_triage.py orchard/skills/delta-triage.md tests/test_delta_triage_template.py tests/test_skills.py
git commit -m "stage 0: record how the research estimate compared with the measured class"
```

---

### Task 10: The opt-in live test, the log and the version

**Files:**
- Create: `tests/test_research_live.py`
- Modify: `CLAUDE.md`, `README.md` (a short "How the base model is found" section), `pyproject.toml`, `orchard/__init__.py`

- [ ] **Step 1: Write the opt-in live test**

`tests/test_research_live.py`:

```python
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
```

- [ ] **Step 2: Run the whole suite, then the live test once**

Run: `python -m pytest -q` (expect the earlier count plus the new tests; the live test is skipped), then
`ORCHARD_LIVE_RESEARCH=1 python -m pytest tests/test_research_live.py -q -s`.

When the plan was checked on 2026-10-12 the same call, run by hand, printed this for `jialinyyzz/humanizer`
(no bundles installed, `gh` logged in, nothing unavailable):

```
stated: ['google/gemma-4-12B']
google/gemma-4-12B-it        inferred  group 2  ['stisiTT/gemma-4-12b-it-p150']  weights-only  24.0 GB  differing: none
google/gemma-4-26B-A4B-it    weak      group 2  ['tt-hous/gemma-4-26B-A4B-it_p150_p150x2_p150x4']  differing: num_hidden_layers, hidden_size, layer_types
google/gemma-4-12B           stated    group 3  no bundle  weights-only  24.0 GB
ports: tt-metal has models/demos/gemma4, gemma4_31b_qb2, gemma4_d_p (inferred); the community catalog lists
       stisiTT/gemma-4-12b-it-p150 (inferred); open GitHub issues and PRs for gemma4 (weak)
```

Read the live output in full, not a truncated view. If the ranking differs because new bundles were published,
describe what came back in the log entry; change the test only when the new result is also right.

- [ ] **Step 3: Check the dry run on this machine**

Run: `tt-orchard bringup jialinyyzz/humanizer --dry-run` and read the base check line. Record the result in the log (what ranked first, what class it expects, what was unavailable).

- [ ] **Step 4: Log, README, version**

Add a dated entry to the Log in `CLAUDE.md`: the original prompt (quoted), the key decisions (deterministic collectors, no model; class rules loaded from the template; CDN redirect allow-list; dry run does the reads but keeps nothing; `gh` only for code search; an inferred base is a warning), the findings from planning (HF CDN redirect; published bundles carry a manifest and tags; GitHub code search needs a login), what was measured in Steps 2 and 3, and what is not done: the UI preflight panel (`_ui_preflight`) does not run research, the sidecar class cannot be estimated from headers, and nothing was run on hardware. Add a README section "How the base model is found" (about 15 lines, plain English) that names the sources, the three flags, the files, and that GitHub code search needs `gh auth login`. Bump `version` in `pyproject.toml` and `__version__` in `orchard/__init__.py` to `0.5.0` (a test ties the two).

- [ ] **Step 5: Final checks and commit**

Run: `python -m pytest -q`, then sweep the new prose for the phrases in the global CLAUDE.md (`, not `, `rather than`, `instead of`, `reads as`, `honest`, `the one`, `which is why`, `This is what`) and fix each hit.

```bash
git add -A tests/test_research_live.py CLAUDE.md README.md pyproject.toml orchard/__init__.py
git commit -m "research: opt-in live test, README, project log, version 0.5.0"
```

---

## Self-review

**Spec coverage.** Section 3 modules: lineage (Task 2), arch_match and headers (Task 4), ports (Task 5), report (Task 6). Shared rules loaded not copied (Task 3). 3.1 confidence and 3.2 ranking (Task 6). 3.3 preflight integration and the `base_candidates` input (Tasks 7, 8, 9). 3.4 flags (Task 8). Section 4 rules: no token, hosts, untrusted text, cache and refresh, run-directory-only writes, header cap (Tasks 1 to 8). Section 5 testing: recorded fixtures (Task 5), mutations (every task), live test (Task 10). Spec amendments (Task 0).

**Checked by running it.** On 2026-10-12 every code block above was written into a scratch worktree of `main` and run.
- Tasks 1 to 9: the new tests pass, and `tests/test_delta_triage_template.py`, `tests/test_nearest.py`, `tests/test_preflight.py`, `tests/test_cli.py`, `tests/test_readme.py` and `tests/test_skills.py` pass with the edits.
- About 70 mutations were applied one at a time (one per guard listed in each task) and each turned the named test red. Eight survived at first and each led to a change in this plan: an unbounded `read()` (a test now checks the argument), an ignored index (a test where the index and the name pattern disagree), a dtype rule that drifted (a test), a missing "unavailable" row after a failed header read (a code change in `report.py` and a test), an unbounded number of estimates (a test), a markdown section that printed no unavailable rows (a stronger assertion), a class estimate that was not type-checked (a test), and an estimate that could overwrite the measured class (placed correctly in the mutation).
- Two plan errors were found by running: `from tests.x import ...` resolves to an unrelated package in the shared venv (use `from test_cli import ...`), and `Check` has `.detail`, not `.message`.
- The live humanizer query (Task 10) was run by hand and its output is recorded in Task 10, Step 2.

**Placeholders.** None left. Run `grep -n "TBD\|TODO\|pytest.fail\|NotImplementedError" docs/superpowers/plans/2026-10-12-base-research.md` to confirm after any edit.

**Type consistency.** `Result.data` keys: lineage `ancestors`, `edges`, `siblings`; headers `tensors`, `shards`, `extra_safetensors`, `errors`; ports `bundles`, `catalog`. Report dict keys are listed once in Task 6 and used by name in Tasks 7 to 9. `check_base(report=...)` and `Signals.research(model_id, hub, base_override, installed)` match between Tasks 6, 7 and 8.

**Review Focus coverage.** Hostile base names: Task 2. Hostile search term: Task 5. Cycles: Task 2. Unsafe shard names: Task 4. Gated base: Tasks 1 and 2. Resume with a cache: Task 1.

**Not in this plan.** The web UI's preflight panel (`_ui_preflight`) runs no research, so the page shows the earlier base check. The sidecar class cannot be estimated from headers. Nothing runs on hardware.
