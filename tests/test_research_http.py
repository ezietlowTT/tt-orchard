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
    g = fetcher(tmp_path, {HF: Resp(b'{"ok": 1}')})
    assert g.json(HF) == {"ok": 1}      # not {}: a cached failure placeholder would show here


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
