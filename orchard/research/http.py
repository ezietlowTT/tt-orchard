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
        # Hosts that failed at the connection level (no route, DNS failure, timeout). Later requests to the
        # same host fail at once, so a machine with no network does not wait out the timeout once per request.
        self._unreachable: set[str] = set()
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
            pass                                    # an unwritable cache costs a repeat request on the next run

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
        if parts.hostname in self._unreachable:
            raise FetchError(f"network unreachable: {parts.hostname}")
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
            if isinstance(exc, OSError):
                self._unreachable.add(parts.hostname)
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
