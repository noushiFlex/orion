"""HTTP downloads: verified TLS (certifi), retries with backoff, atomic writes, manifest entry per file."""

import gzip
import logging
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import certifi

from orion import manifest

log = logging.getLogger(__name__)

_TLS = ssl.create_default_context(cafile=certifi.where())
_RETRY_CODES = {429, 500, 502, 503, 504}
USER_AGENT = "Mozilla/5.0 (orion; personal research)"


def get(url: str, headers: dict | None = None, retries: int = 3, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip", **(headers or {})})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_TLS) as r:
                body = r.read()
                return gzip.decompress(body) if r.headers.get("Content-Encoding") == "gzip" else body
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_CODES or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        log.warning("retry %d/%d for %s", attempt, retries, url)
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download(url: str, dest: Path, source: str, headers: dict | None = None,
             validate: Callable[[bytes], None] | None = None) -> Path:
    """Fetch url into dest. validate(body) must raise on unexpected content; nothing is written then."""
    body = get(url, headers)
    if validate:
        validate(body)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    manifest.record(source, dest, url=url, fetched_at=manifest.now())
    log.info("fetched %s (%d bytes)", dest.name, len(body))
    return dest
