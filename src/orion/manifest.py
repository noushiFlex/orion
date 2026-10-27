"""Append-only provenance log of raw files: source, URL, collection time and SHA-256."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from orion.paths import MANIFEST, RAW, ROOT


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(source: str, path: Path, url: str | None, fetched_at: str | None, note: str | None = None) -> dict:
    """fetched_at is None when the collection time is unknown (file predates the manifest)."""
    entry = {
        "source": source,
        "path": str(path.relative_to(ROOT)),
        "url": url,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "fetched_at": fetched_at,
        "note": note,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def latest() -> dict[str, dict]:
    """Most recent entry per path."""
    if not MANIFEST.exists():
        return {}
    out = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["path"]] = e
    return out


def backfill() -> int:
    """Record raw files that predate the manifest. Only their mtime is known, so fetched_at stays null."""
    known = latest()
    n = 0
    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or ".git" in p.parts or str(p.relative_to(ROOT)) in known:
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        record(p.relative_to(RAW).parts[0], p, url=None, fetched_at=None, note=f"backfilled, file mtime {mtime}")
        n += 1
    return n
