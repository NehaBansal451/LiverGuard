"""Small shared helpers (logging, hashing)."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger; configure the root logger once from LIVERGUARD_LOG_LEVEL."""
    if not logging.getLogger().handlers:
        level = os.environ.get("LIVERGUARD_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
