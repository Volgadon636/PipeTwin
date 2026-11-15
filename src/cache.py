"""Content-addressed cache keys for optional reuse of complete preprocessing runs.

The cache is deliberately opt-in.  A key includes source bytes, config, schema and
rule versions, request cutoffs/scenarios, and the code snapshot, so an artifact is
never reused merely because a filename is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

CACHE_VERSION = "pipetwin-cache-v1"


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_snapshot(root: Path | str, files=("src/parsing.py", "src/format_detector.py",
                                           "src/data_cleaner.py", "src/grid_builder.py",
                                           "src/segment_mapper.py", "src/temporal.py",
                                           "src/schema_registry.py", "src/source_registry.py")) -> dict:
    root = Path(root)
    return {name: file_sha256(root / name) for name in files if (root / name).is_file()}


def cache_key(source_paths: Mapping[str, Path | str], config: Mapping, requests,
              *, schema_version="schema-v2", rule_version="rules-v2",
              code_hashes: Mapping[str, str] | None = None) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "sources": {name: file_sha256(path) for name, path in sorted(source_paths.items())},
        "config": config,
        "requests": sorted([{k: str(v) for k, v in r.items()} for r in requests],
                           key=lambda r: (r.get("id", ""), r.get("prediction_date", ""), r.get("case", ""))),
        "schema_version": schema_version,
        "rule_version": rule_version,
        "code": dict(sorted((code_hashes or {}).items())),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RunCache:
    """Tiny manifest cache; hits require an exact key and a complete artifact list."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / key

    def lookup(self, key: str, required=()) -> Path | None:
        directory = self.path_for(key)
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("key") != key:
            return None
        for relative in required:
            if not (directory / relative).is_file():
                return None
        return directory

    def record(self, key: str, metadata: Mapping) -> Path:
        directory = self.path_for(key)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps({"key": key, **metadata}, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return directory
