"""Content-based reuse of completed DOCX items, without a separate cache."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from humanize_pl.version import __version__

SCHEMA = 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_digest(path: Path) -> Any:
    if path.is_file():
        return file_digest(path)
    if path.is_dir():
        return {
            item.relative_to(path).as_posix(): file_digest(item)
            for item in sorted(path.rglob("*"))
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        }
    return None


def processing_digest(settings: Any, layers: dict[str, Any], *, rewriter: Any = None) -> str:
    """Hash effective settings, code/data and resources; persist only the digest.

    Credentials and endpoint settings influence reuse but never enter reports.
    A provider changing weights under an unchanged model ID cannot be detected.
    """
    package = Path(__file__).resolve().parents[1]
    dependencies = {}
    for name in ("python-docx", "regex", "numpy", "pyyaml", "stanza", "torch", "transformers", "sentence-transformers", "morfeusz2", "wordfreq"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    resources = {}
    for name in ("style_profile", "template", "blueprint", "llm_env_file"):
        value = getattr(settings, name)
        if value is not None:
            resources[name] = _resource_digest(Path(value))
    if settings.llm_env_file is None:
        resources["default_env"] = _resource_digest(Path(".env"))
    payload = {
        "schema": SCHEMA, "version": __version__, "python": sys.version,
        "settings": asdict(settings), "resources": resources,
        "layers": {key: value for key, value in layers.items() if key != "hosted_model"},
        "hosted_model": {
            key: layers.get("hosted_model", {}).get(key)
            for key in ("backend", "model", "status", "supports_response_format", "response_format_kind")
        },
        "dependencies": dependencies,
        "code_and_data": {
            path.relative_to(package).as_posix(): file_digest(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}
        },
        "llm_settings": asdict(rewriter.settings) if rewriter is not None else None,
        "environment": {key: value for key, value in os.environ.items() if key.startswith("HUMANIZE_PL_")},
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_completed(path: Path, identity: dict[str, Any], *, target: Path | None) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            return None
        saved = payload.get("resume", {})
        if not isinstance(saved, dict) or saved.get("reusable") is not True:
            return None
        if any(saved.get(key) != value for key, value in identity.items()):
            return None
        if not isinstance(payload.get("applied_changes"), list):
            return None
        if target is not None and saved.get("output_sha256") != file_digest(target):
            return None
        return payload
    except (OSError, ValueError, TypeError):
        return None
