"""Lifecycle of application-owned UI runs; user source directories are out of scope."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

MARKER = ".humanize-run"
_active: set[Path] = set()
_lock = threading.RLock()


def register_run(path: Path) -> None:
    with _lock:
        (path / MARKER).write_text(str(time.time()), encoding="ascii")
        _active.add(path.resolve())


def finish_run(path: Path) -> None:
    with _lock:
        _active.discard(path.resolve())


def runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    with _lock:
        return sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()
                      and path.resolve().parent == root.resolve() and (path / MARKER).is_file()
                      and path.resolve() not in _active)


def delete_run(root: Path, name: str) -> None:
    with _lock:
        root = root.resolve()
        target = root / name
        if Path(name).name != name or target.resolve().parent != root or target.is_symlink():
            raise ValueError("Usuwanie jest ograniczone do katalogów przebiegów aplikacji.")
        if target.resolve() in _active:
            raise ValueError("Przebieg nadal trwa; najpierw go anuluj lub poczekaj na zakończenie.")
        if not (target / MARKER).is_file():
            raise ValueError("Katalog nie jest oznaczonym przebiegiem aplikacji.")
        # Validate the exact resolved target immediately before recursive removal.
        resolved = target.resolve()
        if resolved.parent != root or resolved == root:
            raise ValueError("Nieprawidłowa ścieżka przebiegu.")
        shutil.rmtree(resolved)
        archive = root / f"{name}.zip"
        if archive.is_file() and not archive.is_symlink():
            archive.unlink()


def expire_runs(root: Path, *, days: int = 7, now: float | None = None) -> list[str]:
    if days < 1:
        raise ValueError("Retencja musi wynosić co najmniej jeden dzień.")
    cutoff = (time.time() if now is None else now) - days * 86400
    removed = []
    for path in runs(root):
        if (path / MARKER).stat().st_mtime < cutoff:
            delete_run(root, path.name)
            removed.append(path.name)
    return removed
