"""Source protection and atomic publication of individual output files.

Existing outputs are replaced only after a successful write. A failure leaves
the previous file intact and removes the staging file. This is not a multi-file
transaction; reports should be published after the document they describe.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


def same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def ensure_distinct_paths(sources: Iterable[Path], outputs: Iterable[Path]) -> None:
    sources = tuple(Path(path) for path in sources)
    seen: list[Path] = []
    for output in outputs:
        output = Path(output)
        if any(same_path(source, output) for source in sources):
            raise ValueError(f"Plik wyjściowy nie może nadpisywać oryginału: {output}")
        if any(same_path(other, output) for other in seen):
            raise ValueError(f"Kolizja ścieżek plików wynikowych: {output}")
        seen.append(output)


@contextmanager
def atomic_output(target: Path, *, sources: Iterable[Path] = ()) -> Iterator[Path]:
    target = Path(target)
    sources = tuple(sources)
    ensure_distinct_paths(sources, [target])
    target.parent.mkdir(parents=True, exist_ok=True)
    # Close before passing the path to Office libraries (required on Windows).
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".humanize-", suffix=target.suffix, delete=False,
    ) as handle:
        staged = Path(handle.name)
    try:
        yield staged
        # Windows _commit needs a writable descriptor even after the writer closed.
        with staged.open("r+b") as handle:
            os.fsync(handle.fileno())
        from humanize_pl.runtime import checkpoint

        checkpoint("publikowanie pliku")
        ensure_distinct_paths(sources, [target])
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str, *, sources: Iterable[Path] = ()) -> None:
    with atomic_output(path, sources=sources) as staged:
        staged.write_text(text, encoding="utf-8")
