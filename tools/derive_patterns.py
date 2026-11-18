"""Measure the slop fingerprint into `humanize_pl/data/slop_fingerprint.yaml`.

Compares each model's AI text against human text of the same genre with
log-odds under an informative Dirichlet prior, keeps what several models
overuse across many documents and what held-out human text does not share,
and folds the result into the shipped YAML. New terms arrive as `candidate`;
a reviewer's `accepted` / `rejected` and notes survive every re-measurement.

Usage:
    python tools/derive_patterns.py --genre legal \
        --ai haiku=docs_tests/corpus/ai_models/haiku \
        --ai opus=docs_tests/corpus/ai_models/opus \
        --ai sonnet=docs_tests/corpus/ai_models/sonnet --glob 'rewrite_*' --split train \
        --human docs_tests/corpus/saos_train.jsonl \
        --human-holdout docs_tests/corpus/saos_holdout.jsonl \
        --update humanize_pl/data/slop_fingerprint.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from humanize_pl.corpus.fingerprint import (
    MIN_DOC_SHARE,
    MIN_HOLDOUT_RATIO,
    MIN_MODELS,
    PRIOR_SCALE,
    merge_candidates,
    scenario_half,
    slop_candidates,
)
from humanize_pl.corpus.logodds import Z_THRESHOLD
from humanize_pl.detect.fingerprint import GENRES, load_fingerprint

# Below this the ranking is dominated by whichever few documents happen to be
# in the AI corpus, and the derived patterns will not generalise.
MIN_AI_DOCUMENTS = 25


def load_texts(
    path: Path, *, split: str = "all", glob: str = "*", limit: int | None = None
) -> list[str]:
    """Texts of a folder (.txt/.md, else .docx) or of a JSONL file with `text`.

    `split` keeps one half of a folder's scenarios, chosen by file name so the
    same scenario lands in the same half for every model. `glob` narrows a
    folder to one kind of file, e.g. `rewrite_*`.
    """
    if path.is_dir():
        files = [
            file
            for file in sorted(path.glob(glob))
            if split == "all" or scenario_half(file.name) == split
        ]
        texts = [
            file.read_text(encoding="utf-8") for file in files if file.suffix.lower() in {".txt", ".md"}
        ]
        if not texts:
            from humanize_pl.io.docx_io import docx_text

            texts = [
                docx_text(file)
                for file in files
                if file.suffix.lower() == ".docx" and not file.name.startswith("~$")
            ]
        return texts[:limit] if limit else texts

    texts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                texts.append(line)
            if limit and len(texts) >= limit:
                break
    return texts


def _model_folder(value: str) -> tuple[str, Path]:
    model, separator, folder = value.partition("=")
    if not separator or not model or not folder:
        raise argparse.ArgumentTypeError("--ai oczekuje postaci model=ścieżka")
    return model, Path(folder)


def _header(path: Path) -> str:
    """The file's leading comment block, kept across rewrites."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genre", choices=GENRES, required=True)
    parser.add_argument("--ai", type=_model_folder, action="append", required=True)
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    parser.add_argument(
        "--glob", default="*", help="Filtr plików w folderach AI, np. rewrite_*"
    )
    parser.add_argument("--human", type=Path, action="append", required=True)
    parser.add_argument("--human-holdout", type=Path, action="append", default=[])
    parser.add_argument("--human-limit", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--update", type=Path, required=True)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    ai_by_model = {
        model: load_texts(folder, split=args.split, glob=args.glob) for model, folder in args.ai
    }
    human = [text for path in args.human for text in load_texts(path, limit=args.human_limit)]
    holdout = [text for path in args.human_holdout for text in load_texts(path)]
    ai_documents = sum(len(texts) for texts in ai_by_model.values())
    if not ai_documents or not human:
        parser.error("Korpus AI i korpus ludzki muszą być niepuste")

    if ai_documents < MIN_AI_DOCUMENTS:
        print(
            f"UWAGA: tylko {ai_documents} dokumentów AI (zalecane >= {MIN_AI_DOCUMENTS}). "
            "Wyniki traktuj jako orientacyjne, nie jako gotowe wzorce."
        )
    if len(ai_by_model) < MIN_MODELS:
        print(
            "UWAGA: korpus AI pochodzi od jednego modelu. Wpisy dostaną flagę "
            "indicative: zgodności między modelami nie dało się sprawdzić."
        )

    candidates = slop_candidates(
        ai_by_model,
        human,
        genre=args.genre,
        human_holdout=holdout or None,
        min_count=args.min_count,
    )
    measurement = {
        "date": date.today().isoformat(),
        "genre": args.genre,
        "split": args.split,
        "glob": args.glob,
        "ai_documents": {model: len(texts) for model, texts in ai_by_model.items()},
        "human_documents": len(human),
        "holdout_documents": len(holdout),
        "z_threshold": Z_THRESHOLD,
        "min_count": args.min_count,
        "prior_scale": PRIOR_SCALE,
        "min_doc_share": MIN_DOC_SHARE,
        "min_models": MIN_MODELS,
        "min_holdout_ratio": MIN_HOLDOUT_RATIO,
        "candidates": len(candidates),
    }

    document = yaml.safe_load(args.update.read_text(encoding="utf-8")) or {}
    merged = merge_candidates(document, candidates, measurement=measurement)
    args.update.write_text(
        _header(args.update)
        + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    load_fingerprint(args.update)

    print(f"{args.genre}: {len(candidates)} kandydatów, {ai_documents} dokumentów AI")
    for candidate in candidates[: args.top]:
        flag = " [chroniony]" if candidate.protected else ""
        print(
            f"  z={candidate.z_score:6.2f}  {candidate.kind:15s} {candidate.term:32s} "
            f"AI {candidate.ai_per_1000:6.3f}/1000  ludzie {candidate.human_per_1000:6.3f}  "
            f"dok. {candidate.ai_doc_share:.0%}  {','.join(candidate.models)}{flag}"
        )
    missing = merged["measurements"][-1].get("not_remeasured")
    if missing:
        print(f"Nie zmierzono ponownie: {', '.join(missing)}")
    print(f"Zapisano: {args.update}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
