"""Katalog kategorii prawniczych i przypisanie dokumentu do jednej z nich.

Two levels, on purpose. `DocumentType` is a *family*: three values that decide
how the engine behaves — which formatting normalization runs, which genre
profile applies, what the gate expects. Families are behaviour, so they stay
small and stable.

A legal category is finer and descriptive: `umowa_najmu`, `wezwanie_do_zaplaty`,
`skarga_kasacyjna`. Categories are what a lawyer actually names a document, and
what a structure blueprint has to attach to, because "required sections" only
means something at that level — an NDA and a lease are both contracts and share
almost no obligatory clauses.

Categories live in `data/categories.yaml` rather than in an enum so that adding
one is a data edit a lawyer can make and review, not a code change. Every
category names exactly one family, so nothing downstream has to learn about
this file to keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from humanize_pl.document import DocumentType

CATALOGUE_PATH = Path(__file__).resolve().parent / "data" / "categories.yaml"

# The category assigned when nothing matched. It exists so that every document
# has a category and none has a guessed one.
UNSPECIFIED = "nieokreslony"

# A strong signal is one that alone nearly settles the category; an ordinary one
# only nudges. Three-to-one keeps a single decisive phrase ahead of three
# incidental ones without letting it win unopposed.
STRONG_WEIGHT = 3
SIGNAL_WEIGHT = 1

# Below this score the winner is not a finding, it is noise from a couple of
# common words, and the document is left unspecified.
MIN_SCORE = 3


class CategoryCatalogueError(RuntimeError):
    pass


@dataclass(frozen=True)
class LegalCategory:
    id: str
    label_pl: str
    family: DocumentType
    signals_strong: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    # Performative markers: phrases by which the document *acts* as this genre.
    # When set, one of them must appear or the category does not qualify at
    # all, however much of its vocabulary is present. Without this gate an
    # essay about employment law scores as an employment contract, because it
    # discusses exactly the same things - topic is not genre.
    requires_any: tuple[str, ...] = ()

    def score(self, lowered_text: str) -> tuple[int, list[str]]:
        """Weighted hits plus the phrases that produced them."""
        if self.requires_any and not any(m in lowered_text for m in self.requires_any):
            return 0, []
        evidence: list[str] = []
        total = 0
        for phrase in self.signals_strong:
            if phrase in lowered_text:
                total += STRONG_WEIGHT
                evidence.append(phrase)
        for phrase in self.signals:
            if phrase in lowered_text:
                total += SIGNAL_WEIGHT
                evidence.append(phrase)
        return total, evidence


@dataclass(frozen=True)
class CategoryGuess:
    category: LegalCategory
    confidence: float
    evidence: tuple[str, ...] = ()

    @property
    def specified(self) -> bool:
        return self.category.id != UNSPECIFIED

    def to_json(self) -> dict:
        return {
            "id": self.category.id,
            "label_pl": self.category.label_pl,
            "family": self.category.family.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def _load(path: Path) -> dict[str, LegalCategory]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise CategoryCatalogueError(f"Nie można wczytać katalogu kategorii: {exc}") from exc
    rows = payload.get("categories")
    if not isinstance(rows, list) or not rows:
        raise CategoryCatalogueError("Katalog kategorii jest pusty.")

    catalogue: dict[str, LegalCategory] = {}
    for row in rows:
        identifier = str(row.get("id", "")).strip()
        if not identifier:
            raise CategoryCatalogueError("Kategoria bez pola `id`.")
        if identifier in catalogue:
            raise CategoryCatalogueError(f"Zduplikowana kategoria: {identifier}")
        try:
            family = DocumentType(str(row["family"]))
        except (KeyError, ValueError) as exc:
            raise CategoryCatalogueError(
                f"Kategoria {identifier} wskazuje nieznaną rodzinę: {row.get('family')!r}"
            ) from exc
        if family is DocumentType.auto:
            raise CategoryCatalogueError(
                f"Kategoria {identifier} nie może wskazywać rodziny `auto`."
            )
        catalogue[identifier] = LegalCategory(
            id=identifier,
            label_pl=str(row.get("label_pl", identifier)),
            family=family,
            signals_strong=tuple(str(v).casefold() for v in row.get("signals_strong") or ()),
            signals=tuple(str(v).casefold() for v in row.get("signals") or ()),
            requires_any=tuple(str(v).casefold() for v in row.get("requires_any") or ()),
        )

    if UNSPECIFIED not in catalogue:
        raise CategoryCatalogueError(
            f"Katalog musi zawierać kategorię {UNSPECIFIED!r} dla nierozpoznanych dokumentów."
        )
    return catalogue


@lru_cache(maxsize=1)
def catalogue() -> dict[str, LegalCategory]:
    """Every known legal category, keyed by id."""
    return _load(CATALOGUE_PATH)


def get(identifier: str) -> LegalCategory:
    try:
        return catalogue()[identifier]
    except KeyError as exc:
        raise CategoryCatalogueError(f"Nieznana kategoria: {identifier}") from exc


def categories_for(family: DocumentType) -> list[LegalCategory]:
    return [row for row in catalogue().values() if row.family is family]


def classify_category(text: str) -> CategoryGuess:
    """Assign the document to exactly one category.

    An unrecognised document is reported as `nieokreslony` rather than pushed
    into the closest-looking bucket. A blueprint chosen from a guessed category
    would report missing sections the document was never supposed to have,
    which is worse than admitting the category is unknown.
    """
    lowered = text.casefold()
    scored: list[tuple[int, LegalCategory, list[str]]] = []
    for row in catalogue().values():
        if row.id == UNSPECIFIED:
            continue
        total, evidence = row.score(lowered)
        if total:
            scored.append((total, row, evidence))

    if not scored:
        return CategoryGuess(get(UNSPECIFIED), 0.0)

    # Ties resolve by id so the same text always yields the same category.
    scored.sort(key=lambda item: (-item[0], item[1].id))
    best_score, winner, evidence = scored[0]
    if best_score < MIN_SCORE:
        return CategoryGuess(get(UNSPECIFIED), 0.0)

    runner_up = scored[1][0] if len(scored) > 1 else 0
    margin = best_score - runner_up
    confidence = min(0.98, 0.45 + 0.05 * best_score + 0.06 * margin)
    return CategoryGuess(winner, round(confidence, 2), tuple(evidence[:5]))
