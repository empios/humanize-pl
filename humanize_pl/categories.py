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
    # What the document calls itself. Found in its title, one of these
    # settles the category before any vocabulary is weighed: on the firm's
    # documents the vocabulary put a deed of gift, two leases of land and a
    # data-processing agreement among the leases, because each says "najemca"
    # or "czynsz" somewhere.
    titles: tuple[str, ...] = ()

    def score(
        self, lowered_text: str, *, shared: frozenset[str] = frozenset()
    ) -> tuple[int, list[str]]:
        """Weighted hits plus the phrases that produced them.

        `shared` gate markers - "zawarta w dniu", "niniejsza umowa", which
        every contract carries - open the gate but score nothing: they say
        "a contract", not which one. Scored, they gave every contract the
        same three points and alphabetical order picked the winner, which is
        how a deed of gift became a lease.
        """
        gate_hits = [phrase for phrase in self.requires_any if phrase in lowered_text]
        if self.requires_any and not gate_hits:
            return 0, []
        # A gate marker is the strongest evidence there is - it is what makes
        # the document perform this genre at all - so it scores as one.
        # Leaving it at zero meant a letter opening "zwracam się z wnioskiem"
        # could fall through as unrecognised for lacking incidental vocabulary.
        own_gate_hits = [phrase for phrase in gate_hits if phrase not in shared]
        evidence: list[str] = list(own_gate_hits)
        total = STRONG_WEIGHT * len(own_gate_hits)
        counted = set(gate_hits)
        for phrase in self.signals_strong:
            if phrase in lowered_text and phrase not in counted:
                total += STRONG_WEIGHT
                counted.add(phrase)
                evidence.append(phrase)
        for phrase in self.signals:
            if phrase in lowered_text and phrase not in counted:
                total += SIGNAL_WEIGHT
                counted.add(phrase)
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
            titles=tuple(str(v).casefold() for v in row.get("titles") or ()),
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


@lru_cache(maxsize=1)
def shared_gate_markers() -> frozenset[str]:
    """Gate phrases three or more categories share: evidence of a genre
    family, not of one category."""
    counts: dict[str, int] = {}
    for row in catalogue().values():
        for phrase in row.requires_any:
            counts[phrase] = counts.get(phrase, 0) + 1
    return frozenset(phrase for phrase, count in counts.items() if count >= 3)


def get(identifier: str) -> LegalCategory:
    try:
        return catalogue()[identifier]
    except KeyError as exc:
        raise CategoryCatalogueError(f"Nieznana kategoria: {identifier}") from exc


def categories_for(family: DocumentType) -> list[LegalCategory]:
    return [row for row in catalogue().values() if row.family is family]


# Words a document's title starts with. A line that opens with one of these,
# near the top and short, is the title; a sentence such as "Umowa zawarta w
# dniu … pomiędzy …" is not, whatever its first word.
_TITLE_WORDS = (
    "umowa", "przedwstępna", "porozumienie", "oświadczenie", "wypowiedzenie", "rozwiązanie",
    "informacja", "klauzula", "aneks", "ugoda", "regulamin", "polityka", "pozew",
)
# Letters put the sender's and addressee's blocks first: the firm's notices
# of termination carry their title on the ninth line.
_TITLE_LINES = 12
_TITLE_MAX_WORDS = 12
_NOT_A_TITLE = ("zawarta", "zawarto", "pomiędzy", "dnia ")
# "UMOWA" alone, or with a number, names no kind: the vocabulary decides.
_BARE_TITLE_WORDS = frozenset({"umowa", "nr", "no."})
TITLE_CONFIDENCE = 0.95
# A contract whose title names no catalogued kind: a loan, an exchange, a
# surety. Its own category has no skeleton, so nothing is checked or drafted.
OTHER_CONTRACT = "umowa_inna"


def document_title(text: str) -> str | None:
    """The document's title, lower-cased, or None when it has none."""
    lines = [line for line in text.split("\n") if line.strip()][:_TITLE_LINES]
    for index, raw in enumerate(lines):
        # Whitespace normalised: a non-breaking space in "Informacja o\xa0prawach"
        # kept the title from matching.
        line = " ".join(raw.strip().strip("#*_„”\"' ").casefold().split())
        if (
            line.startswith(_TITLE_WORDS)
            and len(line.split()) <= _TITLE_MAX_WORDS
            and not line.endswith(".")
            and not any(marker in line for marker in _NOT_A_TITLE)
        ):
            # A title broken over two lines - "PRZEDWSTĘPNA UMOWA" /
            # "SPRZEDAŻY NIERUCHOMOŚCI" - is read whole.
            following = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if following.isupper() and len(following.split()) <= 6:
                line = f"{line} {' '.join(following.casefold().split())}"
            return line
    return None


def classify_category(text: str) -> CategoryGuess:
    """Assign the document to exactly one category.

    What the document calls itself decides first: a title naming a catalogued
    kind settles it, and a contract title naming none makes it `umowa_inna`.
    Without a title, the vocabulary decides as before.

    An unrecognised document is reported as `nieokreslony` rather than pushed
    into the closest-looking bucket. A blueprint chosen from a guessed category
    would report missing sections the document was never supposed to have,
    which is worse than admitting the category is unknown.
    """
    title = document_title(text)
    if title:
        # The kind named first is the kind: "Umowa o dzieło i przeniesienie
        # praw autorskich" is a contract for a work that also transfers
        # rights. Ties go to the longer name.
        named = [
            (title.find(stem), -len(stem), row.id, row)
            for row in catalogue().values()
            for stem in row.titles
            if stem in title
        ]
        if named:
            winner = min(named, key=lambda item: item[:3])[3]
            return CategoryGuess(winner, TITLE_CONFIDENCE, (f"tytuł: {title[:80]}",))
        words = title.split()
        names_a_kind = any(word not in _BARE_TITLE_WORDS and not word[:1].isdigit() for word in words)
        if "umowa" in words[:2] and names_a_kind and OTHER_CONTRACT in catalogue():
            return CategoryGuess(get(OTHER_CONTRACT), TITLE_CONFIDENCE, (f"tytuł: {title[:80]}",))

    lowered = text.casefold()
    scored: list[tuple[int, LegalCategory, list[str]]] = []
    shared = shared_gate_markers()
    for row in catalogue().values():
        if row.id == UNSPECIFIED:
            continue
        total, evidence = row.score(lowered, shared=shared)
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
