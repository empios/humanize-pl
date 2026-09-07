"""Document genres, office style profiles and readiness contracts.

This module deliberately contains no model or DOCX code.  The small public
types are shared by the command line, the flow reports and Python callers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import yaml


class DocumentType(str, Enum):
    auto = "auto"
    client_communication = "client_communication"
    contract = "contract"
    filing_official = "filing_official"


class RewriteBackend(str, Enum):
    rules = "rules"
    hybrid = "hybrid"


class FormatPolicy(str, Enum):
    preserve = "preserve"
    audit = "audit"
    normalize = "normalize"


class ReadinessStatus(str, Enum):
    ready = "ready"
    ready_with_warnings = "ready_with_warnings"
    failed = "failed"


@dataclass(frozen=True)
class GenreProfile:
    document_type: DocumentType
    label_pl: str
    tone: tuple[str, ...]
    terminology: tuple[str, ...]
    structure: tuple[str, ...]
    forbidden_phrases: tuple[str, ...]

    def prompt_text(self) -> str:
        return (
            f"Gatunek: {self.label_pl}. "
            f"Ton: {'; '.join(self.tone)}. "
            f"Terminologia: {'; '.join(self.terminology)}. "
            f"Struktura: {'; '.join(self.structure)}. "
            f"Unikaj: {'; '.join(self.forbidden_phrases)}."
        )


GENRE_PROFILES: dict[DocumentType, GenreProfile] = {
    DocumentType.client_communication: GenreProfile(
        document_type=DocumentType.client_communication,
        label_pl="komunikacja z klientem",
        tone=(
            "formalny, ale zrozumiały dla osoby bez przygotowania prawnego",
            "konkretny i spokojny",
            "wyraźnie oddziela fakt, ocenę ryzyka i rekomendację",
        ),
        terminology=(
            "termin prawny wyjaśnij przy pierwszym użyciu",
            "używaj nazw stron konsekwentnie",
        ),
        structure=(
            "najpierw wniosek i znaczenie dla klienta",
            "następnie podstawa i kolejne kroki",
        ),
        forbidden_phrases=(
            "warto podkreślić",
            "należy zauważyć",
            "kluczowym aspektem jest",
        ),
    ),
    DocumentType.contract: GenreProfile(
        document_type=DocumentType.contract,
        label_pl="umowa",
        tone=(
            "precyzyjny i normatywny",
            "bez perswazyjnych ozdobników",
            "bez zmiany rozkładu praw i obowiązków",
        ),
        terminology=(
            "zachowaj zdefiniowane pojęcia i wielkie litery",
            "nie zmieniaj modalności może/powinien/musi",
        ),
        structure=(
            "jedna norma lub mechanizm w jednostce redakcyjnej",
            "zachowaj numerację i odesłania",
        ),
        forbidden_phrases=(
            "warto podkreślić",
            "co istotne",
            "podsumowując",
        ),
    ),
    DocumentType.filing_official: GenreProfile(
        document_type=DocumentType.filing_official,
        label_pl="pismo procesowe lub urzędowe",
        tone=(
            "formalny, rzeczowy i stanowczy",
            "twierdzenia oddzielone od argumentacji",
            "bez deklaracji o wyniku, którego nie da się przesądzić",
        ),
        terminology=(
            "zachowaj przepisy, sygnatury i nazwy organów",
            "stosuj terminologię proceduralną konsekwentnie",
        ),
        structure=(
            "wnioski przed uzasadnieniem",
            "teza, podstawa, zastosowanie do faktów i wniosek",
        ),
        forbidden_phrases=(
            "warto podkreślić",
            "należy zauważyć",
            "w dzisiejszych czasach",
        ),
    ),
}


@dataclass(frozen=True)
class DocumentTypeGuess:
    document_type: DocumentType
    confidence: float
    evidence: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "type": self.document_type.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


_TYPE_TERMS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.client_communication: (
        "szanown",
        "informujemy",
        "rekomendujemy",
        "dla państwa",
        "w razie pytań",
        "klient",
        "opinia prawna",
    ),
    DocumentType.contract: (
        "umowa",
        "strona",
        "strony postanawiają",
        "zobowiązuje się",
        "wynagrodzenie",
        "wypowiedzenie",
        "odstąpienie",
        "§",
    ),
    DocumentType.filing_official: (
        "sąd",
        "organ",
        "wnoszę",
        "wnosi o",
        "uzasadnienie",
        "sygn. akt",
        "decyzja",
        "postanowienie",
        "na podstawie art.",
    ),
}


def classify_document(text: str) -> DocumentTypeGuess:
    """Return a conservative genre guess and an explicit confidence value."""
    lowered = text.casefold()
    scores: dict[DocumentType, int] = {}
    hits: dict[DocumentType, list[str]] = {}
    for document_type, terms in _TYPE_TERMS.items():
        found = [term for term in terms if term.casefold() in lowered]
        hits[document_type] = found
        scores[document_type] = len(found)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    winner = ordered[0]
    best = scores[winner]
    second = scores[ordered[1]]
    if best == 0:
        # Client communication is the least dangerous default: it does not
        # invite normalization of contractual or procedural formulas.
        return DocumentTypeGuess(DocumentType.client_communication, 0.34, ())
    confidence = min(0.98, 0.52 + 0.08 * best + 0.08 * max(0, best - second))
    return DocumentTypeGuess(winner, round(confidence, 2), tuple(hits[winner][:5]))


@dataclass
class StyleProfile:
    name: str
    document_type: DocumentType
    document_count: int
    word_count: int
    statistics: dict[str, float] = field(default_factory=dict)
    preferred_terms: dict[str, str] = field(default_factory=dict)
    forbidden_phrases: list[str] = field(default_factory=list)
    abbreviations: dict[str, str] = field(default_factory=dict)
    voice: list[str] = field(default_factory=list)
    anonymized_examples: list[str] = field(default_factory=list)
    template: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["document_type"] = self.document_type.value
        return payload

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "profile.json"
        target.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "StyleProfile":
        path = Path(directory)
        source = path / "profile.json" if path.is_dir() else path
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["document_type"] = DocumentType(payload["document_type"])
        return cls(**payload)

    def prompt_text(self) -> str:
        parts = [f"Profil kancelarii: {self.name}."]
        if self.voice:
            parts.append("Głos: " + "; ".join(self.voice) + ".")
        if self.preferred_terms:
            pairs = [f"{old} → {new}" for old, new in self.preferred_terms.items()]
            parts.append("Preferowana terminologia: " + "; ".join(pairs) + ".")
        if self.forbidden_phrases:
            parts.append("Zakazane zwroty: " + "; ".join(self.forbidden_phrases) + ".")
        if self.abbreviations:
            pairs = [f"{name} = {short}" for name, short in self.abbreviations.items()]
            parts.append("Skróty: " + "; ".join(pairs) + ".")
        if self.statistics:
            parts.append(
                "Rytm wzorców: średnio "
                f"{self.statistics.get('mean_sentence_words', 0):g} słów w zdaniu."
            )
        if self.anonymized_examples:
            parts.append(
                "Krótkie zanonimizowane wzorce tonu: "
                + " | ".join(self.anonymized_examples[:3])
            )
        return " ".join(parts)


_WORD_RE = re.compile(r"[A-Za-zĄ-ż]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _anonymize_excerpt(text: str, limit: int = 260) -> str:
    excerpt = _SENTENCE_RE.split(" ".join(text.split()), maxsplit=1)[0][:limit]
    excerpt = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", excerpt)
    excerpt = re.sub(r"\b\d+(?:[.,]\d+)?\b", "[LICZBA]", excerpt)
    excerpt = re.sub(
        r"\b(?:[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:\s+|$)){2,}",
        "[PODMIOT] ",
        excerpt,
    )
    return excerpt.strip()


def _profile_statistics(texts: Iterable[str]) -> dict[str, float]:
    items = list(texts)
    words = [_WORD_RE.findall(text) for text in items]
    sentences = [part for text in items for part in _SENTENCE_RE.split(text) if part.strip()]
    lengths = [len(_WORD_RE.findall(sentence)) for sentence in sentences]
    tokens = [word.casefold() for row in words for word in row]
    return {
        "mean_sentence_words": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "type_token_ratio": round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0,
        "mean_document_words": round(len(tokens) / len(items), 2) if items else 0.0,
    }


def build_style_profile(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    name: str,
    document_type: DocumentType,
    style_guide: str | Path | None = None,
    template: str | Path | None = None,
) -> StyleProfile:
    """Build a privacy-reduced office profile from 5–20 approved documents."""
    from humanize_pl.io.docx_io import docx_text

    source_directory = Path(source_directory)
    files = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".docx"
        and not path.name.startswith("~$")
    )
    if not 5 <= len(files) <= 20:
        raise ValueError("Profil kancelarii wymaga od 5 do 20 zatwierdzonych plików DOCX.")
    texts = [docx_text(path) for path in files]
    guide: dict[str, Any] = {}
    if style_guide:
        loaded = yaml.safe_load(Path(style_guide).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Instrukcja stylu YAML musi być mapą kluczy i wartości.")
        guide = loaded

    preferred = guide.get("preferred_terms", {}) or {}
    forbidden = guide.get("forbidden_phrases", []) or []
    abbreviations = guide.get("abbreviations", {}) or {}
    voice = guide.get("voice", []) or []
    if isinstance(voice, str):
        voice = [voice]

    profile = StyleProfile(
        name=name,
        document_type=document_type,
        document_count=len(files),
        word_count=sum(len(_WORD_RE.findall(text)) for text in texts),
        statistics=_profile_statistics(texts),
        preferred_terms={str(key): str(value) for key, value in dict(preferred).items()},
        forbidden_phrases=[str(value) for value in forbidden],
        abbreviations={str(key): str(value) for key, value in dict(abbreviations).items()},
        voice=[str(value) for value in voice],
        anonymized_examples=[_anonymize_excerpt(text) for text in texts[:8] if text.strip()],
    )
    output_directory = Path(output_directory)
    if template:
        template_path = Path(template)
        if template_path.suffix.lower() not in {".docx", ".dotx"}:
            raise ValueError("Szablon musi być plikiem .docx albo .dotx.")
        output_directory.mkdir(parents=True, exist_ok=True)
        copied = output_directory / f"template{template_path.suffix.lower()}"
        shutil.copy2(template_path, copied)
        profile.template = copied.name
    profile.save(output_directory)
    return profile


def common_terms(texts: Iterable[str], limit: int = 20) -> list[str]:
    """Small helper for profile inspection; never used as an AI threshold."""
    counter = Counter(
        word.casefold()
        for text in texts
        for word in _WORD_RE.findall(text)
        if len(word) > 4
    )
    return [word for word, _count in counter.most_common(limit)]
