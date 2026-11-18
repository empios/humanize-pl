"""Verdicts on a document's references, in words a lawyer can act on.

Four states and no more: confirmed, not found where the source looked,
an act that cannot be resolved or has no structure to check against, and a
source that could not be asked. Only "not found" says something about the
document; the others say something about the check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from .extract import CaseRef, StatuteRef, extract_references, load_acts
from .sources import ActStructure, SourceUnavailable

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class Sources(Protocol):
    def act_structure(self, eli: str) -> ActStructure | None: ...

    def case_found(self, signature: str) -> tuple[bool, list[str]]: ...


@dataclass(frozen=True)
class ReferenceVerdict:
    text: str
    kind: str  # "statute" | "case"
    status: str  # "confirmed" | "not_found" | "unknown_act" | "unavailable"
    source: str
    note_pl: str

    def to_json(self) -> dict[str, str]:
        return asdict(self)


def verify_references(text: str, sources: Sources) -> list[ReferenceVerdict]:
    """One verdict per distinct reference, in the order they first appear."""
    acts = load_acts()
    structures: dict[str, ActStructure | None | SourceUnavailable] = {}
    verdicts: list[ReferenceVerdict] = []
    seen: set[tuple] = set()
    for ref in extract_references(text):
        if isinstance(ref, StatuteRef):
            key = ("statute", ref.act, ref.article, ref.paragraphs, ref.passages, ref.points)
            if key in seen:
                continue
            seen.add(key)
            eli = acts[ref.act].eli if ref.act in acts else ref.act
            if eli not in structures:
                try:
                    structures[eli] = sources.act_structure(eli)
                except SourceUnavailable as exc:
                    structures[eli] = exc
            verdicts.append(_statute_verdict(ref, eli, structures[eli]))
        else:
            if ("case", ref.signature) in seen:
                continue
            seen.add(("case", ref.signature))
            verdicts.append(_case_verdict(ref, sources))
    return verdicts


def _statute_verdict(
    ref: StatuteRef, eli: str, structure: ActStructure | None | SourceUnavailable
) -> ReferenceVerdict:
    if isinstance(structure, SourceUnavailable):
        return ReferenceVerdict(
            ref.text, "statute", "unavailable", "ELI",
            "Nie udało się połączyć z ISAP/ELI; odwołania nie sprawdzono.",
        )
    if structure is None:
        return ReferenceVerdict(
            ref.text, "statute", "unknown_act", "ELI",
            f"ISAP/ELI nie zna aktu {eli}; odwołania nie sprawdzono.",
        )
    if not structure.units:
        return ReferenceVerdict(
            ref.text, "statute", "unknown_act", f"ELI {structure.eli}",
            "ISAP/ELI nie udostępnia struktury tego aktu; odwołania nie sprawdzono.",
        )
    source = f"ELI {structure.eli}" + (" (tekst jednolity)" if structure.eli != eli else "")
    missing = [unit for unit in _required_units(ref, ref.article) if unit not in structure.units]
    if not missing:
        return ReferenceVerdict(
            ref.text, "statute", "confirmed", source,
            f"Potwierdzono w tekście {structure.eli}.",
        )
    note = f"W tekście {structure.eli} nie ma: {_describe(missing[0])}."
    if ref.flattened_superscript:
        alternative = f"{ref.article[:-1]}_{ref.article[-1]}"
        if all(unit in structure.units for unit in _required_units(ref, alternative)):
            note += f" Czy chodziło o art. {_display_article(alternative)}?"
    return ReferenceVerdict(ref.text, "statute", "not_found", source, note)


def _required_units(ref: StatuteRef, article: str) -> list[str]:
    """Every unit the reference names, most specific first in each branch."""
    if ref.paragraphs or ref.passages:
        mark, numbers = ("§", ref.paragraphs) if ref.paragraphs else ("ust", ref.passages)
        bases = [f"{article}{mark}{number}" for number in numbers]
    else:
        bases = [article]
    if ref.points:
        return [f"{base}pkt{point}" for base in bases for point in ref.points]
    return bases


def _describe(unit: str) -> str:
    """'445§2pkt3' -> 'art. 445 § 2 pkt 3'."""
    out = unit
    for mark, label in (("§", " § "), ("ust", " ust. "), ("pkt", " pkt "), ("lit", " lit. ")):
        out = out.replace(mark, label)
    article, _, rest = out.partition(" ")
    return f"art. {_display_article(article)}" + (f" {rest}" if rest else "")


def _display_article(article: str) -> str:
    number, _, index = article.partition("_")
    return number + index.translate(_SUPERSCRIPT)


def _case_verdict(ref: CaseRef, sources: Sources) -> ReferenceVerdict:
    try:
        found, courts = sources.case_found(ref.signature)
    except SourceUnavailable:
        return ReferenceVerdict(
            ref.text, "case", "unavailable", "SAOS",
            "Nie udało się połączyć z SAOS; sygnatury nie sprawdzono.",
        )
    if found:
        return ReferenceVerdict(
            ref.text, "case", "confirmed", "SAOS",
            f"Znaleziono w SAOS: {', '.join(courts)}.",
        )
    return ReferenceVerdict(
        ref.text, "case", "not_found", "SAOS",
        "Nie znaleziono w SAOS (baza nie obejmuje wszystkich orzeczeń).",
    )
