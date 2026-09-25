"""General editing preferences and local editorial observations, never authorship scores."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from enum import Enum
from itertools import pairwise

from humanize_pl.sentence_splitter import split_sentences


class GeneralProfile(str, Enum):
    preserve = "preserve"
    email = "email"
    article = "article"
    product = "product"
    information = "information"
    prose = "prose"


class EditIntensity(str, Enum):
    light = "light"
    style = "style"
    rewrite = "rewrite"


PROFILE_GUIDANCE = {
    GeneralProfile.preserve: "Zachowaj gatunek, rejestr i głos autora.",
    GeneralProfile.email: "Mail: czytelny cel i kolejne kroki; zachowaj relację z adresatem i formy grzecznościowe.",
    GeneralProfile.article: "Artykuł: spójna argumentacja i przejścia; zachowaj tezy, ich kolejność logiczną i źródła.",
    GeneralProfile.product: "Opis produktu: czytelne cechy; nie dopisuj zalet, obietnic, parametrów ani zastosowań.",
    GeneralProfile.information: "Tekst informacyjny: jasne odniesienia i porządek; zachowaj wszystkie fakty, zastrzeżenia i instrukcje.",
    GeneralProfile.prose: "Proza: zachowaj perspektywę narratora, rytm, potoczność, celowe powtórzenia i niedopowiedzenia.",
}


@dataclass(frozen=True)
class GeneralOptions:
    profile: GeneralProfile = GeneralProfile.preserve
    audience: str = ""
    tone: str = "preserve"
    formality: str = "preserve"
    intensity: EditIntensity = EditIntensity.style
    # A ceiling, not a target; losing facts is forbidden at every setting.
    max_shortening: int = 30
    protected_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", GeneralProfile(self.profile))
        object.__setattr__(self, "intensity", EditIntensity(self.intensity))
        object.__setattr__(self, "protected_terms", tuple(self.protected_terms))
        if self.tone not in {"preserve", "neutral", "warm", "direct"}:
            raise ValueError("Ton ogólny: preserve, neutral, warm lub direct.")
        if self.formality not in {"preserve", "casual", "standard", "formal"}:
            raise ValueError("Formalność: preserve, casual, standard lub formal.")
        if not isinstance(self.max_shortening, int) or not 0 <= self.max_shortening <= 50:
            raise ValueError("Maksymalne skrócenie musi być liczbą całkowitą od 0 do 50 procent.")
        if len(self.audience) > 300 or len(self.protected_terms) > 100:
            raise ValueError("Odbiorca: maksymalnie 300 znaków; chronione terminy: maksymalnie 100.")

    def to_json(self) -> dict:
        return {**asdict(self), "profile": self.profile.value, "intensity": self.intensity.value}

    def prompt_text(self) -> str:
        levels = {
            EditIntensity.light: "Lekka korekta: zmieniaj tylko lokalne usterki, zachowaj budowę zdań.",
            EditIntensity.style: "Redakcja stylistyczna: popraw płynność zdań, powtórzenia i przejścia w akapicie.",
            EditIntensity.rewrite: "Przepisanie akapitu: możesz zmienić budowę zdań i kolejność wyjaśnień, zachowując argumentację.",
        }
        return (
            PROFILE_GUIDANCE[self.profile] + " " + levels[self.intensity]
            + f" Odbiorca: {self.audience or 'jak w źródle'}. Ton: {self.tone}. Formalność: {self.formality}."
            + " Wartość preserve oznacza zachowanie źródła."
            + f" Skrócenie najwyżej {self.max_shortening}% słów, tylko bez utraty informacji; nie jest wymagane."
            + " Zachowaj fakty, intencję, związki przyczynowe, zastrzeżenia i terminologię."
            + " Nie zmieniaj cytatów, kodu, dialogów ani celowej ekspresji. Nie dodawaj informacji z kontekstu."
            + (" Chronione terminy: " + "; ".join(self.protected_terms) + "." if self.protected_terms else "")
        )

    def rejection(self, before: str, after: str) -> str | None:
        old, new = re.findall(r"\w+", before), re.findall(r"\w+", after)
        if len(new) < len(old) * (1 - self.max_shortening / 100):
            return "Przekroczono maksymalne skrócenie."
        budget = {EditIntensity.light: .25, EditIntensity.style: .65, EditIntensity.rewrite: 1.0}[self.intensity]
        if 1 - SequenceMatcher(None, old, new, autojunk=False).ratio() > budget:
            return "Przekroczono wybrany poziom ingerencji."
        for term in self.protected_terms:
            if before.count(term) != after.count(term):
                return "Zmieniono chroniony termin."
        return None


def protected_lines(text: str) -> dict[int, str]:
    """Absolute line indices, matching DOCX units and the rewrite engine."""
    result: dict[int, str] = {}
    fence: str | None = None
    quote_open = False
    for index, line in enumerate(text.split("\n")):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        reason = None
        if fence:
            reason = "kod"
            if marker and marker.group()[0] == fence[0] and len(marker.group()) >= len(fence):
                fence = None
        elif marker:
            fence, reason = marker.group(), "kod"
        elif line.startswith(("    ", "\t")) or "`" in line:
            reason = "kod lub wcięcie autora"
        elif quote_open or any(char in line for char in ('„', '”', '"')):
            reason = "cytat"
        elif re.match(r"(?:[—–-]\s|>\s)", stripped):
            reason = "dialog lub cytat"
        if '„' in line and '”' not in line:
            quote_open = True
        if '”' in line:
            quote_open = False
        if line.count('"') % 2:
            quote_open = not quote_open
        if reason:
            result[index] = reason
    return result


def editorial_findings(text: str, *, protected: set[int] | None = None) -> list[dict]:
    """Conservative observations for review, not a claim that text needs changing."""
    excluded = set(protected_lines(text)) | (protected or set())
    rows: list[dict] = []
    for index, line in enumerate(text.split("\n")):
        if index in excluded or not line.strip():
            continue
        sentences = split_sentences(line)
        def add(kind: str, evidence: str, detail: str, index: int = index) -> None:
            rows.append({"kind": kind, "paragraph_index": index, "evidence": evidence[:180], "detail": detail})
        if re.search(r"\b(\w{2,})\s+\1\b", line, re.IGNORECASE):
            add("adjacent_repetition", line, "Sąsiadujące powtórzenie: sprawdź, czy jest zamierzone.")
        if len(sentences) > 1 and any(a.casefold() == b.casefold() for a, b in pairwise(sentences)):
            add("duplicate_sentence", line, "Powtórzone zdanie: sprawdź jego funkcję w tekście.")
        if any(len(re.findall(r"\w+", sentence)) > 40 for sentence in sentences):
            add("dense_sentence", line, "Długie zdanie: sprawdź czytelność i zależności, nie dziel mechanicznie.")
        openings = [" ".join(re.findall(r"\w+", s.casefold())[:2]) for s in sentences]
        if len(openings) >= 3 and len(set(openings)) == 1:
            add("repeated_opening", line, "Jednakowe początki zdań: rozważ urozmaicenie, chyba że to celowa anafora.")
        if re.search(r"\b(to ostatnie|ten drugi|powyższe|wspomniany)\b", line, re.IGNORECASE):
            add("reference_review", line, "Sprawdź, czy odniesienie jest jednoznaczne w kontekście; nie dopowiadaj podmiotu.")
        if re.search(r"\b(w związku z faktem,? że|w celu dokonania|na chwilę obecną|w dniu dzisiejszym)\b", line, re.IGNORECASE):
            add("wordiness", line, "Rozważ uproszczenie rozwlekłej konstrukcji bez zmiany znaczenia.")
    return rows
