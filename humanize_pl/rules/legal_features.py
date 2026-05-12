from __future__ import annotations

from dataclasses import dataclass
import regex as re

from humanize_pl.rules.finite_verbs import has_finite_verb


AI_TRANSITION_PATTERN = re.compile(
    r"\b(?:ponadto|co więcej|dodatkowo|w tym kontekście|warto\s+(?:wskazać|zauważyć|podkreślić)|"
    r"należy\s+(?:wskazać|zauważyć|podkreślić|odnotować))\b",
    re.IGNORECASE,
)
VAGUE_REFERENCE_PATTERN = re.compile(
    r"\b(?:powyższ\p{L}*|niniejsz\p{L}*|przedmiotow\p{L}*|ten|ta|to|taki|taka|"
    r"takie|ona|ono|on)\b",
    re.IGNORECASE,
)
LEGAL_REFERENCE_PATTERN = re.compile(
    r"(?:\bart\.|§|\bust\.|\bpkt\b|\bDz\.\s*U\.|\bKodeks\p{L}*|\bustaw\p{L}*)",
    re.IGNORECASE,
)
MONEY_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}\s*r\.\b|"
    r"\b\d+(?:[,.]\d+)?\s*(?:zł|PLN|EUR|USD|%)\b",
    re.IGNORECASE,
)
LEGAL_ANCHOR_PATTERN = re.compile(
    r"\b(?:pracownik\p{L}*|pracodawc\p{L}*|stron\p{L}*|wykonawc\p{L}*|"
    r"zamawiając\p{L}*|zleceniobiorc\p{L}*|zleceniodawc\p{L}*|wierzyciel\p{L}*|"
    r"dłużnik\p{L}*|konsument\p{L}*|przedsiębiorc\p{L}*|powod\p{L}*|"
    r"pozw\p{L}*|organ\p{L}*|sąd\p{L}*|spółk\p{L}*|umow\p{L}*|"
    r"obowiązk\p{L}*|uprawnien\p{L}*|świadczen\p{L}*|wynagrodzen\p{L}*|"
    r"kar\p{L}*|odsetk\p{L}*|termin\p{L}*|odpowiedzialnoś\p{L}*|"
    r"roszczen\p{L}*|wypowiedzen\p{L}*|odstąpien\p{L}*|rozwiązan\p{L}*|"
    r"zgod\p{L}*|oświadczen\p{L}*|szkod\p{L}*|poufnoś\p{L}*|danych|"
    r"pracy|stosunk\p{L}*|kodeks\p{L}*)\b",
    re.IGNORECASE,
)
DISCOURSE_NALEZY_PATTERN = re.compile(
    r"\bnależy\s+(?:wskazać|zauważyć|podkreślić|dodać|odnotować)\s*,?\s+że\b",
    re.IGNORECASE,
)
PROHIBITION_PATTERN = re.compile(
    r"\b(?:nie\s+może|nie\s+mogą|nie\s+wolno|zakazuje\s+się|zakaz\p{L}*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LegalReviewFeatures:
    has_finite_verb: bool
    normativity_count: int
    legal_anchor_count: int
    legal_reference_count: int
    money_or_date_count: int
    vague_reference_count: int
    ai_transition_count: int
    ai_artifact_score: float
    legal_risk_score: float


def analyze_legal_review_features(text: str) -> LegalReviewFeatures:
    words = re.findall(r"\p{L}+", text)
    word_count = max(len(words), 1)
    normativity = normativity_signature(text)
    normativity_count = sum(normativity.values())
    legal_anchor_count = len(legal_anchor_terms(text))
    legal_reference_count = len(LEGAL_REFERENCE_PATTERN.findall(text))
    money_or_date_count = len(MONEY_DATE_PATTERN.findall(text))
    vague_reference_count = len(VAGUE_REFERENCE_PATTERN.findall(text))
    ai_transition_count = len(AI_TRANSITION_PATTERN.findall(text))

    ai_artifact_score = min(
        1.0,
        ai_transition_count * 0.18
        + min(0.30, vague_reference_count / word_count)
        + min(0.20, _nominalization_count(words) / word_count * 2),
    )
    legal_risk_score = min(
        1.0,
        normativity_count * 0.12
        + legal_reference_count * 0.10
        + money_or_date_count * 0.12
        + legal_anchor_count * 0.025,
    )

    return LegalReviewFeatures(
        has_finite_verb=has_finite_verb(text),
        normativity_count=normativity_count,
        legal_anchor_count=legal_anchor_count,
        legal_reference_count=legal_reference_count,
        money_or_date_count=money_or_date_count,
        vague_reference_count=vague_reference_count,
        ai_transition_count=ai_transition_count,
        ai_artifact_score=round(ai_artifact_score, 4),
        legal_risk_score=round(legal_risk_score, 4),
    )


def normativity_signature(text: str) -> dict[str, int]:
    lower = DISCOURSE_NALEZY_PATTERN.sub(" ", text.lower())
    prohibition_text = PROHIBITION_PATTERN.sub(" ", lower)
    return {
        "prohibition": len(PROHIBITION_PATTERN.findall(lower)),
        "permission": len(re.findall(r"\b(?:może|mogą|uprawnien\p{L}*)\b", prohibition_text)),
        "strict_obligation": len(
            re.findall(
                r"\b(?:musi|muszą|jest\s+zobowiązan\p{L}*|są\s+zobowiązan\p{L}*|"
                r"ma\s+obowiązek|mają\s+obowiązek|obowiązek|obowiązk\p{L}*)\b",
                lower,
            )
        ),
        "soft_obligation": len(
            re.findall(r"\b(?:powinien|powinna|powinni|powinno|należy)\b", lower)
        ),
        "entitlement": len(re.findall(r"\b(?:przysługuje|przysługują|uprawnion\p{L}*)\b", lower)),
        "liability": len(
            re.findall(r"\b(?:podlega|podlegają|ponosi\s+odpowiedzialność|odpowiada)\b", lower)
        ),
    }


def legal_anchor_terms(text: str) -> set[str]:
    return {_legal_stem(match.group(0).lower()) for match in LEGAL_ANCHOR_PATTERN.finditer(text)}


def legal_anchor_retention(original: str, candidate: str) -> float:
    original_terms = legal_anchor_terms(original)
    if not original_terms:
        return 1.0
    candidate_terms = legal_anchor_terms(candidate)
    return len(original_terms & candidate_terms) / len(original_terms)


def lost_legal_anchors(original: str, candidate: str) -> set[str]:
    return legal_anchor_terms(original) - legal_anchor_terms(candidate)


def _nominalization_count(words: list[str]) -> int:
    return sum(
        1
        for word in words
        if len(word) >= 7
        and word.lower().endswith(("anie", "enie", "cie", "ość", "acja", "izja", "yzja"))
    )


def _legal_stem(word: str) -> str:
    for suffix in (
        "owego",
        "owej",
        "owych",
        "ami",
        "ach",
        "ego",
        "emu",
        "owi",
        "owa",
        "owy",
        "owe",
        "enie",
        "ania",
        "anie",
        "ego",
        "ów",
        "om",
        "em",
        "ą",
        "ę",
        "a",
        "u",
        "e",
        "y",
        "i",
    ):
        if len(word) - len(suffix) >= 5 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word
