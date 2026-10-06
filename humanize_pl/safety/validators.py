from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import regex as re

from .anchors import content_anchor_retention, content_anchor_tokens
from .protectors import ProtectedText
from .deontic import extract_deontic_profile, DeonticModality
from .parties import parties_swapped, party_mentions_exchanged
from .speech_acts import speech_acts_changed
from humanize_pl.nlp.morfeusz import try_load_morfeusz
from humanize_pl.rules.finite_verbs import has_finite_verb
from humanize_pl.rules.legal_features import (
    legal_anchor_retention,
    lost_legal_anchors,
)
from humanize_pl.sentence_splitter import split_sentences

BANNED_WORDS = {
    "surprisingly",
    "actually",
    "basically",
    "honestly",
    "literally",
    "overall",
    "moreover",
}

BAD_POLISH_PATTERNS = [
    r"\bich\s+w\s+celu\s+jest\b",
    r"\bjego\s+w\s+celu\s+jest\b",
    r"\bjej\s+w\s+celu\s+jest\b",
    r"\bw\s+celu\s+jest\b",
    r"\bjest\s+(?:przeprowadzić|wykonać|wdrożyć|zastosować|wykorzystać)\b",
    r"\.\s*Ponadto\s+(?:za|w|na|do|od|przy|bez|pod)\b",
    r"\.\s*Z kolei\s+(?:za|w|na|do|od|przy|bez|pod)\b",
    r"\.\s*Przy czym\s+(?:za|w|na|do|od|bez|pod)\b",
]

RELATIVE_PRONOUN_RE = (
    r"który|która|które|którzy|których|którym|którego|której|któremu|"
    r"którymi|którą"
)
STRANDED_RELATIVE_CLAUSE_RE = re.compile(
    rf"\b\p{{L}}{{3,}}(?:ć|c)\s*,\s*(?:{RELATIVE_PRONOUN_RE})\b",
    re.IGNORECASE,
)

# Polarity markers. "brak" counts together with "nie", so "w przypadku braku"
# -> "gdy nie ma" keeps the count.
NEGATION_RE = re.compile(r"\bnie\b|\bbrak\p{L}*|\bbez\b", re.IGNORECASE)
# Rules that resolve a double negation, dropping "nie" without flipping polarity.
DOUBLE_NEGATION_RULES = {"legal_style:ma_granice"}

SENTENCE_TRANSITIONS = (
    "Ponadto",
    "Z kolei",
    "Jednak",
    "Natomiast",
    "Przy czym",
    "Warto",
    "Należy",
    "Wynika to z tego, że",
)

SENSITIVE_INVENTORY_PATTERNS = {
    "legal_references": (
        r"\bart\.\s*\d+[a-zA-Z]?(?:\s*§\s*\d+[a-zA-Z]?)?",
        r"§\s*\d+[a-zA-Z]?",
        r"\b(?:ust\.|pkt)\s*\d+[a-zA-Z]?",
        r"\bDz\.\s*U\.[^,.;)]*",
        # Code abbreviations: k.c., k.p.c., k.p.a., k.k.w., k.r.o., k.s.h., ...
        r"\bk\.(?:p\.[ckaw]|k\.[ws]|r\.o|s\.h|[ckpwm])\.",
    ),
    "dates": (
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}\s*r\.",
    ),
    "amounts": (
        r"\b(?:\d{1,3}(?:[   .]\d{3})+(?:,\d+)?|\d+(?:[,.]\d+)?)\s*"
        r"(?:%|(?:zł(?:ot(?:ych|e|y|ego|ym|ymi))?|PLN|EUR|USD)\b)",
    ),
    "identifiers": (
        r"\b(?:PESEL|NIP|REGON|KRS)\s*[:#]?\s*[A-Z0-9 -]{6,20}\b",
        r"\bPL\d{26}\b",
    ),
    "defined_terms_and_quotes": (r"„[^”]+”", r'"[^"]+"'),
    "party_names": (
        r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:[- ][A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)+\b",
    ),
}


@dataclass
class GateCheck:
    name: str
    ok: bool
    reason: str = ""


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    checks: list[GateCheck] = field(default_factory=list)


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[,.]\d+)?", text)


def legal_sensitive_inventory(text: str) -> dict[str, Counter[str]]:
    """Values whose exact preservation matters more than stylistic gain."""
    inventory: dict[str, Counter[str]] = {}
    for category, patterns in SENSITIVE_INVENTORY_PATTERNS.items():
        values: list[str] = []
        for pattern in patterns:
            if category == "party_names":
                values.extend(_party_names(text, pattern))
            else:
                values.extend(m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE))
        inventory[category] = Counter(value.casefold() for value in values)
    return inventory


def _party_names(text: str, pattern: str) -> list[str]:
    """Runs of capitalised words ("Jan Kowalski"). A sentence starts with a capital
    anyway, so there a first word Morfeusz knows only as a common word is dropped:
    "Niniejsza Umowa" and "Ponadto Strona" are not names."""
    morfeusz = try_load_morfeusz()
    names: list[str] = []
    for match in re.finditer(pattern, text):
        name = match.group(0)
        before = match.start() - 1
        while before >= 0 and text[before] in " \t ":
            before -= 1
        if morfeusz is not None and (before < 0 or text[before] in ".!?:;(\n"):
            first, rest = re.split(r"[- ]", name, maxsplit=1)
            lemmas = {analysis.lemma for analysis in morfeusz.analyses(first)}
            if lemmas and all(lemma[:1].islower() for lemma in lemmas):
                if not re.fullmatch(pattern, rest):
                    continue
                name = rest
        names.append(name)
    return names


def _without_transition(sentence: str) -> str:
    body = sentence.strip()
    for transition in SENTENCE_TRANSITIONS:
        if body.lower().startswith(transition.lower()):
            return body[len(transition):].strip(" ,;")
    return body


def _pair_imbalance(text: str) -> tuple[int, ...]:
    """Unclosed marks per pair; a mark that opens and closes alike counts by parity."""
    pairs = [("(", ")"), ("[", "]"), ("„", "”"), ('"', '"')]
    return tuple(
        text.count(left) % 2 if left == right else text.count(left) - text.count(right)
        for left, right in pairs
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\p{L}+", text))


def has_stranded_relative_clause(text: str) -> bool:
    return bool(STRANDED_RELATIVE_CLAUSE_RE.search(text))


def _sentence_has_required_predication(original: str, candidate: str) -> bool:
    original_parts = split_sentences(original)
    original_has_predication = any(has_finite_verb(_without_transition(part)) for part in original_parts)
    if not original_has_predication:
        return True
    for part in split_sentences(candidate):
        checked = _without_transition(part)
        if _word_count(checked) >= 4 and not has_finite_verb(checked):
            return False
    return True


def _failed(name: str, reason: str, checks: list[GateCheck]) -> ValidationResult:
    checks.append(GateCheck(name, False, reason))
    return ValidationResult(False, reason, checks)


def _passed(name: str, checks: list[GateCheck]) -> None:
    checks.append(GateCheck(name, True))


def validate_candidate(
    original: str,
    candidate: str,
    *,
    protected: ProtectedText,
    max_length_ratio: float,
    rule: str | None = None,
    operation_type: str | None = None,
    nli=None,
) -> ValidationResult:
    checks: list[GateCheck] = []
    if not candidate.strip():
        return _failed("non_empty", "empty candidate", checks)
    _passed("non_empty", checks)

    lower = candidate.lower()
    if any(word in lower for word in BANNED_WORDS):
        return _failed("no_foreign_words", "foreign/informal marker detected", checks)
    _passed("no_foreign_words", checks)

    if any(re.search(pattern, lower) for pattern in BAD_POLISH_PATTERNS):
        return _failed("known_bad_patterns", "known bad Polish pattern", checks)
    _passed("known_bad_patterns", checks)

    if has_stranded_relative_clause(candidate):
        return _failed(
            "no_stranded_relative_clause",
            "relative clause stranded after infinitive",
            checks,
        )
    _passed("no_stranded_relative_clause", checks)

    if re.search(r"\b(?:oraz|i|lub|albo|a także)\s+(?:oraz|i|lub|albo|a także)\b", lower):
        return _failed("no_double_conjunctions", "double conjunction detected", checks)
    _passed("no_double_conjunctions", checks)

    if re.search(r"(?:^|[.!?]\s+)(?:Ponadto|Z kolei|Jednak|Natomiast)\s*[.!?]", candidate):
        return _failed("no_dangling_connectors", "dangling connector detected", checks)
    if re.search(r"\b(?:oraz|lub|albo|a także|ponieważ|jeżeli)\s*[.!?]?$", lower):
        return _failed("no_dangling_connectors", "dangling connector detected", checks)
    _passed("no_dangling_connectors", checks)

    # Relative to the original: inline lists ("1) …, 2) …") and „…“ quotes are
    # unbalanced on purpose; only an imbalance the candidate adds is an error.
    if _pair_imbalance(candidate) != _pair_imbalance(original):
        return _failed("balanced_punctuation", "unbalanced punctuation", checks)
    _passed("balanced_punctuation", checks)

    if len(original) > 40 and len(candidate) > len(original) * max_length_ratio:
        return _failed("length_ratio", "candidate too long", checks)

    if len(original) > 80 and len(candidate) < len(original) * 0.55:
        return _failed("length_ratio", "candidate suspiciously short", checks)
    _passed("length_ratio", checks)

    if _numbers(original) != _numbers(candidate):
        return _failed("numbers_preserved", "numbers changed", checks)
    _passed("numbers_preserved", checks)

    restored_original = protected.restore(original)
    restored_candidate = protected.restore(candidate)
    if "__PROTECTED_" in restored_candidate:
        return _failed("placeholder_restore", "unrestored placeholder leak", checks)
    _passed("placeholder_restore", checks)



    source_inventory = legal_sensitive_inventory(restored_original)
    candidate_inventory = legal_sensitive_inventory(restored_candidate)
    for category in SENSITIVE_INVENTORY_PATTERNS:
        if source_inventory[category] != candidate_inventory[category]:
            return _failed(
                f"{category}_preserved",
                f"legally sensitive inventory changed: {category}",
                checks,
            )
        _passed(f"{category}_preserved", checks)

    # Deontic modality check
    source_deontic = extract_deontic_profile(restored_original)
    candidate_deontic = extract_deontic_profile(restored_candidate)
    
    # We strictly block shifts between OBLIGATION, PROHIBITION and PERMISSION.
    # We also block removing them completely or adding them from nowhere.
    for mod in (DeonticModality.OBLIGATION, DeonticModality.PROHIBITION, DeonticModality.PERMISSION):
        if sum(source_deontic[mod].values()) != sum(candidate_deontic[mod].values()):
            return _failed("deontic_modality_preserved", f"deontic drift in category {mod.value}", checks)
    # For RECOMMENDATION, we just warn if we had a logger, but for now we also require them 
    # to not drift into hard obligations. If sum of recommendations changes without changing hard modals, 
    # we can be a bit more lenient, but to be safe let's ensure it matches too.
    if sum(source_deontic[DeonticModality.RECOMMENDATION].values()) != sum(candidate_deontic[DeonticModality.RECOMMENDATION].values()):
        return _failed("deontic_modality_preserved", "deontic drift in recommendation", checks)
    _passed("deontic_modality_preserved", checks)

    # Polarity can flip with no modal in sight: "odpowiada" -> "nie odpowiada",
    # "bez wypowiedzenia" -> "z zachowaniem wypowiedzenia".
    negation_kept = len(NEGATION_RE.findall(restored_original)) == len(
        NEGATION_RE.findall(restored_candidate)
    )
    if not negation_kept and rule not in DOUBLE_NEGATION_RULES:
        return _failed("negation_preserved", "negation changed", checks)
    _passed("negation_preserved", checks)

    if speech_acts_changed(restored_original, restored_candidate):
        return _failed("speech_acts_preserved", "performative act changed", checks)
    _passed("speech_acts_preserved", checks)

    if parties_swapped(restored_original, restored_candidate):
        return _failed("party_roles_preserved", "parties swapped roles", checks)
    _passed("party_roles_preserved", checks)

    if party_mentions_exchanged(restored_original, restored_candidate):
        return _failed("party_mentions_preserved", "one party named in place of another", checks)
    _passed("party_mentions_preserved", checks)

    anchor_tokens = content_anchor_tokens(protected.restore(original))
    if len(anchor_tokens) >= 4:
        retention = content_anchor_retention(restored_original, restored_candidate)
        if retention < 0.50:
            return _failed("content_anchor_retention", "content anchors changed too much", checks)
    _passed("content_anchor_retention", checks)

    if not _sentence_has_required_predication(original, candidate):
        return _failed("finite_verb_presence", "candidate sentence lacks finite verb", checks)
    _passed("finite_verb_presence", checks)

    legal_retention = legal_anchor_retention(restored_original, restored_candidate)
    redundancy_drop = operation_type == "redundancy_reduction" or (rule or "").startswith("redundancy:")
    legal_retention_floor = 0.50 if redundancy_drop else 0.85
    if legal_retention < legal_retention_floor:
        missing = ", ".join(sorted(lost_legal_anchors(restored_original, restored_candidate))[:4])
        reason = f"legal anchors removed: {missing}" if missing else "legal anchors removed"
        return _failed("legal_anchor_retention", reason, checks)
    _passed("legal_anchor_retention", checks)

    parts = split_sentences(candidate)
    if len(parts) > len(split_sentences(original)):
        if re.search(r"\boraz\b", original, re.IGNORECASE):
            return _failed(
                "sentence_split_safety",
                "sentence split after/around 'oraz' is disabled",
                checks,
            )

        # Reject sentence split if any new sentence is only a complement/list item.
        for idx, part in enumerate(parts):
            checked = _without_transition(part)
            if idx > 0 and not has_finite_verb(checked):
                return _failed(
                    "sentence_split_safety",
                    "split produced fragment without finite verb",
                    checks,
                )
            if re.match(r"^(?:Ponadto|Z kolei|Przy czym)\b", part, re.IGNORECASE) and not has_finite_verb(checked):
                return _failed(
                    "sentence_split_safety",
                    "transition sentence lacks finite verb",
                    checks,
                )
    _passed("sentence_split_safety", checks)

    for placeholder in protected.mapping:
        if placeholder in original and placeholder not in candidate:
            return _failed("protected_fragments", "protected fragment removed", checks)
    _passed("protected_fragments", checks)

    if nli is not None:
        if not nli.check_entailment(restored_original, restored_candidate):
            return _failed("semantic_contradiction", "candidate contradicts original text", checks)
        _passed("semantic_contradiction", checks)

    return ValidationResult(True, checks=checks)
