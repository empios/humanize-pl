from __future__ import annotations

from dataclasses import dataclass, field
import regex as re

from .anchors import content_anchor_retention, content_anchor_tokens
from .protectors import ProtectedText
from humanize_pl.rules.finite_verbs import has_finite_verb
from humanize_pl.rules.legal_features import (
    legal_anchor_retention,
    lost_legal_anchors,
    normativity_signature,
)

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


def _sentence_count(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p])


def _split_sentence_like(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def _without_transition(sentence: str) -> str:
    body = sentence.strip()
    for transition in SENTENCE_TRANSITIONS:
        if body.lower().startswith(transition.lower()):
            return body[len(transition):].strip(" ,;")
    return body


def _balanced_pairs(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("„", "”"), ('"', '"')]
    for left, right in pairs:
        if left == right:
            if text.count(left) % 2:
                return False
            continue
        if text.count(left) != text.count(right):
            return False
    return True


def _word_count(text: str) -> int:
    return len(re.findall(r"\p{L}+", text))


def _sentence_has_required_predication(original: str, candidate: str) -> bool:
    original_parts = _split_sentence_like(original)
    original_has_predication = any(has_finite_verb(_without_transition(part)) for part in original_parts)
    if not original_has_predication:
        return True
    for part in _split_sentence_like(candidate):
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

    if re.search(r"\b(?:oraz|i|lub|albo|a także)\s+(?:oraz|i|lub|albo|a także)\b", lower):
        return _failed("no_double_conjunctions", "double conjunction detected", checks)
    _passed("no_double_conjunctions", checks)

    if re.search(r"(?:^|[.!?]\s+)(?:Ponadto|Z kolei|Jednak|Natomiast)\s*[.!?]", candidate):
        return _failed("no_dangling_connectors", "dangling connector detected", checks)
    if re.search(r"\b(?:oraz|lub|albo|a także|ponieważ|jeżeli)\s*[.!?]?$", lower):
        return _failed("no_dangling_connectors", "dangling connector detected", checks)
    _passed("no_dangling_connectors", checks)

    if not _balanced_pairs(candidate):
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

    if normativity_signature(restored_original) != normativity_signature(restored_candidate):
        return _failed("normativity_preserved", "normativity changed", checks)
    _passed("normativity_preserved", checks)

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

    if _sentence_count(candidate) > _sentence_count(original):
        if re.search(r"\boraz\b", original, re.IGNORECASE):
            return _failed(
                "sentence_split_safety",
                "sentence split after/around 'oraz' is disabled",
                checks,
            )

        # Reject sentence split if any new sentence is only a complement/list item.
        parts = _split_sentence_like(candidate)
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

    return ValidationResult(True, checks=checks)
