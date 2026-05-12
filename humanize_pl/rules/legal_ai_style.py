from __future__ import annotations

import regex as re

from humanize_pl.config import Mode
from humanize_pl.rules.finite_verbs import has_finite_verb
from .base import Candidate
from .features import ParagraphFeatures, SentenceFeatures


def legal_ai_style_candidates(
    sentence: str,
    *,
    mode: Mode,
    features: SentenceFeatures,
    paragraph_features: ParagraphFeatures | None,
    analysis=None,
) -> list[Candidate]:
    """Paragraph-aware candidates for monotone AI-like legal prose."""
    if mode == Mode.conservative or not _paragraph_has_ai_style_issue(features, paragraph_features):
        return []

    nlp_confidence = _nlp_confidence(analysis)
    out: list[Candidate] = []

    out.extend(_drop_empty_emphasis(sentence, nlp_confidence=nlp_confidence))
    out.extend(_abstract_frame_rewrites(sentence, nlp_confidence=nlp_confidence))

    if analysis is not None and nlp_confidence >= 0.55:
        out.extend(_nlp_unlocked_rewrites(sentence, nlp_confidence=nlp_confidence))

    return _unique(out)


def _paragraph_has_ai_style_issue(
    features: SentenceFeatures,
    paragraph_features: ParagraphFeatures | None,
) -> bool:
    score = 0
    if features.ai_artifact_score >= 0.08:
        score += 1
    if features.nominalization_count >= 2:
        score += 1
    if features.vague_reference_count:
        score += 1
    if paragraph_features:
        if paragraph_features.repeated_anchor_count:
            score += 1
        if paragraph_features.transition_count:
            score += 1
        if paragraph_features.avg_sentence_words >= 18:
            score += 1
    return score > 0


def _drop_empty_emphasis(sentence: str, *, nlp_confidence: float | None) -> list[Candidate]:
    out: list[Candidate] = []

    candidate = re.sub(r"\b[Tt]o właśnie\b", "To", sentence, count=1)
    if candidate != sentence:
        out.append(
            _candidate(
                candidate,
                "legal_ai_style:drop_wlasnie",
                "empty_emphasis",
                0.50,
                0.06,
                nlp_confidence,
            )
        )

    candidate = re.sub(r"^\s*Już\s+(?=z\s+\p{L}+\s+wynika\b)", "", sentence, count=1)
    if candidate != sentence:
        candidate = _sentence_case(candidate)
        out.append(_candidate(candidate, "legal_ai_style:drop_juz", "empty_emphasis", 0.49, 0.06, nlp_confidence))

    candidate = re.sub(r"\b(w\s+znacznym\s+stopniu)\s+również\b", r"\1", sentence, count=1)
    if candidate != sentence:
        out.append(
            _candidate(
                candidate,
                "legal_ai_style:drop_redundant_rowniez",
                "transition_monotony",
                0.47,
                0.08,
                nlp_confidence,
            )
        )

    return out


def _abstract_frame_rewrites(sentence: str, *, nlp_confidence: float | None) -> list[Candidate]:
    out: list[Candidate] = []

    replacements: list[tuple[str, str, str, str, float, float]] = [
        (
            r"\bjedną z najważniejszych cech\b",
            "jedną z kluczowych cech",
            "legal_ai_style:key_feature_frame",
            "abstract_frame",
            0.50,
            0.08,
        ),
        (
            r"\bszczególne znaczenie ma\b",
            "duże znaczenie ma",
            "legal_ai_style:special_importance_frame",
            "abstract_frame",
            0.52,
            0.08,
        ),
        (
            r"\bma bardzo istotne znaczenie\b",
            "ma bardzo duże znaczenie",
            "legal_ai_style:very_important_frame",
            "abstract_frame",
            0.51,
            0.08,
        ),
        (
            r"\bma istotne znaczenie\b",
            "ma duże znaczenie",
            "legal_ai_style:important_frame",
            "abstract_frame",
            0.49,
            0.08,
        ),
        (
            r"\bmają istotne znaczenie\b",
            "mają duże znaczenie",
            "legal_ai_style:important_frame_pl",
            "abstract_frame",
            0.49,
            0.08,
        ),
        (
            r"\bma szczególne znaczenie\b",
            "ma duże znaczenie",
            "legal_ai_style:special_importance_direct",
            "abstract_frame",
            0.49,
            0.08,
        ),
        (
            r"\bodgrywa istotną rolę\b",
            "ma duże znaczenie",
            "legal_ai_style:role_frame",
            "abstract_frame",
            0.48,
            0.08,
        ),
        (
            r"\bodgrywa szczególną rolę\b",
            "ma duże znaczenie",
            "legal_ai_style:special_role_frame",
            "abstract_frame",
            0.48,
            0.08,
        ),
        (
            r"\bważne znaczenie mają\b",
            "ważne są",
            "legal_ai_style:important_plural_frame",
            "abstract_frame",
            0.50,
            0.08,
        ),
        (
            r"\bdrugą istotną funkcją jest\b",
            "drugą ważną funkcją jest",
            "legal_ai_style:important_function_frame",
            "abstract_frame",
            0.49,
            0.08,
        ),
        (
            r"\bnajbardziej klasyczną postacią\b",
            "klasyczną postacią",
            "legal_ai_style:drop_most_classic",
            "empty_emphasis",
            0.48,
            0.06,
        ),
        (
            r"\bma\s+(?:\p{L}+\s+)?znaczenie praktyczne\b",
            "ma praktyczne znaczenie",
            "legal_ai_style:practical_importance_frame",
            "abstract_frame",
            0.48,
            0.08,
        ),
    ]
    for pattern, replacement, rule, issue, score, risk in replacements:
        regex = re.compile(pattern, re.IGNORECASE)
        match = regex.search(sentence)
        if not match:
            continue
        candidate = regex.sub(lambda m: _preserve_case(m.group(0), replacement), sentence, count=1)
        if candidate != sentence:
            out.append(_candidate(candidate, rule, issue, score, risk, nlp_confidence))

    return out


def _nlp_unlocked_rewrites(sentence: str, *, nlp_confidence: float | None) -> list[Candidate]:
    out: list[Candidate] = []
    match = re.search(r"(?P<head>^\s*.{6,90}?)\bprzejawia się w\b", sentence, re.IGNORECASE)
    if match and "również" not in match.group(0).lower():
        candidate = (
            sentence[: match.start()]
            + f"{match.group('head').rstrip()} widać w"
            + sentence[match.end() :]
        )
        if candidate != sentence and has_finite_verb(candidate):
            out.append(
                _candidate(
                    candidate,
                    "legal_ai_style:przejawia_sie_w",
                    "abstract_frame",
                    0.46,
                    0.10,
                    nlp_confidence,
                )
            )
    return out


def _candidate(
    text: str,
    rule: str,
    targeted_issue: str,
    score: float,
    risk: float,
    nlp_confidence: float | None,
) -> Candidate:
    return Candidate(
        text,
        rule,
        score,
        stage="legal_rewrite",
        operation_type="legal_ai_style_rewrite",
        risk=risk,
        targeted_issue=targeted_issue,
        nlp_confidence=nlp_confidence,
    )


def _nlp_confidence(analysis) -> float | None:
    if analysis is None:
        return None
    summary = analysis.dependency_summary() if hasattr(analysis, "dependency_summary") else {}
    score = 0.0
    if summary.get("has_finite_verb"):
        score += 0.45
    if summary.get("has_subject"):
        score += 0.35
    if summary.get("has_object"):
        score += 0.20
    return round(score, 4)


def _preserve_case(original: str, replacement: str) -> str:
    if original and original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def _unique(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        unique.append(candidate)
    return unique
