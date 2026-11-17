from __future__ import annotations

from dataclasses import dataclass, replace
import regex as re

from humanize_pl.rules.legal_features import analyze_legal_review_features
from humanize_pl.safety.anchors import content_anchor_tokens
from humanize_pl.nlp.morphology import lix_score, mean_dependency_distance
from humanize_pl.nlp.frequency import sentence_formality


@dataclass(frozen=True)
class SentenceFeatures:
    word_count: int
    comma_count: int
    legal_reference_count: int
    protected_placeholder_count: int
    connective_count: int
    enumeration_count: int
    nominalization_count: int
    modal_count: int
    passive_marker_count: int
    has_finite_verb: bool
    normativity_count: int
    legal_anchor_count: int
    vague_reference_count: int
    ai_transition_count: int
    ai_artifact_score: float
    legal_risk_score: float
    complexity: float
    lix: float = 0.0
    mdd: float | None = None
    formality: float | None = None


@dataclass(frozen=True)
class ParagraphFeatures:
    sentence_count: int
    avg_sentence_words: float
    sentence_length_variance: float
    repeated_anchor_count: int
    transition_count: int
    topic_continuity: float
    repeated_opening_count: int = 0
    repeated_frame_count: int = 0
    monotony_score: float = 0.0


CONNECTIVE_PATTERN = re.compile(
    r"\b(?:ponieważ|dlatego|jednak|natomiast|zarówno|jeżeli|chyba że|o ile|a także|oraz)\b",
    re.IGNORECASE,
)
ENUMERATION_PATTERN = re.compile(r"(?:,|\boraz\b|\ba także\b|\balbo\b|\blub\b)", re.IGNORECASE)
LEGAL_REF_PATTERN = re.compile(
    r"(?:\bart\.|§|\bust\.|\bpkt\b|\bDz\.\s*U\.|\bKodeksu\b|\bKonstytucji\b)|__PROTECTED_\d+__",
    re.IGNORECASE,
)
MODAL_PATTERN = re.compile(
    r"\b(?:należy|może|mogą|powinien|powinna|powinno|obowiązek|uprawnienie|zakaz)\b",
    re.IGNORECASE,
)
PASSIVE_PATTERN = re.compile(
    r"\b(?:został|została|zostało|zostały|jest|są)\s+\p{L}+(?:ny|na|ne|te|ta)\b",
    re.IGNORECASE,
)
NOMINALIZATION_SUFFIXES = ("anie", "enie", "cie", "ość", "acja", "izja", "yzja")
TRANSITION_START_PATTERN = re.compile(
    r"^\s*(?:Ponadto|Jednak|Natomiast|Z kolei|Oznacza to|Warto|Należy)\b",
    re.IGNORECASE,
)
OPENING_FRAME_PATTERN = re.compile(
    r"^\s*(warto\s+(?:wskazać|zauważyć|podkreślić|odnotować|zaznaczyć)|"
    r"należy\s+(?:wskazać|zauważyć|podkreślić|odnotować|zaznaczyć)|"
    r"ponadto|dodatkowo|co\s+więcej|z\s+kolei|oznacza\s+to|"
    r"duże\s+znaczenie\s+ma|istotne\s+znaczenie\s+ma)\b",
    re.IGNORECASE,
)
STYLE_FRAME_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bma\s+(?:bardzo\s+)?(?:istotne|duże|szczególne)\s+znaczenie\b", re.IGNORECASE),
    re.compile(r"\b(?:odgrywa|pełni)\s+(?:istotną|szczególną|ważną)\s+rolę\b", re.IGNORECASE),
    re.compile(r"\bw\s+znacznym\s+stopniu\b", re.IGNORECASE),
)


def analyze_sentence_features(sentence: str) -> SentenceFeatures:
    words = re.findall(r"\p{L}+", sentence)
    word_count = len(words)
    nominalization_count = sum(
        1 for word in words if len(word) >= 7 and word.lower().endswith(NOMINALIZATION_SUFFIXES)
    )
    protected_count = len(re.findall(r"__PROTECTED_\d+__", sentence))
    legal_ref_count = len(LEGAL_REF_PATTERN.findall(sentence))
    connective_count = len(CONNECTIVE_PATTERN.findall(sentence))
    enumeration_count = len(ENUMERATION_PATTERN.findall(sentence))
    modal_count = len(MODAL_PATTERN.findall(sentence))
    passive_count = len(PASSIVE_PATTERN.findall(sentence))
    comma_count = sentence.count(",")
    legal_review = analyze_legal_review_features(sentence)

    if word_count == 0:
        complexity = 0.0
    else:
        density = (
            nominalization_count
            + connective_count
            + modal_count
            + passive_count
            + legal_ref_count * 2
            + legal_review.normativity_count
        ) / word_count
        punctuation_load = min(0.25, comma_count / max(word_count, 1))
        complexity = min(1.0, density + punctuation_load + legal_review.ai_artifact_score * 0.20)

    return SentenceFeatures(
        word_count=word_count,
        comma_count=comma_count,
        legal_reference_count=legal_ref_count,
        protected_placeholder_count=protected_count,
        connective_count=connective_count,
        enumeration_count=enumeration_count,
        nominalization_count=nominalization_count,
        modal_count=modal_count,
        passive_marker_count=passive_count,
        has_finite_verb=legal_review.has_finite_verb,
        normativity_count=legal_review.normativity_count,
        legal_anchor_count=legal_review.legal_anchor_count,
        vague_reference_count=legal_review.vague_reference_count,
        ai_transition_count=legal_review.ai_transition_count,
        ai_artifact_score=legal_review.ai_artifact_score,
        legal_risk_score=legal_review.legal_risk_score,
        complexity=complexity,
        lix=lix_score(sentence),
        formality=sentence_formality(sentence),
    )


def enrich_features_with_analysis(
    features: SentenceFeatures, analysis
) -> SentenceFeatures:
    """Add Stanza-derived metrics (MDD) to already-computed features."""
    if analysis is None:
        return features
    mdd = mean_dependency_distance(analysis)
    if mdd is None:
        return features
    return replace(features, mdd=mdd)


def analyze_paragraph_features(sentences: list[str]) -> ParagraphFeatures:
    lengths = [len(re.findall(r"\p{L}+", sentence)) for sentence in sentences]
    sentence_count = len(lengths)
    avg_sentence_words = sum(lengths) / sentence_count if sentence_count else 0.0
    if sentence_count:
        sentence_length_variance = sum((length - avg_sentence_words) ** 2 for length in lengths) / sentence_count
    else:
        sentence_length_variance = 0.0

    anchor_counts: dict[str, int] = {}
    sentence_anchors: list[set[str]] = []
    for sentence in sentences:
        anchors = content_anchor_tokens(sentence)
        sentence_anchors.append(anchors)
        for anchor in anchors:
            anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

    repeated_anchor_count = sum(1 for count in anchor_counts.values() if count > 1)
    transition_count = sum(1 for sentence in sentences if TRANSITION_START_PATTERN.search(sentence))
    opening_counts: dict[str, int] = {}
    frame_counts: dict[str, int] = {}
    for sentence in sentences:
        opening = _opening_frame(sentence)
        if opening:
            opening_counts[opening] = opening_counts.get(opening, 0) + 1
        for frame in _style_frames(sentence):
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
    repeated_opening_count = sum(count - 1 for count in opening_counts.values() if count > 1)
    repeated_frame_count = sum(count - 1 for count in frame_counts.values() if count > 1)
    overlaps: list[float] = []
    for left, right in zip(sentence_anchors, sentence_anchors[1:]):
        if not left or not right:
            continue
        overlaps.append(len(left & right) / len(left | right))
    topic_continuity = sum(overlaps) / len(overlaps) if overlaps else 0.0
    monotony_score = _paragraph_monotony_score(
        sentence_count=sentence_count,
        transition_count=transition_count,
        repeated_opening_count=repeated_opening_count,
        repeated_frame_count=repeated_frame_count,
    )

    return ParagraphFeatures(
        sentence_count=sentence_count,
        avg_sentence_words=round(avg_sentence_words, 4),
        sentence_length_variance=round(sentence_length_variance, 4),
        repeated_anchor_count=repeated_anchor_count,
        transition_count=transition_count,
        topic_continuity=round(topic_continuity, 4),
        repeated_opening_count=repeated_opening_count,
        repeated_frame_count=repeated_frame_count,
        monotony_score=round(monotony_score, 4),
    )


def _opening_frame(sentence: str) -> str | None:
    match = OPENING_FRAME_PATTERN.search(sentence)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1).lower()).strip()


def _style_frames(sentence: str) -> list[str]:
    frames: list[str] = []
    for pattern in STYLE_FRAME_PATTERNS:
        if pattern.search(sentence):
            frames.append(pattern.pattern)
    return frames


def _paragraph_monotony_score(
    *,
    sentence_count: int,
    transition_count: int,
    repeated_opening_count: int,
    repeated_frame_count: int,
) -> float:
    if sentence_count <= 1:
        return 0.0
    transition_load = transition_count / sentence_count
    repeated_load = (repeated_opening_count + repeated_frame_count) / sentence_count
    return min(1.0, transition_load * 0.45 + repeated_load * 0.55)
