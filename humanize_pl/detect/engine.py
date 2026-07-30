from __future__ import annotations

from dataclasses import replace

import regex as re

from .reference import ReferenceProfile
from humanize_pl.sentence_splitter import split_sentences
from .base import DocumentDiagnosis, FamilySummary, Finding, ParagraphDiagnosis
from .calibration import calibrate
from .signals import WORD_RE, repeated_opening_findings, sentence_findings
from .structural import paragraph_shape_cv

# Weighted findings per 1000 words at which `ai_signal_score` saturates to 1.0.
# Provisional engineering default — replace with a value fitted against a human
# Polish legal reference corpus before treating the score as a threshold.
SATURATION_PER_1000_WORDS = 18.0


def detect_document(text: str, *, profile: ReferenceProfile | None = None) -> DocumentDiagnosis:
    """Locate AI-style signals in `text`.

    Runs independently of `Mode`, of the rule engine, and of whether any
    rewrite is possible. A document with zero accepted changes still gets a
    full diagnosis — that gap is what made the engine report clean documents.

    When a human reference profile is installed the diagnosis also carries a
    calibrated score expressing the document relative to measured human
    writing. Without one, only the raw density score is available.
    """
    paragraphs = [part for part in re.split(r"\n+", text) if part.strip()]

    findings: list[Finding] = []
    indexed_sentences: list[tuple[int, int, str]] = []
    paragraph_rows: list[ParagraphDiagnosis] = []
    total_words = 0
    total_sentences = 0

    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        paragraph_findings: list[Finding] = []
        paragraph_words = 0

        for sentence_index, sentence in enumerate(sentences):
            indexed_sentences.append((paragraph_index, sentence_index, sentence))
            paragraph_words += len(WORD_RE.findall(sentence))
            paragraph_findings.extend(
                sentence_findings(
                    sentence,
                    paragraph_index=paragraph_index,
                    sentence_index=sentence_index,
                )
            )

        findings.extend(paragraph_findings)
        total_words += paragraph_words
        total_sentences += len(sentences)
        paragraph_rows.append(
            ParagraphDiagnosis(
                paragraph_index=paragraph_index,
                word_count=paragraph_words,
                sentence_count=len(sentences),
                signal_score=_score(paragraph_findings, paragraph_words),
                finding_count=len(paragraph_findings),
            )
        )

    findings.extend(repeated_opening_findings(indexed_sentences))
    findings.sort(key=lambda f: (f.paragraph_index, f.sentence_index, f.char_start))

    diagnosis = DocumentDiagnosis(
        ai_signal_score=_score(findings, total_words),
        word_count=total_words,
        sentence_count=total_sentences,
        paragraph_count=len(paragraphs),
        findings=findings,
        families=_family_summaries(findings, total_words),
        paragraphs=paragraph_rows,
        metrics=_metrics(
            indexed_sentences,
            total_words,
            sentences_per_paragraph=[row.sentence_count for row in paragraph_rows],
        ),
    )
    return replace(diagnosis, calibration=calibrate(diagnosis, text, profile=profile))


def _score(findings: list[Finding], word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    density = sum(finding.weight for finding in findings) / word_count * 1000
    return round(min(1.0, density / SATURATION_PER_1000_WORDS), 4)


def _family_summaries(findings: list[Finding], word_count: int) -> list[FamilySummary]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.family, []).append(finding)

    rows = [
        FamilySummary(
            family=family,
            count=len(items),
            per_1000_words=round(len(items) / word_count * 1000, 4) if word_count else 0.0,
            weight_total=round(sum(item.weight for item in items), 4),
            rewritable_count=sum(1 for item in items if item.rewritable),
        )
        for family, items in grouped.items()
    ]
    return sorted(rows, key=lambda row: row.weight_total, reverse=True)


def _metrics(
    indexed_sentences: list[tuple[int, int, str]],
    word_count: int,
    *,
    sentences_per_paragraph: list[int],
) -> dict[str, float]:
    """Descriptive document metrics reported alongside the findings.

    Which of these carry weight is decided in `calibration`, against the human
    reference profile — several point the opposite way to the English-language
    literature for this genre pair.
    """
    lengths = [len(WORD_RE.findall(sentence)) for _, _, sentence in indexed_sentences]
    lengths = [length for length in lengths if length]
    if not lengths:
        return {}

    mean_length = sum(lengths) / len(lengths)
    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    openings = [
        " ".join(sentence.lower().split()[:2]) for _, _, sentence in indexed_sentences
    ]
    tokens = [
        token.lower()
        for _, _, sentence in indexed_sentences
        for token in WORD_RE.findall(sentence)
    ]

    return {
        "mean_sentence_words": round(mean_length, 4),
        "sentence_length_cv": round(variance**0.5 / mean_length, 4) if mean_length else 0.0,
        "opening_diversity": round(len(set(openings)) / len(openings), 4),
        "type_token_ratio": round(len(set(tokens)) / len(tokens), 4) if tokens else 0.0,
        "paragraph_shape_cv": paragraph_shape_cv(sentences_per_paragraph),
        "words": float(word_count),
    }
