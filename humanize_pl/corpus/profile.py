from __future__ import annotations

from datetime import date

from humanize_pl.detect import detect_document
from .normalize import anonymisation_rate
from humanize_pl.detect.reference import Distribution, ReferenceProfile, windowed_ttr


def build_reference_profile(
    texts: list[str],
    *,
    name: str,
    genre: str,
    source: str,
) -> ReferenceProfile:
    """Measure the human baseline by running our own detectors over human text.

    Reusing the detection layer here is deliberate: the resulting rates are
    directly comparable with what the detector reports on a suspect document,
    with no separate feature implementation to drift.
    """
    sentence_words: list[float] = []
    cvs: list[float] = []
    diversities: list[float] = []
    ttrs: list[float] = []
    anonymisations: list[float] = []
    scores: list[float] = []
    family_rates: dict[str, list[float]] = {}

    document_count = 0
    word_count = 0
    sentence_count = 0

    for text in texts:
        diagnosis = detect_document(text)
        if not diagnosis.word_count or not diagnosis.metrics:
            continue

        document_count += 1
        word_count += diagnosis.word_count
        sentence_count += diagnosis.sentence_count

        sentence_words.append(diagnosis.metrics["mean_sentence_words"])
        cvs.append(diagnosis.metrics["sentence_length_cv"])
        diversities.append(diagnosis.metrics["opening_diversity"])
        ttrs.append(windowed_ttr(text))
        anonymisations.append(anonymisation_rate(text))
        scores.append(diagnosis.ai_signal_score)

        seen = {row.family: row.per_1000_words for row in diagnosis.families}
        for family in set(seen) | set(family_rates):
            family_rates.setdefault(family, []).append(seen.get(family, 0.0))

    # Families first seen late in the corpus need zero-padding for the
    # documents that preceded them, otherwise their rates are overstated.
    for values in family_rates.values():
        values.extend([0.0] * (document_count - len(values)))

    return ReferenceProfile(
        name=name,
        genre=genre,
        source=source,
        built_on=date.today().isoformat(),
        document_count=document_count,
        word_count=word_count,
        sentence_count=sentence_count,
        sentence_words=Distribution.of(sentence_words),
        sentence_length_cv=Distribution.of(cvs),
        opening_diversity=Distribution.of(diversities),
        windowed_ttr=Distribution.of(ttrs),
        anonymisation_rate=Distribution.of(anonymisations),
        signal_score=Distribution.of(scores),
        family_rates={
            family: Distribution.of(values) for family, values in sorted(family_rates.items())
        },
    )
