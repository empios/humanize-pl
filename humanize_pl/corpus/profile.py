from __future__ import annotations

from collections.abc import Iterable

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
    families: Iterable[str] | None = None,
) -> ReferenceProfile:
    """Measure the human baseline by running our own detectors over human text.

    Reusing the detection layer here is deliberate: the resulting rates are
    directly comparable with what the detector reports on a suspect document,
    with no separate feature implementation to drift.

    `families` names families to record even when the corpus contains none of
    them, at a rate of zero. Without it a profile only carries families it
    happened to see, and calibration silently stops measuring the rest - so a
    corpus of genuinely clean writing produces the blindest baseline of all,
    which is exactly backwards.
    """
    sentence_words: list[float] = []
    cvs: list[float] = []
    shape_cvs: list[float] = []
    diversities: list[float] = []
    ttrs: list[float] = []
    anonymisations: list[float] = []
    scores: list[float] = []
    family_rates: dict[str, list[float]] = {family: [] for family in families or ()}

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
        shape_cvs.append(diagnosis.metrics.get("paragraph_shape_cv", 0.0))
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
        paragraph_shape_cv=Distribution.of([cv for cv in shape_cvs if cv > 0]),
        opening_diversity=Distribution.of(diversities),
        windowed_ttr=Distribution.of(ttrs),
        anonymisation_rate=Distribution.of(anonymisations),
        signal_score=Distribution.of(scores),
        family_rates={
            family: Distribution.of(values) for family, values in sorted(family_rates.items())
        },
    )
