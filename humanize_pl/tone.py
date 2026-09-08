"""Czy dokument brzmi jak dokumenty tej kancelarii.

A separate question from whether it reads as machine-written, and a document
can fail one while passing the other. An opinion drafted entirely by a person
can still be twice as long-winded as everything else the office sends out; a
clean AI signal says nothing about that.

The comparison is against the office's own measured profile - the same
measurement the detector calibrates against - and it is reported in sentences
rather than numbers, because the person reading it is a lawyer deciding
whether to send the document, not an operator reading a dashboard.

Nothing here blocks anything. House style is a preference, and a preference
that fails a document is a preference nobody keeps switched on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# A metric is only worth mentioning when it sits outside the range the office
# itself spans. Inside p50-p95 the document is doing what the office does.
#
# The floor exists because percentiles from a dozen documents are narrow: on a
# thin profile almost anything lands outside p95, and a report that flags
# every document flags nothing.
MIN_RELATIVE_GAP = 0.15

# When an office's documents are all alike, p95 collapses onto p50 and the
# "usual range" is a point rather than a range. Every document then sits
# outside it, and a report that flags everything says nothing. A distribution
# with no spread has to show a much larger difference before it is worth a
# sentence - we cannot tell a real departure from measurement noise on it.
DEGENERATE_SPREAD = 0.10
DEGENERATE_GAP = 0.50


@dataclass(frozen=True)
class ToneDeviation:
    """One way the document departs from the office's own writing."""

    metric: str
    observed: float
    house_p50: float
    house_p95: float
    direction: str
    sentence_pl: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToneReport:
    profile_name: str | None = None
    documents: int = 0
    indicative: bool = True
    checked: bool = False
    deviations: list[ToneDeviation] = field(default_factory=list)
    note: str | None = None

    @property
    def matches_house_style(self) -> bool:
        return self.checked and not self.deviations

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "documents": self.documents,
            "indicative": self.indicative,
            "checked": self.checked,
            "matches_house_style": self.matches_house_style,
            "deviations": [row.to_json() for row in self.deviations],
            "note": self.note,
        }


def _phrase(metric: str, direction: str, observed: float, typical: float) -> str:
    """Say the deviation the way a person would, with both numbers."""
    if metric == "mean_sentence_words":
        return (
            f"Zdania są {'dłuższe' if direction == 'high' else 'krótsze'} niż w waszych "
            f"dokumentach: średnio {observed:.0f} słów wobec zwykłych {typical:.0f}."
        )
    if metric == "sentence_length_cv":
        if direction == "low":
            return (
                "Zdania są bardziej równej długości niż w waszych dokumentach — "
                "wasze teksty mieszają zdania krótkie i długie wyraźniej."
            )
        return "Długość zdań zmienia się mocniej niż w waszych dokumentach."
    if metric == "paragraph_shape_cv":
        if direction == "low":
            return (
                "Akapity są bardziej jednakowej długości niż u was — wasze dokumenty "
                "mają wyraźniejszy rytm krótszych i dłuższych ustępów."
            )
        return "Akapity różnią się długością bardziej niż w waszych dokumentach."
    if metric == "opening_diversity":
        if direction == "low":
            return (
                "Zdania częściej zaczynają się tak samo niż w waszych dokumentach — "
                "powtarza się ten sam początek."
            )
        return "Początki zdań są bardziej zróżnicowane niż w waszych dokumentach."
    if metric == "type_token_ratio":
        if direction == "low":
            return (
                "Słownictwo jest uboższe niż w waszych dokumentach — więcej powtórzeń "
                "tych samych wyrazów."
            )
        return "Słownictwo jest bogatsze niż w waszych dokumentach."
    return f"{metric}: {observed:.3f} wobec zwykłych {typical:.3f}."


# Which document metric is compared against which distribution on the profile.
_COMPARED: dict[str, str] = {
    "mean_sentence_words": "sentence_words",
    "sentence_length_cv": "sentence_length_cv",
    "paragraph_shape_cv": "paragraph_shape_cv",
    "opening_diversity": "opening_diversity",
    "type_token_ratio": "windowed_ttr",
}


def compare_tone(metrics: dict[str, float], style_profile) -> ToneReport:
    """Compare a document's shape against the office's own writing.

    A profile built before this existed, or from too few documents to measure,
    reports that nothing was checked. "Not compared" and "matches" must not
    read the same in a report someone signs.
    """
    if style_profile is None:
        return ToneReport(note="Brak profilu kancelarii — nie porównano.")

    reference = style_profile.reference_profile()
    if reference is None:
        return ToneReport(
            profile_name=style_profile.name,
            documents=style_profile.document_count,
            note="Profil kancelarii powstał przed wprowadzeniem pomiaru — nie porównano.",
        )
    if not metrics:
        return ToneReport(
            profile_name=style_profile.name,
            documents=style_profile.document_count,
            note="Dokument za krótki, żeby zmierzyć styl.",
        )

    deviations: list[ToneDeviation] = []
    for metric, attribute in _COMPARED.items():
        observed = metrics.get(metric)
        distribution = getattr(reference, attribute, None)
        if observed is None or distribution is None or not distribution.p50:
            continue

        if observed > distribution.p95:
            typical, direction = distribution.p50, "high"
            gap = (observed - distribution.p95) / distribution.p95 if distribution.p95 else 0.0
        elif observed < distribution.p50:
            typical, direction = distribution.p50, "low"
            gap = (distribution.p50 - observed) / distribution.p50
        else:
            continue

        spread = (distribution.p95 - distribution.p50) / distribution.p50
        floor = MIN_RELATIVE_GAP if spread >= DEGENERATE_SPREAD else DEGENERATE_GAP
        if gap < floor:
            continue
        deviations.append(
            ToneDeviation(
                metric=metric,
                observed=round(observed, 4),
                house_p50=distribution.p50,
                house_p95=distribution.p95,
                direction=direction,
                sentence_pl=_phrase(metric, direction, observed, typical),
            )
        )

    return ToneReport(
        profile_name=style_profile.name,
        documents=style_profile.document_count,
        indicative=style_profile.reference_is_indicative,
        checked=True,
        deviations=deviations,
    )
