"""What the rhythm layer is trying to reach, and how far it currently is.

The target is the human band, not the maximum. A document whose sentences all
run the same length reads as machine-written; so does one that alternates
between three words and sixty. The detector only notices the first, because it
scores "how far BELOW the human median" - so a layer that optimised the
detector's own one-sided view would happily overshoot into a second kind of
inhuman and report success.

Weights are copied from `detect.calibration` rather than invented, so the
layer improves the number the engine actually reports. `sentence_burstiness`
is deliberately zero: it is `sentence_length_cv` on a compressed scale
((cv-1)/(cv+1)), so weighting both would chase one quantity twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from humanize_pl.detect.reference import Distribution, ReferenceProfile
from humanize_pl.detect.structural import (
    paragraph_shape_cv,
    sentence_length_cv,
    sentence_length_entropy,
)

# Mirrors the weights in detect/calibration.py for the shape signals. Family
# rates are not here: no rhythm operation can change how often a document says
# "warto wskazać".
RHYTHM_WEIGHTS: dict[str, float] = {
    "sentence_length_cv": 1.5,
    "sentence_entropy": 1.0,
    "paragraph_shape_cv": 1.0,
}

# Below this a step is noise, not progress. `UNCERTAIN_BAND` (0.035) is the
# spread a document's calibrated score shows when the reference profile is
# rebuilt from independent samples; a single rhythm operation moving the
# objective less than this is not measurable.
MIN_LOSS_GAIN = 0.02


@dataclass(frozen=True)
class RhythmObjective:
    metrics: dict[str, float]
    distances: dict[str, float]
    loss: float
    # The same distance without the saturation, used only to rank candidates.
    #
    # `loss` saturates at 1.0 because that is what the detector does, and a
    # layer optimising a different curve would be optimising a number nobody
    # reports. But saturation is flat, and a document far outside the band
    # sits at 1.0 no matter what it does next - so every operation scores a
    # gain of exactly zero and the search stalls at the one place it is most
    # needed. Ranking therefore uses the unsaturated distance while the stop
    # condition keeps the detector's own.
    search_loss: float = 0.0
    active: frozenset[str] = field(default_factory=frozenset)

    @property
    def inside_band(self) -> bool:
        """Every measurable axis sits where human writing sits."""
        return self.loss == 0.0


def band_distance(observed: float, distribution: Distribution) -> float:
    """0.0 inside [p50, p95]; rising on either side.

    The lower branch matches `calibration._exceedance_low`, so minimising it
    minimises exactly what the detector adds to the score. The upper branch
    has no counterpart in the detector and exists only to stop the loop before
    it overshoots the human range - see the module docstring.
    """
    return round(min(1.0, band_distance_raw(observed, distribution)), 4)


def band_distance_raw(observed: float, distribution: Distribution) -> float:
    """`band_distance` without the ceiling. See `RhythmObjective.search_loss`."""
    p50, p95 = distribution.p50, distribution.p95
    if p50 <= 0:
        return 0.0
    if observed < p50:
        return round((p50 - observed) / (p50 * 0.5), 4)
    if p95 > p50 and observed > p95:
        return round((observed - p95) / (p95 - p50), 4)
    return 0.0


def measure_lengths(
    sentence_lengths: list[int], sentences_per_paragraph: list[int]
) -> dict[str, float]:
    """Metrics computed the way the detector computes them.

    Calls the same functions rather than repeating the formulas: a layer
    optimising its own slightly different definition of sentence-length CV
    would move a number that never reaches the report.
    """
    return {
        "sentence_length_cv": sentence_length_cv(sentence_lengths),
        "sentence_entropy": sentence_length_entropy(sentence_lengths),
        "paragraph_shape_cv": paragraph_shape_cv(sentences_per_paragraph),
    }


def evaluate(
    metrics: dict[str, float], profile: ReferenceProfile | None
) -> RhythmObjective:
    """Distance from the human band, as a weighted mean over usable axes.

    An axis whose profile distribution is empty is dropped rather than scored
    against zeros - the same rule calibration applies, and for the same
    reason: a baseline nobody measured cannot say whether a document departs
    from it.
    """
    if profile is None:
        return RhythmObjective(metrics=metrics, distances={}, loss=0.0)

    distributions = {
        "sentence_length_cv": profile.sentence_length_cv,
        "sentence_entropy": profile.sentence_entropy,
        "paragraph_shape_cv": profile.paragraph_shape_cv,
    }
    distances: dict[str, float] = {}
    active: set[str] = set()
    weighted = raw_weighted = total = 0.0
    for name, weight in RHYTHM_WEIGHTS.items():
        distribution = distributions[name]
        observed = metrics.get(name, 0.0)
        # paragraph_shape_cv is 0.0 for documents under three paragraphs,
        # where the metric is undefined rather than bad. Treating that as
        # maximum distance would make the layer split paragraphs to fix a
        # number the detector does not even score there.
        if distribution.is_empty or observed == 0.0:
            continue
        distance = band_distance(observed, distribution)
        distances[name] = distance
        active.add(name)
        weighted += distance * weight
        raw_weighted += band_distance_raw(observed, distribution) * weight
        total += weight

    return RhythmObjective(
        metrics=metrics,
        distances=distances,
        loss=round(weighted / total, 4) if total else 0.0,
        search_loss=round(raw_weighted / total, 4) if total else 0.0,
        active=frozenset(active),
    )
