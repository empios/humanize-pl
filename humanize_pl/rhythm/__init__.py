"""Rhythm humanisation: sentence-length spread and paragraph shape.

The detector has always measured these and the engine could only ever
describe them, handing the caller a regeneration constraint. This layer moves
them instead, within the same safety gates every other rewrite passes.

Read the measured headroom before expecting much of it. On real model output,
closing both axes completely is worth about 0.045 of calibrated score, and
`UNCERTAIN_BAND` - the spread the score shows when the reference profile is
rebuilt from independent samples - is 0.035. In DOCX, where the paragraph
axis is unavailable, the ceiling is 0.014. The layer is built to be correct
and to refuse when unsure, not to be decisive.
"""

from humanize_pl.rhythm.objective import RhythmObjective, band_distance, evaluate
from humanize_pl.rhythm.policy import RhythmScope, decide
from humanize_pl.rhythm.runner import (
    RhythmInvariantError,
    RhythmResult,
    apply_rhythm_pass,
)

__all__ = [
    "RhythmInvariantError",
    "RhythmObjective",
    "RhythmResult",
    "RhythmScope",
    "apply_rhythm_pass",
    "band_distance",
    "decide",
    "evaluate",
]
