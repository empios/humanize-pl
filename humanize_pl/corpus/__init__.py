"""Ingestion of human reference corpora.

Depends on `humanize_pl.detect`, never the other way round: the profile data
model lives with the detector that consumes it, and this package only turns
raw corpora into profiles.
"""

from humanize_pl.detect.lexical import connective_density, mtld
from humanize_pl.detect.reference import Distribution, ReferenceProfile

from .normalize import (
    anonymisation_rate,
    is_usable,
    normalize_judgment,
    strip_markup,
)
from .profile import build_reference_profile

__all__ = [
    "Distribution",
    "ReferenceProfile",
    "anonymisation_rate",
    "build_reference_profile",
    "connective_density",
    "is_usable",
    "mtld",
    "normalize_judgment",
    "strip_markup",
]
