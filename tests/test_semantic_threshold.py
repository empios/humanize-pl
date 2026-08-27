"""Per-operation semantic thresholds.

Removing a leading discourse frame preserves the assertion but moves the
embedding a long way on a short sentence. Measured over 97 candidates on the
benchmark set the two populations separate cleanly: ai_artifact_reduction runs
down to 0.825, every other operation stays above 0.933.
"""

from __future__ import annotations

import pytest

from humanize_pl.config import (
    AI_ARTIFACT_OPERATIONS,
    AI_ARTIFACT_SIMILARITY_OFFSET,
    HumanizeConfig,
    Mode,
)


@pytest.mark.parametrize(
    ("mode", "base", "artifact"),
    [(Mode.conservative, 0.94, 0.84), (Mode.standard, 0.90, 0.80), (Mode.strong, 0.86, 0.76)],
)
def test_artifact_threshold_sits_below_the_general_one(mode, base, artifact) -> None:
    config = HumanizeConfig(mode=mode)

    assert config.similarity_threshold() == pytest.approx(base)
    assert config.similarity_threshold_for("ai_artifact_reduction") == pytest.approx(artifact)


@pytest.mark.parametrize(
    "operation",
    ["voice_transform", "sentence_split", "debureaucratization", "legal_style_rewrite", None],
)
def test_other_operations_keep_the_general_threshold(operation) -> None:
    config = HumanizeConfig(mode=Mode.standard)

    assert config.similarity_threshold_for(operation) == config.similarity_threshold()


def test_offset_follows_an_explicit_threshold_override() -> None:
    """--semantic-threshold moves the base; the relationship must hold."""
    config = HumanizeConfig(mode=Mode.standard, semantic_threshold=0.95)

    assert config.similarity_threshold_for("ai_artifact_reduction") == pytest.approx(
        0.95 - AI_ARTIFACT_SIMILARITY_OFFSET
    )


def test_the_exempt_set_stays_narrow() -> None:
    """Widening this must be a deliberate act, not a side effect."""
    assert AI_ARTIFACT_OPERATIONS == {"ai_artifact_reduction"}


def test_threshold_clears_the_observed_tail_with_margin() -> None:
    """Lowest similarity seen for this operation class on the benchmark set."""
    observed_minimum = 0.825
    threshold = HumanizeConfig(mode=Mode.standard).similarity_threshold_for(
        "ai_artifact_reduction"
    )

    assert threshold < observed_minimum
    assert observed_minimum - threshold < 0.05, "margin should be tight, not permissive"
