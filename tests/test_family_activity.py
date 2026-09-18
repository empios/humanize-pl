"""The measurement of which signal families AI text produces, per document kind."""

from __future__ import annotations

from humanize_pl.detect import AI_FAMILIES
from humanize_pl.detect.activity import activity_for


def test_every_measured_kind_covers_every_family():
    """A family missing from the file would read as "never produced" when it
    was never measured. Regenerate with tools/family_activity.py after adding
    one to AI_FAMILIES."""
    for kind in ("contract", "filing_official", "client_communication"):
        activity = activity_for(kind)
        assert activity is not None, kind
        assert set(activity.hits) == set(AI_FAMILIES), kind
        assert activity.documents > 0 and activity.generators


def test_essay_frames_are_silent_in_contracts_and_filings_but_not_in_essays():
    """What the measurement found, pinned so a regenerated file that says
    otherwise is noticed rather than silently shipped."""
    for kind in ("contract", "filing_official"):
        assert "discourse_frame" in activity_for(kind).silent
    assert "discourse_frame" not in activity_for("client_communication").silent
    assert "nominalization" not in activity_for("contract").silent


def test_an_unmeasured_kind_is_none_not_empty():
    assert activity_for("auto") is None
    assert activity_for(None) is None
