"""The fluency gate must not veto AI-artifact removal.

A masked LM scores "fluent" and "high-probability" identically, and AI-style
discourse frames are high-probability Polish. Left unchecked, the fluency
scorer rejects exactly the edits the engine exists to make.
"""

from __future__ import annotations

from humanize_pl.config import HumanizeConfig, Mode
from humanize_pl.pipeline import FLUENCY_EXEMPT_OPERATIONS, LegalPipeline
from humanize_pl.rules.engine import RuleEngine
from humanize_pl.safety.protectors import protect_text


class _AlwaysDegradingFluency:
    """Stands in for HerBERT scoring a stripped discourse frame as less fluent."""

    def delta(self, original: str, candidate: str) -> float:
        return -99.0


def _pipeline(config: HumanizeConfig, protected):
    return LegalPipeline(
        config=config,
        protected=protected,
        rule_engine=RuleEngine(mode=config.mode),
        fluency=_AlwaysDegradingFluency(),
    )


def test_discourse_frame_removal_survives_a_hostile_fluency_scorer() -> None:
    text = "Warto podkreślić, że ciężar dowodu rozkłada się w tym przypadku nierównomiernie."
    config = HumanizeConfig(mode=Mode.standard)
    protected = protect_text(text)

    result = _pipeline(config, protected).process_paragraph(protected.text, paragraph_index=0)

    assert result.text.startswith("Ciężar dowodu")
    assert any(change.rule.startswith("ai_artifact:") for change in result.changes)


def test_other_operations_remain_subject_to_the_fluency_gate() -> None:
    """The exemption is narrow — it must not disable the gate wholesale."""
    assert "voice_transform" not in FLUENCY_EXEMPT_OPERATIONS
    assert "sentence_split" not in FLUENCY_EXEMPT_OPERATIONS
    assert "debureaucratization" not in FLUENCY_EXEMPT_OPERATIONS
    assert "ai_artifact_reduction" in FLUENCY_EXEMPT_OPERATIONS
