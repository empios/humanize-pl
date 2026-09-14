import pytest
from humanize_pl.safety.validators import validate_candidate
from humanize_pl.safety.protectors import ProtectedText

class FakeContradictingNLI:
    def check_entailment(self, premise: str, hypothesis: str) -> bool:
        return False

def test_nli_gate_rejects_contradiction():
    res = validate_candidate(
        original="Sąd oddalił powództwo.",
        candidate="Sąd uwzględnił powództwo.",
        protected=ProtectedText("", {}),
        max_length_ratio=2.0,
        nli=FakeContradictingNLI()
    )
    assert not res.ok
    assert any(c.name == "semantic_contradiction" and not c.ok for c in res.checks)
