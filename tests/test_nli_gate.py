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


def test_nli_validator_wants_mutual_entailment(monkeypatch):
    """A changed modality rarely reads as a contradiction, but it no longer
    entails its source; the validator wants both directions to hold."""
    from humanize_pl.nlp import semantic

    verdicts = {}

    def fake_pipeline(pair, top_k=None):
        entail, contra = verdicts[(pair["text"], pair["text_pair"])]
        return [
            {"label": "entailment", "score": entail},
            {"label": "neutral", "score": 1 - entail - contra},
            {"label": "contradiction", "score": contra},
        ]

    monkeypatch.setattr(semantic, "_get_nli_pipeline", lambda name, offline: fake_pipeline)
    nli = semantic.NLIValidator()
    verdicts.update({
        ("A", "B"): (0.9, 0.05), ("B", "A"): (0.8, 0.05),   # same meaning
        ("A", "C"): (0.1, 0.8),                              # contradiction
        ("A", "D"): (0.6, 0.1), ("D", "A"): (0.1, 0.1),      # entailed one way only
    })
    assert nli.check_entailment("A", "B")
    assert not nli.check_entailment("A", "C")
    assert not nli.check_entailment("A", "D")


def test_nli_inference_runs_one_call_at_a_time(monkeypatch):
    """Concurrent GPU calls crashed the process; fragments are validated on threads."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from humanize_pl.nlp import semantic

    inside, overlaps = [0], []
    guard = threading.Lock()

    def fake_pipeline(pair, top_k=None):
        with guard:
            inside[0] += 1
            overlaps.append(inside[0] > 1)
        time.sleep(0.01)
        with guard:
            inside[0] -= 1
        return [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]

    monkeypatch.setattr(semantic, "_get_nli_pipeline", lambda name, offline: fake_pipeline)
    nli = semantic.NLIValidator()
    with ThreadPoolExecutor(4) as pool:
        assert all(pool.map(lambda _: nli.check_entailment("A", "B"), range(8)))
    assert overlaps and not any(overlaps)
