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


def test_a_pair_too_long_for_the_model_is_checked_in_aligned_parts(monkeypatch):
    """Past 512 tokens the end of a long rewrite went unread: a sentence
    invented there was caught in 4 of 20. Each aligned part must hold."""
    from humanize_pl.nlp import semantic

    calls = []

    class Tokenizer:
        def __call__(self, premise, hypothesis):
            return {"input_ids": (premise + " " + hypothesis).split()}

    def fake_pipeline(pair, top_k=None):
        calls.append(pair)
        invented = "wymyślone" in pair["text_pair"] and "wymyślone" not in pair["text"]
        entail = 0.05 if invented else 0.9
        return [
            {"label": "entailment", "score": entail},
            {"label": "neutral", "score": 0.95 - entail},
            {"label": "contradiction", "score": 0.05},
        ]

    fake_pipeline.tokenizer = Tokenizer()
    monkeypatch.setattr(semantic, "_get_nli_pipeline", lambda name, offline: fake_pipeline)
    monkeypatch.setattr(semantic, "MAX_PAIR_TOKENS", 24)
    nli = semantic.NLIValidator()
    source = (
        "Spotkanie zespołu projektowego odbędzie się w czwartek. Agenda obejmuje budżet. "
        "Prosimy o przygotowanie raportów kwartalnych. Raporty trafią do zarządu firmy."
    )
    rewrite = (
        "Spotkanie zespołu projektowego będzie w czwartek. Omówimy budżet. "
        "Przygotujcie raporty kwartalne. Raporty trafią do zarządu firmy."
    )
    assert nli.check_entailment(source, rewrite)
    # The pair was split: no call carried the whole of both texts
    assert all(len(call["text"]) < len(source) for call in calls)
    assert not nli.check_entailment(source, rewrite + " To zdanie jest wymyślone.")


def test_aligned_cut_follows_a_merged_sentence():
    from humanize_pl.nlp.semantic import _aligned_cuts

    runs = [
        ["Umowa została podpisana w poniedziałek.", "Strony ustaliły cenę towaru."],
        ["Dostawa nastąpi w przyszłym miesiącu.", "Płatność przelewem bankowym."],
    ]
    # The rewrite merges the first run's two sentences into one
    rewrite = [
        "Umowa została podpisana w poniedziałek, a strony ustaliły cenę towaru.",
        "Dostawa nastąpi w przyszłym miesiącu.",
        "Płatność przelewem bankowym.",
    ]
    assert _aligned_cuts(runs, rewrite) == [1]


def test_only_the_changed_span_is_judged(monkeypatch):
    """Read whole, a long legal paragraph was "neutral" even against itself."""
    from humanize_pl.nlp import semantic

    calls = []

    def fake_pipeline(pair, top_k=None):
        calls.append(pair)
        return [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]

    monkeypatch.setattr(semantic, "_get_nli_pipeline", lambda name, offline: fake_pipeline)
    nli = semantic.NLIValidator()
    source = "Uprzejmie informuję, że termin minął. Sprawa trafi do sądu. Koszty poniesie dłużnik."
    assert nli.check_entailment(source, source)
    assert calls == []

    assert nli.check_entailment(source, "Termin minął. Sprawa trafi do sądu. Koszty poniesie dłużnik.")
    assert {call["text"] for call in calls} | {call["text_pair"] for call in calls} == {
        "Uprzejmie informuję, że termin minął.",
        "Termin minął.",
    }

    # A sentence only added is judged against the whole source
    calls.clear()
    nli.check_entailment(source, source + " Dłużnik zapłaci podwójnie.")
    assert calls[0]["text"] == source
