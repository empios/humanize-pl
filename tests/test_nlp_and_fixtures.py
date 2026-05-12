import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import humanize_pl.core as core
from humanize_pl.cli import app
from humanize_pl.config import Engine, HumanizeConfig, Mode
from humanize_pl.core import humanize_text
from humanize_pl.pipeline import LegalPipeline
from humanize_pl.reports.report import write_json_report
from humanize_pl.rules.base import Candidate
from humanize_pl.results import HumanizeResult
from humanize_pl.rules.engine import RuleEngine
from humanize_pl.nlp.stanza_engine import SentenceAnalysis, TokenInfo
from humanize_pl.safety.protectors import protect_text


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "legal_docs" / "isap_samples.json"


class FakeToken:
    def __init__(self, text: str, upos: str | None = None, feats: str | None = None) -> None:
        self.text = text
        self.lemma = text.lower()
        self.upos = upos
        self.feats = feats
        self.head = None
        self.deprel = None


class FakeAnalysis:
    def __init__(self, tokens: list[FakeToken]) -> None:
        self.tokens = tokens

    def dependency_summary(self):
        return {
            "has_finite_verb": any(token.upos == "VERB" for token in self.tokens),
            "has_subject": any(token.deprel == "nsubj" for token in self.tokens),
            "has_object": any(token.deprel == "obj" for token in self.tokens),
            "finite_verbs": [token.text for token in self.tokens if token.upos == "VERB"],
            "subjects": [token.text for token in self.tokens if token.deprel == "nsubj"],
            "objects": [token.text for token in self.tokens if token.deprel == "obj"],
        }


class FakeStanzaEngine:
    def analyze_sentence(self, sentence: str) -> FakeAnalysis:
        finite_words = {
            "jest",
            "ma",
            "wskazuje",
            "wynika",
            "prowadzi",
            "wykonuje",
            "chroni",
        }
        tokens = []
        for word in sentence.replace(".", "").split():
            lower = word.lower()
            if lower in finite_words:
                tokens.append(FakeToken(word, "VERB", "VerbForm=Fin|Tense=Pres"))
            else:
                tokens.append(FakeToken(word, "NOUN", None))
        return FakeAnalysis(tokens)


class SplitOnlyRuleEngine(RuleEngine):
    def generate_candidates(self, sentence: str, **kwargs):
        return [
            Candidate(
                "Pracownik wykonuje pracę. Ponadto sam wymóg organizuje ocenę.",
                "test:bad_split",
                0.9,
                stage="quality_gate",
                operation_type="sentence_split",
            )
        ]


class OneCandidateRuleEngine(RuleEngine):
    def __init__(self, candidate: Candidate, *, mode: Mode = Mode.standard) -> None:
        super().__init__(mode=mode)
        self.candidate = candidate

    def generate_candidates(self, sentence: str, **kwargs):
        return [self.candidate]


class FakeSimilarity:
    def __init__(self, value: float) -> None:
        self.value = value

    def similarity(self, left: str, right: str) -> float:
        return self.value


class FakeFluency:
    def __init__(self, delta: float) -> None:
        self._delta = delta

    def delta(self, left: str, right: str) -> float:
        return self._delta


def test_stanza_gate_rejects_split_without_finite_verb():
    original = "Pracownik wykonuje pracę i sam wymóg organizuje ocenę."
    protected = protect_text(original)
    pipeline = LegalPipeline(
        config=HumanizeConfig(mode=Mode.standard, engine=Engine.nlp),
        protected=protected,
        rule_engine=SplitOnlyRuleEngine(mode=Mode.standard),
        stanza_engine=FakeStanzaEngine(),
        include_candidates=True,
    )
    result = pipeline.process_paragraph(original, paragraph_index=0)
    assert result.text == original
    assert result.rejected[0].reason == "split produced fragment without Stanza finite verb"
    assert any(
        gate["name"] == "stanza_finite_verb" and not gate["ok"]
        for gate in result.traces[0].gate_results
    )


def test_stanza_dependency_summary_detects_sentence_roles():
    analysis = SentenceAnalysis(
        tokens=[
            TokenInfo("Pracownik", "pracownik", "NOUN", None, 2, "nsubj"),
            TokenInfo("wykonuje", "wykonywać", "VERB", "VerbForm=Fin|Tense=Pres", 0, "root"),
            TokenInfo("pracę", "praca", "NOUN", None, 2, "obj"),
        ]
    )
    summary = analysis.dependency_summary()
    assert summary["has_finite_verb"]
    assert summary["has_subject"]
    assert summary["has_object"]
    assert summary["subjects"] == ["Pracownik"]
    assert summary["objects"] == ["pracę"]


def test_transformer_similarity_blocks_semantic_drift():
    original = "Pracownik wykonuje pracę pod kierownictwem pracodawcy."
    protected = protect_text(original)
    pipeline = LegalPipeline(
        config=HumanizeConfig(mode=Mode.standard, engine=Engine.hybrid, semantic_threshold=0.90),
        protected=protected,
        rule_engine=OneCandidateRuleEngine(
            Candidate(
                "Pracownik wykonuje pracę dla pracodawcy.",
                "test:semantic_drift",
                0.95,
            )
        ),
        semantic=FakeSimilarity(0.20),
        include_candidates=True,
    )
    result = pipeline.process_paragraph(original, paragraph_index=0)
    assert result.text == original
    assert result.rejected[0].reason == "semantic similarity below threshold"
    assert any(
        gate["name"] == "semantic_similarity" and not gate["ok"]
        for gate in result.traces[0].gate_results
    )


def test_fluency_scorer_rejects_degraded_candidate():
    original = "Podsumowując źródła prawa pracy tworzą system."
    protected = protect_text(original)
    pipeline = LegalPipeline(
        config=HumanizeConfig(mode=Mode.standard, engine=Engine.hybrid, min_fluency_delta=-0.5),
        protected=protected,
        rule_engine=OneCandidateRuleEngine(
            Candidate(
                "Podsumowując, źródła prawa pracy tworzą system.",
                "test:comma",
                0.95,
            )
        ),
        fluency=FakeFluency(-2.0),
        include_candidates=True,
    )
    result = pipeline.process_paragraph(original, paragraph_index=0)
    assert result.text == original
    assert result.rejected[0].reason == "fluency score degraded below threshold"
    assert result.traces[0].fluency_delta == -2.0


def test_quality_gate_cannot_be_bypassed_by_transformer_scores():
    original = "Strona może złożyć oświadczenie w terminie 7 dni."
    protected = protect_text(original)
    pipeline = LegalPipeline(
        config=HumanizeConfig(mode=Mode.standard, engine=Engine.hybrid),
        protected=protected,
        rule_engine=OneCandidateRuleEngine(
            Candidate(
                "Strona musi złożyć oświadczenie w terminie 7 dni.",
                "test:normativity_drift",
                0.99,
            )
        ),
        semantic=FakeSimilarity(1.0),
        fluency=FakeFluency(1.0),
        include_candidates=True,
    )
    result = pipeline.process_paragraph(original, paragraph_index=0)
    assert result.text == original
    assert result.rejected[0].reason == "normativity changed"
    assert not any(gate["name"] == "semantic_similarity" for gate in result.traces[0].gate_results)


def test_report_contains_hybrid_model_metadata(tmp_path):
    result = HumanizeResult(
        text="Podsumowując, źródła prawa pracy tworzą system.",
        changed=True,
        engine_used="hybrid",
        model_status={"stanza": "ready", "semantic": "ready", "fluency": "ready"},
        semantic_model="fake-semantic",
        fluency_model="fake-fluency",
    )
    report_path = tmp_path / "hybrid.json"
    write_json_report(result, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["model_status"]["semantic"] == "ready"
    assert payload["semantic_model"] == "fake-semantic"
    assert payload["fluency_model"] == "fake-fluency"


def test_hybrid_fallback_records_model_status(monkeypatch):
    class BrokenModel:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("missing model")

    monkeypatch.setattr(core, "StanzaEngine", BrokenModel)
    monkeypatch.setattr(core, "EmbeddingSimilarityValidator", BrokenModel)
    monkeypatch.setattr(core, "MaskedLMFluencyScorer", BrokenModel)

    result = core.humanize_text("Pracownik wykonuje pracę.", engine="hybrid")
    assert result.engine_used == "basic"
    assert result.model_status["stanza"].startswith("unavailable")
    assert result.model_status["semantic"].startswith("unavailable")
    assert result.model_status["fluency"].startswith("unavailable")
    assert result.warnings


def test_require_models_raises_when_requested_model_is_missing(monkeypatch):
    class BrokenModel:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("missing model")

    monkeypatch.setattr(core, "StanzaEngine", BrokenModel)

    with pytest.raises(RuntimeError, match="Required Stanza model unavailable"):
        core.humanize_text("Pracownik wykonuje pracę.", engine="nlp", require_models=True)


def test_offline_models_flag_is_passed_to_model_loaders(monkeypatch):
    calls: dict[str, bool] = {}

    class FakeStanza:
        def __init__(self, *, offline: bool = False) -> None:
            calls["stanza"] = offline

        def analyze_sentence(self, sentence: str):
            return FakeAnalysis([FakeToken("jest", "VERB", "VerbForm=Fin")])

    class FakeSemantic:
        def __init__(self, *, threshold: float, model_name=None, offline: bool = False) -> None:
            calls["semantic"] = offline
            self.model_name = model_name or "fake-semantic"

        def similarity(self, left: str, right: str) -> float:
            return 1.0

    class FakeFluency:
        def __init__(self, model_name=None, *, offline: bool = False) -> None:
            calls["fluency"] = offline
            self.model_name = model_name or "fake-fluency"

        def delta(self, left: str, right: str) -> float:
            return 0.0

    monkeypatch.setattr(core, "StanzaEngine", FakeStanza)
    monkeypatch.setattr(core, "EmbeddingSimilarityValidator", FakeSemantic)
    monkeypatch.setattr(core, "MaskedLMFluencyScorer", FakeFluency)

    result = core.humanize_text(
        "Podsumowując źródła prawa pracy tworzą system.",
        mode="standard",
        engine="hybrid",
        require_models=True,
        offline_models=True,
    )
    assert result.engine_used == "hybrid"
    assert calls == {"stanza": True, "semantic": True, "fluency": True}


def test_cli_exposes_offline_models_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--offline-models" in result.stdout


def test_intra_sentence_redundancy_reduction_is_safe():
    text = (
        "Pracownik wykonuje pracę pod kierownictwem, "
        "oraz pracownik pozostaje w dyspozycji pracodawcy."
    )
    result = humanize_text(text, mode="standard", include_candidates=True)
    assert "oraz pozostaje w dyspozycji pracodawcy" in result.text
    assert any(
        trace.rule == "redundancy:drop_repeated_subject_in_sentence"
        and trace.status == "accepted"
        for trace in result.all_candidates
    )


def test_tautological_adj_pair_is_reduced():
    from humanize_pl.rules.redundancy import _tautological_adj_candidates
    from humanize_pl.config import Mode

    sentence = "Warunki konieczne i niezbędne muszą być spełnione."
    cands = _tautological_adj_candidates(sentence, mode=Mode.standard)
    assert len(cands) == 1
    assert cands[0].text == "Warunki niezbędne muszą być spełnione."
    assert cands[0].rule == "redundancy:drop_tautological_adj_pair"
    assert cands[0].operation_type == "redundancy_reduction"


def test_tautological_adj_inflected_forms():
    from humanize_pl.rules.redundancy import _tautological_adj_candidates
    from humanize_pl.config import Mode

    # Inflected: koniecznych i niezbędnych (Gen Plur)
    sentence = "Brak dokumentów koniecznych i niezbędnych uniemożliwia rejestrację."
    cands = _tautological_adj_candidates(sentence, mode=Mode.standard)
    assert len(cands) == 1
    assert "niezbędnych" in cands[0].text
    assert "koniecznych" not in cands[0].text


def test_tautological_adj_not_fired_for_non_pair():
    from humanize_pl.rules.redundancy import _tautological_adj_candidates
    from humanize_pl.config import Mode

    # "ważny i prawomocny" — not in the tautology list
    sentence = "Wyrok jest ważny i prawomocny."
    cands = _tautological_adj_candidates(sentence, mode=Mode.standard)
    assert cands == []


def test_tautological_adj_not_fired_in_conservative_mode():
    from humanize_pl.rules.redundancy import redundancy_candidates
    from humanize_pl.config import Mode

    sentence = "Analiza jest kompleksowa i wyczerpująca."
    cands = redundancy_candidates(
        sentence,
        previous_sentence=None,
        mode=Mode.conservative,
        paragraph_features=None,
    )
    assert cands == []


def test_tautological_adj_various_pairs():
    from humanize_pl.rules.redundancy import _tautological_adj_candidates
    from humanize_pl.config import Mode

    cases = [
        ("Wymóg jest jasny i oczywisty.", "oczywisty"),
        ("Przesłanki kluczowe i zasadnicze decydują o wyniku.", "zasadnicze"),
        ("Obowiązek całkowity i zupełny spoczywa na stronach.", "zupełny"),
    ]
    for sentence, expected_word in cases:
        cands = _tautological_adj_candidates(sentence, mode=Mode.standard)
        assert cands, f"Expected candidate for: {sentence}"
        assert expected_word in cands[0].text, (
            f"Expected '{expected_word}' in '{cands[0].text}'"
        )


def test_isap_fixture_documents_smoke(tmp_path):
    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(samples) >= 3
    for sample in samples:
        result = humanize_text(sample["text"], mode="standard", include_candidates=True)
        assert "surprisingly" not in result.text.lower()
        assert "Ponadto za wynagrodzeniem" not in result.text
        assert "__PROTECTED_" not in result.text
        if sample["id"] == "kodeks_pracy":
            assert "art. 22 § 1" in result.text
        report_path = tmp_path / f"{sample['id']}.json"
        write_json_report(result, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert "quality" in payload
        assert payload["quality"]["word_count_estimate"] > 0
        assert "operation_types" in payload["quality"]
