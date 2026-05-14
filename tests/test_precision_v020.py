import json

from humanize_pl.config import Mode
from humanize_pl.config import HumanizeConfig
from humanize_pl.core import humanize_text
from humanize_pl.reports.report import write_json_report
from humanize_pl.rules.base import Candidate
from humanize_pl.rules.features import analyze_sentence_features
from humanize_pl.rules.scoring import score_candidate
from humanize_pl.safety.anchors import content_anchor_retention, content_anchor_tokens
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


def test_no_split_on_oraz_list_fragment():
    text = (
        "Jeżeli ktoś formalnie ma umowę zlecenia, ale faktycznie pracuje jak pracownik, "
        "czyli pod kierownictwem, w określonym miejscu i czasie oraz za wynagrodzeniem, "
        "może istnieć podstawa do uznania, że jest to stosunek pracy."
    )
    result = humanize_text(text, mode="standard", engine="basic")
    assert "Ponadto za wynagrodzeniem" not in result.text
    assert "w określonym miejscu i czasie oraz za wynagrodzeniem" in result.text


def test_podsumowujac_comma():
    result = humanize_text("Podsumowując źródła prawa pracy tworzą system.", mode="conservative")
    assert result.text == "Podsumowując, źródła prawa pracy tworzą system."


def test_bad_split_rejected_reported():
    text = (
        "To jest bardzo długie zdanie, które opisuje różne elementy stosunku pracy, "
        "wskazuje na ich znaczenie w praktyce oraz za wynagrodzeniem pokazuje pewien warunek."
    )
    result = humanize_text(text, mode="standard")
    assert "Ponadto za wynagrodzeniem" not in result.text


def test_legal_style_boundaries():
    result = humanize_text("Podporządkowanie pracownika nie jest jednak nieograniczone.", mode="standard")
    assert result.text == "Podporządkowanie pracownika ma jednak swoje granice."


def test_formal_legal_candidates_are_precise():
    result = humanize_text(
        "Istotne znaczenie ma sposób wykonywania pracy. "
        "Polega ona na tym, że pracownik wykonuje zadania pod kierownictwem. "
        "Zależność przejawia się również w obowiązku stosowania poleceń. "
        "Ocena zależy w dużej mierze od okoliczności.",
        mode="standard",
    )
    assert "Duże znaczenie ma sposób wykonywania pracy." in result.text
    assert "Oznacza to, że pracownik wykonuje zadania pod kierownictwem." in result.text
    assert "Zależność widać także w obowiązku stosowania poleceń" in result.text
    assert "w znacznym stopniu" in result.text


def test_przejawia_sie_rewrite_depends_on_sentence_position():
    result = humanize_text(
        "Szczególny charakter norm prawa pracy przejawia się również w ich funkcjach.",
        mode="standard",
    )
    assert result.text == "Szczególny charakter norm prawa pracy widać także w ich funkcjach."

    result = humanize_text("Przejawia się również w sposobie organizacji pracy.", mode="standard")
    assert result.text == "Widać to także w sposobie organizacji pracy."


def test_legal_terms_are_preserved():
    text = "Co do zasady zgodnie z art. 22 § 1 Kodeksu pracy pracownik wykonuje pracę osobiście."
    result = humanize_text(text, mode="standard")
    # "Co do zasady" is a bureaucratic filler → replaced with "zasadniczo"; legal citation preserved
    assert "art. 22 § 1 Kodeksu pracy" in result.text


def test_role_phrase_rewrite_keeps_polish_syntax():
    result = humanize_text(
        "Szczególną rolę w systemie źródeł prawa pracy odgrywają układy zbiorowe pracy.",
        mode="standard",
    )
    assert result.text == "Duże znaczenie w systemie źródeł prawa pracy mają układy zbiorowe pracy."
    assert "surprisingly" not in result.text.lower()


def test_preserves_outer_whitespace():
    result = humanize_text("\tPodsumowując źródła prawa pracy tworzą system.  ", mode="standard")
    assert result.text == "\tPodsumowując, źródła prawa pracy tworzą system.  "


def test_rejects_fragment_sentence_without_finite_verb():
    original = "Pracownik wykonuje pracę pod kierownictwem i otrzymuje wynagrodzenie."
    candidate = (
        "Pracownik wykonuje pracę pod kierownictwem. "
        "Ponadto sam wymóg osobistego wykonywania pracy."
    )
    protected = protect_text(original)
    validation = validate_candidate(
        original,
        candidate,
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert "finite verb" in validation.reason


def test_report_contains_paragraph_sentence_and_rejection_reason(tmp_path):
    text = (
        "To jest bardzo długie zdanie, które opisuje obowiązki pracownika, "
        "wskazuje podstawy podporządkowania, przedstawia skutki organizacyjne, "
        "omawia zakres odpowiedzialności, pokazuje praktyczne znaczenie tej oceny, "
        "odnosi się do praktyki sądowej i opisuje typowe spory, "
        "oraz wskazuje typowe ryzyka dowodowe, "
        "a także pracownik wykonuje pracę osobiście w zakładzie pracy każdego dnia."
    )
    result = humanize_text(text, mode="standard", include_candidates=True)
    report_path = tmp_path / "report.json"
    write_json_report(result, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"paragraph": 0' in payload
    assert '"sentence": 0' in payload
    assert '"accepted": false' in payload
    assert '"rejection_reason"' in payload
    assert '"skipped"' in payload
    assert '"all_candidates"' in payload
    assert '"status": "rejected"' in payload


def test_candidate_trace_records_unselected_candidates():
    result = humanize_text(
        "Podsumowując podporządkowanie ma także istotne znaczenie przy odróżnianiu stosunku pracy.",
        mode="standard",
        include_candidates=True,
    )
    statuses = {candidate.status for candidate in result.all_candidates}
    assert "accepted" in statuses
    assert "not_evaluated" in statuses


def test_skipped_sentence_without_candidates_is_reported():
    result = humanize_text("Pracownik wykonuje pracę.", mode="standard")
    assert not result.changes
    assert result.skipped[0].reason == "no_candidate"


def test_istotne_znaczenie_is_not_conservative():
    text = "W prawie pracy istotne znaczenie ma art. 9 Kodeksu pracy."
    assert humanize_text(text, mode="conservative").text == text
    assert "duże znaczenie ma" in humanize_text(text, mode="standard").text


def test_validator_rejects_split_when_original_contains_oraz():
    original = "Pracownik wykonuje pracę oraz pracodawca wypłaca wynagrodzenie."
    candidate = "Pracownik wykonuje pracę. Ponadto pracodawca wypłaca wynagrodzenie."
    validation = validate_candidate(
        original,
        candidate,
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert "oraz" in validation.reason


def test_validator_rejects_numbers_dates_english_and_placeholder_leaks():
    original = "W dniu 10.05.2026 r. pracownik otrzymał 5000 zł."
    protected = protect_text(original)

    changed_number = validate_candidate(
        original,
        "W dniu 10.05.2026 r. pracownik otrzymał 6000 zł.",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not changed_number.ok
    assert changed_number.reason == "numbers changed"

    english = validate_candidate(
        original,
        "Surprisingly pracownik otrzymał 5000 zł.",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not english.ok
    assert "foreign" in english.reason

    placeholder = validate_candidate(
        protected.text,
        protected.text + " __PROTECTED_ABCD__",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not placeholder.ok
    assert "placeholder" in placeholder.reason


def test_validator_rejects_length_outliers():
    original = " ".join(["pracownik"] * 20)
    too_long = original + " " + " ".join(["dodatkowo"] * 30)
    validation = validate_candidate(
        original,
        too_long,
        protected=protect_text(original),
        max_length_ratio=1.2,
    )
    assert not validation.ok
    assert validation.reason == "candidate too long"


def test_sentence_feature_analysis_detects_legal_complexity():
    features = analyze_sentence_features(
        "Zgodnie z art. 22 § 1 Kodeksu pracy pracownik wykonuje pracę, "
        "jeżeli pozostaje pod kierownictwem oraz w określonym czasie."
    )
    assert features.legal_reference_count >= 2
    assert features.connective_count >= 2
    assert features.enumeration_count >= 2
    assert features.complexity > 0


def test_paragraph_features_detect_monotone_ai_openings():
    from humanize_pl.rules.features import analyze_paragraph_features
    from humanize_pl.sentence_splitter import split_sentences

    text = (
        "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem. "
        "Warto wskazać, że pracodawca organizuje proces pracy. "
        "Ponadto obowiązek osobistego świadczenia pracy ma istotne znaczenie. "
        "Ponadto wynagrodzenie ma istotne znaczenie dla stron."
    )
    features = analyze_paragraph_features(split_sentences(text))

    assert features.repeated_opening_count >= 2
    assert features.repeated_frame_count >= 1
    assert features.monotony_score > 0


def test_standard_reduces_monotone_ai_openings_and_reports_metrics(tmp_path):
    text = (
        "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem. "
        "Warto wskazać, że pracodawca organizuje proces pracy. "
        "Ponadto obowiązek osobistego świadczenia pracy ma istotne znaczenie. "
        "Ponadto wynagrodzenie ma istotne znaczenie dla stron."
    )
    result = humanize_text(text, mode="standard", include_candidates=True)

    assert result.text.count("Warto wskazać") < text.count("Warto wskazać")
    assert result.text.count("Ponadto") < text.count("Ponadto")
    assert any(
        trace.operation_type == "ai_artifact_reduction" and trace.status == "accepted"
        for trace in result.all_candidates
    )

    report_path = tmp_path / "report.json"
    write_json_report(result, report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    monotony = payload["quality"]["paragraph_monotony"]
    assert monotony["average_score"] > 0
    assert monotony["repeated_openings"] >= 2


def test_algorithmic_scoring_penalizes_splits_in_dense_legal_sentences():
    original = (
        "Zgodnie z art. 22 § 1 Kodeksu pracy pracownik wykonuje pracę, "
        "jeżeli pozostaje pod kierownictwem oraz w określonym czasie."
    )
    features = analyze_sentence_features(original)
    candidate = Candidate(
        "Zgodnie z art. 22 § 1 Kodeksu pracy pracownik wykonuje pracę. "
        "Ponadto pozostaje pod kierownictwem oraz w określonym czasie.",
        "split_long_sentence",
        0.50,
    )
    scored = score_candidate(original, candidate, features=features, mode=Mode.standard)
    assert scored.score < candidate.score


def test_content_anchor_retention_uses_stems_not_fixed_phrases():
    original = "Została przeprowadzona analiza danych."
    candidate = "Przeprowadzono analizę danych."
    assert content_anchor_retention(original, candidate) >= 0.5
    assert "przeprowadz" in content_anchor_tokens(original)


def test_validator_rejects_content_anchor_drift():
    original = (
        "Stosunek pracy obejmuje obowiązek osobistego wykonywania pracy przez pracownika."
    )
    candidate = (
        "Relacja zatrudnienia opisuje ogólne założenia organizacji oraz sytuację stron."
    )
    validation = validate_candidate(
        original,
        candidate,
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    # Either normativity or content-anchor check catches this drift
    assert validation.reason in {"normativity changed", "content anchors changed too much"}


def test_nlp_nominalization_does_not_strand_relative_clause():
    from humanize_pl.rules.nominalization import nominalization_candidates

    class _T:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _A:
        def __init__(self, tokens):
            self.tokens = tokens

    cases = [
        (
            "Użytkownik nie może również podejmować działań, "
            "które zakłócają funkcjonowanie platformy.",
            "działań",
            "działanie",
        ),
        (
            "Organ nie powinien podejmować czynności, które naruszają prawa strony.",
            "czynności",
            "czynność",
        ),
        (
            "Organ nie powinien podejmować decyzji, która narusza prawa strony.",
            "decyzji",
            "decyzja",
        ),
        (
            "Strona nie musi podejmować obowiązku, który wynika z umowy.",
            "obowiązku",
            "obowiązek",
        ),
        (
            "Organ nie powinien podejmować dokumentów, których nie zbadano.",
            "dokumentów",
            "dokument",
        ),
    ]
    for sentence, noun_text, noun_lemma in cases:
        verb_start = sentence.index("podejmować")
        noun_start = sentence.index(noun_text)
        analysis = _A([
            _T(
                id=1,
                text="podejmować",
                lemma="podejmować",
                upos="VERB",
                deprel="xcomp",
                head=0,
                start_char=verb_start,
                end_char=verb_start + len("podejmować"),
            ),
            _T(
                id=2,
                text=noun_text,
                lemma=noun_lemma,
                upos="NOUN",
                deprel="obj",
                head=1,
                start_char=noun_start,
                end_char=noun_start + len(noun_text),
            ),
        ])

        candidates = nominalization_candidates(sentence, mode=Mode.standard, analysis=analysis)

        assert all("ć, któr" not in candidate.text for candidate in candidates)
        assert not any(
            candidate.rule == f"nominalizacja:nlp:podejmować+{noun_lemma}"
            for candidate in candidates
        )


def test_validator_rejects_stranded_relative_clause_after_infinitive():
    original = (
        "Użytkownik nie może również podejmować działań, "
        "które zakłócają funkcjonowanie platformy."
    )
    validation = validate_candidate(
        original,
        "Użytkownik nie może również działać, które zakłócają funkcjonowanie platformy.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason == "relative clause stranded after infinitive"

    valid = validate_candidate(
        original,
        original,
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert valid.ok


def test_safe_nlp_nominalizations_still_work():
    assert (
        humanize_text(
            "Administrator powinien udzielić odpowiedzi bez zbędnej zwłoki, "
            "nie później niż w terminie 30 dni.",
            mode="standard",
            engine="nlp",
        ).text
        == (
            "Administrator powinien odpowiedzieć bez zbędnej zwłoki, "
            "nie później niż w terminie 30 dni."
        )
    )
    assert (
        humanize_text(
            "Brak zapłaty w powyższym terminie może skutkować skierowaniem sprawy do sądu.",
            mode="standard",
            engine="nlp",
        ).text
        == "Brak zapłaty w tym terminie może skutkować skierowaniem sprawy do sądu."
    )


def test_mode_to_intensity_mapping_is_stable():
    assert HumanizeConfig(mode=Mode.conservative).intensity() == 25
    assert HumanizeConfig(mode=Mode.standard).intensity() == 50
    assert HumanizeConfig(mode=Mode.strong).intensity() == 70


def test_quality_gate_rejects_connector_and_punctuation_artifacts():
    original = "Pracownik wykonuje pracę pod kierownictwem."
    protected = protect_text(original)

    double_conjunction = validate_candidate(
        original,
        "Pracownik wykonuje pracę oraz oraz pozostaje pod kierownictwem.",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not double_conjunction.ok
    assert double_conjunction.reason == "double conjunction detected"

    dangling = validate_candidate(
        original,
        "Pracownik wykonuje pracę, ponieważ",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not dangling.ok
    assert dangling.reason == "dangling connector detected"

    unbalanced = validate_candidate(
        original,
        "Pracownik wykonuje pracę (pod kierownictwem.",
        protected=protected,
        max_length_ratio=2.0,
    )
    assert not unbalanced.ok
    assert unbalanced.reason == "unbalanced punctuation"


def test_report_candidate_trace_contains_pipeline_metadata(tmp_path):
    result = humanize_text(
        "Podsumowując podporządkowanie ma także istotne znaczenie przy odróżnianiu stosunku pracy.",
        mode="standard",
        include_candidates=True,
    )
    report_path = tmp_path / "report.json"
    write_json_report(result, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"stage"' in payload
    assert '"operation_type"' in payload
    assert '"risk"' in payload
    assert '"features_before"' in payload
    assert '"features_after"' in payload
    assert '"score_before_gate"' in payload
    assert '"score_after_gate"' in payload
    assert '"gate_results"' in payload


def test_standard_applies_two_safe_steps_in_one_sentence():
    result = humanize_text(
        "Podsumowując podporządkowanie ma także istotne znaczenie przy odróżnianiu stosunku pracy.",
        mode="standard",
        include_candidates=True,
    )
    assert result.text == (
        "Podsumowując, podporządkowanie ma też duże znaczenie przy odróżnianiu stosunku pracy."
    )
    assert [change.step_index for change in result.changes] == [0, 1]
    assert [change.rule for change in result.changes] == [
        "legal_style:comma_after_podsumowujac",
        "legal_style:jest_tez_wazne",
    ]


def test_conservative_still_applies_single_step():
    result = humanize_text(
        "Podsumowując podporządkowanie ma także istotne znaczenie przy odróżnianiu stosunku pracy.",
        mode="conservative",
    )
    assert len(result.changes) == 1
    assert result.changes[0].step_index == 0
    assert "ma także istotne znaczenie" in result.text


def test_same_rule_is_not_applied_twice_in_sequence():
    result = humanize_text(
        "Podsumowując podporządkowanie ma także istotne znaczenie przy odróżnianiu stosunku pracy.",
        mode="standard",
    )
    rules = [change.rule for change in result.changes]
    assert len(rules) == len(set(rules))


def test_redundancy_reduction_drops_repeated_opening_in_standard():
    result = humanize_text(
        "Podporządkowanie pracownika jest cechą stosunku pracy. "
        "Podporządkowanie pracownika wynika z art. 22 Kodeksu pracy.",
        mode="standard",
        include_candidates=True,
    )
    assert "Wynika z art. 22 Kodeksu pracy." in result.text
    assert any(
        trace.operation_type == "redundancy_reduction" and trace.status == "accepted"
        for trace in result.all_candidates
    )


def test_redundancy_reduction_requires_finite_verb_after_drop():
    result = humanize_text(
        "Podporządkowanie pracownika jest cechą stosunku pracy. "
        "Podporządkowanie pracownika w zakładzie pracy.",
        mode="standard",
        include_candidates=True,
    )
    assert result.text.endswith("Podporządkowanie pracownika w zakładzie pracy.")
    assert not any(trace.operation_type == "redundancy_reduction" for trace in result.all_candidates)


def test_redundancy_reduction_does_not_drop_protected_opening():
    result = humanize_text(
        "Zgodnie z art. 22 § 1 Kodeksu pracy obowiązek istnieje. "
        "Art. 22 § 1 Kodeksu pracy wynika z ustawy.",
        mode="standard",
        include_candidates=True,
    )
    assert "Art. 22 § 1 Kodeksu pracy wynika z ustawy." in result.text
    assert not any(trace.operation_type == "redundancy_reduction" for trace in result.all_candidates)


def test_strong_generates_more_redundancy_than_standard():
    text = (
        "Podporządkowanie jest istotne. "
        "Podporządkowanie oznacza wykonywanie pracy pod kierownictwem."
    )
    standard = humanize_text(text, mode="standard", include_candidates=True)
    strong = humanize_text(text, mode="strong", include_candidates=True)
    assert standard.text == text
    assert "Oznacza wykonywanie pracy pod kierownictwem." in strong.text
    assert len(strong.all_candidates) > len(standard.all_candidates)


def test_report_contains_step_index_and_redundancy_operation(tmp_path):
    result = humanize_text(
        "Podporządkowanie pracownika jest cechą stosunku pracy. "
        "Podporządkowanie pracownika wynika z art. 22 Kodeksu pracy.",
        mode="standard",
        include_candidates=True,
    )
    report_path = tmp_path / "report.json"
    write_json_report(result, report_path)
    payload = report_path.read_text(encoding="utf-8")
    assert '"step_index": 0' in payload
    assert '"operation_type": "redundancy_reduction"' in payload
    assert '"gate_results"' in payload
