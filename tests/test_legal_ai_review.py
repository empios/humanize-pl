import json
from pathlib import Path

from typer.testing import CliRunner

from humanize_pl.cli import app
from humanize_pl.config import HumanizeConfig, LegalReviewProfile, Mode
from humanize_pl.core import humanize_text
from humanize_pl.reports.report import write_json_report
from humanize_pl.rules.legal_features import (
    analyze_legal_review_features,
    legal_anchor_retention,
    lost_legal_anchors,
)
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import legal_sensitive_inventory, validate_candidate


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ai_legal_samples.json"


def test_default_profile_is_legal_ai_review():
    assert HumanizeConfig().legal_review_profile == LegalReviewProfile.legal_ai_review
    result = humanize_text("Pracownik wykonuje pracę.")
    assert result.legal_review_profile == "legal_ai_review"


def test_cli_accepts_legal_review_profile():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "Podsumowując źródła prawa pracy tworzą system.",
            "--legal-review-profile",
            "legal_ai_review",
        ],
    )
    assert result.exit_code == 0
    assert "Podsumowując," in result.stdout


def test_validator_rejects_normativity_changes():
    original = "Strona może złożyć oświadczenie w terminie 7 dni."
    validation = validate_candidate(
        original,
        "Strona musi złożyć oświadczenie w terminie 7 dni.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason.startswith("deontic drift")

    original = "Pracownik powinien wykonać obowiązek zgodnie z umową."
    validation = validate_candidate(
        original,
        "Pracownik jest zobowiązany wykonać obowiązek zgodnie z umową.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason.startswith("deontic drift")


def test_validator_rejects_loss_of_legal_party_or_obligation():
    original = "Pracownik wykonuje obowiązek osobiście na rzecz pracodawcy."
    validation = validate_candidate(
        original,
        "Wykonuje obowiązek osobiście na rzecz pracodawcy.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason.startswith("legal anchors removed")


def test_legal_anchors_survive_a_change_of_case():
    # "umowa" -> "umowę", "sąd" -> "sądu": same anchor in another case
    assert legal_anchor_retention(
        "Została zawarta umowa, a sąd ją zatwierdził.",
        "Zawarto umowę, którą zatwierdziło orzeczenie sądu.",
    ) == 1.0
    assert lost_legal_anchors("Strona zapłaci karę umowną.", "Zapłaci karę umowną.")


def test_money_and_dates_count_as_legal_risk():
    features = analyze_legal_review_features("Kara wynosi 10% ceny i jest płatna do 2024 r.")
    assert features.money_or_date_count == 2


def test_validator_rejects_parties_swapping_roles():
    for original, candidate in [
        (
            "Najemca zapłaci Wynajmującemu czynsz w terminie siedmiu dni.",
            "Wynajmujący zapłaci Najemcy czynsz w terminie siedmiu dni.",
        ),
        (
            "Zamawiający zapłaci Wykonawcy wynagrodzenie w terminie siedmiu dni.",
            "Wykonawca zapłaci Zamawiającemu wynagrodzenie w terminie siedmiu dni.",
        ),
    ]:
        validation = validate_candidate(
            original, candidate, protected=protect_text(original), max_length_ratio=2.0
        )
        assert validation.reason == "parties swapped roles", candidate


def test_parties_keeping_their_roles_pass_the_swap_check():
    from humanize_pl.safety.parties import parties_swapped

    for original, candidate in [
        # Same subject, restated
        ("Najemca jest zobowiązany do zapłaty na rzecz Wynajmującego.", "Najemca płaci Wynajmującemu."),
        # Passive to active: a subject appears, none is lost
        ("Czynsz jest płacony przez Najemcę Wynajmującemu.", "Najemca płaci Wynajmującemu czynsz."),
        # Clauses reordered, each party keeps its verb
        (
            "Najemca płaci czynsz, a Wynajmujący wydaje lokal.",
            "Wynajmujący wydaje lokal, a Najemca płaci czynsz.",
        ),
        # The contract is not a party
        ("Umowa określa obowiązki Najemcy.", "Najemca ma obowiązki określone w Umowie."),
    ]:
        assert not parties_swapped(original, candidate), candidate


def test_validator_rejects_one_party_named_in_place_of_another():
    for original, candidate in [
        # The side of the lawsuit liable for costs, swapped
        (
            "Pozwana wniosła o zasądzenie od strony pozwanej kosztów procesu.",
            "Pozwana wniosła o zasądzenie od strony powodowej kosztów procesu.",
        ),
        # A legal role replaced by another one: the payer is not the employer
        (
            "Płatnik zgłosił ubezpieczoną do ubezpieczeń społecznych w terminie.",
            "Pracodawca zgłosił ubezpieczoną do ubezpieczeń społecznych w terminie.",
        ),
    ]:
        validation = validate_candidate(
            original, candidate, protected=protect_text(original), max_length_ratio=2.0
        )
        assert validation.reason == "one party named in place of another", candidate


def test_party_mentions_may_change_without_an_exchange():
    from humanize_pl.safety.parties import party_mentions_exchanged

    for original, candidate in [
        # Passive to active: the same parties, mentioned as often
        ("Pozwane zostały poinformowane przez powoda o zaległości.", "Powód poinformował pozwane o zaległości."),
        # A party named less often, by a pronoun
        ("Pozwany wezwał powoda, a pozwany nie zapłacił.", "Pozwany wezwał powoda, lecz nie zapłacił."),
        # An opener dropped capitalises a word Morfeusz also reads as a surname
        ("Warto zauważyć, że zgoda może zostać cofnięta.", "Zgoda może zostać cofnięta."),
        # Adjectives used as nouns name no party
        ("Inny świadek potwierdził zeznania powoda.", "Kolejny świadek potwierdził zeznania powoda."),
    ]:
        assert not party_mentions_exchanged(original, candidate), candidate


def test_validator_rejects_an_act_turned_into_a_request():
    for original, candidate in [
        # A hosted model's edit in a payment demand; the meaning check passed it
        (
            "Wzywam do zapłaty całej kwoty w terminie 7 dni od dnia doręczenia pisma.",
            "Płatność całej kwoty proszę dokonać w terminie 7 dni od dnia doręczenia pisma.",
        ),
        (
            "Wnoszę o zasądzenie od pozwanego kosztów postępowania według norm przepisanych.",
            "Proszę o zasądzenie od pozwanego kosztów postępowania według norm przepisanych.",
        ),
    ]:
        validation = validate_candidate(
            original, candidate, protected=protect_text(original), max_length_ratio=2.0
        )
        assert validation.reason == "performative act changed", candidate


def test_acts_kept_reported_or_merely_referred_to_pass():
    from humanize_pl.safety.speech_acts import speech_acts_changed

    for original, candidate in [
        ("Niniejszym wypowiadam umowę o pracę.", "Wypowiadam umowę o pracę."),
        # A request phrased another way is still a request
        ("Uprzejmie proszę o przesłanie dokumentów.", "Zwracam się z prośbą o przesłanie dokumentów."),
        # A reference, not an appeal
        ("Umowa odwołuje się do art. 471 k.c.", "Umowa nawiązuje do art. 471 k.c."),
        # The past tense reports an act; it performs none
        ("Powód wezwał pozwanego do zapłaty.", "Powód wezwał pozwaną spółkę do zapłaty."),
        # The same act carried by a noun: "składam odwołanie", "wnoszę odwołanie"
        ("Odwołuję się od decyzji z dnia 5 maja.", "Składam odwołanie od decyzji z dnia 5 maja."),
        ("Odwołuję się od decyzji z dnia 5 maja.", "Wnoszę odwołanie od decyzji z dnia 5 maja."),
        ("Wnoszę o zasądzenie kwoty.", "Składam wniosek o zasądzenie kwoty."),
        ("Niniejszym składam wypowiedzenie umowy.", "Wypowiadam umowę."),
    ]:
        assert not speech_acts_changed(original, candidate), candidate
    # An act made into a report of itself is no longer performed
    assert speech_acts_changed("Składam odwołanie od decyzji.", "Odwołanie od decyzji zostało złożone.")


def test_validator_rejects_changed_code_abbreviation():
    original = "Do szkody stosuje się art. 471 k.c. i strony to akceptują bez zastrzeżeń."
    validation = validate_candidate(
        original,
        "Do szkody stosuje się art. 471 k.p.c. i strony to akceptują bez zastrzeżeń.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert not validation.ok
    assert validation.reason == "legally sensitive inventory changed: legal_references"


def test_party_names_ignore_a_capitalised_common_word_at_sentence_start():
    inventory = legal_sensitive_inventory(
        "Niniejsza Umowa wiąże obie strony.\nPonadto Strona zapłaci karę."
    )
    assert not inventory["party_names"]
    inventory = legal_sensitive_inventory(
        "Jan Kowalski zapłaci karę. Kancelaria Prawna Nowak reprezentuje go w sporze."
    )
    assert set(inventory["party_names"]) == {"jan kowalski", "prawna nowak"}

    original = "Niniejsza Umowa wchodzi w życie z dniem jej podpisania przez obie strony."
    validation = validate_candidate(
        original,
        "Ta Umowa wchodzi w życie z dniem jej podpisania przez obie strony.",
        protected=protect_text(original),
        max_length_ratio=2.0,
    )
    assert validation.ok, validation.reason


def test_inventory_tracks_percent_section_and_journal_reference():
    inventory = legal_sensitive_inventory(
        "Zgodnie z § 5 kara wynosi 10% kwoty (Dz. U. z 2023 r. poz. 1610)."
    )
    assert inventory["amounts"] == {"10%": 1}
    assert {"§ 5", "dz. u. z 2023 r"} <= set(inventory["legal_references"])


def test_only_a_repeated_additive_opener_is_dropped():
    # The first "Dodatkowo" adds a cumulative condition; it is not overuse
    single = (
        "Powodowie wykazali posiadanie nieruchomości. "
        "Dodatkowo powinni udowodnić, że korzystali z trwałego i widocznego urządzenia."
    )
    assert humanize_text(single, mode="standard").text == single

    repeated = (
        "Pozwany nie zapłacił czynszu. Ponadto pozwany nie wydał lokalu. "
        "Ponadto pozwany nie oddał kluczy."
    )
    assert humanize_text(repeated, mode="standard").text == (
        "Pozwany nie zapłacił czynszu. Ponadto pozwany nie wydał lokalu. "
        "Pozwany nie oddał kluczy."
    )


def test_ai_artifact_reduction_preserves_legal_content():
    text = (
        "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem pracodawcy. "
        "Ponadto, pracownik otrzymuje wynagrodzenie w terminie określonym w umowie."
    )
    result = humanize_text(text, mode="standard", include_candidates=True)
    assert result.text.startswith("Pracownik wykonuje pracę")
    assert "pracodawcy" in result.text
    assert "wynagrodzenie" in result.text
    assert any(
        trace.operation_type == "ai_artifact_reduction" and trace.status == "accepted"
        for trace in result.all_candidates
    )


def test_ai_legal_fixtures_smoke_and_report(tmp_path):
    samples = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(samples) == 5
    for sample in samples:
        result = humanize_text(sample["text"], mode="standard", include_candidates=True)
        assert "surprisingly" not in result.text.lower()
        assert "Ponadto za wynagrodzeniem" not in result.text
        assert "__PROTECTED_" not in result.text
        report_path = tmp_path / f"{sample['id']}.json"
        write_json_report(result, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["legal_review_profile"] == "legal_ai_review"
        assert "legal_review" in payload
        assert "ai_artifact_score" in payload["legal_review"]
        assert "legal_risk_score" in payload["legal_review"]


def test_standard_changes_more_than_conservative_without_relaxing_gates():
    text = "Warto wskazać, że pracownik wykonuje pracę pod kierownictwem pracodawcy."
    conservative = humanize_text(text, mode=Mode.conservative, include_candidates=True)
    standard = humanize_text(text, mode=Mode.standard, include_candidates=True)
    assert conservative.text == text
    assert standard.text == "Pracownik wykonuje pracę pod kierownictwem pracodawcy."
    assert all(
        gate.get("ok", True)
        for trace in standard.all_candidates
        if trace.status == "accepted"
        for gate in trace.gate_results
    )
