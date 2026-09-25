"""Adversarial legal edits must fail even with an overconfident NLI model."""

import pytest

from humanize_pl.safety.protectors import ProtectedText, protect_text
from humanize_pl.safety.validators import validate_candidate


class AlwaysEntails:
    def check_entailment(self, left, right):
        return True


CASES = [
    # Roles and actions: inflection or a changed verb defeats bag-of-words checks.
    ("Dostawca przekazuje klientowi dokumentację.", "Klient przekazuje dostawcy dokumentację."),
    ("Pracownik przekazuje pracodawcy wniosek.", "Pracodawca przekazuje pracownikowi wniosek."),
    ("Najemca wpłaca kaucję.", "Najemca odzyskuje kaucję."),
    ("Wierzyciel zwalnia dłużnika z długu.", "Dłużnik zwalnia wierzyciela z długu."),
    # Same number of modals, different scope, action or recipients.
    ("Dostawca może odmówić odbioru, a klient musi zapłacić.",
     "Dostawca musi odmówić odbioru, a klient może zapłacić."),
    ("Może odmówić zapłaty i odbioru towaru.", "Może odmówić zapłaty lub odbioru towaru."),
    ("Musi zwrócić towar.", "Musi zatrzymać towar."),
    ("Prawo przysługuje po akceptacji wniosku.", "Prawo przysługuje przed akceptacją wniosku."),
    # Conditions and exceptions survive unchanged in number but change meaning.
    ("Jeżeli dostawa opóźni się, nalicza się karę.", "Jeżeli płatność opóźni się, nalicza się karę."),
    ("W razie opóźnienia dostawy nalicza się karę.", "W razie opóźnienia zapłaty nalicza się karę."),
    ("Opłata obowiązuje, chyba że przesyłka zaginie.", "Opłata obowiązuje, chyba że przesyłka dotrze."),
    ("Odpowiedzialność obejmuje straty z wyjątkiem kosztów transportu.",
     "Odpowiedzialność obejmuje straty z wyjątkiem kosztów magazynowania."),
    ("Z zastrzeżeniem zgody sądu następuje zwrot kwoty.",
     "Z zastrzeżeniem zgody zarządu następuje zwrot kwoty."),
    ("Pokrywa tylko koszt dostawy oraz koszt montażu.",
     "Pokrywa koszt dostawy oraz tylko koszt montażu."),
    # Numeric preservation alone misses units, counting rules and triggers.
    ("Zapłata następuje w ciągu 14 dni.", "Zapłata następuje w ciągu 14 miesięcy."),
    ("Zwrot następuje w ciągu 7 dni roboczych.", "Zwrot następuje w ciągu 7 dni kalendarzowych."),
    ("Termin wynosi 14 dni od doręczenia wezwania.", "Termin wynosi 14 dni od wysłania wezwania."),
    ("Zwrot następuje po dniu 10.05.2026.", "Zwrot następuje przed dniem 10.05.2026."),
    # Identical article numbers, different provision / act / reference.
    ("Zastosowanie ma art. 22 k.c.", "Zastosowanie ma art. 22 k.p."),
    ("Zastosowanie ma art. 22 Kodeksu pracy.", "Zastosowanie ma art. 22 Kodeksu cywilnego."),
    ("Zastosowanie ma § 2.", "Zastosowanie ma pkt 2."),
    ("Zastosowanie ma ust. 2 lit. a.", "Zastosowanie ma ust. 2 lit. b."),
    ("Zastosowanie ma przepis powyżej.", "Zastosowanie ma przepis poniżej."),
    # Punctuation can change what is a condition of what.
    ("Jeżeli dostawa nastąpi, nalicza się opłatę.", "Jeżeli dostawa nastąpi. Nalicza się opłatę."),
]


@pytest.mark.parametrize("source,candidate", CASES)
@pytest.mark.parametrize("model", [None, AlwaysEntails()])
def test_legal_meaning_change_is_not_automatically_accepted(source, candidate, model):
    result = validate_candidate(
        source, candidate, protected=ProtectedText(source, {}),
        max_length_ratio=1.6, nli=model,
    )
    assert not result.ok, (source, candidate)


@pytest.mark.parametrize("text", [
    "Dostawca przekazuje klientowi dokumentację.",
    "Strona musi zapłacić w terminie 14 dni od doręczenia faktury.",
    "Jeżeli dostawa nastąpi, nalicza się opłatę.",
    "Zwrot następuje zgodnie z § 2 ust. 1 lit. a.",
])
def test_intro_removal_is_still_available_in_legal_editing(text):
    source = "Warto podkreślić, że " + text
    result = validate_candidate(
        source, text, protected=ProtectedText(source, {}), max_length_ratio=1.6,
    )
    assert result.ok, result.reason


def test_masked_reference_keeps_its_act_not_only_its_number():
    source = "Zastosowanie ma art. 22 k.c."
    protected = protect_text(source)
    result = validate_candidate(
        protected.text, protected.text.replace("k.c.", "k.p."),
        protected=protected, max_length_ratio=1.6, nli=AlwaysEntails(),
    )
    assert not result.ok
    assert result.checks[-1].name == "legal_reference_scope_preserved"


def test_legal_limits_are_not_applied_to_general_paraphrases():
    source, candidate = "Wysyłka nastąpi w ciągu 7 dni.", "Przesyłkę nadamy w ciągu 7 dni."
    common = {"protected": ProtectedText(source, {}), "max_length_ratio": 1.6, "nli": AlwaysEntails()}
    assert not validate_candidate(source, candidate, **common).ok
    assert validate_candidate(source, candidate, legal=False, **common).ok


def test_legal_default_preserves_sensitive_assertion_but_removes_frame():
    from humanize_pl import humanize

    source = "Warto podkreślić, że najemca musi dokonać zwrotu lokalu w terminie 7 dni."
    result = humanize(source, track="legal", pdf=False, draft_missing=False)
    assert result.text == "Najemca musi dokonać zwrotu lokalu w terminie 7 dni."


def test_agreement_repair_cannot_change_roles_after_validation(monkeypatch):
    from humanize_pl.config import HumanizeConfig, Mode
    from humanize_pl.pipeline import LegalPipeline
    from humanize_pl.rules.engine import RuleEngine
    from humanize_pl.safety.validators import GateCheck

    source = "Warto podkreślić, że pracownik składa pracodawcy wniosek."
    monkeypatch.setattr(
        "humanize_pl.pipeline.agreement_gate",
        lambda *args, **kwargs: (
            [GateCheck("agreement", True)], "Pracodawca składa pracownikowi wniosek.",
        ),
    )
    pipeline = LegalPipeline(
        config=HumanizeConfig(mode=Mode.standard), protected=protect_text(source),
        rule_engine=RuleEngine(mode=Mode.standard), morfeusz=object(), include_candidates=True,
    )
    result = pipeline.process_paragraph(source, paragraph_index=0)
    assert result.text == source
    assert any(
        gate["name"] == "legal_party_roles_preserved" and not gate["ok"]
        for trace in result.traces for gate in trace.gate_results
    )
