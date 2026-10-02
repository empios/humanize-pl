from humanize_pl.safety.deontic import extract_deontic_profile, DeonticModality
from humanize_pl.safety.validators import validate_candidate
from humanize_pl.safety.protectors import ProtectedText

def test_extract_deontic_profile():
    # Obligations
    profile = extract_deontic_profile("Wykonawca zobowiązuje się do zapłaty. Musi to zrobić.")
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 2
    assert sum(profile[DeonticModality.PERMISSION].values()) == 0

    # Permissions
    profile = extract_deontic_profile("Strona może odstąpić od umowy. Prawo do tego przysługuje stronie.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 2

    # Prohibitions
    profile = extract_deontic_profile("Zabrania się kopiowania. Nie wolno tego robić.")
    assert sum(profile[DeonticModality.PROHIBITION].values()) == 2

    # Recommendations
    profile = extract_deontic_profile("Zamawiający powinien rozważyć. Warto to zrobić.")
    assert sum(profile[DeonticModality.RECOMMENDATION].values()) == 2

    # Negated permissions -> Prohibition
    profile = extract_deontic_profile("Użytkownik nie może kopiować kodu.")
    assert sum(profile[DeonticModality.PROHIBITION].values()) == 1
    assert sum(profile[DeonticModality.PERMISSION].values()) == 0

    # Negated obligations -> Permission
    profile = extract_deontic_profile("Strona nie musi płacić kary.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 0


def test_negation_scope_in_deontic_profile():
    # "nie należy" forbids; it does not leave a choice like "nie musi"
    profile = extract_deontic_profile("Najemcy nie należy wnosić zwierząt.")
    assert sum(profile[DeonticModality.PROHIBITION].values()) == 1
    assert sum(profile[DeonticModality.PERMISSION].values()) == 0

    # Negated prohibition -> Permission
    profile = extract_deontic_profile("Najemcy nie zabrania się trzymania zwierząt.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.PROHIBITION].values()) == 0

    profile = extract_deontic_profile("Najemcy wolno podnajmować lokal.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1

    # A conjunction ends the reach of "nie"
    profile = extract_deontic_profile("Najemca nie pali i może trzymać psa.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.PROHIBITION].values()) == 0

    # "nie tylko" does not negate the modal
    profile = extract_deontic_profile("Najemca nie tylko może, ale i musi zapłacić.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 1


def test_validator_rejects_deontic_drift():
    protected = ProtectedText(text="", mapping={})
    
    # Obligation -> Recommendation
    res = validate_candidate(
        "Strona zobowiązuje się do zapłaty.",
        "Strona powinna zapłacić.",
        protected=protected, max_length_ratio=1.5
    )
    assert not res.ok
    assert any("deontic drift" in c.reason for c in res.checks if not c.ok)

    # Permission -> Obligation
    res = validate_candidate(
        "Użytkownik może złożyć wniosek.",
        "Użytkownik ma obowiązek złożyć wniosek.",
        protected=protected, max_length_ratio=1.5
    )
    assert not res.ok
    
    # Valid rewrite (preserves modality)
    res = validate_candidate(
        "Strona zobowiązuje się do zapłaty.",
        "Strona musi zapłacić.", # zobowiązuje się -> musi (oba to OBLIGATION)
        protected=protected, max_length_ratio=1.5
    )
    assert res.ok

    # Negated permission -> Prohibition drift
    res = validate_candidate(
        "Użytkownik nie może złożyć wniosku.",
        "Użytkownik może złożyć wniosek.",
        protected=protected, max_length_ratio=1.5
    )
    assert not res.ok


def test_validator_rejects_negation_flips():
    protected = ProtectedText(text="", mapping={})
    flips = [
        # Modal category unchanged before the fix, polarity reversed
        ("Najemcy nie zabrania się trzymania w lokalu zwierząt domowych.",
         "Najemcy zabrania się trzymania w lokalu zwierząt domowych."),
        ("Najemca nie pali i może trzymać w lokalu zwierzęta domowe.",
         "Najemca nie pali i nie może trzymać w lokalu zwierząt domowych."),
        ("Najemcy nie należy wnosić do lokalu zwierząt domowych.",
         "Najemca może nie wnosić do lokalu zwierząt domowych."),
        # No modal at all
        ("Najemca odpowiada za szkody wyrządzone w lokalu przez osoby trzecie.",
         "Najemca nie odpowiada za szkody wyrządzone w lokalu przez osoby trzecie."),
        ("Wynajmujący może rozwiązać umowę bez wypowiedzenia w razie zaległości z zapłatą czynszu.",
         "Wynajmujący może rozwiązać umowę z zachowaniem wypowiedzenia w razie zaległości "
         "z zapłatą czynszu."),
        ("Najemca odpowiada nie tylko za szkody wyrządzone przez siebie.",
         "Najemca odpowiada tylko za szkody wyrządzone przez siebie."),
    ]
    for original, candidate in flips:
        res = validate_candidate(original, candidate, protected=protected, max_length_ratio=1.5)
        failed = next((check.name for check in res.checks if not check.ok), "accepted")
        assert failed in {"deontic_modality_preserved", "negation_preserved"}, candidate


def test_validator_counts_brak_as_negation():
    # The rule "przypadek wystąpić brak możliwość" -> "gdy nie ma możliwości"
    # swaps "brak" for "nie ma" without reversing polarity.
    protected = ProtectedText(text="", mapping={})
    res = validate_candidate(
        "W przypadku braku możliwości wykonania umowy Strona niezwłocznie zawiadamia drugą Stronę.",
        "Gdy nie ma możliwości wykonania umowy, Strona niezwłocznie zawiadamia drugą Stronę.",
        protected=protected, max_length_ratio=1.5,
    )
    assert res.ok
    assert any(check.name == "negation_preserved" for check in res.checks)



def test_inflected_modals_are_detected():
    profile = extract_deontic_profile("Najemca mógł wypowiedzieć umowę, a Wynajmujący musiał ją przyjąć.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 1
    profile = extract_deontic_profile("Przyjmujący zamówienie jest obowiązany do noszenia odzieży.")
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 1
    profile = extract_deontic_profile("Najemca nie ma obowiązku zapłaty kary.")
    assert sum(profile[DeonticModality.PERMISSION].values()) == 1
    assert sum(profile[DeonticModality.OBLIGATION].values()) == 0
    # Not modals: in force, belongs to, perhaps, debts
    profile = extract_deontic_profile(
        "Umowa obowiązuje od dziś. Lokal należy do Wynajmującego. Być może zobowiązania wygasły."
    )
    assert not any(profile.values())


def test_validator_rejects_a_past_tense_modal_flip():
    protected = ProtectedText(text="", mapping={})
    res = validate_candidate(
        "Najemca mógł wypowiedzieć umowę w każdym czasie.",
        "Najemca musiał wypowiedzieć umowę w każdym czasie.",
        protected=protected, max_length_ratio=1.5,
    )
    assert not res.ok
    assert res.reason.startswith("deontic drift")
