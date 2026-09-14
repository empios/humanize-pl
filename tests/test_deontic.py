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
