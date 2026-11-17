from humanize_pl.core import humanize_text
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate
from humanize_pl.sentence_splitter import split_sentences


def test_boundary_values_are_protected():
    cases = {
        "Zgodnie z § 5 umowy.": "§ 5",
        "Kara wynosi 10% kwoty.": "10%",
        "Kara wynosi 10 000 zł.": "10 000 zł",
        "Kara wynosi 1.000,00 zł.": "1.000,00 zł",
        "Kara wynosi 100 złotych.": "100 złotych",
        "(Dz. U. z 2023 r. poz. 1610)": "Dz. U. z 2023 r",
    }
    for text, fragment in cases.items():
        assert fragment in protect_text(text).mapping.values(), text


def test_year_placeholder_keeps_sentence_end_visible():
    protected = protect_text("Umowę zawarto w 2024 r. Strony ją podpisały.")
    assert list(protected.mapping.values()) == ["2024 r"]
    assert len(split_sentences(protected.text)) == 2


def test_quote_with_reference_is_one_placeholder():
    text = "Pojęcie „art. 5 ustawy” oznacza przepis szczególny."
    protected = protect_text(text)
    assert protected.text.count("__PROTECTED_") == 1
    assert not any("__PROTECTED_" in value for value in protected.mapping.values())
    assert protected.restore(protected.text) == text


def test_sensitive_name_inside_quote_restores():
    text = "Strona „Jan Kowalski” zapłaci karę."
    protected = protect_text(text, include_sensitive=True)
    assert "Kowalski" not in protected.text
    assert protected.restore(protected.text) == text


def test_re_protect_keeps_repeated_fragments_apart():
    protected = protect_text("Zgodnie z art. 5 umowy oraz art. 5 regulaminu strony współpracują.")
    assert protected.re_protect(protected.restore(protected.text)) == protected.text


def test_candidate_next_to_quoted_reference_is_not_a_leak():
    protected = protect_text("Pojęcie „art. 5 ustawy” oznacza w umowie przepis szczególny.")
    validation = validate_candidate(
        protected.text,
        protected.text.replace("w umowie", "w tej umowie"),
        protected=protected,
        max_length_ratio=2.0,
    )
    assert validation.ok, validation.reason


def test_skipped_sentences_report_restored_text():
    result = humanize_text("Pojęcie „art. 5 ustawy” oznacza przepis szczególny.")
    assert result.skipped
    assert all("__PROTECTED_" not in skip.original for skip in result.skipped)
