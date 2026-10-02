from humanize_pl.core import humanize_text
from humanize_pl.safety.protectors import protect_text
from humanize_pl.safety.validators import validate_candidate


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
