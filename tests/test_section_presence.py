"""The presence gate from 99088db: conservative protocol, not model accuracy."""

import pytest

from humanize_pl.blueprint import Section
from humanize_pl.drafting import PRESENCE_MAX_CHARS, section_presence
from humanize_pl.llm import LlmEndpointError

SECTION = Section("rozwiazanie", "rozwiązanie umowy", ("rozwiązanie umowy",))
PARAPHRASE = "Każda ze stron może zakończyć współpracę po pisemnym zawiadomieniu drugiej strony."


class Answer:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def complete_json(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@pytest.mark.parametrize("answer", [
    {}, [], None, "absent", {"obecna": 0}, {"obecna": "false"},
    {"obecna": False, "cytat": PARAPHRASE},
    {"obecna": True, "cytat": "Zmyślony fragment, którego nie ma w dokumencie."},
    LlmEndpointError("test error"),
])
def test_inconclusive_presence_cannot_authorize_drafting(answer):
    result = section_presence(PARAPHRASE, SECTION, client=Answer(answer))
    assert result.state == "unclear"


def test_gate_accepts_a_literal_quote_for_a_paraphrased_section():
    assert SECTION.found_in(PARAPHRASE.casefold()) is None
    result = section_presence(
        PARAPHRASE, SECTION, client=Answer({"obecna": True, "cytat": PARAPHRASE}),
    )
    assert result.state == "present" and result.quote == PARAPHRASE


def test_long_document_is_not_truncated_to_support_a_missing_section():
    model = Answer({"obecna": False, "cytat": ""})
    document = ("Opis świadczenia usług. " * (PRESENCE_MAX_CHARS // 10)) + PARAPHRASE
    result = section_presence(document, SECTION, client=model)
    assert result.state == "unclear"
    assert model.calls == 0


def test_explicit_absence_is_distinct_from_unknown():
    model = Answer({"obecna": False, "cytat": ""})
    result = section_presence("Przedmiotem umowy jest malowanie biura.", SECTION, client=model)
    assert result.state == "absent"
    assert model.calls == 1
