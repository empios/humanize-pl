"""Tests for the AI corpus generator's prompt grid.

Only the planning half is covered here: it is the part that decides what the
corpus measures, and it runs without touching an endpoint. Whether the model
answers is the endpoint's business; whether we asked it varied questions is
ours.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "build_ai_corpus", Path("tools/build_ai_corpus.py")
)
build_ai_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_ai_corpus)

PROMPTS = Path("tools/corpus_prompts.yaml")


def test_the_shipped_grid_has_all_three_axes() -> None:
    """Scenario x register x temperature, as the file's own header promises.

    The temperature axis was described in the header and absent from the
    data, so a corpus built from this file could only ever use one sampling
    setting - and would measure that setting alongside the model.
    """
    styles, temperatures, categories = build_ai_corpus.load_grid(PROMPTS)

    assert len(styles) >= 3
    assert len(temperatures) >= 2
    assert len(categories) >= 8
    for style in styles:
        assert "{zadanie}" in style["user"], style["id"]


def test_every_category_with_a_blueprint_can_be_generated() -> None:
    """A category we check the structure of is one we must be able to produce."""
    from humanize_pl.blueprint import blueprints

    _styles, _temperatures, categories = build_ai_corpus.load_grid(PROMPTS)
    missing = sorted(set(blueprints()) - set(categories))
    assert missing == [], f"brak scenariuszy dla: {missing}"


def test_plan_varies_register_and_temperature_within_a_category() -> None:
    """Four documents of one category must not be four takes on one prompt."""
    styles, temperatures, categories = build_ai_corpus.load_grid(PROMPTS)
    jobs = build_ai_corpus.plan(categories, styles, temperatures, per_category=3)

    by_category: dict[str, list[dict]] = {}
    for job in jobs:
        by_category.setdefault(job["category"], []).append(job)

    for category, rows in by_category.items():
        assert len({row["scenario"] for row in rows}) == 3, category
        assert len({row["style"]["id"] for row in rows}) == 3, category
        assert len({row["temperature"] for row in rows}) == 3, category


def test_plan_is_deterministic() -> None:
    """Two runs with the same arguments ask for the same things."""
    styles, temperatures, categories = build_ai_corpus.load_grid(PROMPTS)
    first = build_ai_corpus.plan(categories, styles, temperatures, per_category=4)
    second = build_ai_corpus.plan(categories, styles, temperatures, per_category=4)

    assert [job["scenario"] for job in first] == [job["scenario"] for job in second]
    assert [job["temperature"] for job in first] == [job["temperature"] for job in second]


def test_default_run_clears_the_ranking_threshold() -> None:
    """`derive_patterns.py` refuses to rank fewer than 25 AI documents."""
    styles, temperatures, categories = build_ai_corpus.load_grid(PROMPTS)
    jobs = build_ai_corpus.plan(categories, styles, temperatures, per_category=4)

    assert len(jobs) >= 25


def test_messages_fill_the_scenario_into_the_register() -> None:
    styles, temperatures, categories = build_ai_corpus.load_grid(PROMPTS)
    job = build_ai_corpus.plan(categories, styles, temperatures, per_category=1)[0]
    messages = build_ai_corpus.messages_for(job)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "{zadanie}" not in messages[1]["content"]
    assert job["scenario"] in messages[1]["content"]


def test_grid_without_categories_is_refused(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(yaml.safe_dump({"styles": [{"id": "a", "system": "s", "user": "{zadanie}"}]}),
                    encoding="utf-8")

    with pytest.raises(ValueError, match="categories"):
        build_ai_corpus.load_grid(path)


def test_numbering_continues_past_the_existing_fixtures(tmp_path) -> None:
    """The hand-written fixtures stay; six test modules read them by name."""
    (tmp_path / "ai_legal_01_umowa.txt").write_text("x", encoding="utf-8")
    (tmp_path / "ai_legal_15_pozew.txt").write_text("x", encoding="utf-8")

    assert build_ai_corpus.next_index(tmp_path) == 16


def test_a_vanished_server_is_told_from_a_bad_answer() -> None:
    """The runner waits for the first and gives up on the second."""
    assert build_ai_corpus._unreachable("Błąd połączenia z modelem: ConnectTimeout.")
    assert build_ai_corpus._unreachable("Endpoint modelu zwrócił HTTP 503.")
    assert not build_ai_corpus._unreachable("Model zwrócił pustą odpowiedź.")


def test_the_runner_waits_for_the_server_to_come_back(monkeypatch) -> None:
    """The Bielik endpoint dropped twice in one afternoon and the runner raced
    through the remaining jobs, failing 23 of 28. It waits now, asking a fresh
    client each time: `probe()` remembers its first answer."""
    answers = iter([False, False, True])

    class Fresh:
        def __init__(self, settings):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def probe(self):
            return next(answers)

    monkeypatch.setattr(build_ai_corpus, "OpenAICompatibleRewriter", Fresh)
    monkeypatch.setattr(build_ai_corpus, "ENDPOINT_POLL_SECONDS", 0.0)

    assert build_ai_corpus._wait_for_endpoint(settings=None, limit=5.0) is True
    monkeypatch.setattr(build_ai_corpus, "OpenAICompatibleRewriter", type(
        "Never", (Fresh,), {"probe": lambda self: False}
    ))
    assert build_ai_corpus._wait_for_endpoint(settings=None, limit=0.01) is False
