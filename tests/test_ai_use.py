"""Tests for the record of what the language model changed."""

from __future__ import annotations

from humanize_pl.flows.base import ItemOutcome, summarise

LLM = {
    "backend": "openai_compatible_chat_completions",
    "model": "bielik",
    "rejected": 2,
    "decision_reasons": {"rejected:nli": 2},
}


def change(*rules: str) -> dict:
    return {"before": "a", "after": "b", "rules": list(rules)}


def test_item_records_model_and_rule_segments() -> None:
    outcome = ItemOutcome(
        name="x", llm=LLM, applied_changes=[change("kancelaryzmy"), change("llm_rewrite")]
    )
    assert outcome.ai_use == {
        "model": "bielik",
        "model_segments": 1,
        "rule_segments": 1,
        "model_rejections": 2,
        "model_errors": 0,
    }


def test_a_rule_edit_finished_by_the_model_counts_as_model() -> None:
    outcome = ItemOutcome(name="x", llm=LLM, applied_changes=[change("kancelaryzmy", "llm_rewrite")])
    assert (outcome.ai_use["model_segments"], outcome.ai_use["rule_segments"]) == (1, 0)


def test_rules_only_run_records_no_model() -> None:
    outcome = ItemOutcome(name="x", applied_changes=[change("kancelaryzmy")])
    assert outcome.ai_use == {
        "model": None,
        "model_segments": 0,
        "rule_segments": 1,
        "model_rejections": 0,
        "model_errors": 0,
    }


def test_summary_aggregates_model_use() -> None:
    edited = ItemOutcome(name="a", llm=LLM, applied_changes=[change("llm_rewrite"), change("x")])
    rules = ItemOutcome(name="b", applied_changes=[change("x")])
    fallback = ItemOutcome(name="c", llm={"backend": "rules_fallback", "status": "unavailable"})
    assert summarise([edited, rules, fallback])["ai_use"] == {
        "models": ["bielik"],
        "documents_with_model_edits": 1,
        "model_segments": 1,
        "rule_segments": 2,
        "model_rejections": 2,
        "model_errors": 0,
        "model_unavailable": 1,
    }


def test_ai_use_reaches_the_json_row() -> None:
    outcome = ItemOutcome(name="x", llm=LLM, applied_changes=[change("llm_rewrite")])
    assert outcome.to_json()["ai_use"]["model_segments"] == 1
    assert ItemOutcome.from_json(outcome.to_json()).name == "x"


def test_failed_probe_is_counted_as_unavailable_not_used() -> None:
    probe_failed = ItemOutcome(
        name="x", llm={"backend": LLM["backend"], "model": "bielik", "status": "unavailable"}
    )
    assert probe_failed.ai_use["model"] is None
    assert summarise([probe_failed])["ai_use"]["model_unavailable"] == 1


def test_rejections_count_local_checks_and_errors_apart() -> None:
    llm = {
        **LLM,
        "status": "ready_with_errors",
        "rejected": 6,
        "decision_reasons": {
            "rejected:nli": 1,
            "rejected:numbers": 1,
            "rejected:endpoint_or_schema_error": 3,
            "rejected:no_visible_change": 1,
            "accepted": 4,
        },
    }
    use = ItemOutcome(name="x", llm=llm).ai_use
    assert (use["model_rejections"], use["model_errors"]) == (2, 3)


def test_rule_edits_inside_a_paragraph_the_model_rewrote_count_as_model() -> None:
    rule = {**change("drop_discourse_intro"), "paragraph_index": 1, "sentence_index": 0}
    paragraph = {**change("llm_rewrite"), "paragraph_index": 1, "sentence_index": None}
    other = {**change("drop_discourse_intro"), "paragraph_index": 2, "sentence_index": 0}
    use = ItemOutcome(name="x", llm=LLM, applied_changes=[rule, paragraph, other]).ai_use
    assert (use["model_segments"], use["rule_segments"]) == (2, 1)


def test_a_chain_finished_by_the_model_is_labelled_as_the_model_s() -> None:
    from humanize_pl.flows.base import collapse_visible_changes

    first = {"before": "A b.", "after": "B b.", "issue": "debureaucratization",
             "rules": ["r1"], "paragraph_index": 0, "sentence_index": 0}
    second = {"before": "B b.", "after": "C b.", "issue": "llm_rewrite",
              "rules": ["llm_rewrite"], "paragraph_index": 0, "sentence_index": 0}
    [collapsed] = collapse_visible_changes([first, second])
    assert collapsed["issue"] == "llm_rewrite"


def test_a_run_without_finished_items_has_no_register() -> None:
    assert summarise([])["ai_use"] is None
    assert summarise([ItemOutcome(name="x", status="failed")])["ai_use"] is None


def test_a_model_that_answered_without_accepted_edits_is_still_named() -> None:
    use = ItemOutcome(name="x", llm={**LLM, "status": "ready"}).ai_use
    assert use["model"] == "bielik"
    assert use["model_segments"] == 0
