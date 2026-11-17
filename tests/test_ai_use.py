"""Tests for the record of what the language model changed."""

from __future__ import annotations

from humanize_pl.flows.base import ItemOutcome, summarise

LLM = {"backend": "openai_compatible_chat_completions", "model": "bielik", "rejected": 2}


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
        "model_unavailable": 1,
    }


def test_ai_use_reaches_the_json_row() -> None:
    outcome = ItemOutcome(name="x", llm=LLM, applied_changes=[change("llm_rewrite")])
    assert outcome.to_json()["ai_use"]["model_segments"] == 1
    assert ItemOutcome.from_json(outcome.to_json()).name == "x"
