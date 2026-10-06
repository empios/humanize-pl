import json
from pathlib import Path

from humanize_pl.evaluation import DEFAULT_BASELINE, DIRECTION, regressions


def _run(**metrics):
    return {"morfeusz": True, "metrics": metrics}


def test_regressions_follow_each_metric_direction():
    baseline = _run(category_exact=0.7, rulings_given_category=0.5, edits_per_1k_ai=30.0)
    assert regressions(
        _run(category_exact=0.69, rulings_given_category=0.6, edits_per_1k_ai=10.0), baseline
    ) == [
        "kategoria trafiona (wzory): 0.7 → 0.69",
        "orzeczenia uznane za dokument z katalogu: 0.5 → 0.6",
    ]
    # Improvements pass, and an informational metric is never gated
    assert not regressions(
        _run(category_exact=0.8, rulings_given_category=0.1, edits_per_1k_ai=10.0), baseline
    )


def test_a_metric_missing_on_either_side_is_not_compared():
    # Without the local ruling corpus its metrics are absent, not zero
    assert not regressions(_run(category_exact=0.7), _run(category_exact=0.7, unsafe_outputs_rulings=0))
    assert not regressions(_run(category_exact=0.7, unsafe_outputs=3), _run(category_exact=0.7))


def test_a_baseline_from_another_environment_is_refused():
    current = {"morfeusz": False, "metrics": {"category_exact": 0.9}}
    assert regressions(current, _run(category_exact=0.7)) == [
        "bazę zmierzono z Morfeuszem2; ten pomiar nie jest z nią porównywalny"
    ]


def test_committed_baseline_records_every_gated_metric():
    path = Path(__file__).resolve().parents[1] / DEFAULT_BASELINE
    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert set(DIRECTION) <= set(recorded["metrics"])


def test_detector_separation_is_a_rank_probability():
    from humanize_pl.evaluation import _auc, detector_metrics

    assert _auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert _auc([0.1], [0.1]) == 0.5
    assert _auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    # Without the local rulings only the template comparison is measured
    draft = "Umowa zawarta w dniu 1 marca 2026 r. pomiędzy stronami. " * 40
    assert set(detector_metrics([("umowa_najmu", draft)], None, None)) == {"detector_auc_drafts"}
