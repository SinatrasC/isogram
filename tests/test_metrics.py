from __future__ import annotations

from isogram.metrics import compute_classification_report, pr_auc, roc_auc


def test_auc_metrics_for_perfect_ranking() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    assert roc_auc(labels, scores) == 1.0
    assert pr_auc(labels, scores) == 1.0


def test_classification_report_contains_threshold_metrics() -> None:
    report = compute_classification_report([0, 1, 1], [0.1, 0.7, 0.8], threshold=0.5)

    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
    assert report["threshold"] == 0.5
