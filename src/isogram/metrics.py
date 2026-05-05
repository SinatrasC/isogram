from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_arrays(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("labels and scores must have the same shape")
    if y_true.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    return y_true, y_score


def roc_auc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    y_true, y_score = _as_arrays(labels, scores)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return math.nan

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.zeros(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_positive = float(ranks[y_true == 1].sum())
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    y_true, y_score = _as_arrays(labels, scores)
    positives = int(y_true.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    true_positives = np.cumsum(sorted_true == 1)
    false_positives = np.cumsum(sorted_true == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positives
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * recall_delta))


def threshold_counts(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    y_true, y_score = _as_arrays(labels, scores)
    y_pred = (y_score >= threshold).astype(np.int64)
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def threshold_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    counts = threshold_counts(labels, scores, threshold=threshold)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def choose_threshold(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    _, y_score = _as_arrays(labels, scores)
    candidates = sorted(set(float(score) for score in y_score))
    if 0.5 not in candidates:
        candidates.append(0.5)
    best_threshold = 0.5
    best_metrics = {"f1": -1.0, "precision": -1.0, "recall": -1.0}
    for threshold in candidates:
        current = threshold_metrics(labels, scores, threshold=threshold)
        current_key = (current["f1"], current["precision"], current["recall"])
        best_key = (best_metrics["f1"], best_metrics["precision"], best_metrics["recall"])
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold


def brier_score(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> float:
    y_true, y_score = _as_arrays(labels, scores)
    return float(np.mean((y_score - y_true) ** 2))


def compute_classification_report(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    if threshold is None:
        threshold = choose_threshold(labels, scores)
    report = {
        "roc_auc": roc_auc(labels, scores),
        "pr_auc": pr_auc(labels, scores),
        "brier_score": brier_score(labels, scores),
        "threshold": float(threshold),
        **threshold_metrics(labels, scores, threshold=threshold),
        **threshold_counts(labels, scores, threshold=threshold),
    }
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in report.items()
    }
