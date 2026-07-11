"""Metrics, segment analysis, drift and regression evaluations."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

from .data import CustomerRecord, shifted_copy
from .model import LogisticRiskModel


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, float | int]:
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(prediction == 1 and label == 1 for prediction, label in zip(predictions, labels))
    tn = sum(prediction == 0 and label == 0 for prediction, label in zip(predictions, labels))
    fp = sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels))
    fn = sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": len(labels),
        "positive_rate": sum(labels) / len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": mean((label - probability) ** 2 for label, probability in zip(labels, probabilities)),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def evaluate_model(model: LogisticRiskModel, records: list[CustomerRecord], threshold: float) -> dict[str, float | int]:
    return classification_metrics(
        [record.label for record in records],
        [model.predict_probability(record) for record in records],
        threshold,
    )


def evaluate_baseline(
    training_records: list[CustomerRecord], evaluation_records: list[CustomerRecord]
) -> dict[str, object]:
    """Fit a majority-class baseline on training data and apply it unchanged."""
    majority = int(sum(record.label for record in training_records) >= len(training_records) / 2)
    metrics = classification_metrics(
        [record.label for record in evaluation_records],
        [float(majority)] * len(evaluation_records),
        0.5,
    )
    return {
        "baseline_type": "training_majority_class",
        "learned_majority_class": majority,
        "training_positive_rate": sum(record.label for record in training_records) / len(training_records),
        **metrics,
    }


def segment_error_analysis(
    model: LogisticRiskModel, records: list[CustomerRecord], threshold: float
) -> dict[str, object]:
    analyses: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("region", "plan"):
        values = sorted({getattr(record, field) for record in records})
        analyses[field] = {}
        for value in values:
            subset = [record for record in records if getattr(record, field) == value]
            analyses[field][value] = evaluate_model(model, subset, threshold)
    recalls = [
        float(metrics["recall"])
        for groups in analyses.values()
        for metrics in groups.values()
        if int(metrics["true_positives"]) + int(metrics["false_negatives"]) >= 3
    ]
    errors = []
    for record in records:
        probability = model.predict_probability(record)
        predicted = int(probability >= threshold)
        if predicted != record.label:
            errors.append(
                {
                    "customer_id": record.customer_id,
                    "region": record.region,
                    "plan": record.plan,
                    "label": record.label,
                    "prediction": predicted,
                    "probability": round(probability, 6),
                }
            )
    return {
        "segments": analyses,
        "max_segment_recall_gap": max(recalls) - min(recalls),
        "error_count": len(errors),
        "errors": errors,
    }


def _psi(reference: list[float], current: list[float], bins: int = 8) -> float:
    ordered = sorted(reference)
    edges = [ordered[min(len(ordered) - 1, int(len(ordered) * index / bins))] for index in range(1, bins)]

    def bucket(value: float) -> int:
        return sum(value > edge for edge in edges)

    reference_counts = Counter(bucket(value) for value in reference)
    current_counts = Counter(bucket(value) for value in current)
    score = 0.0
    for index in range(bins):
        expected = max(reference_counts[index] / len(reference), 1e-6)
        actual = max(current_counts[index] / len(current), 1e-6)
        score += (actual - expected) * math.log(actual / expected)
    return score


def drift_report(
    reference: list[CustomerRecord], current: list[CustomerRecord], scenario: str
) -> dict[str, object]:
    fields = ("tenure_months", "tickets_90d", "usage_score", "late_payments")
    feature_psi = {
        field: _psi(
            [float(getattr(record, field)) for record in reference],
            [float(getattr(record, field)) for record in current],
        )
        for field in fields
    }
    maximum = max(feature_psi.values())
    return {
        "method": "population_stability_index",
        "alert_threshold": 0.20,
        "features": feature_psi,
        "max_psi": maximum,
        "drift_detected": maximum >= 0.20,
        "scenario": scenario,
    }


def drift_control_report(reference: list[CustomerRecord]) -> dict[str, object]:
    """Evaluate one known-negative and one known-positive monitoring scenario."""
    negative = drift_report(reference, list(reference), "identity copy; no distribution shift")
    positive = drift_report(
        reference,
        shifted_copy(reference),
        "usage -24, tickets +2, late payments +1 (bounded)",
    )
    specificity = float(not negative["drift_detected"])
    sensitivity = float(bool(positive["drift_detected"]))
    return {
        "method": "population_stability_index",
        "alert_threshold": 0.20,
        "negative_control": {"expected_drift": False, **negative},
        "shifted_control": {"expected_drift": True, **positive},
        "control_specificity": specificity,
        "control_sensitivity": sensitivity,
    }


def regression_evaluation(model: LogisticRiskModel, threshold: float) -> dict[str, object]:
    cases = [
        CustomerRecord("REG-1", "east", "premium", 68, 0, 94.0, 0, 0),
        CustomerRecord("REG-2", "west", "pro", 55, 0, 87.0, 0, 0),
        CustomerRecord("REG-3", "north", "basic", 2, 7, 12.0, 3, 1),
        CustomerRecord("REG-4", "south", "basic", 5, 6, 20.0, 2, 1),
        CustomerRecord("REG-5", "east", "premium", 45, 1, 82.0, 0, 0),
        CustomerRecord("REG-6", "south", "basic", 3, 8, 8.0, 3, 1),
    ]
    results = []
    for case in cases:
        probability = model.predict_probability(case)
        prediction = int(probability >= threshold)
        results.append(
            {
                "case_id": case.customer_id,
                "expected": case.label,
                "prediction": prediction,
                "probability": round(probability, 6),
                "passed": prediction == case.label,
            }
        )
    passed = sum(result["passed"] for result in results)
    return {"pass_rate": passed / len(results), "passed": passed, "total": len(results), "cases": results}
