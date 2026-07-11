"""Small dependency-free logistic classifier suitable for reproducible CI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .data import CustomerRecord


FEATURE_NAMES = (
    "tenure_months",
    "tickets_90d",
    "usage_score",
    "late_payments",
    "plan_basic",
    "plan_pro",
    "region_north",
    "region_south",
    "region_east",
)


def feature_vector(record: CustomerRecord) -> list[float]:
    return [
        float(record.tenure_months),
        float(record.tickets_90d),
        record.usage_score,
        float(record.late_payments),
        float(record.plan == "basic"),
        float(record.plan == "pro"),
        float(record.region == "north"),
        float(record.region == "south"),
        float(record.region == "east"),
    ]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass
class LogisticRiskModel:
    means: list[float]
    scales: list[float]
    weights: list[float]
    bias: float

    @classmethod
    def fit(
        cls,
        records: list[CustomerRecord],
        epochs: int = 650,
        learning_rate: float = 0.11,
        l2: float = 0.002,
    ) -> "LogisticRiskModel":
        matrix = [feature_vector(record) for record in records]
        width = len(matrix[0])
        means = [sum(row[column] for row in matrix) / len(matrix) for column in range(width)]
        scales = []
        for column in range(width):
            variance = sum((row[column] - means[column]) ** 2 for row in matrix) / len(matrix)
            scales.append(max(math.sqrt(variance), 1e-8))
        normalized = [
            [(value - means[column]) / scales[column] for column, value in enumerate(row)]
            for row in matrix
        ]
        weights = [0.0] * width
        positive_rate = sum(record.label for record in records) / len(records)
        bias = math.log(positive_rate / (1.0 - positive_rate))
        for _ in range(epochs):
            gradient = [0.0] * width
            bias_gradient = 0.0
            for row, record in zip(normalized, records):
                probability = _sigmoid(sum(w * x for w, x in zip(weights, row)) + bias)
                error = probability - record.label
                for column, value in enumerate(row):
                    gradient[column] += error * value
                bias_gradient += error
            sample_count = len(records)
            weights = [
                weight - learning_rate * (gradient[index] / sample_count + l2 * weight)
                for index, weight in enumerate(weights)
            ]
            bias -= learning_rate * bias_gradient / sample_count
        return cls(means=means, scales=scales, weights=weights, bias=bias)

    def predict_probability(self, record: CustomerRecord) -> float:
        row = feature_vector(record)
        normalized = [
            (value - self.means[column]) / self.scales[column]
            for column, value in enumerate(row)
        ]
        return _sigmoid(sum(w * x for w, x in zip(self.weights, normalized)) + self.bias)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "logistic_regression",
            "feature_names": FEATURE_NAMES,
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "bias": self.bias,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def select_threshold(model: LogisticRiskModel, validation: list[CustomerRecord]) -> float:
    """Select the best validation F1, preferring higher recall then 0.5 proximity."""
    candidates = [value / 100 for value in range(20, 81, 2)]
    scored = []
    for threshold in candidates:
        predictions = [int(model.predict_probability(record) >= threshold) for record in validation]
        tp = sum(prediction == 1 and record.label == 1 for prediction, record in zip(predictions, validation))
        fp = sum(prediction == 1 and record.label == 0 for prediction, record in zip(predictions, validation))
        fn = sum(prediction == 0 and record.label == 1 for prediction, record in zip(predictions, validation))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scored.append((f1, recall, -abs(threshold - 0.5), threshold))
    return max(scored)[-1]

