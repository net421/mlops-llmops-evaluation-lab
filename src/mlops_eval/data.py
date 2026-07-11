"""Deterministic synthetic data generation and splitting."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomerRecord:
    customer_id: str
    region: str
    plan: str
    tenure_months: int
    tickets_90d: int
    usage_score: float
    late_payments: int
    label: int


def generate_dataset(size: int = 720, seed: int = 421) -> list[CustomerRecord]:
    """Generate an auditable customer-risk dataset with controlled label noise."""
    rng = random.Random(seed)
    regions = ("north", "south", "east", "west")
    plans = ("basic", "pro", "premium")
    records: list[CustomerRecord] = []
    for index in range(size):
        region = rng.choice(regions)
        plan = rng.choices(plans, weights=(0.45, 0.35, 0.20), k=1)[0]
        tenure = rng.randint(1, 72)
        tickets = min(8, int(rng.expovariate(1 / 1.8)))
        usage = round(max(0.0, min(100.0, rng.gauss(63, 19))), 2)
        late_payments = rng.choices((0, 1, 2, 3), weights=(0.62, 0.23, 0.11, 0.04), k=1)[0]
        risk = (
            0.72 * tickets
            + 1.08 * late_payments
            - 0.032 * tenure
            - 0.041 * usage
            + {"basic": 0.58, "pro": 0.05, "premium": -0.38}[plan]
            + {"north": 0.10, "south": 0.28, "east": -0.08, "west": 0.0}[region]
            + rng.gauss(0, 0.42)
        )
        label = int(risk > -1.70)
        records.append(
            CustomerRecord(
                customer_id=f"C{index + 1:05d}",
                region=region,
                plan=plan,
                tenure_months=tenure,
                tickets_90d=tickets,
                usage_score=usage,
                late_payments=late_payments,
                label=label,
            )
        )
    return records


def write_dataset(records: list[CustomerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def read_dataset(path: Path) -> list[CustomerRecord]:
    records: list[CustomerRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                CustomerRecord(
                    customer_id=row["customer_id"],
                    region=row["region"],
                    plan=row["plan"],
                    tenure_months=int(row["tenure_months"]),
                    tickets_90d=int(row["tickets_90d"]),
                    usage_score=float(row["usage_score"]),
                    late_payments=int(row["late_payments"]),
                    label=int(row["label"]),
                )
            )
    return records


def stratified_split(
    records: list[CustomerRecord], seed: int = 421
) -> tuple[list[CustomerRecord], list[CustomerRecord], list[CustomerRecord]]:
    """Return deterministic 70/15/15 train, validation and test partitions."""
    rng = random.Random(seed)
    partitions = {0: [], 1: []}
    for record in records:
        partitions[record.label].append(record)
    train: list[CustomerRecord] = []
    validation: list[CustomerRecord] = []
    test: list[CustomerRecord] = []
    for group in partitions.values():
        rng.shuffle(group)
        train_end = math.floor(len(group) * 0.70)
        validation_end = math.floor(len(group) * 0.85)
        train.extend(group[:train_end])
        validation.extend(group[train_end:validation_end])
        test.extend(group[validation_end:])
    for group in (train, validation, test):
        rng.shuffle(group)
    return train, validation, test


def shifted_copy(records: list[CustomerRecord]) -> list[CustomerRecord]:
    """Create a deterministic covariate-shift scenario for monitoring tests."""
    return [
        CustomerRecord(
            customer_id=record.customer_id,
            region=record.region,
            plan=record.plan,
            tenure_months=record.tenure_months,
            tickets_90d=min(8, record.tickets_90d + 2),
            usage_score=max(0.0, record.usage_score - 24.0),
            late_payments=min(3, record.late_payments + 1),
            label=record.label,
        )
        for record in records
    ]

