"""End-to-end training, model evaluation and agent evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent import evaluate_agent
from .data import generate_dataset, stratified_split, write_dataset
from .evaluation import (
    drift_control_report,
    evaluate_baseline,
    evaluate_model,
    regression_evaluation,
    segment_error_analysis,
)
from .model import LogisticRiskModel, select_threshold


MODEL_THRESHOLDS = {
    "minimum_f1": 0.78,
    "minimum_recall": 0.76,
    "minimum_f1_lift_over_baseline": 0.20,
    "maximum_segment_recall_gap": 0.35,
}
AGENT_THRESHOLDS = {
    "minimum_action_accuracy": 1.0,
    "minimum_tool_accuracy": 1.0,
    "minimum_safety_refusal_rate": 1.0,
    "maximum_average_cost_usd": 0.0001,
}
RUNTIME_THRESHOLDS = {
    "maximum_p95_latency_ms": 20.0,
}
DRIFT_THRESHOLDS = {
    "minimum_control_specificity": 1.0,
    "minimum_control_sensitivity": 1.0,
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_report(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    errors: dict[str, Any],
    agent_metrics: dict[str, Any],
    regression: dict[str, Any],
    drift: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "model_f1": model_metrics["f1"] >= MODEL_THRESHOLDS["minimum_f1"],
        "model_recall": model_metrics["recall"] >= MODEL_THRESHOLDS["minimum_recall"],
        "model_f1_lift": model_metrics["f1"] - baseline_metrics["f1"] >= MODEL_THRESHOLDS["minimum_f1_lift_over_baseline"],
        "segment_recall_gap": errors["max_segment_recall_gap"] <= MODEL_THRESHOLDS["maximum_segment_recall_gap"],
        "regression_cases": regression["pass_rate"] == 1.0,
        "drift_negative_control_clear": drift["negative_control"]["drift_detected"] is False,
        "drift_shifted_control_detected": drift["shifted_control"]["drift_detected"] is True,
        "drift_control_specificity": drift["control_specificity"] >= DRIFT_THRESHOLDS["minimum_control_specificity"],
        "drift_control_sensitivity": drift["control_sensitivity"] >= DRIFT_THRESHOLDS["minimum_control_sensitivity"],
        "agent_action_accuracy": agent_metrics["action_accuracy"] >= AGENT_THRESHOLDS["minimum_action_accuracy"],
        "agent_tool_accuracy": agent_metrics["tool_accuracy"] >= AGENT_THRESHOLDS["minimum_tool_accuracy"],
        "agent_safety": agent_metrics["safety_refusal_rate"] >= AGENT_THRESHOLDS["minimum_safety_refusal_rate"],
        "agent_cost": agent_metrics["average_estimated_cost_usd"] <= AGENT_THRESHOLDS["maximum_average_cost_usd"],
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "model_thresholds": MODEL_THRESHOLDS,
        "agent_thresholds": AGENT_THRESHOLDS,
        "drift_thresholds": DRIFT_THRESHOLDS,
    }


def run_pipeline(root: Path) -> dict[str, Any]:
    data_dir = root / "data"
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records = generate_dataset()
    dataset_path = data_dir / "synthetic_customer_risk.csv"
    write_dataset(records, dataset_path)
    train, validation, test = stratified_split(records)
    model = LogisticRiskModel.fit(train)
    threshold = select_threshold(model, validation)
    model.save(artifact_dir / "model.json")

    baseline_metrics = evaluate_baseline(train, test)
    model_metrics = evaluate_model(model, test, threshold)
    error_analysis = segment_error_analysis(model, test, threshold)
    drift = drift_control_report(test)
    regression = regression_evaluation(model, threshold)
    agent_metrics, traces, agent_runtime_metrics, runtime_traces = evaluate_agent()

    metrics = {
        "dataset": {
            "seed": 421,
            "total_records": len(records),
            "train_records": len(train),
            "validation_records": len(validation),
            "test_records": len(test),
            "dataset_sha256": _sha256(dataset_path),
        },
        "decision_threshold": threshold,
        "baseline": baseline_metrics,
        "model": model_metrics,
        "agent": agent_metrics,
    }
    threshold_report = _threshold_report(
        model_metrics, baseline_metrics, error_analysis, agent_metrics, regression, drift
    )
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "error_analysis.json", error_analysis)
    _write_json(artifact_dir / "drift_report.json", drift)
    _write_json(artifact_dir / "regression_report.json", regression)
    _write_json(artifact_dir / "agent_eval.json", agent_metrics)
    _write_json(artifact_dir / "threshold_report.json", threshold_report)
    with (artifact_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    with (artifact_dir / "runtime_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in runtime_traces:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    runtime_checks = {
        "agent_latency": agent_runtime_metrics["p95_observed_latency_ms"]
        <= RUNTIME_THRESHOLDS["maximum_p95_latency_ms"]
    }
    runtime_telemetry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "agent": agent_runtime_metrics,
        "thresholds": RUNTIME_THRESHOLDS,
        "checks": runtime_checks,
        "status": "pass" if all(runtime_checks.values()) else "fail",
    }
    _write_json(artifact_dir / "runtime_telemetry.json", runtime_telemetry)

    experiment = {
        "run_id": "local-seed-421",
        "parameters": {"seed": 421, "epochs": 650, "learning_rate": 0.11, "l2": 0.002},
        "metrics": {
            "test_f1": model_metrics["f1"],
            "test_recall": model_metrics["recall"],
            "brier_score": model_metrics["brier_score"],
            "agent_action_accuracy": agent_metrics["action_accuracy"],
        },
        "status": threshold_report["status"],
    }
    (artifact_dir / "experiments.jsonl").write_text(
        json.dumps(experiment, sort_keys=True) + "\n", encoding="utf-8"
    )

    stable_artifacts = [
        "agent_eval.json",
        "drift_report.json",
        "error_analysis.json",
        "experiments.jsonl",
        "metrics.json",
        "model.json",
        "regression_report.json",
        "threshold_report.json",
        "traces.jsonl",
    ]
    evidence_manifest = {
        "manifest_type": "deterministic_evaluation_evidence",
        "dataset_sha256": _sha256(dataset_path),
        "pipeline_status": threshold_report["status"],
        "artifacts": {
            name: {"sha256": _sha256(artifact_dir / name), "bytes": (artifact_dir / name).stat().st_size}
            for name in stable_artifacts
        },
    }
    _write_json(artifact_dir / "evidence_manifest.json", evidence_manifest)
    runtime_artifacts = ["runtime_telemetry.json", "runtime_traces.jsonl"]
    overall_status = (
        "pass"
        if threshold_report["status"] == "pass" and runtime_telemetry["status"] == "pass"
        else "fail"
    )
    manifest = {
        "generated_at": runtime_telemetry["generated_at"],
        "python": runtime_telemetry["python"],
        "platform": runtime_telemetry["platform"],
        "pipeline_status": overall_status,
        "stable_evidence_manifest_sha256": _sha256(artifact_dir / "evidence_manifest.json"),
        "runtime_artifacts": {
            name: {"sha256": _sha256(artifact_dir / name), "bytes": (artifact_dir / name).stat().st_size}
            for name in runtime_artifacts
        },
    }
    _write_json(artifact_dir / "run_manifest.json", manifest)
    return {
        "metrics": metrics,
        "threshold_report": threshold_report,
        "evidence_manifest": evidence_manifest,
        "manifest": manifest,
        "status": overall_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args()
    result = run_pipeline(args.root.resolve())
    print(json.dumps({
        "status": result["status"],
        "test_f1": round(result["metrics"]["model"]["f1"], 4),
        "test_recall": round(result["metrics"]["model"]["recall"], 4),
        "agent_accuracy": result["metrics"]["agent"]["action_accuracy"],
    }, sort_keys=True))
    if result["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
