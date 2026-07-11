"""Normalize and gate evidence from the portfolio's governed agent systems."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _record(checks: list[Check], name: str, passed: bool, detail: str) -> None:
    checks.append(Check(name=name, passed=bool(passed), detail=detail))


def _artifact_inventory(root: Path, mapping: dict[str, str]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for key, relative in mapping.items():
        path = root / relative
        inventory[key] = {
            "path": relative,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return inventory


def _system_report(
    spec: dict[str, Any],
    actual_commit: str,
    checks: list[Check],
    inventory: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "repository": spec["repository"],
        "expected_commit_sha": spec["commit_sha"],
        "actual_commit_sha": actual_commit,
        "release_decision": "pass" if checks and all(check.passed for check in checks) else "fail",
        "check_count": len(checks),
        "checks": [check.as_dict() for check in checks],
        "metrics": metrics,
        "input_artifacts": inventory,
    }


def evaluate_semantic_agent(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    checks: list[Check] = []
    actual_commit = _git_head(root)
    _record(checks, "commit_pin", actual_commit == spec["commit_sha"], f"actual={actual_commit}")
    artifacts = spec["artifacts"]
    inventory = _artifact_inventory(root, artifacts)
    for key, item in inventory.items():
        _record(checks, f"artifact_{key}", item["exists"] and item["bytes"] > 0, f"bytes={item['bytes']}")
    if not all(item["exists"] and item["bytes"] > 0 for item in inventory.values()):
        return _system_report(spec, actual_commit, checks, inventory, {})

    evaluation = _read_json(root / artifacts["evaluation"])
    traces = _read_jsonl(root / artifacts["traces"])
    manifest = _read_json(root / artifacts["manifest"])
    live = _read_json(root / artifacts["live_upstream"])
    thresholds = spec["thresholds"]
    scalar_checks = {
        "release_decision": evaluation.get("release_decision") == "pass",
        "case_count": evaluation.get("case_count", 0) >= thresholds["minimum_case_count"],
        "answer_case_count": evaluation.get("answer_case_count", 0) >= thresholds["minimum_answer_case_count"],
        "refusal_case_count": evaluation.get("refusal_case_count", 0) >= thresholds["minimum_refusal_case_count"],
        "correctness_rate": evaluation.get("correctness_rate", 0) >= thresholds["minimum_correctness_rate"],
        "grounding_rate": evaluation.get("grounding_rate", 0) >= thresholds["minimum_grounding_rate"],
        "lineage_accuracy": evaluation.get("lineage_accuracy", 0) >= thresholds["minimum_lineage_accuracy"],
        "refusal_accuracy": evaluation.get("refusal_accuracy", 0) >= thresholds["minimum_refusal_accuracy"],
        "cost_bound": evaluation.get("total_estimated_cost_usd", float("inf")) <= thresholds["maximum_total_estimated_cost_usd"],
        "all_cases_passed": bool(evaluation.get("results")) and all(item.get("passed") is True for item in evaluation["results"]),
        "trace_count": len(traces) == evaluation.get("case_count"),
        "manifest_release": manifest.get("evaluation_release_decision") == "pass",
        "live_upstream": live.get("passed") is True and isinstance(live.get("live"), dict) and live["live"].get("passed") is True,
    }
    for name, passed in scalar_checks.items():
        _record(checks, name, passed, f"value={passed}")

    answered = [trace for trace in traces if trace.get("outcome") == "answered"]
    refused = [trace for trace in traces if trace.get("outcome") == "refused"]
    answered_safe = all(
        trace.get("sql")
        and trace.get("rows")
        and trace.get("lineage")
        and trace.get("source_scope")
        and (trace.get("recommendation") is None or trace["recommendation"].get("agent_executed_action") is False)
        for trace in answered
    )
    refused_safe = all(
        trace.get("sql") is None and trace.get("rows") == [] and trace.get("recommendation") is None
        for trace in refused
    )
    _record(checks, "answered_trace_governance", answered_safe, f"answered={len(answered)}")
    _record(checks, "refused_trace_governance", refused_safe, f"refused={len(refused)}")
    metrics = {
        "case_count": evaluation.get("case_count"),
        "answer_case_count": evaluation.get("answer_case_count"),
        "refusal_case_count": evaluation.get("refusal_case_count"),
        "correctness_rate": evaluation.get("correctness_rate"),
        "grounding_rate": evaluation.get("grounding_rate"),
        "lineage_accuracy": evaluation.get("lineage_accuracy"),
        "refusal_accuracy": evaluation.get("refusal_accuracy"),
        "total_estimated_cost_usd": evaluation.get("total_estimated_cost_usd"),
    }
    return _system_report(spec, actual_commit, checks, inventory, metrics)


def evaluate_decision_twin(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    checks: list[Check] = []
    actual_commit = _git_head(root)
    _record(checks, "commit_pin", actual_commit == spec["commit_sha"], f"actual={actual_commit}")
    artifacts = spec["artifacts"]
    inventory = _artifact_inventory(root, artifacts)
    for key, item in inventory.items():
        _record(checks, f"artifact_{key}", item["exists"] and item["bytes"] > 0, f"bytes={item['bytes']}")
    if not all(item["exists"] and item["bytes"] > 0 for item in inventory.values()):
        return _system_report(spec, actual_commit, checks, inventory, {})

    validation = _read_json(root / artifacts["validation"])
    combined = _read_json(root / artifacts["combined_scenario"])
    openapi = _read_json(root / artifacts["openapi"])
    thresholds = spec["thresholds"]
    native_checks = validation.get("checks", [])
    native_by_name = {item.get("name"): item for item in native_checks}
    scalar_checks = {
        "native_status": validation.get("status") == "passed",
        "minimum_checks": validation.get("checks_passed", 0) >= thresholds["minimum_checks_passed"],
        "all_native_checks": bool(native_checks) and all(item.get("passed") is True for item in native_checks),
        "scenario_count": validation.get("scenario_count") == thresholds["required_scenario_count"],
        "pair_count": validation.get("product_location_pairs") == thresholds["required_product_location_pairs"],
        "legacy_contract": native_by_name.get("legacy_dify_endpoint_status", {}).get("passed") is True and native_by_name.get("legacy_dify_endpoint_rows", {}).get("passed") is True,
        "no_autonomous_execution": native_by_name.get("no_autonomous_execution", {}).get("passed") is True,
        "combined_scenario_id": combined.get("scenario_id") == "demand_spike_supplier_delay",
        "combined_pair_count": combined.get("product_location_count") == thresholds["required_product_location_pairs"],
        "combined_non_execution": combined.get("executes_operational_actions") is False,
        "human_approval_boundary": "human approval" in combined.get("claim_boundary", "").lower(),
        "cost_bound": combined.get("total_incremental_cost", float("inf")) <= thresholds["maximum_total_incremental_cost"],
    }
    for name, passed in scalar_checks.items():
        _record(checks, name, passed, f"value={passed}")

    recommendations = combined.get("recommendations", [])
    allowed_actions = set(thresholds["allowed_actions"])
    recommendation_contract = len(recommendations) == thresholds["required_product_location_pairs"] and all(
        row.get("recommended_action") in allowed_actions
        and 0 <= row.get("service_level_after_action", -1) <= 1
        and 0 <= row.get("stockout_risk_after_action", -1) <= 1
        and row.get("incremental_cost", -1) >= 0
        and row.get("decision_score") is not None
        for row in recommendations
    )
    _record(checks, "recommendation_contract", recommendation_contract, f"recommendations={len(recommendations)}")
    paths = openapi.get("paths", {})
    required_paths = {
        "/stockout-risks/high": "get",
        "/decision-twin/scenarios": "get",
        "/decision-twin/recommendations": "get",
        "/decision-twin/run": "post",
        "/decision-twin/runs/{run_id}": "get",
    }
    api_contract = all(path in paths and method in paths[path] for path, method in required_paths.items())
    _record(checks, "openapi_contract", api_contract, f"paths={len(paths)}")
    metrics = {
        "checks_passed": validation.get("checks_passed"),
        "scenario_count": validation.get("scenario_count"),
        "product_location_pairs": validation.get("product_location_pairs"),
        "combined_high_risk_count": combined.get("high_risk_count"),
        "combined_total_incremental_cost": combined.get("total_incremental_cost"),
        "combined_average_service_level": combined.get("average_service_level"),
    }
    return _system_report(spec, actual_commit, checks, inventory, metrics)


def run_shared_evaluation(
    contract_path: Path,
    semantic_root: Path,
    decision_twin_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    systems = contract["systems"]
    semantic = evaluate_semantic_agent(semantic_root, systems["semantic_agent"])
    decision_twin = evaluate_decision_twin(decision_twin_root, systems["decision_twin"])
    report = {
        "contract_version": contract["contract_version"],
        "overall_release_decision": "pass" if semantic["release_decision"] == "pass" and decision_twin["release_decision"] == "pass" else "fail",
        "system_count": 2,
        "systems": {"semantic_agent": semantic, "decision_twin": decision_twin},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "shared_release_report.json"
    trace_path = output_dir / "shared_checks.jsonl"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for system_name in ("semantic_agent", "decision_twin"):
            for check in report["systems"][system_name]["checks"]:
                handle.write(json.dumps({"system": system_name, **check}, sort_keys=True) + "\n")
    manifest = {
        "manifest_type": "shared_agent_evaluation_evidence",
        "contract_sha256": _sha256(contract_path),
        "overall_release_decision": report["overall_release_decision"],
        "outputs": {
            report_path.name: {"sha256": _sha256(report_path), "bytes": report_path.stat().st_size},
            trace_path.name: {"sha256": _sha256(trace_path), "bytes": trace_path.stat().st_size},
        },
    }
    manifest_path = output_dir / "shared_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
