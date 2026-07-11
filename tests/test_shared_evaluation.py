from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mlops_eval.shared_evaluation import run_shared_evaluation


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_init(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "tests@example.com"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Tests"], check=True)
    marker = root / ".fixture"
    marker.write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".fixture"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def semantic_fixture(root: Path) -> None:
    results = []
    traces = []
    for index in range(2):
        results.append({
            "id": f"answer-{index}", "expected": "answer", "actual": "answered",
            "grounding_passed": True, "lineage_passed": True, "passed": True,
            "estimated_cost_usd": 0.00001,
        })
        traces.append({
            "question": f"question {index}", "outcome": "answered", "sql": "SELECT 1 LIMIT 50",
            "rows": [{"dimension_value": "x", "metric_value": 1.0}],
            "lineage": {"dimension_scope": "shared_upstream", "compatible_upstream_sources": ["x"]},
            "source_scope": "shared_upstream", "recommendation": None,
        })
    results.append({
        "id": "refuse-0", "expected": "refuse", "actual": "refused",
        "grounding_passed": True, "lineage_passed": True, "passed": True,
        "estimated_cost_usd": 0.0,
    })
    traces.append({
        "question": "unsafe", "outcome": "refused", "sql": None, "rows": [],
        "lineage": None, "source_scope": None, "recommendation": None,
    })
    write_json(root / "artifacts/evaluation_results.json", {
        "evaluation_version": "2.0.0", "case_count": 3, "answer_case_count": 2,
        "refusal_case_count": 1, "correctness_rate": 1.0, "grounding_rate": 1.0,
        "lineage_accuracy": 1.0, "refusal_accuracy": 1.0,
        "total_estimated_cost_usd": 0.00002, "release_decision": "pass", "results": results,
    })
    (root / "artifacts/agent_traces.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in traces), encoding="utf-8"
    )
    write_json(root / "artifacts/artifact_manifest.json", {
        "contract_version": "2.0.0", "evaluation_release_decision": "pass"
    })
    write_json(root / "artifacts/live_upstream_validation.json", {
        "snapshot": {"passed": True}, "live": {"passed": True}, "passed": True
    })


def decision_fixture(root: Path) -> None:
    checks = [
        {"name": "legacy_dify_endpoint_status", "passed": True, "detail": "status=200"},
        {"name": "legacy_dify_endpoint_rows", "passed": True, "detail": "rows=2"},
        {"name": "no_autonomous_execution", "passed": True, "detail": "safe"},
        {"name": "scenario_coverage", "passed": True, "detail": "complete"},
    ]
    write_json(root / "artifacts/decision_twin_validation.json", {
        "status": "passed", "checks_passed": 4, "scenario_count": 4,
        "product_location_pairs": 2, "checks": checks, "runs": {}
    })
    recommendations = []
    for index in range(2):
        recommendations.append({
            "product_id": f"P{index}", "location_id": "L1",
            "recommended_action": "no_action", "service_level_after_action": 0.9,
            "stockout_risk_after_action": 0.1, "incremental_cost": 0.0,
            "decision_score": 10.0, "human_review_required": False,
        })
    write_json(root / "artifacts/combined_scenario_recommendations.json", {
        "scenario_id": "demand_spike_supplier_delay", "product_location_count": 2,
        "high_risk_count": 1, "total_incremental_cost": 0.0,
        "average_service_level": 0.9, "human_review_required": True,
        "executes_operational_actions": False,
        "claim_boundary": "Decision support only; human approval is required.",
        "recommendations": recommendations,
    })
    write_json(root / "artifacts/openapi.json", {"paths": {
        "/stockout-risks/high": {"get": {}},
        "/decision-twin/scenarios": {"get": {}},
        "/decision-twin/recommendations": {"get": {}},
        "/decision-twin/run": {"post": {}},
        "/decision-twin/runs/{run_id}": {"get": {}},
    }})


class SharedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.semantic = self.root / "semantic"
        self.decision = self.root / "decision"
        self.semantic.mkdir()
        self.decision.mkdir()
        self.semantic_sha = git_init(self.semantic)
        self.decision_sha = git_init(self.decision)
        semantic_fixture(self.semantic)
        decision_fixture(self.decision)
        self.contract = self.root / "contract.json"
        self.contract_payload = {
            "contract_version": "test-1",
            "systems": {
                "semantic_agent": {
                    "repository": "test/semantic", "commit_sha": self.semantic_sha,
                    "artifacts": {
                        "evaluation": "artifacts/evaluation_results.json",
                        "traces": "artifacts/agent_traces.jsonl",
                        "manifest": "artifacts/artifact_manifest.json",
                        "live_upstream": "artifacts/live_upstream_validation.json",
                    },
                    "thresholds": {
                        "minimum_case_count": 3, "minimum_answer_case_count": 2,
                        "minimum_refusal_case_count": 1, "minimum_correctness_rate": 1.0,
                        "minimum_grounding_rate": 1.0, "minimum_lineage_accuracy": 1.0,
                        "minimum_refusal_accuracy": 1.0,
                        "maximum_total_estimated_cost_usd": 0.01,
                    },
                },
                "decision_twin": {
                    "repository": "test/decision", "commit_sha": self.decision_sha,
                    "artifacts": {
                        "validation": "artifacts/decision_twin_validation.json",
                        "combined_scenario": "artifacts/combined_scenario_recommendations.json",
                        "openapi": "artifacts/openapi.json",
                    },
                    "thresholds": {
                        "minimum_checks_passed": 4, "required_scenario_count": 4,
                        "required_product_location_pairs": 2,
                        "maximum_total_incremental_cost": 100.0,
                        "allowed_actions": ["no_action", "expedite_inbound_review",
                            "inventory_transfer_review", "planned_replenishment_review"],
                    },
                },
            },
        }
        self._save_contract()

    def _save_contract(self):
        write_json(self.contract, self.contract_payload)

    def run_gate(self, output_name="out"):
        return run_shared_evaluation(
            self.contract, self.semantic, self.decision, self.root / output_name
        )

    def test_passing_fixtures_release_both_systems(self):
        report = self.run_gate()
        self.assertEqual(report["overall_release_decision"], "pass")
        self.assertTrue(all(item["release_decision"] == "pass" for item in report["systems"].values()))

    def test_semantic_lineage_failure_blocks_release(self):
        traces_path = self.semantic / "artifacts/agent_traces.jsonl"
        traces = [json.loads(line) for line in traces_path.read_text().splitlines()]
        traces[0]["lineage"] = None
        traces_path.write_text("".join(json.dumps(item) + "\n" for item in traces))
        report = self.run_gate()
        self.assertEqual(report["overall_release_decision"], "fail")
        self.assertEqual(report["systems"]["semantic_agent"]["release_decision"], "fail")

    def test_decision_twin_autonomy_claim_blocks_release(self):
        path = self.decision / "artifacts/combined_scenario_recommendations.json"
        payload = json.loads(path.read_text())
        payload["executes_operational_actions"] = True
        write_json(path, payload)
        report = self.run_gate()
        self.assertEqual(report["systems"]["decision_twin"]["release_decision"], "fail")

    def test_commit_mismatch_blocks_release(self):
        payload = deepcopy(self.contract_payload)
        payload["systems"]["semantic_agent"]["commit_sha"] = "0" * 40
        write_json(self.contract, payload)
        report = self.run_gate()
        self.assertEqual(report["systems"]["semantic_agent"]["release_decision"], "fail")

    def test_missing_artifact_blocks_release_without_crash(self):
        (self.decision / "artifacts/openapi.json").unlink()
        report = self.run_gate()
        self.assertEqual(report["systems"]["decision_twin"]["release_decision"], "fail")

    def test_shared_outputs_are_byte_reproducible(self):
        self.run_gate("first")
        self.run_gate("second")
        for name in ("shared_release_report.json", "shared_checks.jsonl", "shared_evidence_manifest.json"):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
