from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from mlops_eval.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_produces_passing_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_pipeline(root)
            report = json.loads((root / "artifacts" / "threshold_report.json").read_text())
            manifest = json.loads((root / "artifacts" / "run_manifest.json").read_text())
            evidence_manifest = json.loads((root / "artifacts" / "evidence_manifest.json").read_text())
            runtime_telemetry = json.loads((root / "artifacts" / "runtime_telemetry.json").read_text())
            self.assertEqual(result["threshold_report"]["status"], "pass")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(report["status"], "pass")
            self.assertEqual(manifest["pipeline_status"], "pass")
            self.assertEqual(evidence_manifest["pipeline_status"], "pass")
            self.assertEqual(len(evidence_manifest["artifacts"]), 9)
            self.assertEqual(len(manifest["runtime_artifacts"]), 2)
            self.assertEqual(runtime_telemetry["status"], "pass")
            self.assertTrue(runtime_telemetry["checks"]["agent_latency"])
            stable_trace = json.loads((root / "artifacts" / "traces.jsonl").read_text().splitlines()[0])
            runtime_trace = json.loads((root / "artifacts" / "runtime_traces.jsonl").read_text().splitlines()[0])
            self.assertNotIn("observed_latency_ms", stable_trace)
            self.assertIn("observed_latency_ms", runtime_trace)


if __name__ == "__main__":
    unittest.main()
