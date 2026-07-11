import unittest

from mlops_eval.data import CustomerRecord, generate_dataset, stratified_split
from mlops_eval.evaluation import (
    drift_control_report,
    evaluate_baseline,
    evaluate_model,
    regression_evaluation,
)
from mlops_eval.model import LogisticRiskModel, select_threshold


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, validation, cls.test = stratified_split(generate_dataset())
        cls.model = LogisticRiskModel.fit(train)
        cls.threshold = select_threshold(cls.model, validation)

    def test_model_improves_on_majority_baseline(self):
        model_metrics = evaluate_model(self.model, self.test, self.threshold)
        train, _, _ = stratified_split(generate_dataset())
        baseline_metrics = evaluate_baseline(train, self.test)
        self.assertGreater(model_metrics["f1"], baseline_metrics["f1"] + 0.20)
        self.assertGreaterEqual(model_metrics["recall"], 0.76)

    def test_regression_cases_pass(self):
        self.assertEqual(regression_evaluation(self.model, self.threshold)["pass_rate"], 1.0)

    def test_baseline_learns_only_from_training_when_majorities_invert(self):
        def record(identifier, label):
            return CustomerRecord(identifier, "east", "basic", 12, 1, 50.0, 0, label)

        training = [record("T1", 1), record("T2", 1), record("T3", 1), record("T4", 0)]
        evaluation = [record("E1", 0), record("E2", 0), record("E3", 0), record("E4", 1)]
        metrics = evaluate_baseline(training, evaluation)
        self.assertEqual(metrics["learned_majority_class"], 1)
        self.assertEqual(metrics["training_positive_rate"], 0.75)
        self.assertEqual(metrics["positive_rate"], 0.25)
        self.assertEqual(metrics["accuracy"], 0.25)

    def test_drift_controls_measure_specificity_and_sensitivity(self):
        report = drift_control_report(self.test)
        self.assertFalse(report["negative_control"]["drift_detected"])
        self.assertEqual(report["negative_control"]["max_psi"], 0.0)
        self.assertTrue(report["shifted_control"]["drift_detected"])
        self.assertEqual(report["control_specificity"], 1.0)
        self.assertEqual(report["control_sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main()
