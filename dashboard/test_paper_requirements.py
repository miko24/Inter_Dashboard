import unittest

from paper_requirements import evaluate_paper_requirements, paper_reproduction_spec


class PaperRequirementTests(unittest.TestCase):
    def test_synthetic_linear_run_exposes_metric_failures(self):
        result = evaluate_paper_requirements(
            "paper_linear", "beta_vae", 2,
            {
                "paper_dto": 0.028,
                "paper_disentanglement": 0.992,
                "paper_delta_kl": 0.0002,
                "paper_polarized_continuous_percent": 96.3,
            },
        )
        statuses = {item["metric"]: item["status"] for item in result["checks"]}
        self.assertEqual(statuses["paper_dto"], "fail")
        self.assertEqual(statuses["paper_disentanglement"], "pass")
        self.assertEqual(statuses["paper_delta_kl"], "pass")
        self.assertEqual(statuses["paper_polarized_continuous_percent"], "fail")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["pass_fraction"], 0.5)

    def test_missing_paper_metrics_make_gate_incomplete(self):
        result = evaluate_paper_requirements("paper_linear", "beta_vae", 2, {})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing"], 4)

    def test_latent_ten_uses_table_two_latent_ten_target(self):
        result = evaluate_paper_requirements(
            "dsprites", "beta_vae", 10,
            {
                "paper_dto": 0.7,
                "paper_disentanglement": 0.3,
                "paper_delta_kl": 0.01,
                "paper_polarized_continuous_percent": 91.0,
            },
        )
        polarized = next(
            item for item in result["checks"]
            if item["metric"] == "paper_polarized_continuous_percent"
        )
        self.assertEqual(polarized["threshold"], 90.6)
        self.assertEqual(polarized["status"], "pass")

    def test_unreported_configuration_is_not_applicable(self):
        result = evaluate_paper_requirements("paper_linear", "vae", 2, {})
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["applicable"], 0)

    def test_random_control_is_checked_for_two_sided_reproduction_agreement(self):
        too_low = evaluate_paper_requirements(
            "paper_nonlinear", "random_decoder", 2, {"paper_dto": 0.53},
        )
        matched = evaluate_paper_requirements(
            "paper_nonlinear", "random_decoder", 2, {"paper_dto": 0.89},
        )
        check = too_low["checks"][0]
        self.assertEqual(check["operator"], "between")
        self.assertEqual(check["threshold"], [0.73, 1.05])
        self.assertEqual(check["status"], "fail")
        self.assertEqual(matched["status"], "pass")

    def test_polarized_duration_accepts_one_measurement_interval(self):
        result = evaluate_paper_requirements(
            "paper_nonlinear", "beta_vae", 2,
            {
                "paper_dto": 0.18,
                "paper_disentanglement": 0.73,
                "paper_delta_kl": 0.001,
                "paper_polarized_continuous_percent": 99.47,
                "paper_polarization_resolution_percent": 0.54,
            },
        )
        polarized = next(
            item for item in result["checks"]
            if item["metric"] == "paper_polarized_continuous_percent"
        )
        self.assertAlmostEqual(polarized["threshold"], 99.36)
        self.assertEqual(polarized["status"], "pass")

    def test_control_spec_only_requires_paper_reported_metrics(self):
        spec = paper_reproduction_spec("paper_nonlinear", "beta_vae_full_cov", 2)
        self.assertEqual(
            [item["aggregate_key"] for item in spec["metrics"]],
            ["dto", "disentanglement"],
        )


if __name__ == "__main__":
    unittest.main()
