import json
import tempfile
import unittest
from pathlib import Path

from flask import Flask

from geometry_lab import register_geometry_lab


class ReproductionSummaryTests(unittest.TestCase):
    def _client_with_suite(self, temporary, label, model_type, metrics):
        root = Path(temporary) / "geometry_lab"
        root.mkdir(parents=True, exist_ok=True)
        for seed in range(1, 21):
            directory = root / f"experiment_{seed:02d}"
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({
                "id": directory.name,
                "name": directory.name,
                "run_label": label,
                "dataset_source": "paper_nonlinear",
                "dataset_type": "linear",
                "latent_dim": 2,
            }))
            (directory / "training_config.json").write_text(json.dumps({
                "model_type": model_type,
                "latent_dim": 2,
                "seed": seed,
                "seed_source": "effective_training_configuration",
            }))
            paper = {"dto": {"mean": metrics["dto"]}}
            if "disentanglement" in metrics:
                paper["disentanglement"] = {"score": metrics["disentanglement"]}
            if "final_delta_kl" in metrics:
                paper["polarization_final"] = {"delta_kl": metrics["final_delta_kl"]}
            (directory / "paper_metrics.json").write_text(json.dumps(paper))
            polarization = {}
            if "continuous_percent_to_end" in metrics:
                polarization = {
                    "continuous_percent_to_end": metrics["continuous_percent_to_end"],
                    "evaluation_interval_batches": 500,
                    "total_optimizer_steps": 94_200,
                }
            (directory / "analysis.json").write_text(json.dumps({"polarization_summary": polarization}))
            (directory / "summary.json").write_text(json.dumps({
                "split_counts": {"train": 40_000, "validation": 5_000, "test": 5_000},
            }))

        app = Flask(__name__)
        app.testing = True
        register_geometry_lab(app, temporary)
        return app.test_client()

    def test_control_suite_only_requires_applicable_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self._client_with_suite(
                temporary, "paper-nonlinear-fullcov-test", "beta_vae_full_cov",
                {"dto": 0.55, "disentanglement": 0.42},
            )
            response = client.get(
                "/api/geometry/reproduction-summary?run_label=paper-nonlinear-fullcov-test"
            )
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertEqual(result["status"], "pass")
            self.assertEqual([item["metric"] for item in result["comparisons"]], ["dto", "disentanglement"])
            self.assertEqual(result["integrity"]["eligible_runs"], 20)

    def test_random_control_that_is_too_low_fails_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self._client_with_suite(
                temporary, "paper-nonlinear-random-test", "random_decoder", {"dto": 0.53},
            )
            result = client.get(
                "/api/geometry/reproduction-summary?run_label=paper-nonlinear-random-test"
            ).get_json()
            self.assertEqual(result["integrity"]["status"], "pass")
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["comparisons"][0]["operator"], "paper_mean_in_ci95")

    def test_polarization_gate_accounts_for_measurement_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self._client_with_suite(
                temporary, "paper-nonlinear-diagonal-test", "beta_vae",
                {
                    "dto": 0.18,
                    "disentanglement": 0.73,
                    "continuous_percent_to_end": 99.47,
                    "final_delta_kl": 0.001,
                },
            )
            result = client.get(
                "/api/geometry/reproduction-summary?run_label=paper-nonlinear-diagonal-test"
            ).get_json()
            self.assertEqual(result["status"], "pass")
            polarized = next(
                item for item in result["comparisons"]
                if item["metric"] == "continuous_percent_to_end"
            )
            self.assertGreater(polarized["resolution_allowance_percent"], 0.53)
            self.assertEqual(polarized["status"], "pass")


if __name__ == "__main__":
    unittest.main()
