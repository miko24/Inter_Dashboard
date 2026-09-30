import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from flask import Flask

from geometry_lab import register_geometry_lab
from zietlow_reproduction import (
    DATASETS,
    aggregate_rows,
    apply_manipulation,
    compute_metrics,
    make_uniform_noise,
    paper_protocol,
)


class ZietlowProtocolTests(unittest.TestCase):
    def test_beta_vae_primary_values_are_dataset_specific(self):
        dsprites = paper_protocol("dsprites", "beta_vae", "original", 3)
        shapes = paper_protocol("shapes3d", "beta_vae", "manipulated", 7)
        self.assertEqual(dsprites["latent_dim"], 5)
        self.assertEqual(dsprites["beta"], 8.0)
        self.assertEqual(shapes["latent_dim"], 6)
        self.assertEqual(shapes["beta"], 32.0)
        self.assertEqual(shapes["dataset_variant"], "manipulated")
        self.assertEqual(shapes["seed"], 7)
        self.assertEqual(dsprites["training_steps"], 300000)
        self.assertEqual(dsprites["reconstruction_loss"], "bernoulli_logits")
        self.assertEqual(dsprites["custom_model"]["conv_encoder_kernel_sizes"], [4, 4, 2, 2])
        self.assertEqual(dsprites["metric_sampling"], "reference")
        self.assertEqual(dsprites["metric_seed"], 0)

    def test_uniform_noise_is_bounded_and_reproducible(self):
        original = np.full((32, 8), 0.5, dtype=np.float32)
        changed_a, stats = make_uniform_noise(original, DATASETS["dsprites"]["epsilon"], 11)
        changed_b, _ = make_uniform_noise(original, DATASETS["dsprites"]["epsilon"], 11)
        np.testing.assert_array_equal(changed_a, changed_b)
        self.assertLessEqual(stats["max_abs_change"], 0.1 + 1e-6)
        self.assertFalse(stats["factor_labels_changed"])
        self.assertFalse(stats["sample_order_changed"])

    def test_manipulation_clips_the_unit_constraint(self):
        original = np.full((4, 3), 0.5, dtype=np.float32)
        changed, stats = apply_manipulation(original, np.full_like(original, 4), 0.1)
        np.testing.assert_allclose(changed, 0.6, atol=1e-6)
        self.assertAlmostEqual(stats["max_abs_change"], 0.1, places=6)


class ZietlowMetricTests(unittest.TestCase):
    @staticmethod
    def aligned_fixture():
        grid = np.asarray([(a, b) for a in range(4) for b in range(4)], dtype=np.float32)
        factors = np.tile(grid, (30, 1))
        rng = np.random.default_rng(19)
        order = rng.permutation(len(factors))
        factors = factors[order]
        latents = factors + rng.normal(0, 0.005, factors.shape)
        split = np.full(len(factors), 2, dtype=np.int8)
        split[:320] = 0
        logvar = np.full_like(latents, -3.0)
        return latents, factors, split, logvar

    def test_aligned_latents_score_high_and_units_are_active(self):
        latents, factors, split, logvar = self.aligned_fixture()
        result = compute_metrics(
            latents, factors, split, seed=5, logvar=logvar,
            vote_batches=250, vote_evaluation_batches=150,
        )
        self.assertGreater(result["mig"]["score"], 0.9)
        self.assertGreater(result["dci"]["disentanglement"], 0.9)
        # SAP is the best-vs-second-best single-coordinate accuracy gap; even
        # perfect factor recovery need not approach one for four-way factors.
        self.assertGreater(result["sap"]["score"], 0.4)
        self.assertGreater(result["factor_vae_score"]["score"], 0.9)
        self.assertEqual(result["active_units"]["count"], 2)
        self.assertFalse(result["active_units"]["over_pruned"])

    def test_reference_metric_sampling_records_library_sample_counts(self):
        latents, factors, split, logvar = self.aligned_fixture()
        result = compute_metrics(
            latents, factors, split, seed=9, logvar=logvar, sampling="reference",
            metric_seed=0, train_samples=1000, test_samples=500, variance_samples=800,
            vote_batches=250, vote_evaluation_batches=150,
        )
        self.assertEqual(result["seed"], 0)
        self.assertEqual(result["training_samples"], 1000)
        self.assertEqual(result["test_samples"], 500)
        self.assertEqual(result["variance_samples"], 800)
        self.assertEqual(result["factor_vae_score"]["variance_estimate_samples"], 800)

    def test_robustness_scales_are_not_merged_with_primary_target(self):
        rows = []
        for scale in (1.0, 1.25):
            for seed, score in ((1, 0.2), (2, 0.3)):
                rows.append({
                    "dataset": "dsprites", "model": "beta_vae", "variant": "original",
                    "seed": seed, "hyperparameter_scale": scale, "paper_comparable": True, "mig": score,
                    "dci": 0.4, "sap": 0.3, "factor_vae_score": 0.5,
                })
        groups = aggregate_rows(rows, bootstrap_samples=100)["groups"]
        self.assertEqual(len(groups), 2)
        primary = next(group for group in groups if group["hyperparameter_scale"] == 1.0)
        sweep = next(group for group in groups if group["hyperparameter_scale"] == 1.25)
        self.assertIsNotNone(primary["paper_mean"])
        self.assertIsNone(sweep["paper_mean"])
        self.assertEqual(sweep["status"], "not_applicable")


class ZietlowReportRouteTests(unittest.TestCase):
    def test_manual_clone_registers_a_fresh_untrained_paper_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "geometry_lab"
            source = root / "geo_source"
            source.mkdir(parents=True)
            observations = np.zeros((20, 4), dtype=np.float32)
            factors = np.tile(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (5, 1))
            split = np.asarray([0] * 16 + [1] * 2 + [2] * 2, dtype=np.uint8)
            np.savez_compressed(source / "dataset.npz", observations=observations, factor_values=factors, split=split)
            source.joinpath("summary.json").write_text(json.dumps({"sample_count": 20, "split_counts": {"train": 16, "validation": 2, "test": 2}}), encoding="utf-8")
            source.joinpath("config.json").write_text(json.dumps({
                "id": "geo_source", "name": "dSprites source", "dataset_source": "dsprites",
                "dataset_variant": "original", "dataset_size": 20, "image_shape": [1, 2, 2],
            }), encoding="utf-8")
            app = Flask(__name__)
            app.testing = True
            register_geometry_lab(app, temporary)
            client = app.test_client()
            response = client.post("/api/geometry/experiments/geo_source/clone", json={
                "name": "Zietlow 2021 manual beta run seed 1",
                "run_label": "zietlow2021-manual",
                "tags": ["reproduction", "zietlow2021"],
                "paper_title": "Demystifying Inductive Biases for (Beta-)VAE Based Architectures",
                "paper_protocol": {"paper_id": "zietlow2021", "model_type": "beta_vae", "beta": 8},
                "analysis_plan": [],
            })
            self.assertEqual(response.status_code, 200)
            cloned = response.get_json()["experiment"]
            self.assertNotEqual(cloned["id"], "geo_source")
            self.assertEqual(cloned["paper_id"], "zietlow2021")
            self.assertEqual(cloned["parent_experiment_id"], "geo_source")
            clone_directory = root / cloned["id"]
            self.assertTrue((clone_directory / "dataset.npz").exists())
            self.assertFalse((clone_directory / "model.pt").exists())

    def test_completed_suite_can_be_summarized_and_downloaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "geometry_lab"
            run = root / "geo_completed"
            run.mkdir(parents=True)
            run.joinpath("config.json").write_text(json.dumps({
                "id": "geo_completed", "suite_id": "zietlow_test", "paper_id": "zietlow2021",
                "dataset_source": "dsprites", "dataset_variant": "original", "effective_training_seed": 1,
                "paper_protocol": {"paper_hyperparameter_scale": 1, "implementation_fidelity": "test fidelity"},
            }), encoding="utf-8")
            run.joinpath("zietlow_metrics.json").write_text(json.dumps({
                "model_type": "beta_vae", "seed": 1,
                "mig": {"score": 0.23}, "dci": {"disentanglement": 0.4},
                "sap": {"score": 0.3}, "factor_vae_score": {"score": 0.5},
                "active_units": {"count": 5, "over_pruned": False},
            }), encoding="utf-8")
            app = Flask(__name__)
            app.testing = True
            register_geometry_lab(app, temporary)
            client = app.test_client()
            summary = client.get("/api/geometry/zietlow/summary?suite_id=zietlow_test")
            self.assertEqual(summary.status_code, 200)
            self.assertEqual(summary.get_json()["aggregate"]["groups"][0]["paper_mean"], 0.23)
            report = client.get("/api/geometry/zietlow/report?suite_id=zietlow_test")
            self.assertEqual(report.status_code, 200)
            text = report.data.decode("utf-8")
            self.assertIn("Primary MIG comparison", text)
            self.assertIn("test fidelity", text)
            report.close()


if __name__ == "__main__":
    unittest.main()
