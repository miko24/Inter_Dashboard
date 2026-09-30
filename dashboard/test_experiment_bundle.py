import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask

from geometry_lab import register_geometry_lab


class ExperimentBundleTests(unittest.TestCase):
    def make_app(self, experiments_root):
        app = Flask(__name__)
        register_geometry_lab(app, experiments_root)
        app.config.update(TESTING=True)
        return app

    def seed_experiment(self, experiments_root, experiment_id="geo_bundle_fixture"):
        directory = Path(experiments_root) / "geometry_lab" / experiment_id
        directory.mkdir(parents=True)
        config = {
            "schema_version": 2,
            "id": experiment_id,
            "name": "Bundle fixture",
            "created_at": "2026-01-01T00:00:00+00:00",
            "dataset_type": "linear",
            "dataset_source": "synthetic_linear",
            "factor_types": ["linear"],
            "k": 1,
            "dataset_size": 4,
            "latent_dim": 1,
        }
        summary = {
            "sample_count": 4,
            "observation_dim": 1,
            "factor_count": 1,
            "factor_types": ["linear"],
            "projection": [[0, 0, 0]] * 4,
            "factor_values": [[0], [1], [2], [3]],
            "factor_active": [[1]] * 4,
            "pca_explained_variance": [1, 0, 0],
            "split_counts": {"train": 3, "validation": 0, "test": 1},
        }
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "config.yaml").write_text("id: geo_bundle_fixture\n", encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (directory / "analysis.json").write_text(json.dumps({"metrics": [], "projection": []}), encoding="utf-8")
        np.savez_compressed(
            directory / "dataset.npz",
            observations=np.arange(4, dtype=np.float32)[:, None],
            factor_values=np.arange(4, dtype=np.float32)[:, None],
        )
        return experiment_id

    def export_bundle(self, app, experiment_id, profile):
        response = app.test_client().get(
            f"/api/geometry/experiments/{experiment_id}/export?format=bundle&profile={profile}"
        )
        if response.status_code != 200:
            self.fail(response.get_data(as_text=True))
        payload = response.get_data()
        response.close()
        return payload

    def import_bundle(self, app, payload):
        return app.test_client().post(
            "/api/geometry/experiments/import-bundle",
            data={"file": (io.BytesIO(payload), "fixture.mslab")},
            content_type="multipart/form-data",
        )

    def test_results_bundle_round_trip_and_deduplication(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as target_root:
            experiment_id = self.seed_experiment(source_root)
            payload = self.export_bundle(self.make_app(source_root), experiment_id, "results")
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["profile"], "results")
                self.assertNotIn("artifacts/dataset.npz", archive.namelist())

            target_app = self.make_app(target_root)
            first = self.import_bundle(target_app, payload)
            self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
            first_result = first.get_json()
            self.assertFalse(first_result["deduplicated"])
            imported = Path(target_root) / "geometry_lab" / first_result["experiment"]["id"]
            self.assertTrue((imported / "analysis.json").exists())
            self.assertFalse((imported / "dataset.npz").exists())
            self.assertTrue(first_result["experiment"]["results_only"])

            second = self.import_bundle(target_app, payload)
            self.assertEqual(second.status_code, 200)
            self.assertTrue(second.get_json()["deduplicated"])

    def test_complete_bundle_restores_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as target_root:
            experiment_id = self.seed_experiment(source_root)
            payload = self.export_bundle(self.make_app(source_root), experiment_id, "complete")
            response = self.import_bundle(self.make_app(target_root), payload)
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            result = response.get_json()
            imported = Path(target_root) / "geometry_lab" / result["experiment"]["id"]
            self.assertTrue((imported / "dataset.npz").exists())
            self.assertFalse(result["experiment"]["results_only"])
            self.assertTrue(result["experiment"]["dataset_available"])

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as source_root, tempfile.TemporaryDirectory() as target_root:
            experiment_id = self.seed_experiment(source_root)
            payload = self.export_bundle(self.make_app(source_root), experiment_id, "results")
            tampered = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(payload), "r") as source, zipfile.ZipFile(tampered, "w") as target:
                for info in source.infolist():
                    content = source.read(info.filename)
                    if info.filename == "artifacts/summary.json":
                        changed = bytearray(content)
                        changed[max(0, len(changed) // 2)] ^= 1
                        content = bytes(changed)
                    target.writestr(info, content)
            response = self.import_bundle(self.make_app(target_root), tampered.getvalue())
            self.assertEqual(response.status_code, 400)
            self.assertIn("Checksum mismatch", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
