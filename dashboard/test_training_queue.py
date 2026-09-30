import json
import tempfile
import time
import unittest
from pathlib import Path

from flask import Flask

from geometry_lab import _validated_image_shape, register_geometry_lab


class PersistentTrainingQueueTests(unittest.TestCase):
    def test_image_shape_recovery_normalizes_grayscale_and_rejects_mismatch(self):
        self.assertEqual(_validated_image_shape([64, 64], 4096), [1, 64, 64])
        self.assertEqual(_validated_image_shape([3, 64, 64], 12288), [3, 64, 64])
        self.assertEqual(_validated_image_shape([64, 64], 128), [])

    def test_clear_history_preserves_pending_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue_directory = Path(temporary) / "geometry_lab"
            queue_directory.mkdir(parents=True)
            queue_path = queue_directory / "training_queue.json"
            queue_path.write_text(json.dumps({
                "schema_version": 1,
                "items": [
                    {"id": "queue_pending", "experiment_id": "exp_pending", "status": "queued"},
                    {"id": "queue_complete", "experiment_id": "exp_complete", "status": "complete"},
                    {"id": "queue_error", "experiment_id": "exp_error", "status": "error"},
                ],
            }))

            app = Flask(__name__)
            app.testing = True
            register_geometry_lab(app, temporary)
            client = app.test_client()

            cleared = client.delete("/api/geometry/training-queue/history")
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.get_json()["removed"], 2)
            self.assertEqual(cleared.get_json()["remaining"], 1)

            persisted = json.loads(queue_path.read_text())
            self.assertEqual([item["id"] for item in persisted["items"]], ["queue_pending"])

    def test_untrained_experiment_runs_and_remains_visible_after_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Flask(__name__)
            app.testing = True
            register_geometry_lab(app, temporary)
            client = app.test_client()

            created = client.post("/api/geometry/experiments", json={
                "name": "queue smoke test",
                "dataset_source": "synthetic_linear",
                "dataset_type": "linear",
                "generator_type": "linear",
                "factor_types": ["linear"],
                "k": 1,
                "latent_dim": 1,
                "dataset_size": 100,
                "observation_dim": 2,
                "noise": 0,
                "train_split": 0.8,
                "validation_split": 0.1,
                "analysis_plan": [],
                "seed": 7,
            })
            self.assertEqual(created.status_code, 200)
            experiment_id = created.get_json()["experiment"]["id"]

            queued = client.post("/api/geometry/training-queue", json={
                "experiment_id": experiment_id,
                "training_config": {
                    "model_type": "linear_ae",
                    "epochs": 2,
                    "training_steps": 3,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "optimizer": "adam",
                    "latent_dim": 1,
                    "encoder_depth": 0,
                    "decoder_depth": 0,
                    "seed": 3,
                },
                "analysis_config": {},
            })
            self.assertEqual(queued.status_code, 200)
            queue_id = queued.get_json()["queue_item"]["id"]

            deadline = time.time() + 10
            item = None
            while time.time() < deadline:
                response = client.get("/api/geometry/training-queue")
                self.assertEqual(response.status_code, 200)
                item = next(entry for entry in response.get_json()["items"] if entry["id"] == queue_id)
                if item["status"] in {"complete", "error"}:
                    break
                time.sleep(0.05)

            self.assertIsNotNone(item)
            self.assertEqual(item["status"], "complete", item.get("error"))
            experiment_directory = Path(temporary) / "geometry_lab" / experiment_id
            self.assertTrue((experiment_directory / "analysis.json").exists())
            training = json.loads((experiment_directory / "training_config.json").read_text())
            analysis = json.loads((experiment_directory / "analysis.json").read_text())
            self.assertEqual(training["training_steps"], 3)
            self.assertEqual(analysis["training_budget"]["completed_steps"], 3)

            # A new client models a browser refresh: the item is read from the
            # server-owned queue rather than browser memory.
            refreshed_client = app.test_client()
            refreshed = refreshed_client.get("/api/geometry/training-queue").get_json()
            refreshed_item = next(entry for entry in refreshed["items"] if entry["id"] == queue_id)
            self.assertEqual(refreshed_item["status"], "complete")

            persisted = json.loads((Path(temporary) / "geometry_lab" / "training_queue.json").read_text())
            self.assertEqual(persisted["items"][0]["status"], "complete")

            duplicate = client.post("/api/geometry/training-queue", json={
                "experiment_id": experiment_id,
                "training_config": {"model_type": "linear_ae", "epochs": 1},
            })
            self.assertEqual(duplicate.status_code, 400)

            cleared = client.delete("/api/geometry/training-queue/history")
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.get_json()["removed"], 1)
            self.assertEqual(cleared.get_json()["remaining"], 0)

            after_clear = client.get("/api/geometry/training-queue").get_json()
            self.assertEqual(after_clear["items"], [])
            persisted_after_clear = json.loads(
                (Path(temporary) / "geometry_lab" / "training_queue.json").read_text()
            )
            self.assertEqual(persisted_after_clear["items"], [])

            cleared_again = client.delete("/api/geometry/training-queue/history")
            self.assertEqual(cleared_again.status_code, 200)
            self.assertEqual(cleared_again.get_json()["removed"], 0)


if __name__ == "__main__":
    unittest.main()
