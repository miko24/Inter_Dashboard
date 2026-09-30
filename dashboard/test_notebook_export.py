import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from notebook_export import build_experiment_notebook


class NotebookExportTests(unittest.TestCase):
    def test_export_is_valid_auditable_notebook(self):
        dashboard_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "experiment_a"
            directory.mkdir()
            config = {"id": "experiment_a", "name": "Notebook test"}
            training = {
                "model_type": "beta_vae", "latent_dim": 2, "architecture": "mlp",
                "custom_model": {"architecture": "mlp"},
            }
            for name, value in (
                ("config.json", config), ("training_config.json", training),
                ("metrics.json", {"loss": 1.0}), ("analysis.json", {"metric": 2.0}),
            ):
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
            np.savez_compressed(
                directory / "dataset.npz", observations=np.zeros((4, 3), dtype=np.float32),
                split=np.asarray([0, 0, 1, 2], dtype=np.int8),
            )
            (directory / "model.pt").write_bytes(b"checkpoint-placeholder")

            destination = build_experiment_notebook(directory, dashboard_root)
            notebook = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(notebook["metadata"]["geometry_lab"]["experiment_id"], "experiment_a")
            self.assertIn("dataset.npz", notebook["metadata"]["geometry_lab"]["artifact_manifest"])
            self.assertIn("geometry_lab.py", notebook["metadata"]["geometry_lab"]["source_manifest"])
            joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            self.assertIn("RUN_DASHBOARD_RETRAIN = False", joined)
            self.assertIn("class GeometryAutoencoder", joined)
            self.assertIn("artifact_integrity", joined)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    def test_export_rejects_untrained_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trained complete experiment"):
                build_experiment_notebook(directory, Path(__file__).resolve().parent)


if __name__ == "__main__":
    unittest.main()
