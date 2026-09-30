import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from flask import Flask

from analysis_extensions import run_custom_analyses
from geometry_lab import GeometryAutoencoder, _normalized_training_config, register_geometry_lab


class ArchitectureAndAblationTests(unittest.TestCase):
    def test_transformer_and_recurrent_models_share_autoencoder_contract(self):
        cases = [
            ("transformer_vae", {"architecture": "transformer", "sequence_length": 4, "d_model": 16, "num_heads": 4, "num_layers": 1}),
            ("rnn_autoencoder", {"architecture": "rnn", "sequence_length": 4, "rnn_type": "gru", "rnn_hidden_size": 12, "num_layers": 1}),
            ("mamba_vae", {"architecture": "mamba", "sequence_length": 4, "d_model": 12, "num_layers": 1, "d_state": 4, "d_conv": 2, "expand": 1}),
        ]
        for model_type, custom in cases:
            config = _normalized_training_config({"model_type": model_type, "latent_dim": 3, "custom_model": custom})
            reconstruction, latent, mean, log_variance = GeometryAutoencoder(12, config)(torch.randn(5, 12))
            self.assertEqual(tuple(reconstruction.shape), (5, 12))
            self.assertEqual(tuple(latent.shape), (5, 3))
            self.assertEqual(tuple(mean.shape), (5, 3))
            self.assertEqual(tuple(log_variance.shape), (5, 3))

    def test_registry_and_ablation_routes_persist_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Flask(__name__)
            register_geometry_lab(app, temporary)
            client = app.test_client()
            definition = client.post("/api/geometry/custom-analyses", json={
                "id": "encoder_profile", "name": "Encoder profile",
                "type": "activation_statistics", "module_pattern": "encoder.*",
                "statistics": ["mean", "std"], "max_samples": 8,
            })
            self.assertEqual(definition.status_code, 200)
            self.assertEqual(client.get("/api/geometry/custom-analyses").json["analyses"][0]["id"], "encoder_profile")

            directory = Path(temporary) / "geometry_lab" / "example"
            directory.mkdir()
            config = _normalized_training_config({
                "model_type": "custom_ae", "latent_dim": 2,
                "custom_model": {"architecture": "mlp", "encoder_layers": [8], "decoder_layers": [8]},
            })
            model = GeometryAutoencoder(4, config)
            torch.save({"model_state_dict": model.state_dict(), "training_config": config, "input_dim": 4}, directory / "model.pt")
            np.savez_compressed(directory / "dataset.npz", observations=np.random.default_rng(1).normal(size=(12, 4)).astype(np.float32))
            custom = run_custom_analyses(
                model, {"observations": np.random.default_rng(2).normal(size=(12, 4)).astype(np.float32)},
                directory, ["custom:encoder_profile"], Path(temporary) / "geometry_lab",
            )
            self.assertIn("encoder_profile", custom["analyses"])
            self.assertTrue((directory / "custom_analysis_results.json").exists())
            targets = client.get("/api/geometry/experiments/example/ablation/targets")
            self.assertEqual(targets.status_code, 200)
            target = targets.json["targets"][0]
            result = client.post("/api/geometry/experiments/example/ablations", json={
                "target": target["name"], "indices": [0], "mode": "zero", "max_samples": 8,
            })
            self.assertEqual(result.status_code, 200, result.json)
            self.assertIn("absolute_mse_delta", result.json["ablation"]["metrics"])
            self.assertTrue(list(directory.glob("ablation_*.json")))


if __name__ == "__main__":
    unittest.main()
