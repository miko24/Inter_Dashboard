import unittest

import numpy as np
import torch

from orthogonality_experiment import (
    GeometryVAE,
    compile_radius_equation,
    gram_analysis,
    normalize_config,
    train_cell,
)


class GramAnalysisTests(unittest.TestCase):
    def test_zero_inputs_probe_posterior_instead_of_null_mean(self):
        torch.manual_seed(7)
        model = GeometryVAE(ambient_dim=3)
        with torch.no_grad():
            model.w_mu.copy_(torch.tensor([[1.0, -0.5, 0.25], [0.5, 1.0, -1.0]]))
            model.w_logvar.zero_()
            model.bias.zero_()

        analysis = gram_analysis(model, torch.zeros(512, 3))

        self.assertGreater(np.linalg.norm(analysis["metric"]), 0.0)
        self.assertGreater(np.linalg.norm(analysis["manifold_gram"]), 0.0)
        self.assertTrue(np.all(np.diag(analysis["manifold_gram"]) > 0.0))

    def test_unit_norm_option_projects_trained_feature_columns(self):
        config = normalize_config({
            "densities": [0.5],
            "importance_decays": [1.0],
            "ambient_dimensions": [3],
            "steps": 2,
            "batch_size": 8,
            "evaluation_size": 16,
            "unit_norm_weights": True,
        })

        result = train_cell(
            config,
            density=0.5,
            decay=1.0,
            ambient_dim=3,
            seed=11,
            device=torch.device("cpu"),
            radius_fn=compile_radius_equation("1"),
        )

        np.testing.assert_allclose(np.linalg.norm(result["weights"], axis=0), 1.0, atol=1e-6)
        self.assertTrue(config["unit_norm_weights"])

    def test_unit_norm_option_defaults_to_disabled(self):
        self.assertFalse(normalize_config({})["unit_norm_weights"])


if __name__ == "__main__":
    unittest.main()
