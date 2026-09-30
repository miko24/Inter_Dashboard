import unittest

import numpy as np
import torch

from geometry_lab import (
    GeometryAutoencoder,
    PAPER_BATCH_SIZE_ASSUMPTION,
    _loss_terms,
    _normalized_training_config,
    _resolved_paper_protocol,
    _three_way_split,
    generate_dataset,
)
from paper_metrics import (
    nearest_signed_permutation,
    paper_disentanglement_score,
    paper_dto_from_jacobian,
    polarization_snapshot,
)


class PaperMetricTests(unittest.TestCase):
    def test_identity_decoder_has_zero_paper_dto(self):
        self.assertAlmostEqual(paper_dto_from_jacobian(np.eye(3)), 0.0, places=12)

    def test_nearest_signed_permutation_is_valid(self):
        angle = np.pi / 5
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        result = nearest_signed_permutation(rotation)
        np.testing.assert_array_equal(np.sum(np.abs(result), axis=0), np.ones(2))
        np.testing.assert_array_equal(np.sum(np.abs(result), axis=1), np.ones(2))
        # Distinct singular values make the rotated right-singular directions
        # identifiable (a pure rotation has a degenerate singular spectrum).
        jacobian = np.diag([2.0, 1.0]) @ rotation.T
        self.assertGreater(paper_dto_from_jacobian(jacobian), 0)

    def test_paper_dto_uses_complete_v_for_wide_jacobian(self):
        jacobian = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(paper_dto_from_jacobian(jacobian), 0.0, places=12)

    def test_disentanglement_score_rewards_axis_alignment(self):
        rng = np.random.default_rng(7)
        factors = rng.uniform(-1, 1, (1200, 2))
        aligned = factors + rng.normal(0, 0.002, factors.shape)
        angle = np.pi / 4
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = aligned @ rotation.T
        aligned_score = paper_disentanglement_score(aligned, factors, ["continuous", "continuous"], seed=3)["score"]
        rotated_score = paper_disentanglement_score(rotated, factors, ["continuous", "continuous"], seed=3)["score"]
        self.assertGreater(aligned_score, 0.9)
        self.assertLessEqual(aligned_score, 1.0)
        self.assertGreater(aligned_score, rotated_score)

    def test_polarized_approximation_matches_active_coordinate(self):
        values = np.linspace(-2, 2, 1000)
        mu = np.column_stack((values, np.zeros_like(values)))
        logvar = np.column_stack((np.full_like(values, -5.0), np.zeros_like(values)))
        result = polarization_snapshot(mu, logvar)
        self.assertEqual(result["active_mask"], [True, False])
        self.assertLess(result["delta_kl"], 0.01)

    def test_polarization_uses_paper_standard_deviation_threshold(self):
        # std(mu)=0.6 is active under the paper's std > 0.5 criterion, while
        # its variance (0.36) would fail the dashboard's former var > 0.5 test.
        mu = np.column_stack((np.tile([-0.6, 0.6], 100), np.zeros(200)))
        result = polarization_snapshot(mu, np.zeros_like(mu))
        self.assertEqual(result["active_mask"], [True, False])
        self.assertEqual(result["active_selection_rule"], "sqrt(var(mu_j(x_i))) > 0.5")


class PaperDatasetAndLossTests(unittest.TestCase):
    def test_paper_linear_generator_and_three_way_split(self):
        data = generate_dataset({
            "seed": 11, "dataset_size": 1000, "generator_type": "paper_linear",
            "noise": 0, "train_split": 0.8, "validation_split": 0.1,
        })
        self.assertEqual(data["observations"].shape, (1000, 3))
        self.assertEqual(data["factor_values"].shape, (1000, 2))
        self.assertEqual([int(np.sum(data["split"] == i)) for i in range(3)], [800, 100, 100])
        self.assertTrue(np.all((data["factor_values"] >= 0) & (data["factor_values"] <= 1)))
        self.assertEqual(float(data["paper_stretch_factor"][0]), 2.0)
        self.assertEqual(data["paper_embedding_dimensions"].tolist(), [2, 3])
        self.assertEqual(float(data["paper_rotation_degrees"][0]), 45.0)
        np.testing.assert_array_equal(data["paper_rotation_axis"], np.asarray([1, -1, 1], dtype=np.float32))

    def test_floating_point_80_10_10_split_is_exact(self):
        validation = (1.0 - 0.8) / 2.0
        split = _three_way_split(50_000, np.random.default_rng(17), 0.8, validation)
        self.assertEqual([int(np.sum(split == i)) for i in range(3)], [40_000, 5_000, 5_000])

    def test_paper_nonlinear_generator_has_reported_width(self):
        data = generate_dataset({
            "seed": 13, "dataset_size": 100, "generator_type": "paper_nonlinear",
            "noise": 0, "train_split": 0.8, "validation_split": 0.1,
        })
        self.assertEqual(data["observations"].shape, (100, 6))
        self.assertEqual(data["nonlinear_w1"].shape, (10, 2))
        self.assertEqual(data["nonlinear_w2"].shape, (6, 10))
        self.assertEqual(data["nonlinear_b1"].shape, (10,))
        self.assertEqual(data["nonlinear_b2"].shape, (6,))

    def test_reconstruction_loss_sums_observation_dimensions(self):
        class Dummy(torch.nn.Module):
            variational = False
            full_covariance = False
            kind = "linear_ae"

            def forward(self, x):
                zeros = torch.zeros_like(x)
                return zeros, zeros, zeros, zeros

        x = torch.ones((2, 3))
        _, reconstruction, _, _, *_ = _loss_terms(Dummy(), x, {"beta": 0})
        self.assertEqual(float(reconstruction), 3.0)

    def test_bernoulli_logits_matches_reference_sum_reduction(self):
        class Dummy(torch.nn.Module):
            variational = False
            full_covariance = False
            kind = "linear_ae"

            def forward(self, x):
                zeros = torch.zeros_like(x)
                return zeros, zeros, zeros, zeros

        x = torch.ones((2, 3))
        _, reconstruction, _, _, *_ = _loss_terms(
            Dummy(), x, {"beta": 0, "reconstruction_loss": "bernoulli_logits"},
        )
        self.assertAlmostEqual(float(reconstruction), 3 * np.log(2), places=6)

    def test_adagrad_and_split_activations_are_preserved(self):
        config = _normalized_training_config({
            "model_type": "beta_vae", "optimizer": "adagrad",
            "encoder_activation": "relu", "decoder_activation": "tanh",
        })
        self.assertEqual(config["optimizer"], "adagrad")
        self.assertEqual(config["encoder_activation"], "relu")
        self.assertEqual(config["decoder_activation"], "tanh")

    def test_paper_batch_size_is_labeled_as_an_assumption(self):
        config = _normalized_training_config({"paper_reproduction": True, "batch_size": 256})
        self.assertEqual(config["batch_size"], 256)
        self.assertEqual(config["batch_size_source"], "declared_reproduction_assumption")
        self.assertEqual(config["batch_size_assumption"], PAPER_BATCH_SIZE_ASSUMPTION)

    def test_protocol_seed_batch_and_bias_resolve_from_effective_training(self):
        training = _normalized_training_config({
            "paper_reproduction": True, "seed": 7, "batch_size": 128, "bias": False,
        })
        protocol = {"seed": 1, "training": {"seed": 2, "batch_size": 256}, "model": {"bias": True}}
        resolved = _resolved_paper_protocol(protocol, training)
        self.assertEqual(resolved["seed"], 7)
        self.assertEqual(resolved["training"]["seed"], 7)
        self.assertEqual(resolved["batch_size"], 128)
        self.assertEqual(resolved["training"]["batch_size"], 128)
        self.assertFalse(resolved["bias"])
        self.assertFalse(resolved["model"]["bias"])
        self.assertEqual(resolved["seed_source"], "effective_training_configuration")
        self.assertEqual(training["seed_source"], "effective_training_configuration")

    def test_bias_configuration_is_applied_to_linear_layers(self):
        config = _normalized_training_config({"model_type": "vae", "bias": False})
        model = GeometryAutoencoder(4, config)
        linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        self.assertTrue(linear_layers)
        self.assertTrue(all(module.bias is None for module in linear_layers))

    def test_celeba_convolutional_preset_shape(self):
        config = _normalized_training_config({
            "model_type": "conv_beta_vae", "latent_dim": 32,
            "image_shape": [3, 64, 64], "conv_channels": [32, 32, 64, 64],
            "encoder_activation": "relu", "decoder_activation": "relu",
        })
        model = GeometryAutoencoder(3 * 64 * 64, config)
        with torch.no_grad():
            reconstruction, latent, mu, logvar = model(torch.rand(2, 3 * 64 * 64))
        self.assertEqual(reconstruction.shape, (2, 3 * 64 * 64))
        self.assertEqual(latent.shape, (2, 32))
        self.assertEqual(mu.shape, logvar.shape)

    def test_disentanglement_library_convolutional_architecture(self):
        custom = {
            "architecture": "convolutional",
            "conv_encoder_channels": [32, 32, 64, 64],
            "conv_encoder_kernel_sizes": [4, 4, 2, 2],
            "conv_encoder_strides": [2, 2, 2, 2],
            "conv_encoder_paddings": [1, 1, 0, 0],
            "conv_encoder_dense_layers": [256],
            "conv_decoder_base_shape": [64, 4, 4],
            "conv_decoder_dense_layers": [256],
            "conv_decoder_channels": [64, 32, 32],
            "conv_decoder_kernel_sizes": [4, 4, 4, 4],
            "conv_decoder_strides": [2, 2, 2, 2],
            "conv_decoder_paddings": [1, 1, 1, 1],
            "posterior": "diagonal",
        }
        config = _normalized_training_config({
            "model_type": "beta_vae", "latent_dim": 5, "image_shape": [1, 64, 64],
            "paper_image_model": True, "custom_model": custom,
            "reconstruction_loss": "bernoulli_logits", "training_steps": 300000,
        })
        model = GeometryAutoencoder(64 * 64, config)
        self.assertEqual(model.encoder.feature_shape, (64, 4, 4))
        self.assertTrue(any(isinstance(module, torch.nn.Linear) and module.out_features == 256 for module in model.encoder.head))
        with torch.no_grad():
            reconstruction, *_ = model(torch.rand(2, 64 * 64))
        self.assertEqual(reconstruction.shape, (2, 64 * 64))
        self.assertEqual(config["training_steps"], 300000)


if __name__ == "__main__":
    unittest.main()
