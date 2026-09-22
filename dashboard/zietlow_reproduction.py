"""Zietlow, Rolinek & Martius (ICML 2021) reproduction utilities.

The functions in this module deliberately keep paper-facing measurements and
dataset transformations separate from the dashboard's exploratory geometry
metrics.  Every public result contains enough configuration to reconstruct the
measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


PAPER = {
    "id": "zietlow2021",
    "title": "Demystifying Inductive Biases for (Beta-)VAE Based Architectures",
    "citation": "D. Zietlow, M. Rolinek, G. Martius, ICML 2021, PMLR 139:12945-12954",
    "url": "https://proceedings.mlr.press/v139/zietlow21a.html",
    "supplement": "https://proceedings.mlr.press/v139/zietlow21a/zietlow21a-supp.pdf",
}

DATASETS = {
    "dsprites": {"latent_dim": 5, "epsilon": 0.1, "published_samples": 737280},
    "shapes3d": {"latent_dim": 6, "epsilon": 0.175, "published_samples": 480000},
}

PRIMARY_HYPERPARAMETERS = {
    "dsprites": {"beta_vae": 8.0, "beta_tcvae": 6.0, "factor_vae": 35.0, "slow_vae": 1.0},
    "shapes3d": {"beta_vae": 32.0, "beta_tcvae": 32.0, "factor_vae": 7.0, "slow_vae": 1.0},
}

MODEL_FIDELITY = {
    "autoencoder": "Disentanglement Library convolutional architecture reimplemented in PyTorch",
    "beta_vae": "Disentanglement Library beta-VAE architecture, Bernoulli objective, and training budget reimplemented in PyTorch",
    "factor_vae": "paper density-ratio total-correlation objective with reference convolutional architecture reimplemented in PyTorch",
    "beta_tcvae": "paper minibatch-weighted ELBO decomposition with reference convolutional architecture reimplemented in PyTorch",
    "slow_vae": "experimental dashboard approximation of the sparse-transition prior",
    "pcl": "experimental dashboard contrastive adapter; not source-code-identical PCL",
    "weakly_supervised_gan": "experimental dashboard pair-adversarial adapter; not source-code-identical Ada-GVAE/GAN",
}

# Table 1 (MIG), main paper.  These are aggregate targets, never per-run gates.
MIG_TARGETS = {
    "dsprites": {
        "autoencoder": {"original": (.09, .06), "manipulated": (.05, .02), "uniform_noise": (.06, .03)},
        "beta_vae": {"original": (.23, .08), "manipulated": (.07, .09), "uniform_noise": (.14, .07)},
        "factor_vae": {"original": (.27, .11), "manipulated": (.20, .12), "uniform_noise": (.16, .08)},
        "beta_tcvae": {"original": (.25, .08), "manipulated": (.14, .10), "uniform_noise": (.20, .04)},
        "slow_vae": {"original": (.39, .08), "manipulated": (.27, .08), "uniform_noise": (.37, .09)},
        "pcl": {"original": (.21, .03), "manipulated": (.24, .07), "uniform_noise": (.24, .07)},
        "weakly_supervised_gan": {"original": (.45, .05), "manipulated": (.36, .02), "uniform_noise": (.36, .01)},
    },
    "shapes3d": {
        "autoencoder": {"original": (.06, .03), "manipulated": (.05, .03), "uniform_noise": (.07, .03)},
        "beta_vae": {"original": (.60, .31), "manipulated": (.09, .14), "uniform_noise": (.66, .05)},
        "factor_vae": {"original": (.27, .18), "manipulated": (.07, .05), "uniform_noise": (.33, .20)},
        "beta_tcvae": {"original": (.58, .20), "manipulated": (.24, .16), "uniform_noise": (.60, .11)},
        "slow_vae": {"original": (.53, .19), "manipulated": (.13, .08), "uniform_noise": (.60, .10)},
        "pcl": {"original": (.44, .06), "manipulated": (.47, .08), "uniform_noise": (.40, .07)},
        "weakly_supervised_gan": {"original": (.69, .12), "manipulated": (.66, .12), "uniform_noise": (.77, .13)},
    },
}


def paper_protocol(dataset: str, model: str, variant: str = "original", seed: int = 1) -> dict:
    dataset = str(dataset).lower()
    model = str(model).lower()
    variant = str(variant).lower()
    if dataset not in DATASETS:
        raise ValueError("Zietlow reproduction dataset must be dsprites or shapes3d")
    if variant not in {"original", "manipulated", "uniform_noise"}:
        raise ValueError("Dataset variant must be original, manipulated, or uniform_noise")
    primary = PRIMARY_HYPERPARAMETERS.get(dataset, {}).get(model)
    protocol = {
        "paper_id": PAPER["id"], "paper_title": PAPER["title"], "paper_url": PAPER["url"],
        "dataset_source": dataset, "dataset_variant": variant,
        "model_type": model, "latent_dim": DATASETS[dataset]["latent_dim"],
        "seed": int(seed), "paper_reproduction": True, "zietlow_metrics": True,
        "maximum_test_samples": 0, "factor_metric_bins": 20,
        "factor_vote_batches": 10000, "factor_vote_evaluation_batches": 5000,
        "metric_sampling": "reference", "metric_seed": 0,
        "metric_train_samples": 10000, "metric_test_samples": 5000,
        "metric_variance_samples": 10000,
        "training_steps": 300000, "training_steps_source": "Disentanglement Library default",
        "learning_rate": 1e-4, "batch_size": 64, "optimizer": "adam",
        "adam_beta1": .9, "adam_beta2": .999, "optimizer_epsilon": 1e-8,
        "weight_decay": 0.0, "reconstruction_loss": "bernoulli_logits",
        "encoder_activation": "relu", "decoder_activation": "relu", "bias": True,
        "paper_image_model": True,
        "custom_model": {
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
            "encoder_activation": "relu", "decoder_activation": "relu",
            "dropout": 0.0, "layer_norm": False, "posterior": "diagonal",
        },
        "paper_result_scope": "aggregate of 10 distinct training seeds",
        "implementation_fidelity": MODEL_FIDELITY.get(model, "dashboard implementation; fidelity not classified"),
    }
    if model == "factor_vae":
        protocol.update({"beta": 1.0, "tc_weight": primary, "factor_discriminator": True})
    elif model in {"beta_vae", "beta_tcvae", "slow_vae"}:
        protocol.update({"beta": primary})
    return protocol


def make_uniform_noise(observations: np.ndarray, epsilon: float, seed: int) -> tuple[np.ndarray, dict]:
    """Apply the paper's matched iid U[-epsilon, epsilon] pixel control."""
    x = np.asarray(observations, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.uniform(-float(epsilon), float(epsilon), size=x.shape).astype(np.float32)
    changed = np.clip(x + noise, 0.0, 1.0).astype(np.float32)
    delta = changed - x
    return changed, transformation_statistics(x, changed, "uniform_noise", epsilon, seed)


def apply_manipulation(observations: np.ndarray, manipulation: np.ndarray, epsilon: float, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Apply x' = x + epsilon*m(w), with ||m(w)||_infinity <= 1."""
    x = np.asarray(observations, dtype=np.float32)
    m = np.asarray(manipulation, dtype=np.float32)
    if m.shape != x.shape:
        raise ValueError("Manipulation and observations must have identical shapes")
    m = np.clip(m, -1.0, 1.0)
    changed = np.clip(x + float(epsilon) * m, 0.0, 1.0).astype(np.float32)
    return changed, transformation_statistics(x, changed, "manipulated", epsilon, seed)


def transformation_statistics(original: np.ndarray, changed: np.ndarray, kind: str, epsilon: float, seed: int) -> dict:
    delta = np.asarray(changed, dtype=np.float64) - np.asarray(original, dtype=np.float64)
    flattened = delta.reshape(len(delta), -1)
    return {
        "variant": str(kind), "epsilon": float(epsilon), "seed": int(seed),
        "max_abs_change": float(np.abs(delta).max(initial=0.0)),
        "mean_l2_change": float(np.linalg.norm(flattened, axis=1).mean()),
        "mean_absolute_change": float(np.abs(delta).mean()),
        "original_variance": float(np.var(original)), "derived_variance": float(np.var(changed)),
        "factor_labels_changed": False, "sample_order_changed": False,
    }


def _discretize(values: np.ndarray, bins: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values, dtype=np.int32)
    for column in range(values.shape[1]):
        x = values[:, column]
        if np.allclose(x, x[0]):
            continue
        edges = np.histogram(x, bins=int(bins))[1][:-1]
        result[:, column] = np.digitize(x, edges[1:], right=False)
    return result


def mutual_information_gap(latents: np.ndarray, factors: np.ndarray, bins: int = 20) -> dict:
    from sklearn.metrics import mutual_info_score

    z = _discretize(np.asarray(latents), bins)
    w = np.asarray(factors)
    if w.ndim == 1:
        w = w[:, None]
    matrix = np.zeros((z.shape[1], w.shape[1]), dtype=np.float64)
    entropy = np.zeros(w.shape[1], dtype=np.float64)
    for factor in range(w.shape[1]):
        labels = np.unique(w[:, factor], return_inverse=True)[1]
        entropy[factor] = mutual_info_score(labels, labels)
        for coordinate in range(z.shape[1]):
            matrix[coordinate, factor] = mutual_info_score(z[:, coordinate], labels)
    ordered = np.sort(matrix, axis=0)[::-1]
    second = ordered[1] if len(ordered) > 1 else np.zeros(w.shape[1])
    per_factor = np.divide(ordered[0] - second, entropy, out=np.zeros_like(entropy), where=entropy > 1e-12)
    return {
        "score": float(per_factor.mean()), "per_factor": per_factor.tolist(),
        "mutual_information_matrix": matrix.tolist(), "factor_entropy": entropy.tolist(),
        "discretization": "equal-width histogram", "bins": int(bins),
    }


def _entropy(probabilities: np.ndarray, axis: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.divide(p, p.sum(axis=axis, keepdims=True), out=np.zeros_like(p), where=p.sum(axis=axis, keepdims=True) > 0)
    count = p.shape[axis]
    if count <= 1:
        return np.zeros(np.delete(p.shape, axis), dtype=np.float64)
    return -(p * np.log(p + 1e-12)).sum(axis=axis) / math.log(count)


def dci_disentanglement(train_z: np.ndarray, train_factors: np.ndarray, test_z: np.ndarray, test_factors: np.ndarray, seed: int) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier

    train_factors = np.asarray(train_factors)
    test_factors = np.asarray(test_factors)
    if train_factors.ndim == 1:
        train_factors, test_factors = train_factors[:, None], test_factors[:, None]
    importance = np.zeros((train_z.shape[1], train_factors.shape[1]), dtype=np.float64)
    train_accuracy, test_accuracy = [], []
    for factor in range(train_factors.shape[1]):
        y_train = np.unique(train_factors[:, factor], return_inverse=True)[1]
        values = np.unique(train_factors[:, factor])
        mapping = {value: index for index, value in enumerate(values)}
        y_test = np.asarray([mapping.get(value, -1) for value in test_factors[:, factor]])
        valid = y_test >= 0
        if len(np.unique(y_train)) < 2:
            train_accuracy.append(1.0); test_accuracy.append(1.0); continue
        model = GradientBoostingClassifier(random_state=int(seed))
        model.fit(train_z, y_train)
        importance[:, factor] = np.abs(model.feature_importances_)
        train_accuracy.append(float(model.score(train_z, y_train)))
        test_accuracy.append(float(model.score(test_z[valid], y_test[valid])) if np.any(valid) else 0.0)
    coordinate_scores = 1.0 - _entropy(importance, axis=1)
    weights = importance.sum(axis=1)
    score = float(np.dot(coordinate_scores, weights) / weights.sum()) if weights.sum() else 0.0
    return {
        "disentanglement": score, "coordinate_scores": coordinate_scores.tolist(),
        "importance_matrix": importance.tolist(), "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy, "estimator": "GradientBoostingClassifier",
    }


def sap_score(train_z: np.ndarray, train_factors: np.ndarray, test_z: np.ndarray, test_factors: np.ndarray, seed: int) -> dict:
    from sklearn.svm import LinearSVC

    train_factors = np.asarray(train_factors)
    test_factors = np.asarray(test_factors)
    if train_factors.ndim == 1:
        train_factors, test_factors = train_factors[:, None], test_factors[:, None]
    scores = np.zeros((train_z.shape[1], train_factors.shape[1]), dtype=np.float64)
    for coordinate in range(train_z.shape[1]):
        feature_train = train_z[:, coordinate:coordinate + 1]
        feature_test = test_z[:, coordinate:coordinate + 1]
        for factor in range(train_factors.shape[1]):
            values = np.unique(train_factors[:, factor])
            mapping = {value: index for index, value in enumerate(values)}
            y_train = np.asarray([mapping[value] for value in train_factors[:, factor]])
            y_test = np.asarray([mapping.get(value, -1) for value in test_factors[:, factor]])
            valid = y_test >= 0
            if len(values) < 2:
                scores[coordinate, factor] = 1.0
                continue
            classifier = LinearSVC(C=.01, class_weight="balanced", random_state=int(seed), max_iter=5000)
            classifier.fit(feature_train, y_train)
            scores[coordinate, factor] = float(classifier.score(feature_test[valid], y_test[valid])) if np.any(valid) else 0.0
    ordered = np.sort(scores, axis=0)[::-1]
    second = ordered[1] if len(ordered) > 1 else np.zeros(scores.shape[1])
    per_factor = ordered[0] - second
    return {"score": float(per_factor.mean()), "per_factor": per_factor.tolist(), "score_matrix": scores.tolist(), "classifier": "LinearSVC(C=0.01, balanced)"}


def factor_vae_score(latents: np.ndarray, factors: np.ndarray, seed: int, batches: int = 10000,
                     evaluation_batches: int = 5000, batch_size: int = 64,
                     variance_latents: np.ndarray | None = None) -> dict:
    """FactorVAE majority-vote metric with disjoint classifier/evaluation votes."""
    z = np.asarray(latents, dtype=np.float64)
    w = np.asarray(factors)
    if w.ndim == 1:
        w = w[:, None]
    rng = np.random.default_rng(int(seed))
    variance_z = z if variance_latents is None else np.asarray(variance_latents, dtype=np.float64)
    global_variance = np.var(variance_z, axis=0, ddof=1)
    # Reference FactorVAE metric removes globally collapsed coordinates before voting.
    active = np.sqrt(np.maximum(global_variance, 0.0)) >= 0.05
    def generate_votes(count):
        votes = np.zeros((w.shape[1], z.shape[1]), dtype=np.int64)
        usable = 0
        for _ in range(int(count)):
            factor = int(rng.integers(w.shape[1]))
            values = np.unique(w[:, factor])
            value = values[int(rng.integers(len(values)))]
            candidates = np.flatnonzero(w[:, factor] == value)
            if len(candidates) < 2 or not np.any(active):
                continue
            chosen = rng.choice(candidates, size=int(batch_size), replace=len(candidates) < batch_size)
            normalized = np.full(z.shape[1], np.inf)
            normalized[active] = np.var(z[chosen], axis=0, ddof=1)[active] / global_variance[active]
            votes[factor, int(np.argmin(normalized))] += 1
            usable += 1
        return votes, usable

    training_votes, usable_training = generate_votes(batches)
    classifier = training_votes.argmax(axis=0)
    evaluation_votes, usable_evaluation = generate_votes(evaluation_batches)
    correct = sum(
        evaluation_votes[classifier[coordinate], coordinate]
        for coordinate in range(evaluation_votes.shape[1])
    )
    return {
        "score": float(correct / max(1, usable_evaluation)),
        "training_votes": training_votes.tolist(), "evaluation_votes": evaluation_votes.tolist(),
        "training_batches": int(batches), "usable_training_batches": int(usable_training),
        "evaluation_batches": int(evaluation_batches), "usable_evaluation_batches": int(usable_evaluation),
        "batch_size": int(batch_size), "classifier": classifier.tolist(),
        "active_coordinate_mask": active.astype(int).tolist(), "global_std_threshold": 0.05,
        "variance_estimate_samples": int(len(variance_z)),
    }


def active_units(logvar: np.ndarray | None, latent_dim: int, threshold: float = .8) -> dict:
    if logvar is None:
        return {"applicable": False, "count": None, "threshold": float(threshold), "over_pruned": False}
    posterior_variance = np.exp(np.asarray(logvar, dtype=np.float64)).mean(axis=0)
    mask = posterior_variance < float(threshold)
    return {
        "applicable": True, "count": int(mask.sum()), "required": int(latent_dim),
        "posterior_variance_mean": posterior_variance.tolist(), "mask": mask.astype(int).tolist(),
        "threshold": float(threshold), "rule": "E_x[sigma_i^2(x)] < 0.8",
        "over_pruned": bool(mask.sum() < int(latent_dim)),
    }


def compute_metrics(latents: np.ndarray, factors: np.ndarray, split: np.ndarray, seed: int, logvar: np.ndarray | None = None,
                    bins: int = 20, vote_batches: int = 10000, vote_evaluation_batches: int = 5000,
                    sampling: str = "heldout", metric_seed: int | None = None,
                    train_samples: int = 10000, test_samples: int = 5000,
                    variance_samples: int = 10000) -> dict:
    """Compute paper metrics with either dashboard splits or reference-library sampling."""
    z = np.asarray(latents, dtype=np.float64)
    w = np.asarray(factors)
    split = np.asarray(split)
    train_mask = split == 0
    test_mask = split == (2 if np.any(split == 2) else 1)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Zietlow metrics require non-empty train and held-out test partitions")
    sampling = str(sampling).lower()
    evaluation_seed = int(seed if metric_seed is None else metric_seed)
    if sampling == "reference":
        rng = np.random.default_rng(evaluation_seed)
        sample = lambda count: rng.choice(len(z), size=int(count), replace=True)
        train_indices = sample(train_samples)
        test_indices = sample(test_samples)
        mig_indices = sample(train_samples)
        variance_indices = sample(variance_samples)
        factor_z, factor_w = z, w
        evaluation_scope = "reference-library resampling from available ground-truth dataset"
    elif sampling == "heldout":
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        mig_indices = test_indices
        variance_indices = test_indices
        factor_z, factor_w = z[test_indices], w[test_indices]
        evaluation_scope = "held-out test partition"
    else:
        raise ValueError("sampling must be heldout or reference")
    mig = mutual_information_gap(z[mig_indices], w[mig_indices], bins)
    dci = dci_disentanglement(z[train_indices], w[train_indices], z[test_indices], w[test_indices], evaluation_seed)
    sap = sap_score(z[train_indices], w[train_indices], z[test_indices], w[test_indices], evaluation_seed)
    factor_score = factor_vae_score(
        factor_z, factor_w, evaluation_seed, batches=vote_batches,
        evaluation_batches=vote_evaluation_batches,
        variance_latents=z[variance_indices],
    )
    units = active_units(None if logvar is None else np.asarray(logvar)[variance_indices], w.shape[1])
    return {
        "schema_version": 2, "paper_id": PAPER["id"], "evaluation_scope": evaluation_scope,
        "metric_sampling": sampling, "training_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)), "mig_samples": int(len(mig_indices)),
        "variance_samples": int(len(variance_indices)), "seed": evaluation_seed,
        "mig": mig, "dci": dci, "sap": sap, "factor_vae_score": factor_score,
        "active_units": units,
    }


def aggregate_rows(rows: list[dict], bootstrap_seed: int = 2021, bootstrap_samples: int = 10000) -> dict:
    """Aggregate homogeneous run rows by dataset/model/variant."""
    groups: dict[tuple[str, str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["dataset"], row["model"], row["variant"], float(row.get("hyperparameter_scale", 1.0)))
        groups.setdefault(key, []).append(row)
    rng = np.random.default_rng(int(bootstrap_seed))
    output = []
    for (dataset, model, variant, hyperparameter_scale), group in sorted(groups.items()):
        values = np.asarray([float(item["mig"]) for item in group], dtype=np.float64)
        samples = rng.choice(values, size=(int(bootstrap_samples), len(values)), replace=True).mean(axis=1)
        target = MIG_TARGETS.get(dataset, {}).get(model, {}).get(variant) if hyperparameter_scale == 1.0 else None
        paper_comparable = all(bool(item.get("paper_comparable", True)) for item in group)
        interval = np.quantile(samples, [.025, .975])
        supplemental = {}
        for metric in ("dci", "sap", "factor_vae_score"):
            metric_values = np.asarray(
                [float(item[metric]) for item in group if item.get(metric) is not None],
                dtype=np.float64,
            )
            if not len(metric_values):
                continue
            metric_samples = rng.choice(
                metric_values,
                size=(int(bootstrap_samples), len(metric_values)),
                replace=True,
            ).mean(axis=1)
            supplemental[metric] = {
                "mean": float(metric_values.mean()),
                "sample_sd": float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0,
                "bootstrap_ci95": np.quantile(metric_samples, [.025, .975]).tolist(),
                "n": int(len(metric_values)),
            }
        output.append({
            "dataset": dataset, "model": model, "variant": variant,
            "hyperparameter_scale": float(hyperparameter_scale), "n": int(len(values)),
            "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "bootstrap_ci95": interval.tolist(), "paper_mean": None if target is None else target[0],
            "paper_sd": None if target is None else target[1],
            "paper_comparable": paper_comparable,
            "status": (
                "not_applicable" if target is None else
                "pilot_only" if not paper_comparable else
                "pass" if interval[0] <= target[0] <= interval[1] else "fail"
            ),
            "supplemental_metrics": supplemental,
        })
    return {"paper_id": PAPER["id"], "groups": output, "bootstrap_seed": int(bootstrap_seed), "bootstrap_samples": int(bootstrap_samples)}
