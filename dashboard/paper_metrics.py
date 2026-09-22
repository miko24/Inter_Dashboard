"""Exact metrics used by Rolinek, Zietlow, and Martius (CVPR 2019).

These metrics intentionally live beside, rather than replace, the dashboard's
broader interpretability metrics.  In particular, ``paper_dto`` implements
Equation 29 and is not the normalized off-diagonal Gram score historically
shown by the dashboard as "DtO".
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def nearest_signed_permutation(matrix: np.ndarray) -> np.ndarray:
    """Return the L1-nearest signed permutation matrix.

    For an orthogonal matrix every entry is in [-1, 1].  Relative to the
    all-zero baseline, selecting entry (i, j) changes the L1 objective by
    ``1 - 2 * abs(V[i, j])``.  The paper's MILP therefore reduces exactly to
    a maximum-weight bipartite assignment.
    """

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("V must be a square matrix")
    rows, columns = linear_sum_assignment(-np.abs(value))
    permutation = np.zeros_like(value)
    signs = np.sign(value[rows, columns])
    signs[signs == 0] = 1
    permutation[rows, columns] = signs
    return permutation


def paper_dto_from_jacobian(jacobian: np.ndarray) -> float:
    """Equation 29's per-sample distance to orthogonality."""

    # Equation 29 compares the complete d x d right-singular-vector matrix.
    _, _, vh = np.linalg.svd(np.asarray(jacobian, dtype=np.float64), full_matrices=True)
    v = vh.T
    return float(np.linalg.norm(v - nearest_signed_permutation(v), ord="fro"))


def _metric_kinds(factors: np.ndarray, supplied: Iterable[str] | None) -> list[str]:
    if supplied is not None:
        kinds = [str(item).lower() for item in supplied]
        if len(kinds) == factors.shape[1] and all(item in {"continuous", "categorical"} for item in kinds):
            return kinds
    # Conservative fallback: only low-cardinality integer labels are treated
    # as categorical.  Explicit metadata is preferred for paper runs.
    result = []
    for column in factors.T:
        unique = np.unique(column)
        integer_like = np.allclose(unique, np.round(unique))
        result.append("categorical" if integer_like and len(unique) <= 10 else "continuous")
    return result


def paper_disentanglement_score(
    latents: np.ndarray,
    factors: np.ndarray,
    factor_metric_types: Iterable[str] | None = None,
    seed: int = 42,
) -> dict:
    """Implement Equations 65-70 with one k=5 probe per latent coordinate."""

    z = np.asarray(latents, dtype=np.float64)
    w = np.asarray(factors)
    if z.ndim != 2 or w.ndim != 2 or len(z) != len(w):
        raise ValueError("latents and factors must be aligned two-dimensional arrays")
    if len(z) < 10:
        raise ValueError("at least 10 held-out samples are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(z))
    cut = max(5, min(len(z) - 1, int(0.8 * len(z))))
    train, test = order[:cut], order[cut:]
    kinds = _metric_kinds(w, factor_metric_types)
    sensitivities = np.zeros((w.shape[1], z.shape[1]), dtype=np.float64)
    normalizers = np.zeros(w.shape[1], dtype=np.float64)
    factor_scores = np.zeros(w.shape[1], dtype=np.float64)

    for factor_index, kind in enumerate(kinds):
        target = w[:, factor_index]
        if kind == "categorical":
            values, counts = np.unique(target[train], return_counts=True)
            constant = values[int(np.argmax(counts))]
            normalizer = float(np.mean(target[test] == constant))
            for latent_index in range(z.shape[1]):
                probe = KNeighborsClassifier(n_neighbors=5)
                probe.fit(z[train, latent_index, None], target[train])
                sensitivities[factor_index, latent_index] = max(0.0, float(
                    probe.score(z[test, latent_index, None], target[test])
                ))
        else:
            normalizer = float(np.sqrt(np.var(target[test])))
            for latent_index in range(z.shape[1]):
                probe = KNeighborsRegressor(n_neighbors=5)
                probe.fit(z[train, latent_index, None], target[train])
                prediction = probe.predict(z[test, latent_index, None])
                rmse = float(np.sqrt(np.mean((target[test] - prediction) ** 2)))
                # A coordinate that is worse than the constant regressor has
                # zero useful sensitivity. This preserves the paper's stated
                # [0, 1] score range on finite held-out samples.
                sensitivities[factor_index, latent_index] = max(0.0, normalizer - rmse)
        normalizers[factor_index] = normalizer
        ordered = np.sort(sensitivities[factor_index])
        best = ordered[-1]
        second = ordered[-2] if len(ordered) > 1 else 0.0
        factor_scores[factor_index] = np.clip((best - second) / max(normalizer, 1e-12), 0.0, 1.0)

    return {
        "score": float(np.mean(factor_scores)),
        "factor_scores": factor_scores.tolist(),
        "sensitivities": sensitivities.tolist(),
        "normalizers": normalizers.tolist(),
        "factor_metric_types": kinds,
        "probe_train_samples": int(len(train)),
        "probe_test_samples": int(len(test)),
        "neighbors": 5,
        "seed": int(seed),
    }


def polarization_snapshot(mu: np.ndarray, logvar: np.ndarray) -> dict:
    """Return the paper's implemented and polarized KL values and Delta_KL."""

    means = np.asarray(mu, dtype=np.float64)
    logs = np.asarray(logvar, dtype=np.float64)
    if means.shape != logs.shape or means.ndim != 2:
        raise ValueError("mu and logvar must be aligned [samples, latent] arrays")
    # Equation 30 selects active variables with sqrt(var(mu_j(x_i))) > 0.5.
    # Comparing the variance itself to 0.5 would incorrectly use a threshold
    # of sqrt(0.5) on the reported standard-deviation scale.
    latent_standard_deviation = np.sqrt(np.var(means, axis=0))
    active = latent_standard_deviation > 0.5
    exact_per_sample = 0.5 * np.sum(means**2 + np.exp(logs) - logs - 1.0, axis=1)
    if np.any(active):
        approximate_per_sample = 0.5 * np.sum(means[:, active] ** 2 - logs[:, active] - 1.0, axis=1)
    else:
        approximate_per_sample = np.zeros(len(means), dtype=np.float64)
    exact = float(np.mean(exact_per_sample))
    approximate = float(np.mean(approximate_per_sample))
    delta = abs(exact - approximate) / max(abs(exact), 1e-12)
    return {
        "kl": exact,
        "approximate_kl": approximate,
        "delta_kl": float(delta),
        "active_mask": active.astype(bool).tolist(),
        "active_dimensions": int(np.sum(active)),
        "latent_standard_deviation": latent_standard_deviation.tolist(),
        "active_standard_deviation_threshold": 0.5,
        "active_selection_rule": "sqrt(var(mu_j(x_i))) > 0.5",
    }


def _encode(model, observations: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    means, logs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[start:start + batch_size]).float()
            mu, logvar = model.encode(batch)
            means.append(mu.cpu().numpy())
            logs.append(logvar.cpu().numpy())
    return np.concatenate(means), np.concatenate(logs)


def _decoder_jacobian_batches(model, latents: np.ndarray, batch_size: int):
    """Yield decoder Jacobians without retaining the complete image-scale tensor."""

    model.eval()
    for start in range(0, len(latents), batch_size):
        batch = torch.tensor(latents[start:start + batch_size], dtype=torch.float32)
        try:
            from torch.func import jacrev, vmap

            def decode_one(value):
                return model.decoder(value[None])[0]

            yield vmap(jacrev(decode_one))(batch).detach().cpu().numpy()
        except (ImportError, RuntimeError, NotImplementedError):
            rows = []
            for value in batch:
                value = value.detach().requires_grad_(True)
                jacobian = torch.autograd.functional.jacobian(
                    lambda item: model.decoder(item[None])[0], value, vectorize=True
                )
                rows.append(jacobian.detach().cpu().numpy())
            yield np.stack(rows)


def run_paper_reproduction_metrics(model, data: dict, config: dict | None = None) -> dict:
    """Compute exact paper metrics on the held-out test split.

    ``maximum_test_samples=0`` means the complete test split.  A positive value
    is an explicitly labeled pilot/subsample and must not be used for final
    reproduction tables.
    """

    config = config or {}
    split = np.asarray(data["split"])
    test_label = 2 if np.any(split == 2) else 1
    indices = np.flatnonzero(split == test_label)
    if not len(indices):
        raise ValueError("the dataset has no held-out test samples")
    maximum = max(0, int(config.get("maximum_test_samples", 0)))
    seed = int(config.get("seed", 42))
    if maximum and len(indices) > maximum:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=maximum, replace=False))
    observations = np.asarray(data["observations"])[indices].astype(np.float32, copy=False)
    factors = np.asarray(data["factor_values"])[indices]
    requested_encode_batch = max(8, int(config.get("encode_batch_size", 512)))
    adaptive_encode_batch = max(8, 1_000_000 // max(1, observations.shape[1]))
    encode_batch = min(requested_encode_batch, adaptive_encode_batch)
    jacobian_batch = max(1, int(config.get("jacobian_batch_size", 8)))
    latents, logvar = _encode(model, observations, encode_batch)

    dto_values = []
    for jacobians in _decoder_jacobian_batches(model, latents, jacobian_batch):
        dto_values.extend(paper_dto_from_jacobian(item) for item in jacobians)
    dto_values = np.asarray(dto_values, dtype=np.float64)

    metric_types = None
    if "factor_metric_types" in data:
        metric_types = [str(item) for item in np.asarray(data["factor_metric_types"]).tolist()]
    dataset_source = str(config.get("dataset_source", "")).lower()
    if dataset_source in {"mnist", "fashion_mnist", "celeba"}:
        disentanglement = {
            "score": None,
            "not_applicable_reason": "The paper does not treat class/attribute labels as the complete generating factors for this dataset.",
            "factor_metric_types": metric_types or [],
        }
    else:
        disentanglement = paper_disentanglement_score(latents, factors, metric_types, seed)
    polarization = polarization_snapshot(latents, logvar) if getattr(model, "variational", False) and not getattr(model, "full_covariance", False) else None
    return {
        "definition": "Rolinek et al. CVPR 2019, Equations 29 and 65-70",
        "evaluation_scope": "full_test" if maximum == 0 else "seeded_test_subsample",
        "test_split_label": int(test_label),
        "test_samples": int(len(indices)),
        "test_indices": indices.astype(int).tolist(),
        "dto": {
            "mean": float(np.mean(dto_values)),
            "sample_standard_deviation": float(np.std(dto_values, ddof=1)) if len(dto_values) > 1 else 0.0,
            "minimum": float(np.min(dto_values)),
            "maximum": float(np.max(dto_values)),
            "values": dto_values.tolist(),
        },
        "disentanglement": disentanglement,
        "polarization_final": polarization,
        "mean_posterior_sigma": np.mean(np.exp(0.5 * logvar), axis=0).tolist() if getattr(model, "variational", False) else [],
        "seed": seed,
        "seed_source": str(config.get("seed_source", "effective_training_configuration")),
    }
