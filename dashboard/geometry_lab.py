"""Reproducible manifold experiments for the dashboard Geometry Lab.

This module is intentionally isolated from the legacy superposition trainer.  It
owns dataset generation, the small model library, asynchronous training jobs,
and exports used by the Data Geometry/Training/Explorer/Traversal screens.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import stat
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import jsonify, request, send_file

from concept_catalog import CONCEPTS
from metrics_engine import run_scientific_analysis
from paper_metrics import polarization_snapshot
from paper_requirements import evaluate_paper_requirements, model_family, paper_reproduction_spec
from zietlow_reproduction import (
    DATASETS as ZIETLOW_DATASETS,
    MIG_TARGETS as ZIETLOW_MIG_TARGETS,
    PAPER as ZIETLOW_PAPER,
    PRIMARY_HYPERPARAMETERS as ZIETLOW_HYPERPARAMETERS,
    aggregate_rows as aggregate_zietlow_rows,
    apply_manipulation as apply_zietlow_manipulation,
    compute_metrics as compute_zietlow_metrics,
    make_uniform_noise as make_zietlow_uniform_noise,
    paper_protocol as zietlow_paper_protocol,
)


FACTOR_SPECS = {
    "linear": {"label": "Linear factor", "intrinsic": 1, "embedding": 1},
    "circle": {"label": "Circle S1", "intrinsic": 1, "embedding": 2},
    "sphere": {"label": "Sphere S2", "intrinsic": 2, "embedding": 3},
    "torus": {"label": "Torus S1 x S1", "intrinsic": 2, "embedding": 4},
    "swiss_roll": {"label": "Swiss roll", "intrinsic": 2, "embedding": 3},
    "dsprites": {"label": "dSprites-style", "intrinsic": 1, "embedding": 3},
}

MODEL_LABELS = {
    "autoencoder": "Zietlow convolutional autoencoder",
    "linear_ae": "Linear Autoencoder",
    "nonlinear_ae": "Nonlinear Autoencoder",
    "vae": "Standard VAE",
    "beta_vae": "beta-VAE",
    "beta_vae_full_cov": "beta-VAE — full covariance posterior",
    "conv_beta_vae": "Convolutional beta-VAE",
    "random_decoder": "Random untrained encoder/decoder control",
    "sparse_ae": "Sparse Autoencoder",
    "factor_vae": "FactorVAE",
    "beta_tcvae": "beta-TCVAE",
    "slow_vae": "SlowVAE",
    "pcl": "Permutation Contrastive Learning",
    "weakly_supervised_gan": "Weakly supervised GAN — full sharing",
    "hyperspherical_vae": "Hyperspherical VAE",
    "riemannian_vae": "Geometry-aware / Riemannian VAE",
    "custom_ae": "Custom declarative Autoencoder",
    "custom_vae": "Custom declarative VAE",
}

DATASET_CATALOG = {
    "synthetic_linear": {"label": "Synthetic Linear", "kind": "generated", "factors": True, "official_format": "generated NPZ"},
    "synthetic_nonlinear": {"label": "Synthetic NonLinear", "kind": "generated", "factors": True, "official_format": "generated NPZ"},
    "paper_linear": {"label": "Rolinek 2019 Synthetic Linear", "kind": "generated", "factors": True, "official_format": "exact 2D-to-3D paper generator"},
    "paper_nonlinear": {"label": "Rolinek 2019 Synthetic Nonlinear", "kind": "generated", "factors": True, "official_format": "seeded 2-10-6 tanh generator"},
    "dsprites": {
        "label": "dSprites", "kind": "official_fetch", "factors": True, "format": ".npz",
        "official_format": "native NPZ", "native_npz": True,
        "official_source": "https://github.com/google-deepmind/dsprites-dataset",
        "official_file": "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
    },
    "shapes3d": {
        "label": "Shapes3D", "kind": "official_fetch", "factors": True, "format": ".h5 or .npz",
        "official_format": "native HDF5", "native_npz": False,
        "official_source": "https://github.com/google-deepmind/3d-shapes",
        "official_file": "https://storage.googleapis.com/3d-shapes/3dshapes.h5",
    },
    "mnist": {
        "label": "MNIST", "kind": "official_fetch", "factors": False,
        "official_format": "IDX gzip → standardized NPZ", "native_npz": False,
        "official_source": "https://yann.lecun.com/exdb/mnist/",
    },
    "fashion_mnist": {
        "label": "Fashion-MNIST", "kind": "official_fetch", "factors": False,
        "official_format": "IDX gzip → standardized NPZ", "native_npz": False,
        "official_source": "https://github.com/zalandoresearch/fashion-mnist",
    },
    "celeba": {
        "label": "CelebA", "kind": "official_fetch", "factors": True,
        "official_format": "image/annotation archives → standardized NPZ", "native_npz": False,
        "official_source": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
    },
    "uploaded": {"label": "My dataset", "kind": "upload", "factors": "optional", "format": ".npz or .csv"},
}

ANALYSIS_IDS = {
    "representation", "decoder_geometry", "precision_polarization", "tangent_metrics",
    "topology_folding", "factor_probes", "causal_interference", "latent_explorer", "factor_traversal",
}
SCIENTIFIC_ANALYSIS_IDS = ANALYSIS_IDS - {"latent_explorer", "factor_traversal"}
DEFAULT_ANALYSIS_PLAN = ["representation", "decoder_geometry", "precision_polarization", "tangent_metrics", "latent_explorer"]

PAPER_BATCH_SIZE_ASSUMPTION = (
    "The paper does not specify batch size. This value is a declared dashboard "
    "reproduction assumption and must be included in sensitivity analysis."
)

PAPER_LINEAR_DATASET_PROTOCOL = {
    "factor_domain": "[0, 1]^2",
    "stretch_factor": 2.0,
    "stretched_factor": 1,
    "embedding": "R^2 -> R^3 with an appended zero coordinate",
    "rotation_degrees": 45.0,
    "rotation_axis": [1.0, -1.0, 1.0],
    "transformation_order": ["stretch", "embed", "rotate"],
    "paper_location": "Supplement B.5",
}

PAPER_NONLINEAR_DATASET_PROTOCOL = {
    "factor_domain": "[0, 1]^2",
    "mapping": "R^2 -> R^6 random multilayer perceptron",
    "hidden_layers": [10],
    "hidden_activation": "tanh",
    "output_activation": "linear",
    "biases": True,
    "implementation_initialization": {
        "first_layer_weights": "Normal(0, 1/sqrt(2))",
        "first_layer_biases": "Normal(0, 0.15)",
        "second_layer_weights": "Normal(0, 1/sqrt(10))",
        "second_layer_biases": "Normal(0, 0.15)",
        "seed_source": "dataset seed",
    },
    "saved_generator_tensors": ["nonlinear_w1", "nonlinear_b1", "nonlinear_w2", "nonlinear_b2"],
    "paper_location": "Supplement B.5",
}


def _analysis_plan(value):
    supplied = value if isinstance(value, list) else DEFAULT_ANALYSIS_PLAN
    return list(dict.fromkeys(str(item) for item in supplied if str(item) in ANALYSIS_IDS))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _safe_id(raw: str) -> str:
    value = "".join(c for c in str(raw) if c.isalnum() or c in "-_")
    if not value or value != raw:
        raise ValueError("Invalid experiment identifier")
    return value


def _yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _to_yaml(value, indent=0):
    pad = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(value)}"


def _factor_types(config):
    k = max(1, min(32, int(config.get("k", 3))))
    dataset_type = str(config.get("dataset_type", "circle"))
    supplied = config.get("factor_types") or []
    if dataset_type == "mixed":
        types = [str(x) for x in supplied]
        if not types:
            types = ["linear", "circle"]
        types = (types * math.ceil(k / len(types)))[:k]
    elif dataset_type == "multiple_circles":
        types = ["circle"] * k
    else:
        normalized = dataset_type if dataset_type in FACTOR_SPECS else "linear"
        types = [normalized] * k
    if any(t not in FACTOR_SPECS for t in types):
        raise ValueError("Unknown factor type")
    return types


def _embed_factor(kind, params):
    """Map intrinsic coordinates (n, 2) to a Euclidean factor embedding."""
    u, v = params[:, 0], params[:, 1]
    if kind == "linear":
        return u[:, None]
    if kind == "circle":
        return np.column_stack((np.cos(u), np.sin(u)))
    if kind == "sphere":
        return np.column_stack((np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)))
    if kind == "torus":
        return np.column_stack((np.cos(u), np.sin(u), np.cos(v), np.sin(v))) / math.sqrt(2)
    if kind == "swiss_roll":
        radius = u / (4 * math.pi)
        return np.column_stack((radius * np.cos(u), v, radius * np.sin(u)))
    # A compact, continuous dSprites-style pose embedding: position, scale/rotation.
    return np.column_stack((u, np.sin(math.pi * u), np.cos(math.pi * u)))


def _sample_params(kind, n, rng):
    out = np.zeros((n, 2), dtype=np.float32)
    if kind == "linear":
        out[:, 0] = rng.uniform(-1, 1, n)
    elif kind == "circle":
        out[:, 0] = rng.uniform(0, 2 * math.pi, n)
    elif kind == "sphere":
        out[:, 0] = np.arccos(rng.uniform(-1, 1, n))
        out[:, 1] = rng.uniform(0, 2 * math.pi, n)
    elif kind == "torus":
        out[:, :] = rng.uniform(0, 2 * math.pi, (n, 2))
    elif kind == "swiss_roll":
        out[:, 0] = rng.uniform(1.5 * math.pi, 4.5 * math.pi, n)
        out[:, 1] = rng.uniform(-1, 1, n)
    else:
        out[:, 0] = rng.uniform(-1, 1, n)
    return out


def _three_way_split(n, rng, train_fraction=0.8, validation_fraction=0.1):
    train_fraction = float(np.clip(train_fraction, 0.5, 0.9))
    validation_fraction = float(np.clip(validation_fraction, 0.05, 0.25))
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a non-empty test split")
    fractions = np.asarray(
        [train_fraction, validation_fraction, 1.0 - train_fraction - validation_fraction],
        dtype=np.float64,
    )
    exact = fractions * int(n)
    counts = np.floor(exact).astype(int)
    # Largest-remainder allocation is immune to values such as
    # 0.09999999999999998 and guarantees that all samples are assigned.
    remaining = int(n) - int(counts.sum())
    order_by_remainder = np.argsort(-(exact - counts), kind="stable")
    counts[order_by_remainder[:remaining]] += 1
    order = rng.permutation(n)
    train_end = int(counts[0])
    validation_end = train_end + int(counts[1])
    labels = np.full(n, 2, dtype=np.uint8)
    labels[order[:train_end]] = 0
    labels[order[train_end:validation_end]] = 1
    return labels


def _validated_image_shape(raw_shape, input_dim):
    """Return a channel-first image shape only when it matches flattened data."""
    if raw_shape is None:
        return []
    shape = [int(item) for item in np.asarray(raw_shape).reshape(-1).tolist()]
    if len(shape) == 2:
        shape = [1, *shape]
    if len(shape) != 3 or any(item <= 0 for item in shape):
        return []
    return shape if int(np.prod(shape)) == int(input_dim) else []


def _axis_angle_rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross


def _pca(points, dimensions=3):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.empty((0, dimensions), dtype=np.float32), []
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    dims = min(dimensions, vt.shape[0])
    projected = centered @ vt[:dims].T
    if dims < dimensions:
        projected = np.pad(projected, ((0, 0), (0, dimensions - dims)))
    variance = singular ** 2
    ratio = variance / max(float(variance.sum()), 1e-12)
    return projected.astype(np.float32), ratio[:dimensions].tolist()


def generate_dataset(config):
    seed = int(config.get("seed", 42))
    rng = np.random.default_rng(seed)
    n = max(100, min(200000, int(config.get("dataset_size", 5000))))
    p = float(np.clip(config.get("sparsity", 0.35), 0.0, 1.0))
    noise = max(0.0, float(config.get("noise", 0.02)))
    generator_type = str(config.get("generator_type", "linear"))
    if generator_type in {"paper_linear", "paper_nonlinear"}:
        factor_types = ["linear", "linear"]
        k = 2
        active = np.ones((n, k), dtype=np.float32)
        raw_factors = rng.uniform(0, 1, (n, k)).astype(np.float32)
        params = np.zeros((n, k, 2), dtype=np.float32)
        params[:, :, 0] = raw_factors
        factors = raw_factors.copy()
        component_slices = [[0, 1], [1, 2]]
        observation_dim = 3 if generator_type == "paper_linear" else 6
        extra = {}
        if generator_type == "paper_linear":
            rotation = _axis_angle_rotation((1, -1, 1), math.pi / 4).astype(np.float32)
            mixing = rotation @ np.asarray([[2, 0], [0, 1], [0, 0]], dtype=np.float32)
            observations = factors @ mixing.T
            extra.update({
                "paper_rotation_matrix": rotation,
                "paper_stretch_factor": np.asarray([2.0], dtype=np.float32),
                "paper_embedding_dimensions": np.asarray([2, 3], dtype=np.int16),
                "paper_rotation_degrees": np.asarray([45.0], dtype=np.float32),
                "paper_rotation_axis": np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
            })
        else:
            w1 = rng.normal(0, 1 / math.sqrt(2), (10, 2)).astype(np.float32)
            b1 = rng.normal(0, 0.15, 10).astype(np.float32)
            w2 = rng.normal(0, 1 / math.sqrt(10), (6, 10)).astype(np.float32)
            b2 = rng.normal(0, 0.15, 6).astype(np.float32)
            observations = np.tanh(factors @ w1.T + b1) @ w2.T + b2
            mixing = np.zeros((6, 2), dtype=np.float32)
            extra = {"nonlinear_w1": w1, "nonlinear_b1": b1, "nonlinear_w2": w2, "nonlinear_b2": b2}
        if noise:
            observations += rng.normal(0, noise, observations.shape).astype(np.float32)
        split_labels = _three_way_split(n, rng, config.get("train_split", 0.8), config.get("validation_split", 0.1))
        return {
            "observations": observations.astype(np.float32), "factor_parameters": params,
            "factor_values": raw_factors, "factor_active": active,
            "factor_embeddings": factors, "mixing_matrix": mixing,
            "split": split_labels, "factor_types": np.asarray(factor_types),
            "factor_metric_types": np.asarray(["continuous", "continuous"]),
            "component_slices": np.asarray(component_slices, dtype=np.int32),
            "generator_type": np.asarray([generator_type]), "degeneracy": np.asarray([1.0], dtype=np.float32),
            **extra,
        }

    factor_types = _factor_types(config)
    k = len(factor_types)

    active = rng.binomial(1, p, size=(n, k)).astype(np.float32)

    params = np.zeros((n, k, 2), dtype=np.float32)
    embeddings = []
    component_slices = []
    start = 0
    for index, kind in enumerate(factor_types):
        params[:, index] = _sample_params(kind, n, rng)
        embedded = _embed_factor(kind, params[:, index]).astype(np.float32)
        embedded *= active[:, index:index + 1]
        embeddings.append(embedded)
        component_slices.append([start, start + embedded.shape[1]])
        start += embedded.shape[1]
    factors = np.concatenate(embeddings, axis=1)

    observation_dim = max(2, min(512, int(config.get("observation_dim", max(8, factors.shape[1])))))
    supplied_matrix = config.get("mixing_matrix")
    if supplied_matrix is not None:
        mixing = np.asarray(supplied_matrix, dtype=np.float32)
        if mixing.shape != (observation_dim, factors.shape[1]):
            raise ValueError(f"Mixing matrix must have shape [{observation_dim}, {factors.shape[1]}]")
    else:
        mixing = rng.normal(0, 1 / math.sqrt(max(1, factors.shape[1])), (observation_dim, factors.shape[1])).astype(np.float32)
        if bool(config.get("orthogonal_mixing", True)):
            if observation_dim >= factors.shape[1]:
                mixing, _ = np.linalg.qr(mixing, mode="reduced")  # orthonormal columns
            else:
                rows, _ = np.linalg.qr(mixing.T, mode="reduced")  # orthonormal rows
                mixing = rows.T
            mixing = mixing.astype(np.float32)

    degeneracy = float(np.clip(config.get("degeneracy", 1.0), 1e-6, 1.0))
    if degeneracy < 1 and supplied_matrix is None:
        left, singular, right = np.linalg.svd(mixing, full_matrices=False)
        profile = np.geomspace(1.0, degeneracy, len(singular))
        mixing = ((left * profile[None, :]) @ right).astype(np.float32)

    extra = {}
    if generator_type == "nonlinear":
        nonlinear_hidden = max(observation_dim, min(512, factors.shape[1] * 3))
        w1 = rng.normal(0, 1 / math.sqrt(max(1, factors.shape[1])), (nonlinear_hidden, factors.shape[1])).astype(np.float32)
        b1 = rng.normal(0, .15, nonlinear_hidden).astype(np.float32)
        w2 = rng.normal(0, 1 / math.sqrt(nonlinear_hidden), (observation_dim, nonlinear_hidden)).astype(np.float32)
        if degeneracy < 1:
            left, singular, right = np.linalg.svd(w2, full_matrices=False)
            profile = np.geomspace(1.0, degeneracy, len(singular))
            w2 = ((left * profile[None, :]) @ right).astype(np.float32)
        observations = np.tanh(factors @ w1.T + b1) @ w2.T
        extra = {"nonlinear_w1": w1, "nonlinear_b1": b1, "nonlinear_w2": w2}
    else:
        observations = factors @ mixing.T
    if noise:
        observations += rng.normal(0, noise, observations.shape).astype(np.float32)

    split_labels = _three_way_split(n, rng, config.get("train_split", 0.8), config.get("validation_split", 0.1))

    return {
        "observations": observations.astype(np.float32),
        "factor_parameters": params,
        "factor_values": params[:, :, 0],
        "factor_active": active,
        "factor_embeddings": factors.astype(np.float32),
        "mixing_matrix": mixing.astype(np.float32),
        "split": split_labels,
        "factor_types": np.asarray(factor_types),
        "factor_metric_types": np.asarray(["continuous"] * k),
        "component_slices": np.asarray(component_slices, dtype=np.int32),
        "generator_type": np.asarray([generator_type]),
        "degeneracy": np.asarray([degeneracy], dtype=np.float32),
        **extra,
    }


def _dataset_summary(data):
    x = data["observations"]
    active = data["factor_active"]
    params = data["factor_parameters"]
    factor_types = data["factor_types"].tolist()
    sample_count = min(900, len(x))
    indices = np.linspace(0, len(x) - 1, sample_count, dtype=int)
    # Dataset cards only need a stable preview.  Exact SVD on image benchmarks
    # (10k x 12,288 for Shapes3D) is unnecessarily expensive and can make an
    # otherwise completed import appear hung.  Sketch rows and features
    # deterministically; the original full-resolution observations remain the
    # immutable training artifact.
    feature_count = min(512, x.shape[1])
    feature_indices = np.linspace(0, x.shape[1] - 1, feature_count, dtype=int)
    sketch = np.asarray(x[np.ix_(indices, feature_indices)], dtype=np.float32)
    centered = sketch - sketch.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(3, vt.shape[0])
    projection = centered @ vt[:dimensions].T
    if dimensions < 3:
        projection = np.pad(projection, ((0, 0), (0, 3 - dimensions)))
    eigen = singular ** 2
    explained = eigen / max(float(eigen.sum()), 1e-12)
    covariance = np.cov(params[:, :, 0], rowvar=False)
    correlation = np.corrcoef(params[:, :, 0], rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[covariance]])
        correlation = np.asarray([[1.0]])
    participation = float(eigen.sum() ** 2 / max(float((eigen ** 2).sum()), 1e-12))
    known_intrinsic = int(sum(FACTOR_SPECS[t]["intrinsic"] for t in factor_types))
    split = np.asarray(data.get("split", np.zeros(len(x), dtype=np.uint8)))
    return {
        "sample_count": int(len(x)),
        "observation_dim": int(x.shape[1]),
        "factor_count": int(active.shape[1]),
        "factor_types": factor_types,
        "known_intrinsic_dimension": known_intrinsic,
        "empirical_dimension": round(participation, 3),
        "mean_active_factors": round(float(active.sum(1).mean()), 3),
        "inactive_rate": (1 - active.mean(0)).round(4).tolist(),
        "covariance": np.nan_to_num(covariance).round(5).tolist(),
        "correlation": np.nan_to_num(correlation).round(5).tolist(),
        "projection": projection.round(6).tolist(),
        "factor_values": params[indices, :, 0].round(6).tolist(),
        "factor_active": active[indices].astype(int).tolist(),
        "pca_explained_variance": [round(float(v), 5) for v in explained[:3]],
        "dimension_estimator": {
            "method": "deterministic row/feature sketch SVD",
            "rows": int(sample_count),
            "features": int(feature_count),
            "exact": bool(sample_count == len(x) and feature_count == x.shape[1]),
        },
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)) if np.any(split == 2) else 0,
            "test": int(np.sum(split == 2)) if np.any(split == 2) else int(np.sum(split == 1)),
        },
    }


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activation, linear=False,
                 hidden_layers=None, dropout=0.0, layer_norm=False, bias=True):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        act = acts.get(activation, nn.GELU)
        if linear or (depth <= 0 and not hidden_layers):
            self.net = nn.Linear(input_dim, output_dim, bias=bool(bias))
        else:
            layers = []
            current = input_dim
            widths = hidden_layers or [hidden_dim] * depth
            for width in widths:
                width = int(width)
                layers.append(nn.Linear(current, width, bias=bool(bias)))
                if layer_norm:
                    layers.append(nn.LayerNorm(width))
                layers.append(act())
                if dropout:
                    layers.append(nn.Dropout(float(dropout)))
                current = width
            layers.append(nn.Linear(current, output_dim, bias=bool(bias)))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _conv_output_size(size, kernel, stride, padding):
    return math.floor((int(size) + 2 * int(padding) - int(kernel)) / int(stride) + 1)


class ConvEncoder(nn.Module):
    def __init__(self, image_shape, output_dim, channels, activation="relu", bias=True,
                 kernel_sizes=None, strides=None, paddings=None, dense_layers=None):
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("Convolutional models require image_shape=[channels,height,width]")
        depth = len(channels)
        kernel_sizes = list(kernel_sizes or [4] * depth)
        strides = list(strides or [2] * depth)
        paddings = list(paddings or [1] * depth)
        if not (len(kernel_sizes) == len(strides) == len(paddings) == depth):
            raise ValueError("Encoder channels, kernels, strides, and paddings must have equal lengths")
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        act = acts.get(activation, nn.ReLU)
        layers = []
        current = int(image_shape[0])
        height, width_px = int(image_shape[1]), int(image_shape[2])
        for width, kernel, stride, padding in zip(channels, kernel_sizes, strides, paddings):
            layers.extend((nn.Conv2d(current, int(width), kernel_size=int(kernel), stride=int(stride),
                                     padding=int(padding), bias=bool(bias)), act()))
            current = int(width)
            height = _conv_output_size(height, kernel, stride, padding)
            width_px = _conv_output_size(width_px, kernel, stride, padding)
            if height < 1 or width_px < 1:
                raise ValueError("Encoder convolution settings collapse the image below one pixel")
        self.features = nn.Sequential(*layers)
        self.feature_shape = (current, height, width_px)
        head = []
        head_input = int(np.prod(self.feature_shape))
        for dense_width in dense_layers or []:
            head.extend((nn.Linear(head_input, int(dense_width), bias=bool(bias)), act()))
            head_input = int(dense_width)
        head.append(nn.Linear(head_input, output_dim, bias=bool(bias)))
        self.head = nn.Sequential(*head)
        self.image_shape = tuple(int(item) for item in image_shape)

    def forward(self, x):
        image = x.reshape((-1,) + self.image_shape)
        return self.head(self.features(image).flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, image_shape, base_shape, channels, activation="relu", bias=True,
                 kernel_sizes=None, strides=None, paddings=None, dense_layers=None):
        super().__init__()
        acts = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}
        act = acts.get(activation, nn.ReLU)
        self.feature_shape = tuple(int(item) for item in base_shape)
        depth = len(channels) + 1
        kernel_sizes = list(kernel_sizes or [4] * depth)
        strides = list(strides or [2] * depth)
        paddings = list(paddings or [1] * depth)
        if not (len(kernel_sizes) == len(strides) == len(paddings) == depth):
            raise ValueError("Decoder kernels, strides, and paddings must match the number of transpose-convolution layers")
        connect = []
        connect_input = int(latent_dim)
        for dense_width in dense_layers or []:
            connect.extend((nn.Linear(connect_input, int(dense_width), bias=bool(bias)), act()))
            connect_input = int(dense_width)
        connect.extend((nn.Linear(connect_input, int(np.prod(self.feature_shape)), bias=bool(bias)), act()))
        self.connect = nn.Sequential(*connect)
        layers = []
        source = self.feature_shape[0]
        targets = [int(item) for item in channels] + [int(image_shape[0])]
        for index, (target, kernel, stride, padding) in enumerate(zip(targets, kernel_sizes, strides, paddings)):
            layers.append(nn.ConvTranspose2d(source, target, kernel_size=int(kernel), stride=int(stride),
                                             padding=int(padding), bias=bool(bias)))
            if index + 1 < len(targets):
                layers.append(act())
            source = target
        self.images = nn.Sequential(*layers)
        self.image_shape = tuple(int(item) for item in image_shape)

    def forward(self, z):
        features = self.connect(z).reshape((-1,) + self.feature_shape)
        images = self.images(features)
        if tuple(images.shape[1:]) != self.image_shape:
            raise ValueError(f"Decoder produced {tuple(images.shape[1:])}, expected {self.image_shape}")
        return images.flatten(1)


class FactorDiscriminator(nn.Module):
    """FactorVAE density-ratio discriminator from joint/permuted latents."""

    def __init__(self, latent_dim, width=1000, depth=6):
        super().__init__()
        layers = []
        size = int(latent_dim)
        for _ in range(int(depth)):
            layers.extend([nn.Linear(size, int(width)), nn.LeakyReLU(.2)])
            size = int(width)
        layers.append(nn.Linear(size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class PairDiscriminator(nn.Module):
    """Contrastive classifier used by PCL and the weakly supervised control."""

    def __init__(self, latent_dim, width=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * int(latent_dim), int(width)), nn.LeakyReLU(.2),
            nn.Linear(int(width), int(width)), nn.LeakyReLU(.2), nn.Linear(int(width), 1),
        )

    def forward(self, left, right):
        return self.network(torch.cat([left, right], dim=-1)).squeeze(-1)


class GeometryAutoencoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.kind = config["model_type"]
        self.latent_dim = int(config["latent_dim"])
        hidden = int(config["hidden_dim"])
        enc_depth = int(config["encoder_depth"])
        dec_depth = int(config["decoder_depth"])
        activation = config["activation"]
        custom = config.get("custom_model") or {}
        bias = bool(config.get("bias", True))
        linear = self.kind == "linear_ae"
        variational = self.kind in {"vae", "beta_vae", "beta_vae_full_cov", "conv_beta_vae", "factor_vae", "beta_tcvae", "slow_vae", "hyperspherical_vae", "riemannian_vae", "custom_vae"}
        self.variational = variational
        self.full_covariance = self.kind == "beta_vae_full_cov" or (self.kind == "custom_vae" and custom.get("posterior") == "full_covariance")
        covariance_parameters = self.latent_dim * (self.latent_dim + 1) // 2
        encoder_output = self.latent_dim + covariance_parameters if self.full_covariance else self.latent_dim * (2 if variational else 1)
        common = {"dropout": custom.get("dropout", 0.0), "layer_norm": custom.get("layer_norm", False)}
        encoder_activation = str(config.get("encoder_activation", custom.get("encoder_activation", activation)))
        decoder_activation = str(config.get("decoder_activation", custom.get("decoder_activation", activation)))
        architecture = str(custom.get("architecture", config.get("architecture", "auto"))).lower()
        if architecture == "convolutional" or self.kind == "conv_beta_vae" or bool(config.get("paper_image_model", False)):
            image_shape = config.get("image_shape") or custom.get("image_shape")
            channels = (custom.get("conv_encoder_channels") or custom.get("conv_channels")
                        or config.get("conv_encoder_channels") or config.get("conv_channels") or [32, 32, 64, 64])
            if not image_shape or int(np.prod(image_shape)) != input_dim:
                raise ValueError("Convolutional beta-VAE requires image_shape matching the flattened observations")
            self.encoder = ConvEncoder(
                image_shape, encoder_output, channels, encoder_activation, bias=bias,
                kernel_sizes=custom.get("conv_encoder_kernel_sizes") or config.get("conv_encoder_kernel_sizes"),
                strides=custom.get("conv_encoder_strides") or config.get("conv_encoder_strides"),
                paddings=custom.get("conv_encoder_paddings") or config.get("conv_encoder_paddings"),
                dense_layers=custom.get("conv_encoder_dense_layers") or config.get("conv_encoder_dense_layers"),
            )
            decoder_base_shape = (custom.get("conv_decoder_base_shape") or config.get("conv_decoder_base_shape")
                                  or list(self.encoder.feature_shape))
            decoder_channels = (custom.get("conv_decoder_channels") or config.get("conv_decoder_channels")
                                or list(reversed(channels))[1:])
            self.decoder = ConvDecoder(
                self.latent_dim, image_shape, decoder_base_shape, decoder_channels, decoder_activation, bias=bias,
                kernel_sizes=custom.get("conv_decoder_kernel_sizes") or config.get("conv_decoder_kernel_sizes"),
                strides=custom.get("conv_decoder_strides") or config.get("conv_decoder_strides"),
                paddings=custom.get("conv_decoder_paddings") or config.get("conv_decoder_paddings"),
                dense_layers=custom.get("conv_decoder_dense_layers") or config.get("conv_decoder_dense_layers"),
            )
        else:
            self.encoder = MLP(input_dim, encoder_output, hidden, enc_depth, encoder_activation, linear,
                               hidden_layers=custom.get("encoder_layers"), bias=bias, **common)
            self.decoder = MLP(self.latent_dim, input_dim, hidden, dec_depth, decoder_activation, linear,
                               hidden_layers=custom.get("decoder_layers"), bias=bias, **common)
        self._last_cholesky = None

    def _cholesky(self, raw):
        batch = raw.shape[0]
        matrix = raw.new_zeros((batch, self.latent_dim, self.latent_dim))
        rows, cols = torch.tril_indices(self.latent_dim, self.latent_dim, device=raw.device)
        matrix[:, rows, cols] = raw
        diagonal = torch.arange(self.latent_dim, device=raw.device)
        matrix[:, diagonal, diagonal] = F.softplus(matrix[:, diagonal, diagonal]) + 1e-4
        return matrix

    def encode(self, x):
        encoded = self.encoder(x)
        if not self.variational:
            return encoded, torch.zeros_like(encoded)
        if self.full_covariance:
            mu, raw = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
            cholesky = self._cholesky(raw)
            self._last_cholesky = cholesky
            logvar = torch.log(torch.diagonal(cholesky @ cholesky.transpose(-1, -2), dim1=-2, dim2=-1).clamp_min(1e-8))
        else:
            mu, logvar = encoded.chunk(2, dim=-1)
            logvar = logvar.clamp(-10, 8)
            self._last_cholesky = None
        if self.kind == "hyperspherical_vae":
            mu = F.normalize(mu, dim=-1)
        return mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        if not self.variational:
            z = mu
        elif self.full_covariance:
            z = mu + torch.bmm(self._last_cholesky, torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        if self.kind == "hyperspherical_vae":
            z = F.normalize(z, dim=-1)
        return self.decoder(z), z, mu, logvar

    def decode(self, z):
        return self.decoder(z)


def _log_density_gaussian(sample, mean, logvar):
    normalization = math.log(2 * math.pi)
    return -.5 * (normalization + logvar + (sample - mean).pow(2) * torch.exp(-logvar))


def _beta_tcvae_terms(z, mu, logvar, dataset_size):
    """Minibatch-weighted decomposition from Chen et al. (NeurIPS 2018)."""
    batch = z.shape[0]
    if batch < 2:
        zero = z.new_tensor(0.0)
        return zero, zero, zero
    log_q_z_given_x = _log_density_gaussian(z, mu, logvar).sum(1)
    matrix = _log_density_gaussian(z[:, None, :], mu[None, :, :], logvar[None, :, :])
    # Stratified minibatch importance weights (MWS/MSS estimator used by beta-TCVAE).
    n = max(int(dataset_size), batch)
    weight = z.new_full((batch, batch), 1.0 / max(1, batch - 1))
    weight.view(-1)[::batch + 1] = 1.0 / n
    weight.view(-1)[1::batch + 1] = max(1.0 / n, (n - batch + 1) / (n * max(1, batch - 1)))
    log_weight = torch.log(weight.clamp_min(1e-12))
    log_q_z = torch.logsumexp(matrix.sum(2) + log_weight, dim=1)
    log_prod_q_z = torch.logsumexp(matrix + log_weight[:, :, None], dim=1).sum(1)
    log_p_z = _log_density_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
    mutual_information = (log_q_z_given_x - log_q_z).mean()
    total_correlation = (log_q_z - log_prod_q_z).mean()
    dimensionwise_kl = (log_prod_q_z - log_p_z).mean()
    return mutual_information, total_correlation, dimensionwise_kl


def _factor_pair_lookup(factors, eligible_indices, seed):
    """Choose a deterministic neighbor differing in exactly one factor."""
    factors = np.asarray(factors)
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    inverse_columns, cardinalities = [], []
    for column in range(factors.shape[1]):
        _, inverse = np.unique(factors[:, column], return_inverse=True)
        inverse_columns.append(inverse.astype(np.int64))
        cardinalities.append(int(inverse.max()) + 1)
    strides = np.cumprod([1] + cardinalities[:-1], dtype=np.int64)
    total = int(np.prod(cardinalities, dtype=np.int64))
    if total > 20_000_000:
        # Defensive fallback for non-grid uploads.
        lookup = np.arange(len(factors), dtype=np.int64)
        lookup[eligible] = eligible
        return lookup
    codes = sum(inverse_columns[column] * strides[column] for column in range(len(cardinalities)))
    index_by_code = np.full(total, -1, dtype=np.int64)
    index_by_code[codes[eligible]] = eligible
    lookup = np.arange(len(factors), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for index in eligible:
        order = rng.permutation(len(cardinalities))
        for factor in order:
            current = inverse_columns[factor][index]
            values = rng.permutation(cardinalities[factor])
            found = -1
            for value in values:
                if value == current:
                    continue
                candidate_code = int(codes[index] + (value - current) * strides[factor])
                found = int(index_by_code[candidate_code])
                if found >= 0:
                    break
            if found >= 0:
                lookup[index] = found
                break
    return lookup


def _loss_terms(model, x, config, dataset_size=None):
    recon, z, mu, logvar = model(x)
    reconstruction_kind = str(config.get("reconstruction_loss", "mse_sum"))
    if reconstruction_kind == "bernoulli_logits":
        reconstruction = F.binary_cross_entropy_with_logits(recon, x, reduction="none").flatten(1).sum(1).mean()
    elif reconstruction_kind == "mse_mean":
        reconstruction = (recon - x).pow(2).flatten(1).mean(1).mean()
    elif reconstruction_kind == "l1_sum":
        reconstruction = (recon - x).abs().flatten(1).sum(1).mean()
    else:
        # Sum observation dimensions per sample, then average the batch.
        reconstruction = (recon - x).pow(2).flatten(1).sum(1).mean()
    if model.full_covariance:
        cholesky = model._last_cholesky
        trace = cholesky.pow(2).sum(dim=(-2, -1))
        logdet = 2 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
        kl = 0.5 * torch.mean(trace + mu.pow(2).sum(-1) - model.latent_dim - logdet)
    else:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean() if model.variational else x.new_tensor(0.0)
    beta = float(config.get("beta", 1.0))
    regularizer = x.new_tensor(0.0)
    if model.kind == "sparse_ae":
        regularizer = float(config.get("sparse_lambda", 1e-3)) * z.abs().mean()
    elif model.kind == "beta_tcvae":
        mi, tc, dimensionwise = _beta_tcvae_terms(z, mu, logvar, dataset_size or len(x))
        regularizer = mi + beta * tc + dimensionwise - kl
    elif model.kind == "riemannian_vae" and len(x) > 1:
        count = min(64, len(x))
        dx = torch.cdist(x[:count], x[:count])
        dz = torch.cdist(mu[:count], mu[:count])
        dx = dx / dx.mean().clamp_min(1e-6)
        dz = dz / dz.mean().clamp_min(1e-6)
        regularizer = float(config.get("geometry_weight", 0.1)) * F.mse_loss(dz, dx)
    if model.kind == "beta_tcvae":
        total = reconstruction + kl + regularizer
    elif model.kind == "factor_vae":
        # The discriminator-derived TC term is added in the training loop.
        total = reconstruction + kl
    else:
        total = reconstruction + (beta * kl if model.variational else 0) + regularizer
    return total, reconstruction, kl, regularizer, recon, z, mu


def _reconstruction_mean(raw_reconstruction, config):
    """Map decoder parameters to observation means for reports and diagnostics."""
    if str(config.get("reconstruction_loss", "mse_sum")) == "bernoulli_logits":
        return torch.sigmoid(raw_reconstruction)
    return raw_reconstruction


def _make_optimizer(parameters, config):
    common = {"lr": config["learning_rate"], "weight_decay": config.get("weight_decay", 0.0)}
    if config["optimizer"] == "adam":
        return torch.optim.Adam(parameters, betas=(config["adam_beta1"], config["adam_beta2"]),
                                eps=config["optimizer_epsilon"], **common)
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(parameters, betas=(config["adam_beta1"], config["adam_beta2"]),
                                 eps=config["optimizer_epsilon"], **common)
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, **common)
    return torch.optim.Adagrad(parameters, eps=config["optimizer_epsilon"], **common)


def _normalized_training_config(raw):
    kind = str(raw.get("model_type", "beta_vae"))
    if kind not in MODEL_LABELS:
        raise ValueError("Unknown model type")
    requested_latent = max(1, min(256, int(raw.get("latent_dim", 3))))
    custom = raw.get("custom_model") or {}
    if not isinstance(custom, dict):
        raise ValueError("custom_model must be a JSON object")
    for key in ("encoder_layers", "decoder_layers", "conv_encoder_channels", "conv_encoder_kernel_sizes",
                "conv_encoder_strides", "conv_encoder_dense_layers", "conv_decoder_channels",
                "conv_decoder_kernel_sizes", "conv_decoder_strides", "conv_decoder_dense_layers",
                "conv_decoder_base_shape"):
        if key in custom:
            if not isinstance(custom[key], list) or not custom[key] or len(custom[key]) > 12:
                raise ValueError(f"{key} must be a non-empty list with at most 12 positive integers")
            minimum = 1 if key not in {"encoder_layers", "decoder_layers", "conv_encoder_dense_layers", "conv_decoder_dense_layers"} else 2
            custom[key] = [max(minimum, min(8192, int(value))) for value in custom[key]]
    for key in ("conv_encoder_paddings", "conv_decoder_paddings"):
        if key in custom:
            if not isinstance(custom[key], list) or not custom[key] or len(custom[key]) > 12:
                raise ValueError(f"{key} must be a non-empty list with at most 12 non-negative integers")
            custom[key] = [max(0, min(128, int(value))) for value in custom[key]]
    architecture = str(custom.get("architecture", raw.get("architecture", "auto"))).lower()
    if architecture not in {"auto", "mlp", "convolutional"}:
        raise ValueError("architecture must be auto, mlp, or convolutional")
    custom["architecture"] = architecture
    custom["dropout"] = max(0.0, min(0.9, float(custom.get("dropout", 0.0))))
    custom["layer_norm"] = bool(custom.get("layer_norm", False))
    if custom.get("posterior", "diagonal") not in {"diagonal", "full_covariance"}:
        raise ValueError("Custom posterior must be diagonal or full_covariance")
    allowed_activations = {"relu", "gelu", "tanh", "silu"}
    activation = str(custom.get("activation", raw.get("activation", "gelu"))).lower()
    encoder_activation = str(custom.get("encoder_activation", raw.get("encoder_activation", activation))).lower()
    decoder_activation = str(custom.get("decoder_activation", raw.get("decoder_activation", activation))).lower()
    if activation not in allowed_activations or encoder_activation not in allowed_activations or decoder_activation not in allowed_activations:
        raise ValueError("Activations must be relu, gelu, tanh, or silu")
    if (kind == "beta_vae_full_cov" or custom.get("posterior") == "full_covariance") and requested_latent > 64:
        raise ValueError("Full-covariance posterior is capped at latent dimension 64")
    optimizer = str(raw.get("optimizer", "adam")).lower()
    if optimizer not in {"adam", "adamw", "sgd", "adagrad"}:
        raise ValueError("Optimizer must be adam, adamw, sgd, or adagrad")
    paper_reproduction = bool(raw.get("paper_reproduction", False))
    batch_size = max(8, min(8192, int(raw.get("batch_size", 256))))
    batch_size_source = str(raw.get("batch_size_source", "")).strip()
    if not batch_size_source:
        batch_size_source = "declared_reproduction_assumption" if paper_reproduction else "user_configured"
    batch_size_assumption = str(raw.get("batch_size_assumption", "")).strip()
    if paper_reproduction and not batch_size_assumption and str(raw.get("paper_id", "")) != ZIETLOW_PAPER["id"]:
        batch_size_assumption = PAPER_BATCH_SIZE_ASSUMPTION
    image_shape = [int(item) for item in (raw.get("image_shape") or [])]
    if len(image_shape) == 2:
        image_shape = [1, *image_shape]
    reconstruction_loss = str(raw.get("reconstruction_loss", custom.get("reconstruction_loss", "mse_sum"))).lower()
    if reconstruction_loss not in {"mse_sum", "mse_mean", "bernoulli_logits", "l1_sum"}:
        raise ValueError("reconstruction_loss must be mse_sum, mse_mean, bernoulli_logits, or l1_sum")
    training_steps = max(0, min(10_000_000, int(raw.get("training_steps", 0) or 0)))
    metric_sampling = str(raw.get("metric_sampling", "heldout")).lower()
    if metric_sampling not in {"heldout", "reference"}:
        raise ValueError("metric_sampling must be heldout or reference")
    return {
        "model_type": kind,
        "epochs": max(1, min(2000, int(raw.get("epochs", 40)))),
        "training_steps": training_steps,
        "training_budget_unit": "optimizer_steps" if training_steps else "epochs",
        "learning_rate": max(1e-6, min(1.0, float(raw.get("learning_rate", 1e-3)))),
        "batch_size": batch_size,
        "batch_size_source": batch_size_source,
        "batch_size_assumption": batch_size_assumption,
        "optimizer": optimizer,
        "adam_beta1": max(0.0, min(.999999, float(raw.get("adam_beta1", .9)))),
        "adam_beta2": max(0.0, min(.999999, float(raw.get("adam_beta2", .999)))),
        "optimizer_epsilon": max(1e-12, min(.1, float(raw.get("optimizer_epsilon", 1e-8)))),
        "weight_decay": max(0.0, min(10.0, float(raw.get("weight_decay", 0.0)))),
        "reconstruction_loss": reconstruction_loss,
        "latent_dim": requested_latent,
        "encoder_depth": max(0, min(8, int(raw.get("encoder_depth", 2)))),
        "decoder_depth": max(0, min(8, int(raw.get("decoder_depth", 2)))),
        "hidden_dim": max(4, min(4096, int(raw.get("hidden_dim", 128)))),
        "activation": activation,
        "encoder_activation": encoder_activation,
        "decoder_activation": decoder_activation,
        "bias": bool(raw.get("bias", True)),
        "beta": max(0.0, min(100.0, float(raw.get("beta", 4.0)))),
        "sparse_lambda": max(0.0, float(raw.get("sparse_lambda", 1e-3))),
        "tc_weight": max(0.0, float(raw.get("tc_weight", 6.0))),
        "geometry_weight": max(0.0, float(raw.get("geometry_weight", 0.1))),
        "paper_id": str(raw.get("paper_id", ""))[:80],
        "dataset_variant": str(raw.get("dataset_variant", "original"))[:40],
        "paper_image_model": bool(raw.get("paper_image_model", str(raw.get("paper_id", "")) == "zietlow2021")),
        "zietlow_metrics": bool(raw.get("zietlow_metrics", str(raw.get("paper_id", "")) == "zietlow2021")),
        "factor_metric_bins": max(5, min(100, int(raw.get("factor_metric_bins", 20)))),
        "factor_vote_batches": max(100, min(50000, int(raw.get("factor_vote_batches", 10000)))),
        "factor_vote_evaluation_batches": max(100, min(50000, int(raw.get("factor_vote_evaluation_batches", 5000)))),
        "metric_sampling": metric_sampling,
        "metric_seed": int(raw.get("metric_seed", 0 if metric_sampling == "reference" else raw.get("seed", 42))),
        "metric_train_samples": max(100, min(1_000_000, int(raw.get("metric_train_samples", 10000)))),
        "metric_test_samples": max(100, min(1_000_000, int(raw.get("metric_test_samples", 5000)))),
        "metric_variance_samples": max(100, min(1_000_000, int(raw.get("metric_variance_samples", 10000)))),
        "factor_discriminator_width": max(32, min(2000, int(raw.get("factor_discriminator_width", 1000)))),
        "factor_discriminator_depth": max(1, min(8, int(raw.get("factor_discriminator_depth", 6)))),
        "factor_discriminator_lr": max(1e-7, min(1.0, float(raw.get("factor_discriminator_lr", 1e-4)))),
        "slow_gamma": max(0.0, min(1000.0, float(raw.get("slow_gamma", raw.get("beta", 1.0))))),
        "slow_alpha": max(0.1, min(2.0, float(raw.get("slow_alpha", 0.5)))),
        "pair_weight": max(0.0, min(1000.0, float(raw.get("pair_weight", 1.0)))),
        "paper_hyperparameter_scale": max(0.01, min(100.0, float(raw.get("paper_hyperparameter_scale", 1.0)))),
        "custom_model": custom,
        "seed": int(raw.get("seed", 42)),
        "seed_source": "effective_training_configuration",
        "paper_reproduction": paper_reproduction,
        "polarization_max_samples": max(0, int(raw.get("polarization_max_samples", 4096))),
        "image_shape": image_shape,
        "conv_channels": [int(item) for item in (raw.get("conv_channels") or [32, 32, 64, 64])],
        "architecture": architecture,
        "conv_encoder_channels": custom.get("conv_encoder_channels") or raw.get("conv_encoder_channels"),
        "conv_encoder_kernel_sizes": custom.get("conv_encoder_kernel_sizes") or raw.get("conv_encoder_kernel_sizes"),
        "conv_encoder_strides": custom.get("conv_encoder_strides") or raw.get("conv_encoder_strides"),
        "conv_encoder_paddings": custom.get("conv_encoder_paddings") or raw.get("conv_encoder_paddings"),
        "conv_encoder_dense_layers": custom.get("conv_encoder_dense_layers") or raw.get("conv_encoder_dense_layers"),
        "conv_decoder_base_shape": custom.get("conv_decoder_base_shape") or raw.get("conv_decoder_base_shape"),
        "conv_decoder_channels": custom.get("conv_decoder_channels") or raw.get("conv_decoder_channels"),
        "conv_decoder_kernel_sizes": custom.get("conv_decoder_kernel_sizes") or raw.get("conv_decoder_kernel_sizes"),
        "conv_decoder_strides": custom.get("conv_decoder_strides") or raw.get("conv_decoder_strides"),
        "conv_decoder_paddings": custom.get("conv_decoder_paddings") or raw.get("conv_decoder_paddings"),
        "conv_decoder_dense_layers": custom.get("conv_decoder_dense_layers") or raw.get("conv_decoder_dense_layers"),
    }


def _resolved_paper_protocol(protocol, training_config):
    """Resolve run-specific provenance from the effective training config."""
    resolved = dict(protocol) if isinstance(protocol, dict) else {}
    if not resolved and not training_config.get("paper_reproduction"):
        return resolved
    effective = {
        "seed": int(training_config["seed"]),
        "seed_source": "effective_training_configuration",
        "batch_size": int(training_config["batch_size"]),
        "batch_size_source": training_config.get("batch_size_source", "user_configured"),
        "batch_size_assumption": training_config.get("batch_size_assumption", ""),
        "bias": bool(training_config.get("bias", True)),
        "training_steps": int(training_config.get("training_steps", 0)),
        "epochs": int(training_config.get("epochs", 1)),
        "learning_rate": float(training_config.get("learning_rate", 0.0)),
        "optimizer": training_config.get("optimizer"),
        "adam_beta1": float(training_config.get("adam_beta1", .9)),
        "adam_beta2": float(training_config.get("adam_beta2", .999)),
        "optimizer_epsilon": float(training_config.get("optimizer_epsilon", 1e-8)),
        "weight_decay": float(training_config.get("weight_decay", 0.0)),
        "reconstruction_loss": training_config.get("reconstruction_loss"),
        "encoder_activation": training_config.get("encoder_activation"),
        "decoder_activation": training_config.get("decoder_activation"),
        "custom_model": training_config.get("custom_model", {}),
        "metric_sampling": training_config.get("metric_sampling"),
        "metric_seed": int(training_config.get("metric_seed", 0)),
        "metric_train_samples": int(training_config.get("metric_train_samples", 10000)),
        "metric_test_samples": int(training_config.get("metric_test_samples", 5000)),
        "metric_variance_samples": int(training_config.get("metric_variance_samples", 10000)),
    }
    resolved.update(effective)
    if isinstance(resolved.get("training"), dict):
        resolved["training"] = {
            **resolved["training"],
            **{key: effective[key] for key in (
                "seed", "seed_source", "batch_size", "batch_size_source", "batch_size_assumption"
            )},
        }
    if isinstance(resolved.get("model"), dict):
        resolved["model"] = {**resolved["model"], "bias": effective["bias"]}
    return resolved


def register_geometry_lab(app, experiments_dir):
    root = Path(experiments_dir) / "geometry_lab"
    root.mkdir(parents=True, exist_ok=True)
    models_root = root / "model_library"
    models_root.mkdir(parents=True, exist_ok=True)
    datasets_root = root / "_datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    official_root = datasets_root / "official"
    official_root.mkdir(parents=True, exist_ok=True)
    exports_root = root / "_exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    jobs = {}
    job_lock = threading.Lock()
    queue_condition = threading.Condition(job_lock)
    queue_path = root / "training_queue.json"
    queue_worker_thread = None
    queue_state = {"schema_version": 1, "items": []}

    if queue_path.exists():
        try:
            saved_queue = json.loads(queue_path.read_text(encoding="utf-8"))
            if isinstance(saved_queue, dict) and isinstance(saved_queue.get("items"), list):
                queue_state = saved_queue
        except Exception:
            traceback.print_exc()

    def persist_queue_locked():
        """Persist queue state atomically. Caller must hold job_lock."""
        temporary = queue_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")
        os.replace(temporary, queue_path)

    def queue_item_locked(queue_id):
        return next((item for item in queue_state["items"] if item.get("id") == queue_id), None)

    # A process restart interrupts Python worker threads. Recover unfinished
    # entries instead of silently losing them; completed training can resume at
    # analysis, while an interrupted optimizer run is restarted deterministically.
    queue_recovered = False
    for queued_item in queue_state["items"]:
        if queued_item.get("status") not in {"running", "analyzing"}:
            continue
        directory = root / str(queued_item.get("experiment_id", ""))
        analysis_ready = (directory / "analysis.json").exists() and (directory / "model.pt").exists()
        needs_science = bool(queued_item.get("run_scientific"))
        science_ready = (directory / "scientific_metrics.json").exists()
        if analysis_ready and (not needs_science or science_ready):
            queued_item.update({"status": "complete", "stage": "recovered complete", "progress": 1.0})
        else:
            queued_item.update({
                "status": "queued", "stage": "recovered after server restart", "progress": 0.0,
                "resume_analysis": bool(analysis_ready and needs_science and not science_ready),
            })
        queue_recovered = True
    if queue_recovered:
        with job_lock:
            persist_queue_locked()

    def update_job(job_id, **values):
        """Update an in-memory job and mirror compact queue progress to disk."""
        with job_lock:
            job = jobs.get(job_id)
            if job is None:
                return
            job.update(values)
            queue_id = job.get("queue_id")
            if not queue_id:
                return
            item = queue_item_locked(queue_id)
            if item is None:
                return
            phase = job.get("queue_phase", "training")
            raw_progress = float(values.get("progress", job.get("progress", 0)) or 0)
            requires_science = bool(item.get("run_scientific"))
            if phase == "training":
                item["progress"] = raw_progress * (0.85 if requires_science else 1.0)
                item["stage"] = "training"
                if values.get("epoch") is not None:
                    item["epoch"] = int(values["epoch"])
                if values.get("metrics"):
                    item["latest_metrics"] = values["metrics"][-1]
            else:
                item["progress"] = 0.85 + raw_progress * 0.15
                item["stage"] = str(values.get("stage") or "scientific analysis")
            if values.get("status") == "error":
                item.update({
                    "status": "error", "stage": "failed", "error": str(values.get("error", "Unknown error")),
                    "completed_at": _utc_now(),
                })
            else:
                # The queue worker owns the final complete transition because a
                # training completion may still be followed by scientific work.
                item["status"] = "running"
            persist_queue_locked()

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def stream_sha256(handle):
        digest = hashlib.sha256()
        position = handle.tell()
        handle.seek(0)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(position)
        return digest.hexdigest()

    def download_official_file(url, filename):
        """Cache an upstream artifact atomically and return its path and fetch state."""
        target = official_root / filename
        if target.exists() and target.stat().st_size > 0:
            with target.open("rb") as cached:
                if target.suffix.lower() != ".npz" or cached.read(4) == b"PK\x03\x04":
                    return target, False
            target.unlink()
        partial = target.with_suffix(target.suffix + ".part")
        request_object = urllib.request.Request(
            url,
            headers={"User-Agent": "ManifoldSuperpositionLab/1.0 (+reproducible-dataset-fetch)"},
        )
        try:
            with urllib.request.urlopen(request_object, timeout=120) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if partial.stat().st_size == 0:
                raise ValueError("The official dataset download was empty")
            if target.suffix.lower() == ".npz":
                with partial.open("rb") as downloaded_file:
                    if downloaded_file.read(4) != b"PK\x03\x04":
                        raise ValueError("The publisher response was not a valid NPZ archive")
            os.replace(partial, target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return target, True

    def exp_dir(exp_id):
        return root / _safe_id(exp_id)

    def display_dataset_type(experiment):
        """Prefer provenance over a stale legacy presentation label."""
        source = str(experiment.get("dataset_source", ""))
        if source.startswith("paper_"):
            return source
        return str(experiment.get("dataset_type", source or "dataset"))

    def presentation_config(experiment):
        presented = dict(experiment)
        presented["dataset_type"] = display_dataset_type(experiment)
        return presented

    def add_paper_requirements(directory, experiment, scientific):
        """Attach requirement calculations to new and legacy scientific results."""
        if scientific is None:
            return scientific
        summary = scientific.setdefault("summary", {})
        requirements = scientific.get("paper_requirements")
        if requirements is None or int(requirements.get("schema_version", 0)) < 2:
            training_path = directory / "training_config.json"
            training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else {}
            requirement_metrics = dict(summary)
            analysis_path = directory / "analysis.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
            polarization = analysis.get("polarization_summary") or {}
            interval = float(polarization.get("evaluation_interval_batches") or 0)
            steps = float(polarization.get("total_optimizer_steps") or 0)
            if interval > 0 and steps > 0:
                requirement_metrics["paper_polarization_resolution_percent"] = 100.0 * interval / steps
            requirements = evaluate_paper_requirements(
                experiment.get("dataset_source", ""), training.get("model_type", ""),
                training.get("latent_dim", experiment.get("latent_dim", 0)), requirement_metrics,
            )
            scientific["paper_requirements"] = requirements
        summary.update({
            "paper_requirement_status": requirements["status"],
            "paper_requirement_pass_fraction": requirements["pass_fraction"],
            "paper_requirement_passed": requirements["passed"],
            "paper_requirement_applicable": requirements["applicable"],
        })
        return scientific

    def load_data(exp_id):
        path = exp_dir(exp_id) / "dataset.npz"
        if not path.exists():
            raise FileNotFoundError("Dataset not found")
        loaded = np.load(path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}

    def save_empirical_experiment(name, dataset_source, observations, factors, factor_names,
                                  seed=42, image_shape=None, source_filename="", metadata=None,
                                  factor_metric_types=None):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim < 2:
            raise ValueError("Dataset observations must have at least two dimensions")
        original_shape = list(observations.shape[1:])
        observations = observations.reshape(len(observations), -1)
        factors = np.asarray(factors, dtype=np.float32)
        if factors.ndim == 1:
            factors = factors[:, None]
        if len(factors) != len(observations):
            raise ValueError("Ground-truth factors must have the same sample count as observations")
        if factors.shape[1] == 0:
            factors = np.zeros((len(observations), 1), dtype=np.float32)
            factor_names = ["sample_group"]
        rng = np.random.default_rng(seed)
        params = np.zeros((len(observations), factors.shape[1], 2), dtype=np.float32)
        params[:, :, 0] = factors
        active = np.ones_like(factors, dtype=np.float32)
        standardized = (factors - factors.mean(0)) / (factors.std(0) + 1e-8)
        split = _three_way_split(len(observations), rng, 0.8, 0.1)
        metric_types = factor_metric_types or [
            "categorical" if np.allclose(np.unique(column), np.round(np.unique(column))) and len(np.unique(column)) <= 10 else "continuous"
            for column in factors.T
        ]
        factor_types = np.asarray(["dsprites"] * factors.shape[1])
        shape = list(image_shape) if image_shape else (original_shape if len(original_shape) in {2, 3} else [])
        data = {
            "observations": observations, "factor_parameters": params, "factor_values": factors,
            "factor_active": active, "factor_embeddings": standardized.astype(np.float32),
            "mixing_matrix": np.zeros((observations.shape[1], factors.shape[1]), dtype=np.float32),
            "split": split, "factor_types": factor_types,
            "factor_metric_types": np.asarray(metric_types),
            "component_slices": np.asarray([[i, i + 1] for i in range(factors.shape[1])], dtype=np.int32),
            "empirical": np.asarray([1], dtype=np.uint8), "image_shape": np.asarray(shape, dtype=np.int16),
        }
        exp_id = "geo_data_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        config = {
            "schema_version": 2, "id": exp_id, "name": str(name).strip()[:120] or dataset_source,
            "description": f"Empirical {dataset_source} dataset", "created_at": _utc_now(),
            "dataset_source": dataset_source, "dataset_type": dataset_source, "factor_types": factor_types.tolist(),
            "k": factors.shape[1], "latent_dim": 8, "sparsity": 1.0, "dataset_size": len(observations),
            "observation_dim": observations.shape[1], "noise": 0, "train_split": .8,
            "validation_split": .1, "test_split": .1, "seed": seed,
            "empirical": True, "ground_truth_labels": list(factor_names), "image_shape": shape,
            "source_filename": source_filename, "paper_title": "", "paper_citation": "", "paper_url": "",
            "run_label": "baseline", "hypothesis": "", "tags": [dataset_source],
            "analysis_plan": DEFAULT_ANALYSIS_PLAN, "paper_protocol": {}, **(metadata or {}),
        }
        directory = exp_dir(exp_id)
        directory.mkdir(parents=True, exist_ok=False)
        dataset_path = directory / "dataset.npz"
        np.savez_compressed(dataset_path, **data)
        config["dataset_npz_sha256"] = file_sha256(dataset_path)
        config["dataset_artifact"] = "dataset.npz"
        summary = _dataset_summary(data)
        config["split_counts"] = dict(summary["split_counts"])
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return config, summary

    @app.route("/api/geometry/specs", methods=["GET"])
    def geometry_specs():
        return jsonify({
            "success": True, "factor_types": FACTOR_SPECS, "models": MODEL_LABELS,
            "datasets": DATASET_CATALOG, "concepts": CONCEPTS,
            "custom_model_schema": {
                "architecture": "mlp", "encoder_layers": [256, 128], "decoder_layers": [128, 256],
                "activation": "gelu", "encoder_activation": "gelu", "decoder_activation": "gelu",
                "dropout": 0.0, "layer_norm": False, "bias": True,
                "posterior": "diagonal",
                "convolutional_example": {
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
                },
            },
        })

    @app.route("/api/geometry/models", methods=["GET", "POST"])
    def geometry_model_library():
        """Persist declarative model definitions without accepting executable code."""
        if request.method == "GET":
            models = []
            for path in models_root.glob("*.json"):
                try:
                    models.append(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
            models.sort(key=lambda item: item.get("created_at", ""), reverse=True)
            return jsonify({"success": True, "models": models})
        try:
            raw = request.get_json(force=True) or {}
            name = str(raw.get("name", "")).strip()[:120]
            if not name:
                raise ValueError("Give the custom model a reusable name")
            model_type = str(raw.get("model_type", "custom_vae"))
            if model_type not in {"custom_ae", "custom_vae"}:
                raise ValueError("Saved custom models must use custom_ae or custom_vae")
            normalized = _normalized_training_config({
                "model_type": model_type,
                "latent_dim": raw.get("latent_dim", 8),
                "beta": raw.get("beta", 1.0),
                "custom_model": raw.get("custom_model") or {},
            })
            model_id = "model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            item = {
                "schema_version": 1, "id": model_id, "name": name,
                "model_type": model_type, "custom_model": normalized["custom_model"],
                "latent_dim": normalized["latent_dim"], "beta": normalized["beta"],
                "created_at": _utc_now(),
            }
            (models_root / f"{model_id}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
            return jsonify({"success": True, "model": item})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments", methods=["GET"])
    def geometry_experiments():
        items = []
        for path in root.iterdir():
            config_path = path / "config.json"
            if path.is_dir() and config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    items.append({
                        "id": cfg["id"], "name": cfg["name"], "created_at": cfg["created_at"],
                        "dataset_type": display_dataset_type(cfg), "k": cfg["k"], "dataset_size": cfg["dataset_size"],
                        "paper_title": cfg.get("paper_title", ""), "run_label": cfg.get("run_label", ""),
                        "trained": (path / "analysis.json").exists(),
                        "bundle_profile": cfg.get("bundle_profile"),
                        "results_only": bool(cfg.get("results_only", False)),
                    })
                except Exception:
                    continue
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return jsonify({"success": True, "experiments": items})

    @app.route("/api/geometry/experiments", methods=["POST"])
    def geometry_create_experiment():
        try:
            raw = request.get_json(force=True) or {}
            factor_types = _factor_types(raw)
            exp_id = "geo_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            config = {
                "schema_version": 2,
                "id": exp_id,
                "name": str(raw.get("name", "Untitled manifold experiment")).strip()[:120] or "Untitled manifold experiment",
                "description": str(raw.get("description", "")).strip()[:2000],
                "paper_title": str(raw.get("paper_title", "")).strip()[:300],
                "paper_citation": str(raw.get("paper_citation", "")).strip()[:1000],
                "paper_url": str(raw.get("paper_url", "")).strip()[:1000],
                "run_label": str(raw.get("run_label", "baseline")).strip()[:120] or "baseline",
                "hypothesis": str(raw.get("hypothesis", "")).strip()[:2000],
                "tags": [str(tag)[:80] for tag in (raw.get("tags") or [])[:30]],
                "analysis_plan": _analysis_plan(raw.get("analysis_plan")),
                "paper_protocol": raw.get("paper_protocol") if isinstance(raw.get("paper_protocol"), dict) else {},
                "created_at": _utc_now(),
                "dataset_source": str(raw.get("dataset_source", "synthetic_linear")),
                "dataset_type": str(raw.get("dataset_type", "circle")),
                "generator_type": str(raw.get("generator_type", "linear")),
                "factor_types": factor_types,
                "k": len(factor_types),
                "latent_dim": max(1, min(256, int(raw.get("latent_dim", 3)))),
                "sparsity": float(np.clip(raw.get("sparsity", 0.35), 0, 1)),
                "dataset_size": max(100, min(200000, int(raw.get("dataset_size", 5000)))),
                "observation_dim": max(2, min(512, int(raw.get("observation_dim", 12)))),
                "noise": max(0.0, float(raw.get("noise", 0.02))),
                "train_split": float(np.clip(raw.get("train_split", 0.8), 0.5, 0.95)),
                "validation_split": float(np.clip(raw.get("validation_split", 0.1), 0.05, 0.25)),
                "orthogonal_mixing": bool(raw.get("orthogonal_mixing", True)),
                "degeneracy": float(np.clip(raw.get("degeneracy", 1.0), 1e-6, 1.0)),
                "mixing_matrix": raw.get("mixing_matrix"),
                "seed": int(raw.get("seed", 42)),
            }
            data = generate_dataset(config)
            directory = exp_dir(exp_id)
            directory.mkdir(parents=True, exist_ok=False)
            config["factor_types"] = data["factor_types"].tolist()
            config["k"] = int(data["factor_values"].shape[1])
            config["observation_dim"] = int(data["observations"].shape[1])
            config["validation_split"] = float(config.get("validation_split", 0.1))
            config["test_split"] = float(1 - config["train_split"] - config["validation_split"])
            config["factor_embedding_dim"] = int(data["factor_embeddings"].shape[1])
            config["mixing_matrix"] = data["mixing_matrix"].round(8).tolist()
            if config["generator_type"] == "paper_linear":
                config["dataset_protocol"] = dict(PAPER_LINEAR_DATASET_PROTOCOL)
            elif config["generator_type"] == "paper_nonlinear":
                config["dataset_protocol"] = dict(PAPER_NONLINEAR_DATASET_PROTOCOL)
            dataset_path = directory / "dataset.npz"
            np.savez_compressed(dataset_path, **data)
            config["dataset_npz_sha256"] = file_sha256(dataset_path)
            config["dataset_artifact"] = "dataset.npz"
            summary = _dataset_summary(data)
            config["split_counts"] = dict(summary["split_counts"])
            (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return jsonify({"success": True, "experiment": config, "summary": summary})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/dsprites/ingest", methods=["POST"])
    def geometry_ingest_dsprites():
        """Fetch or ingest the native official dSprites NPZ, then derive a bounded run artifact."""
        try:
            upload = request.files.get("file")
            official_requested = str(request.form.get("official", "false")).lower() == "true"
            source_url = DATASET_CATALOG["dsprites"]["official_file"]
            downloaded = False
            if official_requested:
                source_filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
                source_path, downloaded = download_official_file(source_url, source_filename)
                source_sha256 = file_sha256(source_path)
                loaded = np.load(source_path, allow_pickle=False)
                source_mode = "official_fetch"
            else:
                if upload is None:
                    raise ValueError("Attach the official dSprites .npz or use Fetch official NPZ")
                source_filename = upload.filename or "dsprites.npz"
                source_sha256 = stream_sha256(upload.stream)
                loaded = np.load(upload.stream, allow_pickle=False)
                source_mode = "manual_official_upload"
            seed = int(request.form.get("seed", 42))
            paper_fidelity = str(request.form.get("paper_fidelity", "false")).lower() == "true"
            maximum = max(100, min(737280, int(request.form.get("dataset_size", 10000))))
            observation_dim = max(16, min(4096, int(request.form.get("observation_dim", 128))))
            if "imgs" not in loaded or "latents_values" not in loaded:
                raise ValueError("Expected dSprites keys: imgs and latents_values")
            total = len(loaded["imgs"])
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, size=min(maximum, total), replace=False))
            images = loaded["imgs"][indices]
            if images.ndim != 3:
                raise ValueError("dSprites imgs must have shape [samples, height, width]")
            height, width = images.shape[1:]
            if paper_fidelity or observation_dim >= height * width:
                images_processed = images.astype(np.float32)
            else:
                # Compact exploratory mode; paper-fidelity mode never downsamples.
                if height % 16 == 0 and width % 16 == 0:
                    images_processed = images.reshape(len(images), 16, height // 16, 16, width // 16).mean((2, 4)).astype(np.float32)
                else:
                    row = np.linspace(0, height - 1, 16, dtype=int)
                    col = np.linspace(0, width - 1, 16, dtype=int)
                    images_processed = images[:, row][:, :, col].astype(np.float32)
            flattened = images_processed.reshape(len(images), -1)
            if paper_fidelity:
                observations = flattened
                observation_dim = flattened.shape[1]
            elif observation_dim < flattened.shape[1]:
                projection = rng.normal(0, 1 / math.sqrt(observation_dim), (flattened.shape[1], observation_dim)).astype(np.float32)
                observations = flattened @ projection
            else:
                observations = flattened
                observation_dim = flattened.shape[1]

            values = loaded["latents_values"][indices]
            if values.shape[1] < 6:
                raise ValueError("dSprites latents_values must contain color, shape, scale, orientation, x and y")
            selected_values = values[:, [1, 2, 3, 4, 5]].astype(np.float32)
            loaded.close()
            params = np.zeros((len(images), 5, 2), dtype=np.float32)
            params[:, :, 0] = selected_values
            active = np.ones((len(images), 5), dtype=np.float32)
            standardized = (selected_values - selected_values.mean(0)) / (selected_values.std(0) + 1e-8)
            split = _three_way_split(len(images), rng, 0.8, 0.1)
            factor_types = np.asarray(["dsprites", "linear", "circle", "linear", "linear"])
            data = {
                "observations": observations.astype(np.float32), "factor_parameters": params,
                "factor_values": selected_values, "factor_active": active,
                "factor_embeddings": standardized.astype(np.float32),
                "mixing_matrix": np.zeros((observation_dim, 5), dtype=np.float32), "split": split,
                "factor_types": factor_types, "component_slices": np.asarray([[i, i + 1] for i in range(5)], dtype=np.int32),
                "factor_metric_types": np.asarray(["categorical", "continuous", "continuous", "continuous", "continuous"]),
                "empirical": np.asarray([1], dtype=np.uint8), "image_samples": images_processed[:128].astype(np.float32),
                "image_shape": np.asarray(images_processed.shape[1:], dtype=np.int16),
            }
            exp_id = "geo_dsprites_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            config = {
                "schema_version": 2, "id": exp_id, "name": str(request.form.get("name", "dSprites validation"))[:120],
                "description": "Official dSprites benchmark ingestion", "created_at": _utc_now(),
                "dataset_source": "dsprites",
                "dataset_type": "dsprites_benchmark", "factor_types": factor_types.tolist(), "k": 5,
                "latent_dim": int(request.form.get("latent_dim", 8)), "sparsity": 1.0,
                "dataset_size": len(images), "observation_dim": observation_dim, "noise": 0,
                "train_split": .8, "validation_split": .1, "test_split": .1, "seed": seed, "empirical": True,
                "paper_fidelity": paper_fidelity,
                "image_shape": [1, int(images_processed.shape[1]), int(images_processed.shape[2])],
                "ground_truth_labels": ["shape", "scale", "orientation", "x_position", "y_position"],
                "source_filename": source_filename,
                "source_provenance": {
                    "publisher": "Google DeepMind",
                    "project_url": DATASET_CATALOG["dsprites"]["official_source"],
                    "artifact_url": source_url,
                    "published_format": "native NPZ",
                    "native_npz": True,
                    "source_sha256": source_sha256,
                    "acquisition": source_mode,
                    "downloaded_now": downloaded,
                    "derived_artifact": "dataset.npz",
                },
                "analysis_plan": DEFAULT_ANALYSIS_PLAN, "paper_protocol": {},
            }
            directory = exp_dir(exp_id)
            directory.mkdir(parents=True, exist_ok=False)
            dataset_path = directory / "dataset.npz"
            np.savez_compressed(dataset_path, **data)
            config["dataset_npz_sha256"] = file_sha256(dataset_path)
            config["dataset_artifact"] = "dataset.npz"
            (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            summary = _dataset_summary(data)
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return jsonify({
                "success": True, "experiment": config, "summary": summary,
                "source": {"filename": source_filename, "sha256": source_sha256, "downloaded_now": downloaded},
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/shapes3d/ingest", methods=["POST"])
    def geometry_ingest_shapes3d():
        """Fetch or ingest the official Shapes3D HDF5/NPZ artifact."""
        try:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("h5py is required to import the official Shapes3D dataset") from exc
            upload = request.files.get("file")
            official_requested = str(request.form.get("official", "false")).lower() == "true"
            source_url = DATASET_CATALOG["shapes3d"]["official_file"]
            downloaded = False
            if official_requested:
                source_path, downloaded = download_official_file(source_url, "3dshapes.h5")
                source_sha256 = file_sha256(source_path)
                source_filename = source_path.name
                handle = h5py.File(source_path, "r")
                source_mode = "official_fetch"
            else:
                if upload is None:
                    raise ValueError("Attach the official Shapes3D .h5/.npz or request the official fetch")
                source_filename = upload.filename or "3dshapes.h5"
                source_sha256 = stream_sha256(upload.stream)
                if Path(source_filename).suffix.lower() == ".npz":
                    handle = np.load(upload.stream, allow_pickle=False)
                else:
                    handle = h5py.File(upload.stream, "r")
                source_mode = "manual_official_upload"
            image_key = "images" if "images" in handle else "imgs" if "imgs" in handle else None
            factor_key = "labels" if "labels" in handle else "latents_values" if "latents_values" in handle else None
            if image_key is None or factor_key is None:
                raise ValueError("Shapes3D artifact must contain images and labels")
            total = len(handle[image_key])
            seed = int(request.form.get("seed", 42))
            maximum = max(100, min(480000, int(request.form.get("dataset_size", 10000))))
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(total, size=min(maximum, total), replace=False))
            images = np.asarray(handle[image_key][indices])
            factors = np.asarray(handle[factor_key][indices], dtype=np.float32)
            handle.close()
            if images.ndim != 4:
                raise ValueError("Shapes3D images must have shape [samples, height, width, channels]")
            if images.shape[-1] in {1, 3, 4}:
                images = np.moveaxis(images[..., :3], -1, 1)
            images = images.astype(np.float32)
            if images.max(initial=0) > 1.0:
                images /= 255.0
            if factors.shape[1] != 6:
                raise ValueError("Shapes3D labels must contain six ground-truth factors")
            names = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
            config, summary = save_empirical_experiment(
                request.form.get("name", "Shapes3D original"), "shapes3d", images, factors, names,
                seed=seed, image_shape=list(images.shape[1:]), source_filename=source_filename,
                factor_metric_types=["categorical"] * 6,
                metadata={
                    "dataset_variant": "original", "paper_id": ZIETLOW_PAPER["id"],
                    "paper_title": ZIETLOW_PAPER["title"], "paper_citation": ZIETLOW_PAPER["citation"],
                    "paper_url": ZIETLOW_PAPER["url"], "latent_dim": 6,
                    "source_provenance": {
                        "publisher": "Google DeepMind", "project_url": DATASET_CATALOG["shapes3d"]["official_source"],
                        "artifact_url": source_url, "published_format": "native HDF5",
                        "source_sha256": source_sha256, "acquisition": source_mode,
                        "downloaded_now": downloaded, "selected_samples": len(images), "total_samples": total,
                        "derived_artifact": "dataset.npz",
                    },
                },
            )
            return jsonify({"success": True, "experiment": config, "summary": summary, "source": config["source_provenance"]})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/zietlow-variant", methods=["POST"])
    def geometry_create_zietlow_variant(exp_id):
        """Create a factor-preserving manipulated or matched-noise child dataset."""
        try:
            parent_directory = exp_dir(exp_id)
            parent_config = json.loads((parent_directory / "config.json").read_text(encoding="utf-8"))
            if parent_config.get("dataset_source") not in ZIETLOW_DATASETS:
                raise ValueError("Zietlow variants require a dSprites or Shapes3D parent dataset")
            data = load_data(exp_id)
            variant = str(request.form.get("variant", "uniform_noise")).lower()
            epsilon = float(request.form.get("epsilon", ZIETLOW_DATASETS[parent_config["dataset_source"]]["epsilon"]))
            if not np.isfinite(epsilon) or epsilon <= 0 or epsilon > 1:
                raise ValueError("Epsilon must be in (0, 1]")
            seed = int(request.form.get("seed", parent_config.get("seed", 42)))
            if variant == "uniform_noise":
                observations, statistics = make_zietlow_uniform_noise(data["observations"], epsilon, seed)
                source_filename = "generated uniform pixel noise"
            elif variant == "manipulated":
                upload = request.files.get("file")
                if upload is None:
                    raise ValueError("Attach an NPZ containing 'manipulation' or 'observations' for the manipulated variant")
                loaded = np.load(upload.stream, allow_pickle=False)
                if "manipulation" in loaded:
                    observations, statistics = apply_zietlow_manipulation(data["observations"], loaded["manipulation"], epsilon, seed)
                elif "observations" in loaded:
                    observations = np.asarray(loaded["observations"], dtype=np.float32)
                    if observations.shape != data["observations"].shape:
                        raise ValueError("Manipulated observations must have exactly the parent's sample order and shape")
                    from zietlow_reproduction import transformation_statistics
                    statistics = transformation_statistics(data["observations"], observations, "manipulated", epsilon, seed)
                else:
                    raise ValueError("Manipulated NPZ must contain 'manipulation' or 'observations'")
                loaded.close()
                source_filename = upload.filename or "authors_manipulated_dataset.npz"
            else:
                raise ValueError("Variant must be manipulated or uniform_noise")
            if not np.all(np.isfinite(observations)) or observations.min(initial=0) < 0 or observations.max(initial=0) > 1:
                raise ValueError("Derived observations must be finite and remain in the normalized [0, 1] range")
            if statistics["max_abs_change"] > epsilon + 1e-6:
                raise ValueError(f"The uploaded manipulation exceeds epsilon={epsilon:g}")
            child_id = "geo_zietlow_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            child_directory = exp_dir(child_id)
            child_directory.mkdir(parents=True, exist_ok=False)
            child_data = {key: np.asarray(value) for key, value in data.items()}
            child_data["observations"] = observations.astype(np.float32)
            dataset_path = child_directory / "dataset.npz"
            np.savez_compressed(dataset_path, **child_data)
            config = {
                **parent_config, "id": child_id,
                "name": str(request.form.get("name", f"{parent_config.get('name', exp_id)} · {variant}"))[:120],
                "created_at": _utc_now(), "dataset_variant": variant,
                "parent_experiment_id": exp_id, "parent_dataset_sha256": parent_config.get("dataset_npz_sha256"),
                "dataset_npz_sha256": file_sha256(dataset_path), "dataset_artifact": "dataset.npz",
                "transformation": statistics, "source_filename": source_filename,
                "factor_values_sha256": hashlib.sha256(np.ascontiguousarray(data["factor_values"]).tobytes()).hexdigest(),
                "split_sha256": hashlib.sha256(np.ascontiguousarray(data["split"]).tobytes()).hexdigest(),
                "paper_id": ZIETLOW_PAPER["id"], "paper_title": ZIETLOW_PAPER["title"],
                "paper_citation": ZIETLOW_PAPER["citation"], "paper_url": ZIETLOW_PAPER["url"],
                "run_label": variant, "tags": list(dict.fromkeys([*(parent_config.get("tags") or []), "zietlow2021", variant])),
            }
            summary = _dataset_summary(child_data)
            config["split_counts"] = dict(summary["split_counts"])
            (child_directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (child_directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            (child_directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (child_directory / "transformation.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
            return jsonify({"success": True, "experiment": config, "summary": summary, "transformation": statistics})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/datasets/upload", methods=["POST"])
    def geometry_upload_dataset():
        """Import user observations from NPZ or CSV with optional ground truth."""
        try:
            upload = request.files.get("file")
            if upload is None:
                raise ValueError("Attach an .npz or .csv dataset")
            seed = int(request.form.get("seed", 42))
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix == ".npz":
                loaded = np.load(upload.stream, allow_pickle=False)
                observation_key = "observations" if "observations" in loaded else "x" if "x" in loaded else None
                if observation_key is None:
                    raise ValueError("NPZ must contain 'observations' or 'x'")
                observations = loaded[observation_key]
                factor_key = "factor_values" if "factor_values" in loaded else "factors" if "factors" in loaded else "labels" if "labels" in loaded else None
                factors = loaded[factor_key] if factor_key else np.zeros((len(observations), 1), dtype=np.float32)
            elif suffix == ".csv":
                matrix = np.genfromtxt(upload.stream, delimiter=",", names=True, dtype=np.float32)
                if matrix.dtype.names:
                    matrix = np.column_stack([matrix[name] for name in matrix.dtype.names])
                observations = np.asarray(matrix, dtype=np.float32)
                factors = np.zeros((len(observations), 1), dtype=np.float32)
            else:
                raise ValueError("Dataset upload supports .npz and .csv")
            names = [item.strip() for item in request.form.get("factor_names", "").split(",") if item.strip()]
            if len(names) != (factors.shape[1] if factors.ndim > 1 else 1):
                names = [f"factor_{index + 1}" for index in range(factors.shape[1] if factors.ndim > 1 else 1)]
            config, summary = save_empirical_experiment(
                request.form.get("name", Path(upload.filename).stem), "uploaded", observations, factors,
                names, seed=seed, source_filename=upload.filename,
            )
            return jsonify({"success": True, "experiment": config, "summary": summary})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/datasets/benchmark", methods=["POST"])
    def geometry_import_benchmark():
        """Fetch published benchmark artifacts and convert a seeded subset to NPZ."""
        try:
            raw = request.get_json(force=True) or {}
            dataset_name = str(raw.get("dataset", "mnist")).lower()
            if dataset_name not in {"mnist", "fashion_mnist", "celeba"}:
                raise ValueError("Benchmark must be MNIST, Fashion-MNIST, or CelebA")
            try:
                from torchvision import datasets, transforms
            except ImportError as exc:
                raise RuntimeError("torchvision is required for benchmark datasets") from exc
            seed = int(raw.get("seed", 42))
            maximum = max(100, min(250000, int(raw.get("dataset_size", 10000))))
            download = bool(raw.get("download", False))
            image_size = int(raw.get("image_size", 64 if dataset_name == "celeba" else 28))
            if dataset_name == "celeba":
                image_size = max(32, min(128, image_size))
            else:
                image_size = 28
            transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
            if dataset_name == "mnist":
                dataset = torch.utils.data.ConcatDataset([
                    datasets.MNIST(datasets_root, train=True, transform=transform, download=download),
                    datasets.MNIST(datasets_root, train=False, transform=transform, download=download),
                ])
                factor_names = ["digit"]
            elif dataset_name == "fashion_mnist":
                dataset = torch.utils.data.ConcatDataset([
                    datasets.FashionMNIST(datasets_root, train=True, transform=transform, download=download),
                    datasets.FashionMNIST(datasets_root, train=False, transform=transform, download=download),
                ])
                factor_names = ["fashion_class"]
            else:
                dataset = datasets.CelebA(datasets_root, split="all", target_type="attr", transform=transform, download=download)
                selected_attributes = ["Smiling", "Male", "Eyeglasses", "Blond_Hair", "Young"]
                attribute_indices = [dataset.attr_names.index(name) for name in selected_attributes]
                factor_names = selected_attributes
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(len(dataset), size=min(maximum, len(dataset)), replace=False))
            images, labels = [], []
            for index in indices:
                image, target = dataset[int(index)]
                images.append(image.numpy())
                if dataset_name == "celeba":
                    labels.append(target.numpy()[attribute_indices])
                else:
                    labels.append([int(target)])
            observations = np.stack(images).astype(np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            catalog_item = DATASET_CATALOG[dataset_name]
            config, summary = save_empirical_experiment(
                raw.get("name", DATASET_CATALOG[dataset_name]["label"]), dataset_name, observations,
                labels, factor_names, seed=seed, image_shape=list(observations.shape[1:]),
                factor_metric_types=["categorical"] * len(factor_names),
                metadata={
                    "download_requested": download, "torchvision_split": "all",
                    "source_provenance": {
                        "project_url": catalog_item["official_source"],
                        "published_format": catalog_item["official_format"].split(" → ")[0],
                        "native_npz": False,
                        "acquisition": "torchvision official resources",
                        "source_verification": "torchvision resource checksum validation",
                        "conversion": "seeded full-dataset subset; 80/10/10 split; resize; float tensor; no relabeling",
                        "image_size": image_size,
                        "derived_artifact": "dataset.npz",
                    },
                },
            )
            return jsonify({
                "success": True, "experiment": config, "summary": summary,
                "source": config["source_provenance"],
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>", methods=["GET"])
    def geometry_get_experiment(exp_id):
        try:
            directory = exp_dir(exp_id)
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else _dataset_summary(load_data(exp_id))
            analysis_path = directory / "analysis.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else None
            scientific_path = directory / "scientific_metrics.json"
            scientific = json.loads(scientific_path.read_text(encoding="utf-8")) if scientific_path.exists() else None
            scientific = add_paper_requirements(directory, config, scientific)
            zietlow_path = directory / "zietlow_metrics.json"
            zietlow = json.loads(zietlow_path.read_text(encoding="utf-8")) if zietlow_path.exists() else None
            return jsonify({
                "success": True, "experiment": presentation_config(config),
                "summary": summary, "analysis": analysis, "scientific": scientific, "zietlow": zietlow,
            })
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/geometry/experiments/<exp_id>/identity", methods=["PATCH"])
    def geometry_update_identity(exp_id):
        try:
            directory = exp_dir(exp_id)
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw = request.get_json(force=True) or {}
            limits = {"name": 120, "run_label": 120, "paper_title": 300, "paper_citation": 1000,
                      "paper_url": 1000, "description": 2000, "hypothesis": 2000}
            for key, limit in limits.items():
                if key in raw:
                    config[key] = str(raw[key]).strip()[:limit]
            if not config.get("name"):
                raise ValueError("Experiment name cannot be empty")
            if "tags" in raw:
                config["tags"] = [str(tag).strip()[:80] for tag in raw["tags"][:30] if str(tag).strip()]
            if config.get("paper_id") == ZIETLOW_PAPER["id"] and "run_label" in raw:
                config["suite_id"] = config["run_label"]
            config["identity_updated_at"] = _utc_now()
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            return jsonify({"success": True, "experiment": config})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/clone", methods=["POST"])
    def geometry_clone_experiment_dataset(exp_id):
        """Register a fresh untrained experiment over an existing immutable dataset."""
        try:
            source_directory = exp_dir(exp_id)
            source_config_path = source_directory / "config.json"
            source_dataset_path = source_directory / "dataset.npz"
            if not source_config_path.exists() or not source_dataset_path.exists():
                raise FileNotFoundError("The active experiment dataset is unavailable")
            source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
            raw = request.get_json(force=True) or {}
            experiment_id = "geo_run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            directory = exp_dir(experiment_id)
            directory.mkdir(parents=True, exist_ok=False)
            try:
                os.link(source_dataset_path, directory / "dataset.npz")
            except OSError:
                shutil.copy2(source_dataset_path, directory / "dataset.npz")
            if (source_directory / "summary.json").exists():
                shutil.copy2(source_directory / "summary.json", directory / "summary.json")
            if (source_directory / "transformation.json").exists():
                shutil.copy2(source_directory / "transformation.json", directory / "transformation.json")
            config = {
                **source_config,
                "id": experiment_id,
                "name": str(raw.get("name", f"{source_config.get('name', exp_id)} · new run")).strip()[:120],
                "created_at": _utc_now(),
                "parent_experiment_id": exp_id,
                "dataset_npz_sha256": source_config.get("dataset_npz_sha256") or file_sha256(directory / "dataset.npz"),
                "analysis_plan": _analysis_plan(raw.get("analysis_plan")),
                "paper_protocol": raw.get("paper_protocol") if isinstance(raw.get("paper_protocol"), dict) else {},
                "run_label": str(raw.get("run_label", source_config.get("run_label", "manual"))).strip()[:120],
                "tags": [str(tag).strip()[:80] for tag in (raw.get("tags") or source_config.get("tags") or [])[:30] if str(tag).strip()],
            }
            for stale_key in ("effective_training_seed", "training_provenance_resolved_at", "identity_updated_at", "workflow_updated_at"):
                config.pop(stale_key, None)
            if config["paper_protocol"].get("paper_id"):
                config["paper_id"] = str(config["paper_protocol"]["paper_id"])[:80]
            if config.get("paper_id") == ZIETLOW_PAPER["id"] and config.get("run_label"):
                config["suite_id"] = str(raw.get("suite_id", config["run_label"])).strip()[:120]
            # Paper presets may make dataset semantics explicit even when an
            # older imported parent predates the dataset_variant field.
            for protocol_key in ("dataset_source", "dataset_variant"):
                if config["paper_protocol"].get(protocol_key):
                    config[protocol_key] = str(config["paper_protocol"][protocol_key])[:80]
            if config["paper_protocol"].get("latent_dim") is not None:
                config["latent_dim"] = int(config["paper_protocol"]["latent_dim"])
            for key, limit in {
                "paper_title": 300, "paper_citation": 1000, "paper_url": 1000,
                "description": 2000, "hypothesis": 2000,
            }.items():
                if key in raw:
                    config[key] = str(raw[key]).strip()[:limit]
            if not config["name"]:
                raise ValueError("Experiment name cannot be empty")
            (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else _dataset_summary(load_data(experiment_id))
            if not summary_path.exists():
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return jsonify({"success": True, "experiment": config, "summary": summary})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/workflow", methods=["PATCH"])
    def geometry_update_workflow(exp_id):
        try:
            directory = exp_dir(exp_id)
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            raw = request.get_json(force=True) or {}
            config["analysis_plan"] = _analysis_plan(raw.get("analysis_plan"))
            protocol = raw.get("paper_protocol", {})
            if not isinstance(protocol, dict):
                raise ValueError("Paper protocol must be a JSON object")
            config["paper_protocol"] = protocol
            if protocol.get("paper_id"):
                config["paper_id"] = str(protocol["paper_id"])[:80]
            if protocol.get("paper_title") and not config.get("paper_title"):
                config["paper_title"] = str(protocol["paper_title"])[:300]
            if protocol.get("paper_url") and not config.get("paper_url"):
                config["paper_url"] = str(protocol["paper_url"])[:1000]
            config["workflow_updated_at"] = _utc_now()
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
            return jsonify({"success": True, "experiment": config})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    def write_results_report(exp_id):
        """Build a portable, human-readable report from persisted run artifacts."""
        directory = exp_dir(exp_id)

        def read_json(name, default=None):
            path = directory / name
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

        experiment = read_json("config.json", {})
        training = read_json("training_config.json")
        analysis = read_json("analysis.json")
        if not training or not analysis:
            raise FileNotFoundError("Train the selected experiment before exporting its results report")
        dataset_summary = read_json("summary.json", {})
        scientific = read_json("scientific_metrics.json", {})
        paper = read_json("paper_metrics.json", {})
        zietlow = read_json("zietlow_metrics.json", {})
        environment = read_json("environment.json", {})
        report_training = dict(training)
        report_training.setdefault("bias", True)
        paper_run = bool(report_training.get("paper_reproduction") or experiment.get("paper_protocol"))
        report_training.setdefault(
            "batch_size_source",
            "declared_reproduction_assumption" if paper_run else "user_configured",
        )
        report_training.setdefault(
            "batch_size_assumption",
            PAPER_BATCH_SIZE_ASSUMPTION if paper_run else "",
        )
        report_training.setdefault("seed_source", "effective_training_configuration")
        resolved_protocol = _resolved_paper_protocol(experiment.get("paper_protocol"), report_training)
        split_counts = dataset_summary.get("split_counts") or experiment.get("split_counts") or {}
        generator_type = experiment.get("generator_type") or experiment.get("dataset_source")
        dataset_protocol = experiment.get("dataset_protocol") or (
            PAPER_LINEAR_DATASET_PROTOCOL if generator_type == "paper_linear"
            else PAPER_NONLINEAR_DATASET_PROTOCOL if generator_type == "paper_nonlinear"
            else {}
        )
        final_metric = (analysis.get("metrics") or [{}])[-1]
        scientific = add_paper_requirements(directory, experiment, scientific)
        scientific_summary = scientific.get("summary", {})
        paper_requirements = scientific.get("paper_requirements", {})

        def value(item, digits=6):
            if item is None:
                return "not available"
            if isinstance(item, float):
                return f"{item:.{digits}g}"
            return str(item)

        lines = [
            f"# Results report: {experiment.get('name', exp_id)}", "",
            f"Generated: {_utc_now()}", "",
            "## Study identity", "",
            f"- Experiment ID: `{exp_id}`",
            f"- Run label: `{experiment.get('run_label', '')}`",
            f"- Paper: {experiment.get('paper_title') or 'not specified'}",
            f"- Citation: {experiment.get('paper_citation') or 'not specified'}",
            f"- Paper URL: {experiment.get('paper_url') or 'not specified'}",
            f"- Hypothesis: {experiment.get('hypothesis') or 'not specified'}", "",
            "## Dataset", "",
            "| Field | Effective value |", "|---|---|",
            f"| Source | `{experiment.get('dataset_source', '')}` |",
            f"| Dataset type | `{display_dataset_type(experiment)}` |",
            f"| Samples | {experiment.get('dataset_size', dataset_summary.get('sample_count', ''))} |",
            f"| Observation dimension | {experiment.get('observation_dim', dataset_summary.get('observation_dim', ''))} |",
            f"| Factors | {experiment.get('k', dataset_summary.get('factor_count', ''))} |",
            f"| Nominal train / validation / test fractions | {experiment.get('train_split', '')} / {experiment.get('validation_split', '')} / {experiment.get('test_split', '')} |",
            f"| Exact train / validation / test samples | {split_counts.get('train', 'not available')} / {split_counts.get('validation', 'not available')} / {split_counts.get('test', 'not available')} |",
            f"| Dataset seed | {experiment.get('seed', '')} |",
            f"| Dataset SHA-256 | `{experiment.get('dataset_npz_sha256', 'not available')}` |", "",
        ]
        if (
            (experiment.get("generator_type") == "paper_linear" or experiment.get("dataset_source") == "paper_linear")
            and split_counts.get("test") != 5_000
        ):
            lines.extend([
                "Legacy split warning: this persisted dataset predates exact integer allocation. "
                f"Its paper metrics used {split_counts.get('test', 'an unknown number of')} test samples; "
                "regenerate, retrain, and reanalyze it before treating it as a corrected reproduction.", "",
            ])
        if dataset_protocol:
            lines.extend(["## Dataset construction protocol", "", "| Field | Effective value |", "|---|---|"])
            if generator_type == "paper_nonlinear":
                lines.extend([
                    f"| Factor domain | `{dataset_protocol.get('factor_domain', 'not available')}` |",
                    f"| Mapping | `{dataset_protocol.get('mapping', 'not available')}` |",
                    f"| Hidden layers | `{dataset_protocol.get('hidden_layers', 'not available')}` |",
                    f"| Hidden activation | `{dataset_protocol.get('hidden_activation', 'not available')}` |",
                    f"| Output activation | `{dataset_protocol.get('output_activation', 'not available')}` |",
                    f"| Biases | `{dataset_protocol.get('biases', 'not available')}` |",
                    f"| Implementation initialization | `{json.dumps(dataset_protocol.get('implementation_initialization', {}), sort_keys=True)}` |",
                    f"| Exact generator tensors | `{', '.join(dataset_protocol.get('saved_generator_tensors', []))}` in `dataset.npz` |",
                    f"| Paper location | `{dataset_protocol.get('paper_location', 'not available')}` |", "",
                ])
            else:
                lines.extend([
                    f"| Factor domain | `{dataset_protocol.get('factor_domain', 'not available')}` |",
                    f"| Stretch | factor {dataset_protocol.get('stretched_factor', 'not available')} multiplied by {value(dataset_protocol.get('stretch_factor'))} |",
                    f"| Embedding | `{dataset_protocol.get('embedding', 'not available')}` |",
                    f"| Rotation | {value(dataset_protocol.get('rotation_degrees'))} degrees about axis `{dataset_protocol.get('rotation_axis', 'not available')}` |",
                    f"| Transformation order | `{' -> '.join(dataset_protocol.get('transformation_order', []))}` |",
                    f"| Paper location | `{dataset_protocol.get('paper_location', 'not available')}` |", "",
                ])
        lines.extend([
            "## Effective model and training configuration", "",
            "| Field | Effective value |", "|---|---|",
        ])
        training_fields = [
            ("Model", "model_type"), ("Latent dimension", "latent_dim"),
            ("Epoch limit/fallback", "epochs"), ("Requested optimizer steps", "training_steps"),
            ("Training budget unit", "training_budget_unit"), ("Optimizer", "optimizer"),
            ("Learning rate", "learning_rate"), ("Batch size", "batch_size"),
            ("Adam beta 1", "adam_beta1"), ("Adam beta 2", "adam_beta2"),
            ("Optimizer epsilon", "optimizer_epsilon"), ("Weight decay", "weight_decay"),
            ("Reconstruction objective", "reconstruction_loss"),
            ("Batch size provenance", "batch_size_source"), ("Layer biases enabled", "bias"),
            ("Beta / KL weight", "beta"), ("Encoder activation", "encoder_activation"),
            ("Decoder activation", "decoder_activation"), ("Training seed", "seed"),
            ("Training seed provenance", "seed_source"),
            ("Paper reproduction diagnostics", "paper_reproduction"),
            ("Polarization sample cap", "polarization_max_samples"),
        ]
        lines.extend(f"| {label} | `{value(report_training.get(key))}` |" for label, key in training_fields)
        if report_training.get("batch_size_assumption"):
            lines.extend([
                "", f"Batch-size assumption: {report_training['batch_size_assumption']}",
                "The selected value is an experimental assumption, not a value attributed to the paper.",
            ])
        lines.extend(["", "## Final training results", "", "| Metric | Value |", "|---|---:|"])
        final_fields = [
            ("Total loss", "total_loss"), ("Reconstruction loss", "reconstruction_loss"),
            ("KL divergence", "kl_divergence"), ("Regularizer", "regularizer"),
            ("Validation reconstruction", "validation_reconstruction"),
            ("Mean latent variance", "latent_variance"),
            ("Active latent dimensions", "active_latent_dimensions"),
        ]
        lines.extend(f"| {label} | {value(final_metric.get(key))} |" for label, key in final_fields)
        lines.extend([
            f"| Test reconstruction error | {value(analysis.get('test_reconstruction_error'))} |",
            "", "## Paper-defined results", "",
        ])
        if zietlow:
            metric_scope = zietlow.get("evaluation_scope", "not recorded")
            lines.extend([
                "### Zietlow et al. (2021) metrics", "",
                "| Metric | Value | Scope |", "|---|---:|---|",
                f"| MIG | {value((zietlow.get('mig') or {}).get('score'))} | {metric_scope} |",
                f"| DCI Disentanglement | {value((zietlow.get('dci') or {}).get('disentanglement'))} | {metric_scope} |",
                f"| FactorVAE score | {value((zietlow.get('factor_vae_score') or {}).get('score'))} | {metric_scope} |",
                f"| SAP | {value((zietlow.get('sap') or {}).get('score'))} | {metric_scope} |",
                f"| Active units | {value((zietlow.get('active_units') or {}).get('count'))} | rule E[sigma_i^2] < 0.8 |",
                f"| Over-pruned | {value((zietlow.get('active_units') or {}).get('over_pruned'))} | retained, excluded only from robustness aggregate |", "",
                "The Zietlow reproduction verdict uses these paper-defined quantities. DtO and Jacobian diagnostics below are additional analyses and do not affect that verdict.", "",
            ])
        if paper:
            dto = paper.get("dto", {})
            disentanglement = paper.get("disentanglement", {})
            polarization = paper.get("polarization_final") or {}
            lines.extend([
                "| Metric | Value | Scope |", "|---|---:|---|",
                f"| DtO, Equation 29 | {value(dto.get('mean'))} | {paper.get('evaluation_scope', '')}; n={paper.get('test_samples', '')} |",
                f"| DtO sample standard deviation | {value(dto.get('sample_standard_deviation'))} | held-out test |",
                f"| Disentanglement, Equations 65–70 | {value(disentanglement.get('score'))} | held-out test, internal 80/20 probe split |",
                f"| Final Delta_KL, Equation 30 | {value(polarization.get('delta_kl'))} | final diagnostic |",
            ])
            if disentanglement.get("not_applicable_reason"):
                lines.extend(["", f"Disentanglement is not applicable: {disentanglement['not_applicable_reason']}"])
        else:
            lines.append("Paper-defined scientific metrics have not been computed. Run **Scientific Metrics** with the paper checkbox enabled, then export this report again.")
        polarization_summary = analysis.get("polarization_summary", {})
        lines.extend([
            "", "### Polarized-regime training result", "",
            f"- Delta_KL threshold: {value(polarization_summary.get('threshold_delta_kl'))}",
            f"- Active-coordinate rule: `{polarization_summary.get('active_selection_rule', 'sqrt(var(mu_j(x_i))) > 0.5')}`",
            f"- Evaluated every: {value(polarization_summary.get('evaluation_interval_batches'))} minibatches",
            f"- Continuously below threshold until the end: {value(polarization_summary.get('continuous_percent_to_end'))}%", "",
            "## Paper reproduction requirements", "",
        ])
        if paper_requirements.get("checks"):
            lines.extend([
                f"Overall status: **{paper_requirements.get('status', 'not available')}** "
                f"({paper_requirements.get('passed', 0)}/{paper_requirements.get('applicable', 0)} passed).", "",
                "| Requirement | Observed | Threshold | Status | Source |", "|---|---:|---:|---|---|",
            ])
            operator_labels = {"at_most": "≤", "at_least": "≥", "between": "within"}
            for check in paper_requirements["checks"]:
                threshold = check.get("threshold")
                if check.get("operator") == "between" and isinstance(threshold, list):
                    threshold = f"[{value(threshold[0])}, {value(threshold[1])}]"
                else:
                    threshold = value(threshold)
                lines.append(
                    f"| {check['label']} | {value(check.get('actual'))} | "
                    f"{operator_labels.get(check.get('operator'), '')} {threshold} | "
                    f"{check.get('status', '')} | {check.get('source', '')} |"
                )
            lines.extend(["", paper_requirements.get("note", ""), ""])
        else:
            lines.extend(["No published requirement target applies to this dataset/model/latent-dimension combination.", ""])
        lines.extend([
            "## Dashboard interpretability extensions", "",
            "| Metric | Value |", "|---|---:|",
        ])
        auxiliary_fields = [
            ("Gram off-diagonal", "decoder_gram_offdiagonal"),
            ("Tangent overlap", "tangent_overlap"), ("Grassmann distance", "grassmann_distance"),
            ("Factor recoverability", "factor_recoverability"),
            ("Topology preservation", "topology_preservation"),
            ("Manifold distortion", "manifold_distortion"),
            ("Causal interference", "causal_interference"),
            ("Mean Jacobian rank", "mean_jacobian_rank"),
            ("Mean condition number", "mean_condition_number"),
        ]
        lines.extend(f"| {label} | {value(scientific_summary.get(key))} |" for label, key in auxiliary_fields)
        lines.extend(["", "## Reproducibility artifacts", "", "| Artifact | Bytes | SHA-256 |", "|---|---:|---|"])
        artifact_names = [
            "config.json", "config.yaml", "dataset.npz", "training_config.json", "environment.json",
            "model.pt", "metrics.json", "analysis.json", "latent.npz", "polarization_trace.json",
            "scientific_metrics.json", "paper_metrics.json", "jacobians.npz",
            "zietlow_metrics.json", "transformation.json",
        ]
        for name in artifact_names:
            path = directory / name
            if path.exists():
                lines.append(f"| `{name}` | {path.stat().st_size} | `{file_sha256(path)}` |")
        lines.extend([
            "", "## Runtime environment", "", "```json",
            json.dumps(environment, indent=2, sort_keys=True), "```", "",
            "## Resolved paper protocol", "",
            "This protocol is resolved against the effective training configuration, so its seed, batch size, and bias fields describe the run that produced the checkpoint.", "",
            "```json", json.dumps(resolved_protocol, indent=2, sort_keys=True), "```", "",
            "## Interpretation note", "",
            "Paper-defined quantities and dashboard extensions are reported separately. A numerical discrepancy should be retained and investigated rather than removed by selecting a favorable seed.", "",
        ])
        target = directory / "reproduction_report.md"
        target.write_text("\n".join(lines), encoding="utf-8")
        return target

    BUNDLE_FORMAT = "manifold-superposition-experiment"
    BUNDLE_SUFFIXES = {".json", ".yaml", ".yml", ".npz", ".pt", ".md", ".txt"}
    BUNDLE_EXCLUDED = {"import_manifest.json"}

    def experiment_bundle(exp_id, profile):
        """Build or reuse a content-addressed, portable experiment bundle."""
        profile = str(profile or "results").lower()
        if profile not in {"results", "complete"}:
            raise ValueError("Bundle profile must be results or complete")
        directory = exp_dir(exp_id)
        if not (directory / "config.json").exists():
            raise FileNotFoundError("Experiment configuration not found")
        candidates = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name in BUNDLE_EXCLUDED or path.suffix.lower() not in BUNDLE_SUFFIXES:
                continue
            if profile == "results" and path.suffix.lower() in {".npz", ".pt"}:
                continue
            candidates.append(path)
        required = {"config.json", "summary.json"}
        present = {path.name for path in candidates}
        if not required.issubset(present):
            raise ValueError("Experiment is missing its configuration or summary")
        if profile == "complete" and "dataset.npz" not in present:
            raise ValueError("A complete bundle requires dataset.npz")

        artifacts = []
        for path in candidates:
            suffix = path.suffix.lower()
            role = ("dataset" if path.name == "dataset.npz" else
                    "checkpoint" if path.name == "model.pt" else
                    "raw_results" if suffix == ".npz" else
                    "metadata")
            artifacts.append({
                "name": path.name, "size": path.stat().st_size,
                "sha256": file_sha256(path), "role": role,
                "zip_compression": "stored" if suffix in {".npz", ".pt"} else "deflate",
            })
        fingerprint_payload = json.dumps(
            [{"name": item["name"], "sha256": item["sha256"]} for item in artifacts],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        manifest = {
            "format": BUNDLE_FORMAT, "schema_version": 1, "profile": profile,
            "created_at": _utc_now(), "content_fingerprint": fingerprint,
            "experiment": {
                "id": exp_id, "name": config.get("name", exp_id),
                "dataset_type": config.get("dataset_type", config.get("dataset_source", "")),
                "trained": (directory / "analysis.json").exists(),
            },
            "artifacts": artifacts,
            "capabilities": {
                "inspect_results": (directory / "analysis.json").exists(),
                "reanalyze": profile == "complete" and "model.pt" in present,
                "retrain": profile == "complete" and "dataset.npz" in present,
            },
        }
        target = exports_root / f"{exp_id}_{profile}_{fingerprint[:12]}.mslab"
        if target.exists() and target.stat().st_size > 0:
            return target, manifest
        temporary = target.with_suffix(".mslab.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2))
                for path, artifact in zip(candidates, artifacts):
                    compression = zipfile.ZIP_STORED if artifact["zip_compression"] == "stored" else zipfile.ZIP_DEFLATED
                    archive.write(path, f"artifacts/{path.name}", compress_type=compression)
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target, manifest

    def _matching_imported_experiment(fingerprint):
        for directory in root.iterdir():
            manifest_path = directory / "import_manifest.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                imported = json.loads(manifest_path.read_text(encoding="utf-8"))
                if imported.get("content_fingerprint") == fingerprint:
                    return directory.name
            except Exception:
                continue
        return None

    @app.route("/api/geometry/experiments/import-bundle", methods=["POST"])
    def geometry_import_bundle():
        """Validate and atomically restore a portable experiment bundle."""
        upload = request.files.get("file")
        if upload is None:
            return jsonify({"success": False, "error": "Attach an .mslab experiment bundle"}), 400
        try:
            with zipfile.ZipFile(upload.stream, "r", allowZip64=True) as archive:
                infos = archive.infolist()
                if len(infos) > 129:
                    raise ValueError("Bundle contains too many files")
                by_name = {info.filename: info for info in infos}
                if len(by_name) != len(infos) or "manifest.json" not in by_name:
                    raise ValueError("Bundle manifest is missing or file names are duplicated")
                manifest_info = by_name["manifest.json"]
                if manifest_info.file_size > 2 * 1024 * 1024:
                    raise ValueError("Bundle manifest is too large")
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                if manifest.get("format") != BUNDLE_FORMAT or manifest.get("schema_version") != 1:
                    raise ValueError("Unsupported experiment bundle format")
                profile = manifest.get("profile")
                if profile not in {"results", "complete"}:
                    raise ValueError("Invalid bundle profile")
                artifacts = manifest.get("artifacts")
                if not isinstance(artifacts, list) or not (1 <= len(artifacts) <= 128):
                    raise ValueError("Bundle artifact list is invalid")

                expected_members = {"manifest.json"}
                seen_names = set()
                total_size = 0
                for artifact in artifacts:
                    name = str(artifact.get("name", ""))
                    pure_name = PurePosixPath(name)
                    if (not name or pure_name.name != name or name in seen_names or
                            Path(name).suffix.lower() not in BUNDLE_SUFFIXES):
                        raise ValueError("Bundle contains an unsafe artifact name")
                    seen_names.add(name)
                    member = f"artifacts/{name}"
                    expected_members.add(member)
                    info = by_name.get(member)
                    if info is None or info.is_dir():
                        raise ValueError(f"Bundle artifact is missing: {name}")
                    if info.flag_bits & 0x1:
                        raise ValueError("Encrypted experiment bundles are not supported")
                    mode = (info.external_attr >> 16) & 0xFFFF
                    if stat.S_ISLNK(mode):
                        raise ValueError("Symbolic links are not allowed in experiment bundles")
                    if info.file_size != int(artifact.get("size", -1)):
                        raise ValueError(f"Artifact size mismatch: {name}")
                    total_size += info.file_size
                if set(by_name) != expected_members:
                    raise ValueError("Bundle contains files not declared by its manifest")
                if total_size > 64 * 1024 ** 3:
                    raise ValueError("Bundle expands beyond the 64 GiB safety limit")
                if not {"config.json", "summary.json"}.issubset(seen_names):
                    raise ValueError("Bundle must contain config.json and summary.json")
                if profile == "complete" and "dataset.npz" not in seen_names:
                    raise ValueError("Complete bundle is missing dataset.npz")

                for artifact in artifacts:
                    digest = hashlib.sha256()
                    with archive.open(f"artifacts/{artifact['name']}", "r") as source:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != artifact.get("sha256"):
                        raise ValueError(f"Checksum mismatch: {artifact['name']}")

                expected_fingerprint = hashlib.sha256(json.dumps(
                    [{"name": item["name"], "sha256": item["sha256"]} for item in artifacts],
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                if expected_fingerprint != manifest.get("content_fingerprint"):
                    raise ValueError("Bundle content fingerprint is invalid")
                existing_import = _matching_imported_experiment(expected_fingerprint)
                if existing_import:
                    config = json.loads((exp_dir(existing_import) / "config.json").read_text(encoding="utf-8"))
                    return jsonify({
                        "success": True, "experiment": config, "deduplicated": True,
                        "profile": profile, "content_fingerprint": expected_fingerprint,
                    })

                imported_config = json.loads(archive.read("artifacts/config.json").decode("utf-8"))
                imported_summary = json.loads(archive.read("artifacts/summary.json").decode("utf-8"))
                required_config_fields = {"name", "created_at", "k", "dataset_size"}
                if not isinstance(imported_config, dict) or not required_config_fields.issubset(imported_config):
                    raise ValueError("Bundle experiment configuration is incomplete")
                if not isinstance(imported_summary, dict):
                    raise ValueError("Bundle summary.json must contain an object")
                original_id = str(manifest.get("experiment", {}).get("id") or imported_config.get("id") or "")
                try:
                    target_id = _safe_id(original_id)
                except ValueError:
                    target_id = "imported_" + uuid.uuid4().hex[:12]
                target_directory = exp_dir(target_id)
                if target_directory.exists():
                    local_match = all(
                        (target_directory / item["name"]).is_file()
                        and file_sha256(target_directory / item["name"]) == item["sha256"]
                        for item in artifacts
                    )
                    if local_match:
                        config = json.loads((target_directory / "config.json").read_text(encoding="utf-8"))
                        return jsonify({
                            "success": True, "experiment": config, "deduplicated": True,
                            "profile": profile, "content_fingerprint": expected_fingerprint,
                        })
                    target_id = f"{target_id}_import_{uuid.uuid4().hex[:6]}"
                    target_directory = exp_dir(target_id)

                staging = Path(tempfile.mkdtemp(prefix=".bundle_import_", dir=root))
                try:
                    for artifact in artifacts:
                        destination = staging / artifact["name"]
                        with archive.open(f"artifacts/{artifact['name']}", "r") as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                    imported_config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
                    if not isinstance(imported_config, dict):
                        raise ValueError("Imported config.json must contain an object")
                    imported_config.update({
                        "id": target_id, "imported_at": _utc_now(),
                        "imported_from_experiment_id": original_id,
                        "bundle_profile": profile,
                        "bundle_content_fingerprint": expected_fingerprint,
                        "dataset_available": (staging / "dataset.npz").exists(),
                        "checkpoint_available": (staging / "model.pt").exists(),
                        "results_only": profile == "results",
                    })
                    (staging / "config.json").write_text(json.dumps(imported_config, indent=2), encoding="utf-8")
                    (staging / "config.yaml").write_text(_to_yaml(imported_config) + "\n", encoding="utf-8")
                    import_record = {
                        **manifest, "imported_at": imported_config["imported_at"],
                        "source_bundle_filename": upload.filename,
                        "restored_experiment_id": target_id,
                    }
                    (staging / "import_manifest.json").write_text(json.dumps(import_record, indent=2), encoding="utf-8")
                    os.replace(staging, target_directory)
                except Exception:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            return jsonify({
                "success": True, "experiment": imported_config, "deduplicated": False,
                "profile": profile, "content_fingerprint": expected_fingerprint,
                "artifact_count": len(artifacts), "total_bytes": total_size,
            })
        except (zipfile.BadZipFile, zipfile.LargeZipFile, UnicodeDecodeError, json.JSONDecodeError,
                KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/export", methods=["GET"])
    def geometry_export(exp_id):
        try:
            directory = exp_dir(exp_id)
            fmt = request.args.get("format", "npz").lower()
            allowed = {
                "npz": "dataset.npz", "json": "config.json", "yaml": "config.yaml",
                "jacobians": "jacobians.npz", "metrics": "scientific_metrics.json",
                "paper_metrics": "paper_metrics.json", "polarization": "polarization_trace.json",
                "checkpoint": "model.pt", "training_config": "training_config.json",
                "environment": "environment.json", "latents": "latent.npz",
                "zietlow_metrics": "zietlow_metrics.json", "transformation": "transformation.json",
            }
            if fmt == "bundle":
                profile = request.args.get("profile", "results").lower()
                filename, manifest = experiment_bundle(exp_id, profile)
                return send_file(
                    filename, as_attachment=True,
                    download_name=f"{exp_id}_{profile}_{manifest['content_fingerprint'][:12]}.mslab",
                    mimetype="application/zip",
                )
            if fmt == "report":
                filename = write_results_report(exp_id)
            elif fmt == "csv":
                target = directory / "dataset.csv"
                if not target.exists():
                    data = load_data(exp_id)
                    with target.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        xdim, k = data["observations"].shape[1], data["factor_values"].shape[1]
                        writer.writerow([f"x_{i}" for i in range(xdim)] + [f"factor_{i + 1}" for i in range(k)] + [f"active_{i + 1}" for i in range(k)] + ["split"])
                        for x, values, active, split in zip(data["observations"], data["factor_values"], data["factor_active"], data["split"]):
                            split_name = {0: "train", 1: "validation", 2: "test"}.get(int(split), "unknown")
                            writer.writerow(x.tolist() + values.tolist() + active.astype(int).tolist() + [split_name])
                filename = target
            else:
                if fmt not in allowed:
                    raise ValueError("Unsupported export format")
                filename = directory / allowed[fmt]
            return send_file(filename, as_attachment=True, download_name=f"{exp_id}_{filename.name}")
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    def train_job(job_id, exp_id, raw_config):
        try:
            experiment_config = json.loads((exp_dir(exp_id) / "config.json").read_text(encoding="utf-8"))
            # The saved paper protocol carries non-visible scientific settings
            # (paper_id, image architecture, exact metrics). Explicit run-form
            # values override it, but cannot accidentally discard it.
            merged_config = {**(experiment_config.get("paper_protocol") or {}), **dict(raw_config)}
            # Image-backed experiments own the tensor shape. Supplying it here
            # makes convolutional checkpoints self-contained and prevents a UI
            # client from accidentally training against an incompatible shape.
            merged_config["image_shape"] = experiment_config.get("image_shape", [])
            config = _normalized_training_config(merged_config)
            config["dataset_version"] = exp_id
            config["analysis_plan"] = _analysis_plan(experiment_config.get("analysis_plan"))
            config["paper_protocol"] = _resolved_paper_protocol(
                experiment_config.get("paper_protocol", {}), config,
            )
            experiment_config["paper_protocol"] = dict(config["paper_protocol"])
            experiment_config["effective_training_seed"] = int(config["seed"])
            experiment_config["training_provenance_resolved_at"] = _utc_now()
            experiment_directory = exp_dir(exp_id)
            (experiment_directory / "config.json").write_text(
                json.dumps(experiment_config, indent=2), encoding="utf-8",
            )
            (experiment_directory / "config.yaml").write_text(_to_yaml(experiment_config) + "\n", encoding="utf-8")
            config["provenance"] = {key: experiment_config.get(key, "") for key in (
                "name", "run_label", "paper_title", "paper_citation", "paper_url", "hypothesis", "created_at"
            )}
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            data = load_data(exp_id)
            observations = torch.from_numpy(data["observations"]).float()
            configured_image_shape = _validated_image_shape(config.get("image_shape"), observations.shape[1])
            dataset_image_shape = _validated_image_shape(data.get("image_shape"), observations.shape[1])
            resolved_image_shape = configured_image_shape or dataset_image_shape
            if resolved_image_shape:
                config["image_shape"] = resolved_image_shape
                experiment_config["image_shape"] = resolved_image_shape
                experiment_config["paper_protocol"]["image_shape"] = resolved_image_shape
                (experiment_directory / "config.json").write_text(
                    json.dumps(experiment_config, indent=2), encoding="utf-8",
                )
                (experiment_directory / "config.yaml").write_text(
                    _to_yaml(experiment_config) + "\n", encoding="utf-8",
                )
            train_indices = np.flatnonzero(data["split"] == 0)
            validation_indices = np.flatnonzero(data["split"] == 1)
            test_mask = data["split"] == (2 if np.any(data["split"] == 2) else 1)
            test_indices = np.flatnonzero(test_mask)
            train_tensor = torch.utils.data.TensorDataset(
                observations[torch.from_numpy(train_indices)], torch.from_numpy(train_indices.astype(np.int64)),
            )
            evaluation_batch_size = max(8, min(config["batch_size"], 1_000_000 // max(1, observations.shape[1])))
            validation_sample = observations[torch.from_numpy(validation_indices[:evaluation_batch_size])]
            model = GeometryAutoencoder(observations.shape[1], config)
            optimizer = None if config["model_type"] == "random_decoder" else _make_optimizer(model.parameters(), config)
            factor_discriminator = None
            factor_discriminator_optimizer = None
            pair_discriminator = None
            pair_discriminator_optimizer = None
            if config["model_type"] == "factor_vae":
                factor_discriminator = FactorDiscriminator(
                    config["latent_dim"], config["factor_discriminator_width"], config["factor_discriminator_depth"],
                )
                factor_discriminator_optimizer = torch.optim.Adam(
                    factor_discriminator.parameters(), lr=config["factor_discriminator_lr"], betas=(.5, .9),
                )
            if config["model_type"] in {"pcl", "weakly_supervised_gan"}:
                pair_discriminator = PairDiscriminator(config["latent_dim"])
                pair_discriminator_optimizer = torch.optim.Adam(
                    pair_discriminator.parameters(), lr=config["learning_rate"], betas=(.5, .999),
                )
            pair_lookup = None
            if config["model_type"] in {"slow_vae", "pcl", "weakly_supervised_gan"}:
                pair_lookup = _factor_pair_lookup(data["factor_values"], train_indices, config["seed"])
            generator = torch.Generator().manual_seed(config["seed"])
            loader = torch.utils.data.DataLoader(train_tensor, batch_size=config["batch_size"], shuffle=True, generator=generator)
            metrics = []
            polarization_trace = []
            global_step = 0
            started = time.time()
            steps_per_epoch = max(1, len(loader))
            target_steps = (steps_per_epoch if config["model_type"] == "random_decoder" else
                            (config["training_steps"] or config["epochs"] * steps_per_epoch))
            epochs_to_run = max(1, math.ceil(target_steps / steps_per_epoch))
            completed_epochs = 0

            def encoded_statistics(tensor):
                maximum = config["polarization_max_samples"]
                if maximum and len(tensor) > maximum:
                    diagnostic_indices = torch.linspace(0, len(tensor) - 1, maximum).long()
                    tensor = tensor[diagnostic_indices]
                means, logs = [], []
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(tensor), evaluation_batch_size):
                        mu_batch, log_batch = model.encode(tensor[start:start + evaluation_batch_size])
                        means.append(mu_batch.cpu().numpy())
                        logs.append(log_batch.cpu().numpy())
                return np.concatenate(means), np.concatenate(logs)

            for epoch in range(1, epochs_to_run + 1):
                model.train()
                sums = np.zeros(4, dtype=np.float64)
                batches = 0
                for batch, batch_indices in loader:
                    if global_step >= target_steps:
                        break
                    if optimizer is None:
                        with torch.no_grad():
                            values = _loss_terms(model, batch, config, len(train_indices))
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        values = list(_loss_terms(model, batch, config, len(train_indices)))
                        if config["model_type"] == "factor_vae":
                            for parameter in factor_discriminator.parameters():
                                parameter.requires_grad_(False)
                            logits = factor_discriminator(values[5])
                            tc_penalty = float(config["tc_weight"]) * (logits[:, 0] - logits[:, 1]).mean()
                            values[0] = values[0] + tc_penalty
                            values[3] = tc_penalty
                        elif config["model_type"] == "slow_vae":
                            paired = observations[torch.from_numpy(pair_lookup[batch_indices.numpy()])]
                            _, paired_z, _, _ = model(paired)
                            delta = (values[5] - paired_z).abs().clamp_min(1e-8)
                            slow_penalty = float(config["slow_gamma"]) * delta.pow(float(config["slow_alpha"])).sum(1).mean()
                            values[0] = values[0] + slow_penalty
                            values[3] = slow_penalty
                        elif config["model_type"] in {"pcl", "weakly_supervised_gan"}:
                            pair_discriminator_optimizer.zero_grad(set_to_none=True)
                            paired = observations[torch.from_numpy(pair_lookup[batch_indices.numpy()])]
                            positive, _ = model.encode(paired)
                            negative = positive[torch.randperm(len(positive))]
                            real_logits = pair_discriminator(values[6], positive)
                            false_logits = pair_discriminator(values[6], negative)
                            pair_loss = .5 * (
                                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                                + F.binary_cross_entropy_with_logits(false_logits, torch.zeros_like(false_logits))
                            )
                            if config["model_type"] == "pcl":
                                values[0] = float(config["pair_weight"]) * pair_loss
                            else:
                                # Full-sharing control keeps reconstruction while enforcing shared-pair information.
                                values[0] = values[0] + float(config["pair_weight"]) * pair_loss
                            values[3] = pair_loss
                        values[0].backward()
                        optimizer.step()
                        if config["model_type"] in {"pcl", "weakly_supervised_gan"}:
                            pair_discriminator_optimizer.step()
                        if config["model_type"] == "factor_vae":
                            for parameter in factor_discriminator.parameters():
                                parameter.requires_grad_(True)
                            factor_discriminator_optimizer.zero_grad(set_to_none=True)
                            joint = values[5].detach()
                            permuted = torch.stack([
                                joint[torch.randperm(len(joint)), coordinate]
                                for coordinate in range(joint.shape[1])
                            ], dim=1)
                            disc_loss = .5 * (
                                F.cross_entropy(factor_discriminator(joint), torch.zeros(len(joint), dtype=torch.long))
                                + F.cross_entropy(factor_discriminator(permuted), torch.ones(len(joint), dtype=torch.long))
                            )
                            disc_loss.backward()
                            factor_discriminator_optimizer.step()
                    sums += [float(values[i].detach()) for i in range(4)]
                    batches += 1
                    global_step += 1
                    if (config["paper_reproduction"] and config.get("paper_id") != ZIETLOW_PAPER["id"] and model.variational and not model.full_covariance
                            and global_step % 500 == 0):
                        diagnostic_mu, diagnostic_logvar = encoded_statistics(observations)
                        snapshot = polarization_snapshot(diagnostic_mu, diagnostic_logvar)
                        snapshot.update({"step": global_step, "epoch": epoch, "dataset_scope": "all" if config["polarization_max_samples"] == 0 else "seeded_bounded"})
                        polarization_trace.append(snapshot)
                        model.train()
                model.eval()
                with torch.no_grad():
                    val_loss = float(_loss_terms(model, validation_sample, config, len(train_indices))[1]) if len(validation_sample) else 0.0
                    mu, _ = model.encode(observations[: min(len(observations), 4096)])
                    variance = mu.var(0)
                row = {
                    "epoch": epoch, "total_loss": sums[0] / max(1, batches),
                    "reconstruction_loss": sums[1] / max(1, batches), "kl_divergence": sums[2] / max(1, batches),
                    "regularizer": sums[3] / max(1, batches), "validation_reconstruction": val_loss,
                    "latent_variance": float(variance.mean()), "active_latent_dimensions": int((variance > 0.01).sum()),
                    "optimizer_step": int(global_step), "target_optimizer_steps": int(target_steps),
                }
                metrics.append(row)
                completed_epochs = epoch
                update_job(job_id, status="running", epoch=epoch, progress=global_step / max(1, target_steps), metrics=metrics[:])
                if global_step >= target_steps:
                    break

            if config["paper_reproduction"] and config.get("paper_id") != ZIETLOW_PAPER["id"] and model.variational and not model.full_covariance:
                diagnostic_mu, diagnostic_logvar = encoded_statistics(observations)
                snapshot = polarization_snapshot(diagnostic_mu, diagnostic_logvar)
                snapshot.update({"step": global_step, "epoch": completed_epochs, "dataset_scope": "all" if config["polarization_max_samples"] == 0 else "seeded_bounded"})
                if not polarization_trace or polarization_trace[-1]["step"] != global_step:
                    polarization_trace.append(snapshot)

            polarization_evaluated = config["paper_reproduction"] and model.variational and not model.full_covariance
            continuous_start = None
            if polarization_evaluated:
                for snapshot in reversed(polarization_trace):
                    if snapshot["delta_kl"] < 0.03:
                        continuous_start = int(snapshot["step"])
                    else:
                        break
            continuous_percent = None if not polarization_evaluated else (
                100.0 * (global_step - continuous_start) / max(1, global_step)
                if continuous_start is not None else 0.0
            )
            polarization_summary = {
                "evaluation_interval_batches": 500,
                "threshold_delta_kl": 0.03,
                "active_selection_rule": "sqrt(var(mu_j(x_i))) > 0.5",
                "active_standard_deviation_threshold": 0.5,
                "continuous_start_step": continuous_start,
                "continuous_fraction_to_end": continuous_percent / 100.0 if continuous_percent is not None else None,
                "continuous_percent_to_end": continuous_percent,
                "total_optimizer_steps": global_step,
                "target_optimizer_steps": target_steps,
            }

            model.eval()
            latent_batches, logvar_batches, error_batches, kl_batches = [], [], [], []
            with torch.no_grad():
                for start in range(0, len(observations), evaluation_batch_size):
                    batch = observations[start:start + evaluation_batch_size]
                    raw_reconstructed_batch, _, mu_batch, logvar_batch = model(batch)
                    reconstructed_batch = _reconstruction_mean(raw_reconstructed_batch, config)
                    error_batches.append(((reconstructed_batch - batch) ** 2).flatten(1).sum(1).cpu().numpy())
                    latent_batches.append(mu_batch.cpu().numpy())
                    logvar_batches.append(logvar_batch.cpu().numpy())
                    if model.full_covariance:
                        cholesky = model._last_cholesky
                        trace = cholesky.pow(2).sum(dim=(-2, -1))
                        logdet = 2 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
                        kl_batch = 0.5 * (trace + mu_batch.pow(2).sum(-1) - model.latent_dim - logdet)
                    elif model.variational:
                        kl_batch = -0.5 * (1 + logvar_batch - mu_batch.pow(2) - logvar_batch.exp()).sum(1)
                    else:
                        kl_batch = torch.zeros(len(mu_batch))
                    kl_batches.append(kl_batch.cpu().numpy())
            latent = np.concatenate(latent_batches)
            latent_logvar = np.concatenate(logvar_batches)
            errors = np.concatenate(error_batches)
            kl_contrib = np.concatenate(kl_batches)
            projection, explained = _pca(latent, 3)
            visualization_limit = max(64, min(1500, 1_500_000 // max(1, observations.shape[1])))
            count = min(visualization_limit, len(latent))
            indices = np.linspace(0, len(latent) - 1, count, dtype=int)
            reconstructed_samples = []
            with torch.no_grad():
                for start in range(0, len(indices), evaluation_batch_size):
                    sample_batch = observations[torch.from_numpy(indices[start:start + evaluation_batch_size])]
                    raw_reconstructed_batch, _, _, _ = model(sample_batch)
                    reconstructed_samples.append(_reconstruction_mean(raw_reconstructed_batch, config).cpu().numpy())
            reconstructed_sample = np.concatenate(reconstructed_samples)
            paired = np.concatenate((observations.numpy()[indices], reconstructed_sample), axis=0)
            paired_projection, _ = _pca(paired, 3)
            analysis = {
                "model_type": config["model_type"], "model_label": MODEL_LABELS[config["model_type"]],
                "latent_dimension": config["latent_dim"], "projection_method": "direct" if config["latent_dim"] <= 3 else "PCA",
                "projection": projection[indices].round(6).tolist(), "latent_vectors": latent[indices, : min(12, latent.shape[1])].round(6).tolist(),
                "factor_values": data["factor_values"][indices].round(6).tolist(),
                "factor_active": data["factor_active"][indices].astype(int).tolist(),
                "reconstruction_error": errors[indices].round(7).tolist(),
                "kl_contribution": kl_contrib[indices].round(7).tolist(),
                "observation_projection": paired_projection[:count].round(6).tolist(),
                "reconstruction_projection": paired_projection[count:].round(6).tolist(),
                "manifold_membership": data["factor_types"].tolist(),
                "pca_explained_variance": explained, "metrics": metrics,
                "training_seed": int(config["seed"]),
                "seed_source": config["seed_source"],
                "training_budget": {"unit": config["training_budget_unit"], "requested_steps": int(config["training_steps"]),
                                    "completed_steps": int(global_step), "completed_epochs": int(completed_epochs)},
                "split_counts": {"train": int(len(train_indices)), "validation": int(len(validation_indices)), "test": int(len(test_indices))},
                "test_reconstruction_error": float(errors[test_mask].mean()) if np.any(test_mask) else None,
                "polarization_trace": polarization_trace,
                "polarization_summary": polarization_summary,
            }
            directory = exp_dir(exp_id)
            zietlow_metrics = None
            if config.get("zietlow_metrics"):
                zietlow_metrics = compute_zietlow_metrics(
                    latent, data["factor_values"], data["split"], config["seed"],
                    latent_logvar if model.variational else None,
                    bins=config["factor_metric_bins"], vote_batches=config["factor_vote_batches"],
                    vote_evaluation_batches=config["factor_vote_evaluation_batches"],
                    sampling=config["metric_sampling"], metric_seed=config["metric_seed"],
                    train_samples=config["metric_train_samples"], test_samples=config["metric_test_samples"],
                    variance_samples=config["metric_variance_samples"],
                )
                zietlow_metrics.update({
                    "dataset": experiment_config.get("dataset_source"),
                    "dataset_variant": experiment_config.get("dataset_variant", config.get("dataset_variant", "original")),
                    "model_type": config["model_type"],
                    "paper_target": ZIETLOW_MIG_TARGETS.get(experiment_config.get("dataset_source", ""), {}).get(config["model_type"], {}).get(experiment_config.get("dataset_variant", "original")),
                })
                (directory / "zietlow_metrics.json").write_text(json.dumps(zietlow_metrics, indent=2), encoding="utf-8")
                analysis["zietlow_metrics"] = zietlow_metrics
            np.savez_compressed(directory / "latent.npz", latent=latent.astype(np.float32), split=data["split"], factor_values=data["factor_values"])
            auxiliary_state = {}
            if factor_discriminator is not None:
                auxiliary_state["factor_discriminator"] = factor_discriminator.state_dict()
            if pair_discriminator is not None:
                auxiliary_state["pair_discriminator"] = pair_discriminator.state_dict()
            torch.save({"model_state_dict": model.state_dict(), "training_config": config, "input_dim": observations.shape[1], "auxiliary_state": auxiliary_state}, directory / "model.pt")
            (directory / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            (directory / "polarization_trace.json").write_text(json.dumps({
                "training_seed": int(config["seed"]), "seed_source": config["seed_source"],
                "summary": polarization_summary, "trace": polarization_trace,
            }, indent=2), encoding="utf-8")
            environment = {
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda, "device": "cpu",
            }
            (directory / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
            (directory / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
            update_job(
                job_id, status="complete", progress=1.0, epoch=completed_epochs,
                analysis=analysis, elapsed_seconds=round(time.time() - started, 3),
            )
        except Exception as exc:
            traceback.print_exc()
            update_job(job_id, status="error", error=str(exc))

    @app.route("/api/geometry/experiments/<exp_id>/train", methods=["POST"])
    def geometry_train(exp_id):
        try:
            directory = exp_dir(exp_id)
            if not (directory / "dataset.npz").exists():
                raise FileNotFoundError("Experiment dataset not found")
            raw = request.get_json(force=True) or {}
            _normalized_training_config(raw)
            job_id = uuid.uuid4().hex
            with job_lock:
                active = any(
                    job.get("experiment_id") == exp_id and job.get("status") in {"queued", "running"}
                    for job in jobs.values()
                ) or any(
                    item.get("experiment_id") == exp_id and item.get("status") in {"queued", "running"}
                    for item in queue_state["items"]
                )
                if active:
                    raise ValueError("This experiment is already queued or running")
                jobs[job_id] = {"id": job_id, "experiment_id": exp_id, "status": "queued", "epoch": 0, "progress": 0, "metrics": []}
            thread = threading.Thread(target=train_job, args=(job_id, exp_id, raw), daemon=True)
            thread.start()
            return jsonify({"success": True, "job_id": job_id})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/training/<job_id>", methods=["GET"])
    def geometry_training_status(job_id):
        with job_lock:
            job = jobs.get(job_id)
            if not job:
                return jsonify({"success": False, "error": "Training job not found"}), 404
            return jsonify({"success": True, "job": job})

    def scientific_job(job_id, exp_id, raw_config):
        try:
            directory = exp_dir(exp_id)
            experiment_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            raw_config = dict(raw_config)
            raw_config["dataset_source"] = experiment_config.get("dataset_source", "")
            checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            raw_config["seed"] = int(checkpoint["training_config"]["seed"])
            raw_config["seed_source"] = "effective_training_configuration"
            model = GeometryAutoencoder(int(checkpoint["input_dim"]), checkpoint["training_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            data = load_data(exp_id)
            analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
            update_job(job_id, status="running", stage="autograd Jacobians", progress=0.1)
            scientific = run_scientific_analysis(model, data, analysis, directory, raw_config)
            update_job(job_id, status="complete", stage="complete", progress=1.0, scientific=scientific)
        except Exception as exc:
            traceback.print_exc()
            update_job(job_id, status="error", error=str(exc))

    @app.route("/api/geometry/experiments/<exp_id>/scientific-analysis", methods=["POST"])
    def geometry_scientific_analysis(exp_id):
        try:
            directory = exp_dir(exp_id)
            if not (directory / "model.pt").exists():
                raise FileNotFoundError("Train a model before computing scientific metrics")
            raw = request.get_json(silent=True) or {}
            job_id = uuid.uuid4().hex
            jobs[job_id] = {"id": job_id, "experiment_id": exp_id, "kind": "scientific", "status": "queued", "stage": "queued", "progress": 0}
            threading.Thread(target=scientific_job, args=(job_id, exp_id, raw), daemon=True).start()
            return jsonify({"success": True, "job_id": job_id})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    def queue_worker():
        """Execute persistent experiment entries in FIFO order, one at a time."""
        while True:
            with queue_condition:
                item = next((entry for entry in queue_state["items"] if entry.get("status") == "queued"), None)
                if item is None:
                    queue_condition.wait(timeout=30)
                    continue
                queue_id = item["id"]
                experiment_id = item["experiment_id"]
                item.update({
                    "status": "running", "stage": "starting", "started_at": item.get("started_at") or _utc_now(),
                    "completed_at": None, "error": None,
                })
                jobs[queue_id] = {
                    "id": queue_id, "queue_id": queue_id, "queue_phase": "training",
                    "kind": "queued_experiment", "experiment_id": experiment_id,
                    "status": "queued", "epoch": 0, "progress": 0, "metrics": [],
                }
                persist_queue_locked()

            try:
                resume_analysis = bool(item.get("resume_analysis"))
                if not resume_analysis:
                    train_job(queue_id, experiment_id, item["training_config"])
                    with job_lock:
                        failed = jobs[queue_id].get("status") == "error"
                    if failed:
                        continue

                if item.get("run_scientific"):
                    with job_lock:
                        jobs[queue_id].update({
                            "queue_phase": "analysis", "status": "queued", "stage": "scientific analysis", "progress": 0,
                        })
                        active_item = queue_item_locked(queue_id)
                        if active_item:
                            active_item.update({
                                "status": "running", "stage": "scientific analysis", "progress": 0.85,
                                "resume_analysis": True,
                            })
                            persist_queue_locked()
                    scientific_job(queue_id, experiment_id, item.get("analysis_config") or {})
                    with job_lock:
                        failed = jobs[queue_id].get("status") == "error"
                    if failed:
                        continue

                with job_lock:
                    completed = queue_item_locked(queue_id)
                    if completed:
                        completed.update({
                            "status": "complete", "stage": "complete", "progress": 1.0,
                            "completed_at": _utc_now(), "resume_analysis": False,
                            "elapsed_seconds": jobs[queue_id].get("elapsed_seconds"),
                        })
                        persist_queue_locked()
            except Exception as exc:
                traceback.print_exc()
                update_job(queue_id, status="error", error=str(exc))

    def ensure_queue_worker():
        nonlocal queue_worker_thread
        with job_lock:
            if queue_worker_thread is not None and queue_worker_thread.is_alive():
                return
            queue_worker_thread = threading.Thread(target=queue_worker, name="geometry-training-queue", daemon=True)
            queue_worker_thread.start()

    def public_queue_locked():
        queued_position = 0
        public_items = []
        for item in queue_state["items"]:
            copy = {key: value for key, value in item.items() if key not in {"training_config", "analysis_config"}}
            config = item.get("training_config") or {}
            copy["configuration"] = {
                "model_type": config.get("model_type"), "epochs": config.get("epochs"),
                "training_steps": config.get("training_steps"), "reconstruction_loss": config.get("reconstruction_loss"),
                "learning_rate": config.get("learning_rate"), "batch_size": config.get("batch_size"),
                "latent_dim": config.get("latent_dim"), "beta": config.get("beta"), "seed": config.get("seed"),
            }
            if item.get("status") == "queued":
                queued_position += 1
                copy["position"] = queued_position
            else:
                copy["position"] = None
            public_items.append(copy)
        return public_items

    @app.route("/api/geometry/training-queue", methods=["GET", "POST"])
    def geometry_training_queue():
        if request.method == "GET":
            ensure_queue_worker()
            with job_lock:
                items = public_queue_locked()
                counts = {
                    status: sum(item.get("status") == status for item in queue_state["items"])
                    for status in ("queued", "running", "complete", "error")
                }
            return jsonify({"success": True, "items": items, "counts": counts, "server_time": _utc_now()})

        try:
            raw = request.get_json(force=True) or {}
            experiment_id = _safe_id(str(raw.get("experiment_id", "")))
            directory = exp_dir(experiment_id)
            config_path = directory / "config.json"
            if not config_path.exists() or not (directory / "dataset.npz").exists():
                raise FileNotFoundError("Experiment dataset not found")
            if (directory / "analysis.json").exists():
                raise ValueError("Only untrained experiments can be added to the queue")
            experiment = json.loads(config_path.read_text(encoding="utf-8"))
            supplied_training = raw.get("training_config") if isinstance(raw.get("training_config"), dict) else {}
            training_config = _normalized_training_config({
                **(experiment.get("paper_protocol") or {}),
                **supplied_training,
            })
            analysis_plan = _analysis_plan(experiment.get("analysis_plan"))
            supplied_analysis = raw.get("analysis_config") if isinstance(raw.get("analysis_config"), dict) else {}
            analysis_config = {
                "maximum_samples": max(16, min(1024, int(supplied_analysis.get("maximum_samples", 256)))),
                "topology_points": max(32, min(600, int(supplied_analysis.get("topology_points", 250)))),
                "seed": int(training_config["seed"]),
                "seed_source": "effective_training_configuration",
                "analysis_plan": analysis_plan,
                "paper_metrics": bool(supplied_analysis.get("paper_metrics", False)),
                "maximum_test_samples": max(0, int(supplied_analysis.get("maximum_test_samples", 0))),
            }
            run_scientific = bool(set(analysis_plan) & SCIENTIFIC_ANALYSIS_IDS)
            queue_id = "queue_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            item = {
                "id": queue_id, "experiment_id": experiment_id, "experiment_name": experiment.get("name", experiment_id),
                "status": "queued", "stage": "waiting", "progress": 0.0, "epoch": 0,
                "created_at": _utc_now(), "started_at": None, "completed_at": None, "error": None,
                "training_config": training_config, "analysis_config": analysis_config,
                "analysis_plan": analysis_plan, "run_scientific": run_scientific, "resume_analysis": False,
            }
            with queue_condition:
                duplicate = any(
                    entry.get("experiment_id") == experiment_id and entry.get("status") in {"queued", "running"}
                    for entry in queue_state["items"]
                ) or any(
                    job.get("experiment_id") == experiment_id and job.get("status") in {"queued", "running"}
                    for job in jobs.values()
                )
                if duplicate:
                    raise ValueError("This experiment is already queued or running")
                queue_state["items"].append(item)
                persist_queue_locked()
                public_item = public_queue_locked()[-1]
                queue_condition.notify_all()
            ensure_queue_worker()
            return jsonify({"success": True, "queue_item": public_item})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/training-queue/<queue_id>", methods=["DELETE"])
    def geometry_remove_queue_item(queue_id):
        try:
            queue_id = _safe_id(queue_id)
            with queue_condition:
                item = queue_item_locked(queue_id)
                if item is None:
                    raise FileNotFoundError("Queue item not found")
                if item.get("status") == "running":
                    raise ValueError("A running experiment cannot be removed")
                queue_state["items"].remove(item)
                persist_queue_locked()
            return jsonify({"success": True})
        except FileNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/training-queue/history", methods=["DELETE"])
    def geometry_clear_queue_history():
        """Remove terminal queue records while preserving pending work."""
        with queue_condition:
            terminal = [
                item for item in queue_state["items"]
                if item.get("status") in {"complete", "error"}
            ]
            terminal_ids = {item.get("id") for item in terminal}
            if terminal:
                queue_state["items"] = [
                    item for item in queue_state["items"]
                    if item.get("id") not in terminal_ids
                ]
                persist_queue_locked()
            remaining = len(queue_state["items"])
        return jsonify({
            "success": True,
            "removed": len(terminal),
            "remaining": remaining,
            "preserved_statuses": ["queued", "running"],
        })

    @app.route("/api/geometry/zietlow/suite", methods=["POST"])
    def geometry_create_zietlow_suite():
        """Clone linked dataset variants into ordinary experiments, optionally enqueueing them."""
        try:
            raw = request.get_json(force=True) or {}
            enqueue = bool(raw.get("enqueue", True))
            stage = str(raw.get("stage", "pilot")).lower()
            models_by_stage = {
                "pilot": ["beta_vae"],
                "vae": ["autoencoder", "beta_vae", "factor_vae", "beta_tcvae", "slow_vae"],
                "complete": ["autoencoder", "beta_vae", "factor_vae", "beta_tcvae", "slow_vae", "pcl", "weakly_supervised_gan"],
            }
            if stage not in models_by_stage:
                raise ValueError("Stage must be pilot, vae, or complete")
            requested_seeds = raw.get("seeds")
            seeds = [int(value) for value in requested_seeds] if isinstance(requested_seeds, list) else list(range(1, 4 if stage == "pilot" else 11))
            if not seeds or len(set(seeds)) != len(seeds) or len(seeds) > 20:
                raise ValueError("Supply between one and twenty unique training seeds")
            requested_scales = raw.get("scales", [raw.get("hyperparameter_scale", 1.0)])
            scales = [float(value) for value in requested_scales] if isinstance(requested_scales, list) else [float(requested_scales)]
            if not scales or len(scales) > 12 or any(not np.isfinite(value) or value <= 0 for value in scales):
                raise ValueError("Supply between one and twelve positive hyperparameter scales")
            if len(set(scales)) != len(scales):
                raise ValueError("Hyperparameter scales must be unique")
            datasets = raw.get("datasets") if isinstance(raw.get("datasets"), dict) else {}
            sources = []
            for dataset, variants in datasets.items():
                if dataset not in ZIETLOW_DATASETS or not isinstance(variants, dict):
                    continue
                for variant in ("original", "manipulated", "uniform_noise"):
                    experiment_id = variants.get(variant)
                    if experiment_id:
                        sources.append((dataset, variant, _safe_id(str(experiment_id))))
            if not sources:
                raise ValueError("Select at least one original/manipulated/noise dataset variant")
            suite_id = "zietlow_" + datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
            created, queued = [], []
            now = _utc_now()
            with queue_condition:
                for dataset, variant, source_id in sources:
                    source_directory = exp_dir(source_id)
                    source_config_path = source_directory / "config.json"
                    source_dataset_path = source_directory / "dataset.npz"
                    if not source_config_path.exists() or not source_dataset_path.exists():
                        raise FileNotFoundError(f"Dataset experiment {source_id} is unavailable")
                    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
                    if source_config.get("dataset_source") != dataset:
                        raise ValueError(f"{source_id} is not a {dataset} dataset")
                    for model_type in models_by_stage[stage]:
                        for scale in scales:
                            for seed in seeds:
                                scale_slug = str(scale).replace(".", "p")
                                experiment_id = "geo_" + suite_id + "_" + dataset + "_" + variant + "_" + model_type + f"_h{scale_slug}_s{seed:02d}"
                                directory = exp_dir(experiment_id)
                                directory.mkdir(parents=True, exist_ok=False)
                                try:
                                    os.link(source_dataset_path, directory / "dataset.npz")
                                except OSError:
                                    shutil.copy2(source_dataset_path, directory / "dataset.npz")
                                protocol = zietlow_paper_protocol(dataset, model_type, variant, seed)
                                protocol.update({
                                    "paper_image_model": True,
                                    "training_steps": int(raw.get("training_steps", protocol["training_steps"])),
                                    "training_steps_source": "Disentanglement Library default",
                                    "batch_size": int(raw.get("batch_size", 64)), "batch_size_source": "Disentanglement Library default",
                                    "learning_rate": float(raw.get("learning_rate", 1e-4)),
                                    "learning_rate_source": "Disentanglement Library default",
                                    "optimizer": "adam", "image_shape": source_config.get("image_shape", []),
                                    "factor_vote_batches": int(raw.get("factor_vote_batches", 10000)),
                                    "factor_vote_evaluation_batches": int(raw.get("factor_vote_evaluation_batches", 5000)),
                                    "paper_hyperparameter_scale": scale,
                                    "paper_dataset_complete": int(source_config.get("dataset_size", 0)) == int(ZIETLOW_DATASETS[dataset]["published_samples"]),
                                    "published_dataset_samples": int(ZIETLOW_DATASETS[dataset]["published_samples"]),
                                })
                                if model_type == "factor_vae":
                                    protocol["tc_weight"] = float(protocol["tc_weight"]) * scale
                                elif model_type in {"beta_vae", "beta_tcvae", "slow_vae"}:
                                    protocol["beta"] = float(protocol["beta"]) * scale
                                training_config = _normalized_training_config(protocol)
                                config = {
                                    **source_config, "id": experiment_id,
                                    "name": f"Zietlow 2021 · {dataset} · {variant} · {MODEL_LABELS[model_type]} · h×{scale:g} · seed {seed}",
                                    "created_at": now, "dataset_source": dataset, "dataset_variant": variant,
                                    "parent_experiment_id": source_id, "suite_id": suite_id, "suite_stage": stage,
                                    "paper_id": ZIETLOW_PAPER["id"], "paper_title": ZIETLOW_PAPER["title"],
                                    "paper_citation": ZIETLOW_PAPER["citation"], "paper_url": ZIETLOW_PAPER["url"],
                                    "run_label": suite_id, "effective_training_seed": seed,
                                    "analysis_plan": [], "paper_protocol": protocol,
                                    "tags": ["zietlow2021", dataset, variant, model_type, f"scale-{scale:g}", f"seed-{seed}"],
                                    "dataset_npz_sha256": source_config.get("dataset_npz_sha256") or file_sha256(directory / "dataset.npz"),
                                }
                                (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
                                (directory / "config.yaml").write_text(_to_yaml(config) + "\n", encoding="utf-8")
                                if (source_directory / "summary.json").exists():
                                    shutil.copy2(source_directory / "summary.json", directory / "summary.json")
                                created.append(experiment_id)
                                if enqueue:
                                    queue_id = "queue_" + uuid.uuid4().hex
                                    item = {
                                        "id": queue_id, "experiment_id": experiment_id, "experiment_name": config["name"],
                                        "status": "queued", "stage": "waiting", "progress": 0.0, "epoch": 0,
                                        "created_at": now, "started_at": None, "completed_at": None, "error": None,
                                        "training_config": training_config,
                                        "analysis_config": {"seed": seed, "analysis_plan": [], "paper_metrics": False, "maximum_test_samples": 0},
                                        "analysis_plan": [], "run_scientific": False, "resume_analysis": False,
                                        "suite_id": suite_id, "paper_id": ZIETLOW_PAPER["id"],
                                    }
                                    queue_state["items"].append(item)
                                    queued.append(queue_id)
                persist_queue_locked()
                if enqueue:
                    queue_condition.notify_all()
            if enqueue:
                ensure_queue_worker()
            return jsonify({
                "success": True, "suite_id": suite_id, "stage": stage,
                "created_experiments": len(created), "queued": len(queued), "enqueue": enqueue, "seeds": seeds,
                "models": models_by_stage[stage], "source_variants": len(sources), "scales": scales,
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/zietlow/summary", methods=["GET"])
    def geometry_zietlow_summary():
        """Aggregate completed suite runs and compute paired variant contrasts."""
        try:
            suite_id = str(request.args.get("suite_id", "")).strip()
            rows = []
            for directory in root.iterdir():
                if not directory.is_dir() or not (directory / "config.json").exists() or not (directory / "zietlow_metrics.json").exists():
                    continue
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                if suite_id and config.get("suite_id") != suite_id:
                    continue
                if config.get("paper_id") != ZIETLOW_PAPER["id"]:
                    continue
                metrics = json.loads((directory / "zietlow_metrics.json").read_text(encoding="utf-8"))
                rows.append({
                    "experiment_id": directory.name, "suite_id": config.get("suite_id"),
                    "dataset": config.get("dataset_source"), "variant": config.get("dataset_variant", "original"),
                    "model": metrics.get("model_type"), "seed": int(metrics.get("seed", config.get("effective_training_seed", 0))),
                    "hyperparameter_scale": float((config.get("paper_protocol") or {}).get("paper_hyperparameter_scale", 1.0)),
                    "paper_comparable": bool((config.get("paper_protocol") or {}).get(
                        "paper_dataset_complete",
                        int(config.get("dataset_size", 0)) == int(ZIETLOW_DATASETS[config.get("dataset_source")]["published_samples"]),
                    )),
                    "mig": float((metrics.get("mig") or {}).get("score")),
                    "dci": float((metrics.get("dci") or {}).get("disentanglement")),
                    "sap": float((metrics.get("sap") or {}).get("score")),
                    "factor_vae_score": float((metrics.get("factor_vae_score") or {}).get("score")),
                    "active_units": (metrics.get("active_units") or {}).get("count"),
                    "over_pruned": bool((metrics.get("active_units") or {}).get("over_pruned")),
                })
            aggregate = aggregate_zietlow_rows(rows) if rows else {"paper_id": ZIETLOW_PAPER["id"], "groups": []}
            paired = []
            keys = sorted({(row["dataset"], row["model"], row["hyperparameter_scale"]) for row in rows})
            for dataset, model, hyperparameter_scale in keys:
                by_variant = {
                    variant: {row["seed"]: row for row in rows if row["dataset"] == dataset and row["model"] == model and row["hyperparameter_scale"] == hyperparameter_scale and row["variant"] == variant}
                    for variant in ("original", "manipulated", "uniform_noise")
                }
                for variant in ("manipulated", "uniform_noise"):
                    seeds = sorted(set(by_variant["original"]) & set(by_variant[variant]))
                    differences = np.asarray([by_variant[variant][seed]["mig"] - by_variant["original"][seed]["mig"] for seed in seeds], dtype=float)
                    paired.append({
                        "dataset": dataset, "model": model, "hyperparameter_scale": hyperparameter_scale,
                        "contrast": f"{variant}-original", "n": len(seeds),
                        "mean_difference": float(differences.mean()) if len(differences) else None,
                        "sample_sd": float(differences.std(ddof=1)) if len(differences) > 1 else (0.0 if len(differences) else None),
                    })
            suites = sorted({row["suite_id"] for row in rows if row.get("suite_id")})
            return jsonify({"success": True, "paper": ZIETLOW_PAPER, "suite_id": suite_id, "suites": suites, "runs": rows, "aggregate": aggregate, "paired_contrasts": paired})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/zietlow/report", methods=["GET"])
    def geometry_zietlow_report():
        """Export an auditable Markdown comparison for one completed suite."""
        try:
            suite_id = str(request.args.get("suite_id", "")).strip()
            if not suite_id:
                raise ValueError("Select a Zietlow suite before downloading its report")
            rows = []
            fidelity = set()
            for directory in root.iterdir():
                config_path = directory / "config.json"
                metrics_path = directory / "zietlow_metrics.json"
                if not directory.is_dir() or not config_path.exists() or not metrics_path.exists():
                    continue
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if config.get("suite_id") != suite_id or config.get("paper_id") != ZIETLOW_PAPER["id"]:
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                protocol = config.get("paper_protocol") or {}
                fidelity.add((metrics.get("model_type"), protocol.get("implementation_fidelity", "not recorded")))
                rows.append({
                    "experiment_id": directory.name, "suite_id": suite_id,
                    "dataset": config.get("dataset_source"), "variant": config.get("dataset_variant", "original"),
                    "model": metrics.get("model_type"), "seed": int(metrics.get("seed", config.get("effective_training_seed", 0))),
                    "hyperparameter_scale": float(protocol.get("paper_hyperparameter_scale", 1.0)),
                    "paper_comparable": bool(protocol.get(
                        "paper_dataset_complete",
                        int(config.get("dataset_size", 0)) == int(ZIETLOW_DATASETS[config.get("dataset_source")]["published_samples"]),
                    )),
                    "mig": float((metrics.get("mig") or {}).get("score")),
                    "dci": float((metrics.get("dci") or {}).get("disentanglement")),
                    "sap": float((metrics.get("sap") or {}).get("score")),
                    "factor_vae_score": float((metrics.get("factor_vae_score") or {}).get("score")),
                    "active_units": (metrics.get("active_units") or {}).get("count"),
                    "over_pruned": bool((metrics.get("active_units") or {}).get("over_pruned")),
                })
            if not rows:
                raise ValueError("This suite has no completed paper-metric runs yet")
            aggregate = aggregate_zietlow_rows(rows)
            fmt = lambda value: "n/a" if value is None else f"{float(value):.6f}"
            lines = [
                f"# Zietlow 2021 reproduction comparison — {suite_id}", "",
                f"Paper: [{ZIETLOW_PAPER['title']}]({ZIETLOW_PAPER['url']})", "",
                f"Completed runs: {len(rows)}", "",
                "## Implementation fidelity", "",
            ]
            lines.extend(f"- `{model}`: {label}" for model, label in sorted(fidelity))
            lines.extend(["", "## Primary MIG comparison", "",
                          "| Dataset | Model | Variant | h scale | Paper mean ± SD | Observed mean ± SD | Bootstrap 95% CI | n | Status |",
                          "|---|---|---:|---:|---:|---:|---:|---:|---|"])
            for group in aggregate["groups"]:
                ci = group["bootstrap_ci95"]
                lines.append(
                    f"| {group['dataset']} | {group['model']} | {group['variant']} | {group['hyperparameter_scale']:g} | "
                    f"{fmt(group['paper_mean'])} ± {fmt(group['paper_sd'])} | {fmt(group['mean'])} ± {fmt(group['sample_sd'])} | "
                    f"[{fmt(ci[0])}, {fmt(ci[1])}] | {group['n']} | {group['status']} |"
                )
            lines.extend(["", "## Supplemental metrics", "",
                          "| Dataset | Model | Variant | h scale | DCI | SAP | FactorVAE score |",
                          "|---|---|---|---:|---:|---:|---:|"])
            for group in aggregate["groups"]:
                supplemental = group.get("supplemental_metrics", {})
                lines.append(
                    f"| {group['dataset']} | {group['model']} | {group['variant']} | {group['hyperparameter_scale']:g} | "
                    f"{fmt((supplemental.get('dci') or {}).get('mean'))} | {fmt((supplemental.get('sap') or {}).get('mean'))} | "
                    f"{fmt((supplemental.get('factor_vae_score') or {}).get('mean'))} |"
                )
            over_pruned = sum(bool(row["over_pruned"]) for row in rows)
            lines.extend(["", "## Interpretation guardrails", "",
                          f"- Over-pruned runs retained: {over_pruned}; exclude them only from the declared robustness aggregate.",
                          "- A pilot with three seeds validates direction and plumbing; it is not the final ten-seed reproduction.",
                          "- `pass` means the paper MIG mean lies inside this reproduction's bootstrap CI. It does not prove distributional equivalence.",
                          "- Subsampled dataset cells are marked `pilot_only`; their scores are never presented as a paper-compatible pass/fail.",
                          "- Hyperparameter scales other than 1 have no Table 1 point target and therefore report `not_applicable`.", ""])
            report_directory = root / "_zietlow_reports"
            report_directory.mkdir(parents=True, exist_ok=True)
            report_path = report_directory / f"{_safe_id(suite_id)}_comparison.md"
            report_path.write_text("\n".join(lines), encoding="utf-8")
            return send_file(report_path, as_attachment=True, download_name=report_path.name)
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/reproduction-summary", methods=["GET"])
    def geometry_reproduction_summary():
        """Aggregate a homogeneous paper suite using its dataset/model targets."""
        try:
            run_label = str(request.args.get("run_label", "")).strip()
            if not run_label:
                raise ValueError("Choose a reproduction suite")

            def read_json(directory, name, default=None):
                path = directory / name
                return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

            rows = []
            for directory in root.iterdir():
                config_path = directory / "config.json"
                if not directory.is_dir() or not config_path.exists():
                    continue
                config = read_json(directory, "config.json", {})
                if config.get("run_label") != run_label:
                    continue
                training = read_json(directory, "training_config.json", {})
                analysis = read_json(directory, "analysis.json", {})
                paper = read_json(directory, "paper_metrics.json", {})
                counts = (read_json(directory, "summary.json", {}) or {}).get("split_counts") or config.get("split_counts") or {}
                seed = training.get("seed")
                split_valid = all(
                    int(counts.get(key, -1)) == expected
                    for key, expected in (("train", 40_000), ("validation", 5_000), ("test", 5_000))
                )
                dto = (paper.get("dto") or {}).get("mean")
                disentanglement = (paper.get("disentanglement") or {}).get("score")
                delta_kl = (paper.get("polarization_final") or {}).get("delta_kl")
                polarization = analysis.get("polarization_summary") or {}
                continuous = polarization.get("continuous_percent_to_end")
                interval = float(polarization.get("evaluation_interval_batches") or 0)
                steps = float(polarization.get("total_optimizer_steps") or 0)
                resolution = 100.0 * interval / steps if interval > 0 and steps > 0 else 0.0
                rows.append({
                    "id": config.get("id", directory.name),
                    "name": config.get("name", directory.name),
                    "dataset_source": config.get("dataset_source", ""),
                    "model_type": training.get("model_type", ""),
                    "model_family": model_family(training.get("model_type", "")),
                    "latent_dimension": int(training.get("latent_dim", config.get("latent_dim", 0)) or 0),
                    "seed": int(seed) if seed is not None else None,
                    "seed_source": training.get("seed_source"),
                    "split_counts": counts,
                    "split_valid": split_valid,
                    "dto": dto,
                    "disentanglement": disentanglement,
                    "continuous_percent_to_end": continuous,
                    "final_delta_kl": delta_kl,
                    "polarization_resolution_percent": resolution,
                })

            if not rows:
                raise FileNotFoundError("No experiments use that run label")

            datasets = sorted({row["dataset_source"] for row in rows})
            families = sorted({row["model_family"] for row in rows})
            latent_dimensions = sorted({row["latent_dimension"] for row in rows})
            homogeneous = len(datasets) == len(families) == len(latent_dimensions) == 1
            spec = paper_reproduction_spec(
                datasets[0] if len(datasets) == 1 else "",
                families[0] if len(families) == 1 else "",
                latent_dimensions[0] if len(latent_dimensions) == 1 else 0,
            )
            definitions = spec["metrics"]
            required_keys = [item["aggregate_key"] for item in definitions]
            for row in rows:
                row["metrics_complete"] = bool(definitions) and all(
                    row.get(key) is not None for key in required_keys
                )

            expected_seeds = set(range(1, 21))
            seed_groups = {}
            for row in rows:
                seed_groups.setdefault(row["seed"], []).append(row)
            present_seeds = {seed for seed in seed_groups if seed in expected_seeds}
            duplicate_seeds = sorted(seed for seed, group in seed_groups.items() if seed is not None and len(group) > 1)
            unexpected_seeds = sorted(seed for seed in seed_groups if seed not in expected_seeds and seed is not None)
            missing_seeds = sorted(expected_seeds - present_seeds)
            invalid_split_ids = [row["id"] for row in rows if not row["split_valid"]]
            incomplete_ids = [row["id"] for row in rows if not row["metrics_complete"]]
            invalid_seed_source_ids = [
                row["id"] for row in rows
                if row["seed_source"] != "effective_training_configuration"
            ]
            eligible = [
                row for row in rows
                if row["seed"] in expected_seeds
                and len(seed_groups[row["seed"]]) == 1
                and row["split_valid"]
                and row["metrics_complete"]
                and row["seed_source"] == "effective_training_configuration"
            ]

            def sample_statistics(key):
                values = np.asarray([float(row[key]) for row in eligible], dtype=float)
                if len(values) == 0:
                    return {"n": 0, "mean": None, "sample_standard_deviation": None, "ci95": [None, None]}
                mean = float(values.mean())
                standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                if len(values) == 1:
                    interval = [mean, mean]
                else:
                    generator = np.random.default_rng(2019)
                    samples = generator.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
                    interval = np.quantile(samples, [0.025, 0.975]).astype(float).tolist()
                return {
                    "n": int(len(values)), "mean": mean,
                    "sample_standard_deviation": standard_deviation,
                    "ci95": interval,
                }

            comparisons = []
            resolution_allowance = max(
                (row["polarization_resolution_percent"] for row in eligible),
                default=0.0,
            )
            for definition in definitions:
                key = definition["aggregate_key"]
                observed = sample_statistics(key)
                value = observed["mean"]
                operator = ""
                threshold = None
                assessment = ""
                if value is None:
                    status = "missing"
                elif definition["kind"] == "reported_mean":
                    operator = "paper_mean_in_ci95"
                    threshold = definition["mean"]
                    lower, upper = observed["ci95"]
                    numerical_tolerance = 1e-12 * max(1.0, abs(threshold))
                    status = "pass" if lower - numerical_tolerance <= threshold <= upper + numerical_tolerance else "fail"
                    assessment = "The paper mean must lie inside the observed bootstrap 95% CI of the suite mean."
                elif definition["kind"] == "maximum":
                    operator = "mean_at_most"
                    threshold = definition["threshold"]
                    status = "pass" if value <= threshold else "fail"
                    assessment = "The observed suite mean must not exceed the paper threshold."
                else:
                    operator = "mean_at_least_resolution_adjusted"
                    threshold = max(0.0, definition["threshold"] - resolution_allowance)
                    status = "pass" if value >= threshold else "fail"
                    assessment = "The observed suite mean must reach the paper percentage within one dashboard evaluation interval."
                comparisons.append({
                    "metric": key, "label": definition["label"],
                    "paper_target": definition["paper_target"],
                    "observed": observed, "operator": operator,
                    "threshold": threshold, "status": status,
                    "source": definition["source"], "assessment": assessment,
                    "resolution_allowance_percent": resolution_allowance if definition.get("resolution_aware") else 0.0,
                })

            integrity_pass = (
                len(rows) == 20 and not missing_seeds and not duplicate_seeds
                and not unexpected_seeds and not invalid_split_ids and not incomplete_ids
                and not invalid_seed_source_ids and homogeneous and bool(definitions)
                and len(eligible) == 20
            )
            metric_pass = bool(comparisons) and all(item["status"] == "pass" for item in comparisons)
            rows.sort(key=lambda row: (row["seed"] is None, row["seed"] or 0, row["id"]))
            return jsonify({
                "success": True,
                "schema_version": 2,
                "run_label": run_label,
                "paper": (
                    "Rolinek, Zietlow & Martius (CVPR 2019), "
                    f"{spec['dataset_source'].replace('paper_', 'synthetic-').replace('_', '-')} "
                    f"{spec['model_family'].replace('_', '-')}"
                ),
                "condition": spec,
                "status": "pass" if integrity_pass and metric_pass else ("incomplete" if not integrity_pass else "fail"),
                "integrity": {
                    "status": "pass" if integrity_pass else "incomplete",
                    "expected_runs": 20, "found_runs": len(rows), "eligible_runs": len(eligible),
                    "expected_seeds": list(range(1, 21)), "present_seeds": sorted(present_seeds),
                    "missing_seeds": missing_seeds, "duplicate_seeds": duplicate_seeds,
                    "unexpected_seeds": unexpected_seeds, "invalid_split_experiment_ids": invalid_split_ids,
                    "incomplete_metric_experiment_ids": incomplete_ids,
                    "invalid_seed_provenance_experiment_ids": invalid_seed_source_ids,
                    "homogeneous_condition": homogeneous,
                    "datasets": datasets, "model_families": families,
                    "latent_dimensions": latent_dimensions,
                },
                "aggregation": {
                    "mean": "arithmetic mean", "spread": "sample standard deviation (ddof=1)",
                    "ci95": "deterministic 10,000-resample percentile bootstrap; RNG seed 2019",
                    "table1_acceptance": "published mean must be inside the observed bootstrap 95% CI",
                    "polarization_resolution_allowance_percent": resolution_allowance,
                },
                "comparisons": comparisons,
                "runs": rows,
            })
        except FileNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/comparison", methods=["POST"])
    def geometry_comparison():
        try:
            ids = (request.get_json(force=True) or {}).get("experiment_ids", [])[:4]
            if len(ids) < 2:
                raise ValueError("Select at least two experiments")
            columns = []
            for experiment_id in ids:
                directory = exp_dir(experiment_id)
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                analysis = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
                scientific_path = directory / "scientific_metrics.json"
                scientific = json.loads(scientific_path.read_text(encoding="utf-8")) if scientific_path.exists() else {"summary": {}}
                scientific = add_paper_requirements(directory, config, scientific)
                columns.append({
                    "id": experiment_id, "name": config["name"], "config": presentation_config(config),
                    "metrics": scientific.get("summary", {}),
                    "projection": analysis.get("projection", [])[:800],
                    "factor_values": analysis.get("factor_values", [])[:800],
                })
            return jsonify({"success": True, "experiments": columns})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/intervention", methods=["POST"])
    def geometry_intervention(exp_id):
        try:
            payload = request.get_json(force=True) or {}
            factor = int(payload.get("factor_index", 0))
            alpha = float(payload.get("alpha", 0.5))
            requested_sample = int(payload.get("sample_index", 0))
            directory = exp_dir(exp_id)
            checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            model = GeometryAutoencoder(int(checkpoint["input_dim"]), checkpoint["training_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            data = load_data(exp_id)
            stored = np.load(directory / "jacobians.npz", allow_pickle=False)
            sample_indices = stored["sample_indices"]
            row = int(np.argmin(np.abs(sample_indices - requested_sample)))
            sample_index = int(sample_indices[row])
            factor = max(0, min(factor, stored["factor_jacobians"].shape[1] - 1))
            direction = stored["factor_jacobians"][row, factor, :, 0].astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            observation = torch.from_numpy(data["observations"][sample_index:sample_index + 1]).float()
            with torch.no_grad():
                z, _ = model.encode(observation)
                changed_z = z + alpha * torch.from_numpy(direction)[None]
                baseline = model.decoder(z)
                intervened = model.decoder(changed_z)
            zmean, zstd = stored["probe_zmean"], stored["probe_zstd"]
            coefficients, target_dims = stored["probe_coefficients"], stored["probe_target_dims"]
            before = ((z[0].numpy() - zmean) / zstd)
            after = ((changed_z[0].numpy() - zmean) / zstd)
            effects = []
            for measured in range(len(coefficients)):
                dims = int(target_dims[measured])
                first = before @ coefficients[measured, :, :dims]
                second = after @ coefficients[measured, :, :dims]
                effects.append(float(np.linalg.norm(second - first)))
            return jsonify({
                "success": True, "sample_index": sample_index, "factor_index": factor, "alpha": alpha,
                "latent_before": z[0].numpy().round(6).tolist(), "latent_after": changed_z[0].numpy().round(6).tolist(),
                "decoded_before": baseline[0].numpy().round(6).tolist(), "decoded_after": intervened[0].numpy().round(6).tolist(),
                "factor_effects": effects,
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    def train_sweep_model(data, raw_config, run_directory):
        config = _normalized_training_config(raw_config)
        torch.manual_seed(config["seed"])
        observations = torch.from_numpy(data["observations"]).float()
        train_tensor = observations[torch.from_numpy(data["split"] == 0)]
        val_tensor = observations[torch.from_numpy(data["split"] == 1)]
        model = GeometryAutoencoder(observations.shape[1], config)
        optimizer = None if config["model_type"] == "random_decoder" else _make_optimizer(model.parameters(), config)
        loader = torch.utils.data.DataLoader(
            train_tensor, batch_size=config["batch_size"], shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        metrics = []
        epochs_to_run = 1 if config["model_type"] == "random_decoder" else config["epochs"]
        for epoch in range(1, epochs_to_run + 1):
            model.train()
            sums, batches = np.zeros(4), 0
            for batch in loader:
                if optimizer is None:
                    with torch.no_grad():
                        values = _loss_terms(model, batch, config)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    values = _loss_terms(model, batch, config)
                    values[0].backward()
                    optimizer.step()
                sums += [float(values[i].detach()) for i in range(4)]
                batches += 1
            model.eval()
            with torch.no_grad():
                validation = float(_loss_terms(model, val_tensor, config)[1]) if len(val_tensor) else 0
                mu, _ = model.encode(observations[:min(4096, len(observations))])
                variance = mu.var(0)
            metrics.append({
                "epoch": epoch, "total_loss": sums[0] / max(1, batches),
                "reconstruction_loss": sums[1] / max(1, batches), "kl_divergence": sums[2] / max(1, batches),
                "regularizer": sums[3] / max(1, batches), "validation_reconstruction": validation,
                "latent_variance": float(variance.mean()), "active_latent_dimensions": int((variance > .01).sum()),
            })
        run_directory.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "training_config": config, "input_dim": observations.shape[1]}, run_directory / "model.pt")
        with torch.no_grad():
            mu, _ = model.encode(observations[:min(600, len(observations))])
        projection, _ = _pca(mu.numpy(), 3)
        analysis = {"metrics": metrics, "projection": projection.round(6).tolist()}
        scientific = run_scientific_analysis(model, data, analysis, run_directory, {
            "maximum_samples": min(96, len(observations)), "topology_points": min(96, len(observations)), "seed": config["seed"],
            "paper_metrics": bool(raw_config.get("paper_metrics", False)),
            "maximum_test_samples": int(raw_config.get("maximum_test_samples", 0)),
        })
        return model, metrics, scientific, projection

    def _as_list(value, default):
        value = default if value is None else value
        return value if isinstance(value, list) else [value]

    def sweep_job(job_id, sweep_id, specification):
        directory = root / "_sweeps" / sweep_id
        try:
            dataset_spec = specification.get("dataset", {})
            model_spec = specification.get("model", {})
            training_spec = specification.get("training", {})
            manifolds = _as_list(dataset_spec.get("manifold"), ["circle"])
            factors = _as_list(dataset_spec.get("factors"), [4])
            sparsities = _as_list(dataset_spec.get("sparsity"), [1, .5, .1])
            degeneracies = _as_list(dataset_spec.get("degeneracy"), [1.0])
            generator_types = _as_list(dataset_spec.get("generator_type"), ["linear"])
            model_types = _as_list(model_spec.get("type"), ["beta_vae"])
            latent_dims = _as_list(model_spec.get("latent_dim"), [2, 4])
            betas = _as_list(model_spec.get("beta"), [.1, 1, 4])
            seeds = _as_list(training_spec.get("seeds"), [1, 2, 3])
            combinations = list(itertools.product(manifolds, factors, sparsities, degeneracies, generator_types, model_types, latent_dims, betas, seeds))
            maximum_runs = max(1, min(512, int(specification.get("maximum_runs", 128))))
            if len(combinations) > maximum_runs:
                raise ValueError(f"Sweep expands to {len(combinations)} runs; maximum_runs is {maximum_runs}")
            results = []
            dataset_cache = {}
            for run_index, (manifold, k, sparsity, degeneracy, generator_type, model_type, latent_dim, beta, seed) in enumerate(combinations):
                key = (str(manifold), int(k), float(sparsity), float(degeneracy), str(generator_type))
                if key not in dataset_cache:
                    dataset_cache[key] = generate_dataset({
                        "dataset_type": manifold, "k": int(k), "sparsity": float(sparsity),
                        "dataset_size": int(dataset_spec.get("size", 1500)),
                        "observation_dim": int(dataset_spec.get("observation_dim", max(12, int(k) * 2))),
                        "noise": float(dataset_spec.get("noise", .02)), "train_split": float(dataset_spec.get("train_split", .8)),
                        "validation_split": float(dataset_spec.get("validation_split", .1)),
                        "seed": int(dataset_spec.get("seed", 42)), "orthogonal_mixing": True,
                        "degeneracy": float(degeneracy), "generator_type": str(generator_type),
                    })
                run_config = {
                    "model_type": model_type, "latent_dim": int(latent_dim), "beta": float(beta), "seed": int(seed),
                    "epochs": int(training_spec.get("epochs", 20)), "batch_size": int(training_spec.get("batch_size", 256)),
                    "learning_rate": float(training_spec.get("learning_rate", .001)), "optimizer": training_spec.get("optimizer", "adam"),
                    "hidden_dim": int(model_spec.get("hidden_dim", 96)), "encoder_depth": int(model_spec.get("encoder_depth", 2)),
                    "decoder_depth": int(model_spec.get("decoder_depth", 2)), "activation": model_spec.get("activation", "gelu"),
                    "encoder_activation": model_spec.get("encoder_activation", model_spec.get("activation", "gelu")),
                    "decoder_activation": model_spec.get("decoder_activation", model_spec.get("activation", "gelu")),
                    "custom_model": model_spec.get("custom_model", {}),
                    "paper_metrics": bool(specification.get("analysis", {}).get("paper_metrics", False)),
                    "maximum_test_samples": int(specification.get("analysis", {}).get("maximum_test_samples", 0)),
                }
                run_dir = directory / f"run_{run_index:04d}"
                _, metrics, scientific, projection = train_sweep_model(dataset_cache[key], run_config, run_dir)
                intrinsic = scientific["summary"]["required_intrinsic_dimensions"]
                result = {
                    "run_id": run_index, "manifold": manifold, "factors": int(k), "sparsity": float(sparsity),
                    "degeneracy": float(degeneracy), "generator_type": str(generator_type),
                    "model_type": model_type, "latent_dim": int(latent_dim), "beta": float(beta), "seed": int(seed),
                    "load_ratio": float(int(latent_dim) / max(1, intrinsic)), "metrics": scientific["summary"],
                    "projection": projection[:300].round(5).tolist(),
                }
                results.append(result)
                (run_dir / "run.json").write_text(json.dumps(result), encoding="utf-8")
                with job_lock:
                    jobs[job_id].update({"status": "running", "progress": (run_index + 1) / len(combinations), "completed_runs": run_index + 1, "total_runs": len(combinations), "latest": result})
            aggregate = {"id": sweep_id, "created_at": _utc_now(), "specification": specification, "runs": results}
            (directory / "sweep.json").write_text(json.dumps(aggregate), encoding="utf-8")
            (directory / "sweep.yaml").write_text(_to_yaml(aggregate) + "\n", encoding="utf-8")
            with job_lock:
                jobs[job_id].update({"status": "complete", "progress": 1, "completed_runs": len(results), "total_runs": len(results), "sweep": aggregate})
        except Exception as exc:
            traceback.print_exc()
            with job_lock:
                jobs[job_id].update({"status": "error", "error": str(exc)})

    @app.route("/api/geometry/sweeps", methods=["POST"])
    def geometry_create_sweep():
        try:
            content_type = request.content_type or ""
            if "yaml" in content_type or "text/plain" in content_type:
                import yaml
                specification = yaml.safe_load(request.get_data(as_text=True)) or {}
            else:
                specification = request.get_json(force=True) or {}
            sweep_id = "sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
            directory = root / "_sweeps" / sweep_id
            directory.mkdir(parents=True, exist_ok=False)
            job_id = uuid.uuid4().hex
            jobs[job_id] = {"id": job_id, "kind": "sweep", "sweep_id": sweep_id, "status": "queued", "progress": 0, "completed_runs": 0}
            threading.Thread(target=sweep_job, args=(job_id, sweep_id, specification), daemon=True).start()
            return jsonify({"success": True, "job_id": job_id, "sweep_id": sweep_id})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/sweeps", methods=["GET"])
    def geometry_list_sweeps():
        sweep_root = root / "_sweeps"
        items = []
        if sweep_root.exists():
            for directory in sweep_root.iterdir():
                path = directory / "sweep.json"
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    items.append({"id": payload["id"], "created_at": payload["created_at"], "runs": len(payload.get("runs", []))})
        return jsonify({"success": True, "sweeps": sorted(items, key=lambda x: x["created_at"], reverse=True)})

    @app.route("/api/geometry/sweeps/<sweep_id>", methods=["GET"])
    def geometry_get_sweep(sweep_id):
        try:
            path = root / "_sweeps" / _safe_id(sweep_id) / "sweep.json"
            return jsonify({"success": True, "sweep": json.loads(path.read_text(encoding="utf-8"))})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

    @app.route("/api/geometry/experiments/<exp_id>/projection", methods=["POST"])
    def geometry_projection(exp_id):
        try:
            method = str((request.get_json(silent=True) or {}).get("method", "pca")).lower()
            if method not in {"direct", "pca", "umap", "tsne"}:
                raise ValueError("Projection must be direct, PCA, UMAP, or t-SNE")
            loaded = np.load(exp_dir(exp_id) / "latent.npz", allow_pickle=False)
            latent = loaded["latent"]
            if method == "direct":
                projection = latent[:, :3]
                if projection.shape[1] < 3:
                    projection = np.pad(projection, ((0, 0), (0, 3 - projection.shape[1])))
            elif method == "pca":
                projection, _ = _pca(latent, 3)
            elif method == "umap":
                try:
                    import umap
                except ImportError as exc:
                    raise RuntimeError("UMAP is optional. Install dashboard requirements to enable it.") from exc
                projection = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, max(2, len(latent) - 1))).fit_transform(latent)
            else:
                try:
                    from sklearn.manifold import TSNE
                except ImportError as exc:
                    raise RuntimeError("t-SNE is optional. Install dashboard requirements to enable it.") from exc
                perplexity = min(30, max(2, (len(latent) - 1) // 3))
                projection = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(latent)
            return jsonify({"success": True, "method": method.upper(), "projection": np.asarray(projection).round(6).tolist()})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/traversal", methods=["POST"])
    def geometry_traversal(exp_id):
        try:
            directory = exp_dir(exp_id)
            checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            config = checkpoint["training_config"]
            model = GeometryAutoencoder(int(checkpoint["input_dim"]), config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            data = load_data(exp_id)
            if bool(np.asarray(data.get("empirical", False)).reshape(-1)[0]):
                raise ValueError("Controlled decoder traversal requires a differentiable synthetic generator; use interventions for empirical dSprites")
            factor_index = int((request.get_json(silent=True) or {}).get("factor_index", 0))
            factor_index = max(0, min(factor_index, data["factor_values"].shape[1] - 1))
            sample_index = int((request.get_json(silent=True) or {}).get("sample_index", 0)) % len(data["observations"])
            steps = max(12, min(240, int((request.get_json(silent=True) or {}).get("steps", 64))))
            kind = str(data["factor_types"][factor_index])
            params = np.repeat(data["factor_parameters"][sample_index:sample_index + 1], steps, axis=0)
            if kind in {"circle", "torus"}:
                values = np.linspace(0, 2 * math.pi, steps, endpoint=True)
            elif kind == "sphere":
                values = np.linspace(0, math.pi, steps)
            elif kind == "swiss_roll":
                values = np.linspace(1.5 * math.pi, 4.5 * math.pi, steps)
            else:
                values = np.linspace(-1, 1, steps)
            params[:, factor_index, 0] = values
            embeddings = []
            for index, fkind in enumerate(data["factor_types"].tolist()):
                embedded = _embed_factor(fkind, params[:, index])
                mask = np.repeat(data["factor_active"][sample_index:sample_index + 1, index:index + 1], steps, axis=0)
                if index == factor_index:
                    mask[:] = 1
                embeddings.append(embedded * mask)
            factor_embedding = np.concatenate(embeddings, axis=1).astype(np.float32)
            observations = torch.from_numpy(factor_embedding @ data["mixing_matrix"].T).float()
            with torch.no_grad():
                recon, _, mu, _ = model(observations)
                reference = torch.from_numpy(data["observations"][sample_index:sample_index + 1]).float()
                reference_recon, _, _, _ = model(reference)
            latent_projection, _ = _pca(mu.numpy(), 3)
            decoder_projection, _ = _pca(recon.numpy(), 3)
            original_projection, _ = _pca(observations.numpy(), 3)
            return jsonify({
                "success": True, "factor_index": factor_index, "factor_type": kind, "factor_values": values.round(6).tolist(),
                "original_trajectory": original_projection.round(6).tolist(), "latent_trajectory": latent_projection.round(6).tolist(),
                "decoder_trajectory": decoder_projection.round(6).tolist(),
                "original_sample": data["observations"][sample_index].round(6).tolist(),
                "reconstructed_sample": reference_recon[0].numpy().round(6).tolist(),
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 400

    return jobs
