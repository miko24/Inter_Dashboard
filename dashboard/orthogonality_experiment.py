"""Density/geometry VAE sweep and linear-vs-manifold Gram analysis.

The experiment intentionally lives outside the legacy sweep implementation.  Its
default grid is the exact 11 x 7 x 8 design requested by the dashboard user and
the stored NPZ artifact keeps every matrix available without returning a huge
JSON response from the run endpoint.
"""

from __future__ import annotations

import ast
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import jsonify, request


EPS = 1e-10
DEFAULT_DENSITIES = [value / 100.0 for value in range(0, 101, 10)]
DEFAULT_IMPORTANCE = [value / 2.0 for value in range(0, 7)]
DEFAULT_DIMENSIONS = list(range(3, 11))


class GeometryVAE(nn.Module):
    """The dashboard's regular tied-weight VAE, fixed to a 2D latent space."""

    def __init__(self, ambient_dim: int, hidden_dim: int = 2):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(hidden_dim, ambient_dim))
        self.w_logvar = nn.Parameter(torch.empty(hidden_dim, ambient_dim))
        self.bias = nn.Parameter(torch.zeros(ambient_dim))
        nn.init.xavier_normal_(self.w_mu)
        nn.init.xavier_normal_(self.w_logvar)

    def encode(self, x):
        mu = x @ self.w_mu.t()
        logvar = torch.clamp(x @ self.w_logvar.t(), -10.0, 10.0)
        return mu, logvar

    def decode(self, z):
        return F.relu(z @ self.w_mu + self.bias)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar, z


_EQUATION_FUNCTIONS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
}
_ALLOWED_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)


def compile_radius_equation(expression: str):
    """Compile r(theta) after rejecting attributes, indexing and unknown names."""

    expression = (expression or "1").strip()
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError(f"Unsupported custom-equation syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in {"theta", "pi", *_EQUATION_FUNCTIONS}:
            raise ValueError(f"Unknown custom-equation name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _EQUATION_FUNCTIONS:
                raise ValueError("Only sin, cos, tan, exp, sqrt, abs and log calls are allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in custom equations")
    code = compile(tree, "<geometry-equation>", "eval")

    def radius(theta):
        namespace = {**_EQUATION_FUNCTIONS, "theta": theta, "pi": np.pi}
        value = eval(code, {"__builtins__": {}}, namespace)
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:
            result = np.full_like(theta, float(result))
        if result.shape != theta.shape or not np.all(np.isfinite(result)):
            raise ValueError("Custom equation must produce one finite radius per theta")
        return np.clip(result, -10.0, 10.0)

    # Fail early so an invalid equation never starts a long sweep.
    radius(np.linspace(0.1, 2.0, 8))
    return radius


def geometry_batch(batch_size, ambient_dim, density, geometry, radius_fn, rng, device):
    """Sample a Fourier curve, then apply feature-wise Bernoulli sparsity."""

    theta = rng.uniform(0.0, 2.0 * np.pi, size=batch_size)
    radius = np.ones(batch_size) if geometry == "circle" else radius_fn(theta)
    points = np.empty((batch_size, ambient_dim), dtype=np.float32)
    for feature in range(ambient_dim):
        harmonic = feature // 2 + 1
        trig = np.cos(harmonic * theta) if feature % 2 == 0 else np.sin(harmonic * theta)
        signal = radius * trig
        # The legacy VAE decoder is ReLU-valued, so retain angular geometry while
        # mapping every coordinate to the model's [0, 1] observation convention.
        points[:, feature] = np.clip(0.5 + 0.5 * signal, 0.0, 1.0)
    mask = rng.random(points.shape) < density
    points *= mask
    return torch.as_tensor(points, dtype=torch.float32, device=device)


def importance_weights(ambient_dim: int, decay: float, device):
    values = [(1.0 - index / ambient_dim) ** decay for index in range(ambient_dim)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def normalize_feature_weights_(model):
    """Project each tied W_mu feature column onto the unit L2 sphere."""

    with torch.no_grad():
        norms = model.w_mu.norm(dim=0, keepdim=True).clamp_min(EPS)
        model.w_mu.div_(norms)


def _normalize_gram(gram):
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norms, norms)
    normalized = np.divide(np.abs(gram), denominator, out=np.zeros_like(gram), where=denominator > EPS)
    np.fill_diagonal(normalized, np.where(norms > EPS, 1.0, 0.0))
    return np.clip(normalized, 0.0, 1.0)


def _off_diagonal_mean(matrix):
    if matrix.shape[0] < 2:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def gram_analysis(model, evaluation_x):
    """Calculate Gram matrices over latents drawn from the encoder posterior.

    The VAE decodes posterior samples during training, so the decoder-induced
    metric must be measured at those samples as well.  Probing only at ``mu``
    makes sparse inputs (and every zero-density input) land at the ReLU kink or
    in an inactive decoder region, spuriously turning every Jacobian and the
    resulting manifold Gram matrix into zero.
    """

    with torch.no_grad():
        mu, logvar = model.encode(evaluation_x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        preactivation = z @ model.w_mu + model.bias
        active = (preactivation > 0).to(model.w_mu.dtype)
        w = model.w_mu.detach().cpu().numpy().astype(np.float64)  # (2, n)
        active_np = active.cpu().numpy().astype(np.float64)

    # J_s = diag(active_s) W^T, so G_s = J_s^T J_s.
    jacobians = active_np[:, :, None] * w.T[None, :, :]
    metrics = np.einsum("sni,snj->sij", jacobians, jacobians)
    mean_jacobian = jacobians.mean(axis=0)
    mean_metric = metrics.mean(axis=0)

    linear_gram = w.T @ w
    manifold_grams = np.einsum("ni,sij,jm->snm", w.T, metrics, w)
    manifold_gram = manifold_grams.mean(axis=0)

    determinants = np.linalg.det(metrics + EPS * np.eye(metrics.shape[-1])[None, :, :])
    factors = np.sqrt(np.maximum(determinants, 0.0))
    singular_values = np.linalg.svd(mean_jacobian, compute_uv=False)
    condition = float(singular_values[0] / max(singular_values[-1], EPS)) if singular_values.size else 0.0

    linear_normalized = _normalize_gram(linear_gram)
    manifold_normalized = _normalize_gram(manifold_gram)
    weight_norms = np.linalg.norm(w, axis=0)
    return {
        "weights": w,
        "linear_gram": linear_gram,
        "linear_normalized": linear_normalized,
        "jacobian": mean_jacobian,
        "metric": mean_metric,
        "manifold_gram": manifold_gram,
        "manifold_normalized": manifold_normalized,
        "jacobian_singular_values": singular_values,
        "jacobian_condition": condition,
        "factor_mean": float(np.mean(factors)),
        "factor_std": float(np.std(factors)),
        "linear_offdiag": _off_diagonal_mean(linear_normalized),
        "manifold_offdiag": _off_diagonal_mean(manifold_normalized),
        "weight_norm_min": float(np.min(weight_norms)),
        "weight_norm_max": float(np.max(weight_norms)),
    }


def train_cell(config, density, decay, ambient_dim, seed, device, radius_fn):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = GeometryVAE(ambient_dim, hidden_dim=2).to(device)
    if config["unit_norm_weights"]:
        normalize_feature_weights_(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    importance = importance_weights(ambient_dim, decay, device)
    final = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    for step in range(config["steps"]):
        x = geometry_batch(
            config["batch_size"], ambient_dim, density, config["geometry"], radius_fn, rng, device
        )
        x_hat, mu, logvar, _ = model(x)
        recon = (importance * (x - x_hat).pow(2)).sum(dim=1).mean()
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        beta_t = config["beta"] * min(1.0, (step + 1) / max(config["beta_warmup"], 1))
        loss = recon + beta_t * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if config["unit_norm_weights"]:
            normalize_feature_weights_(model)
        final = {"loss": float(loss.item()), "recon": float(recon.item()), "kl": float(kl.item())}

    evaluation_x = geometry_batch(
        config["evaluation_size"], ambient_dim, density, config["geometry"], radius_fn, rng, device
    )
    analysis = gram_analysis(model, evaluation_x)
    analysis.update(final)
    return analysis


def _float_values(raw, default, name, minimum, maximum):
    values = default if raw is None else [float(value) for value in raw]
    if not values or any(not math.isfinite(value) or value < minimum or value > maximum for value in values):
        raise ValueError(f"{name} values must be finite and between {minimum} and {maximum}")
    return values


def _int_values(raw, default, name, minimum, maximum):
    values = default if raw is None else [int(value) for value in raw]
    if not values or any(value < minimum or value > maximum for value in values):
        raise ValueError(f"{name} values must be between {minimum} and {maximum}")
    return values


def _bool_value(raw, default, name):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
        return raw.strip().lower() == "true"
    raise ValueError(f"{name} must be a boolean")


def normalize_config(raw):
    geometry = str(raw.get("geometry", "circle"))
    if geometry not in {"circle", "custom"}:
        raise ValueError("geometry must be 'circle' or 'custom'")
    config = {
        "densities": _float_values(raw.get("densities"), DEFAULT_DENSITIES, "density", 0.0, 1.0),
        "importance_decays": _float_values(
            raw.get("importance_decays"), DEFAULT_IMPORTANCE, "importance decay", 0.0, 3.0
        ),
        "ambient_dimensions": _int_values(
            raw.get("ambient_dimensions"), DEFAULT_DIMENSIONS, "ambient dimension", 3, 10
        ),
        "hidden_dimension": 2,
        "geometry": geometry,
        "custom_equation": str(raw.get("custom_equation", "1 + 0.25*cos(3*theta)")),
        "steps": int(raw.get("steps", 200)),
        "batch_size": int(raw.get("batch_size", 128)),
        "evaluation_size": int(raw.get("evaluation_size", 256)),
        "learning_rate": float(raw.get("learning_rate", 0.003)),
        "beta": float(raw.get("beta", 0.05)),
        "beta_warmup": int(raw.get("beta_warmup", 100)),
        "seed": int(raw.get("seed", 42)),
        "unit_norm_weights": _bool_value(raw.get("unit_norm_weights"), False, "unit_norm_weights"),
    }
    if not 1 <= config["steps"] <= 5000:
        raise ValueError("steps must be between 1 and 5000")
    if not 8 <= config["batch_size"] <= 4096 or not 16 <= config["evaluation_size"] <= 4096:
        raise ValueError("batch_size/evaluation_size are outside supported limits")
    total = len(config["densities"]) * len(config["importance_decays"]) * len(config["ambient_dimensions"])
    if total > 1000:
        raise ValueError("experiment contains more than 1000 VAE fits")
    compile_radius_equation(config["custom_equation"] if geometry == "custom" else "1")
    return config


def _json_matrix(matrix):
    return np.asarray(matrix).round(7).tolist()


def _cell_payload(npz, index, summaries):
    summary = summaries[index]
    ambient_dim = int(summary["ambient_dimension"])
    return {
        **summary,
        "weights": _json_matrix(npz["weights"][index, :, :ambient_dim]),
        "linear_gram": _json_matrix(npz["linear_gram"][index, :ambient_dim, :ambient_dim]),
        "linear_normalized": _json_matrix(npz["linear_normalized"][index, :ambient_dim, :ambient_dim]),
        "jacobian": _json_matrix(npz["jacobian"][index, :ambient_dim, :]),
        "metric": _json_matrix(npz["metric"][index]),
        "manifold_gram": _json_matrix(npz["manifold_gram"][index, :ambient_dim, :ambient_dim]),
        "manifold_normalized": _json_matrix(npz["manifold_normalized"][index, :ambient_dim, :ambient_dim]),
        "jacobian_singular_values": _json_matrix(npz["jacobian_singular_values"][index]),
    }


def register_orthogonality_experiment(app, experiments_dir):
    root = os.path.join(experiments_dir, "orthogonality")
    os.makedirs(root, exist_ok=True)

    @app.route("/api/orthogonality/run", methods=["POST"])
    def run_orthogonality_experiment():
        try:
            config = normalize_config(request.get_json(silent=True) or {})
            radius_fn = compile_radius_equation(
                config["custom_equation"] if config["geometry"] == "custom" else "1"
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            experiment_id = f"gram_{timestamp}"
            path = os.path.join(root, experiment_id)
            os.makedirs(path, exist_ok=False)

            maximum_dim = max(config["ambient_dimensions"])
            total = len(config["densities"]) * len(config["importance_decays"]) * len(config["ambient_dimensions"])
            arrays = {
                "weights": np.full((total, 2, maximum_dim), np.nan, dtype=np.float32),
                "linear_gram": np.full((total, maximum_dim, maximum_dim), np.nan, dtype=np.float32),
                "linear_normalized": np.full((total, maximum_dim, maximum_dim), np.nan, dtype=np.float32),
                "jacobian": np.full((total, maximum_dim, 2), np.nan, dtype=np.float32),
                "metric": np.full((total, 2, 2), np.nan, dtype=np.float32),
                "manifold_gram": np.full((total, maximum_dim, maximum_dim), np.nan, dtype=np.float32),
                "manifold_normalized": np.full((total, maximum_dim, maximum_dim), np.nan, dtype=np.float32),
                "jacobian_singular_values": np.full((total, 2), np.nan, dtype=np.float32),
            }
            summaries = []
            index = 0
            for ambient_dim in config["ambient_dimensions"]:
                for density in config["densities"]:
                    for decay in config["importance_decays"]:
                        cell_seed = config["seed"] + index
                        result = train_cell(config, density, decay, ambient_dim, cell_seed, device, radius_fn)
                        arrays["weights"][index, :, :ambient_dim] = result["weights"]
                        for name in ("linear_gram", "linear_normalized", "manifold_gram", "manifold_normalized"):
                            arrays[name][index, :ambient_dim, :ambient_dim] = result[name]
                        arrays["jacobian"][index, :ambient_dim, :] = result["jacobian"]
                        arrays["metric"][index] = result["metric"]
                        arrays["jacobian_singular_values"][index] = result["jacobian_singular_values"]
                        summaries.append({
                            "index": index,
                            "ambient_dimension": ambient_dim,
                            "hidden_dimension": 2,
                            "density": density,
                            "density_percent": density * 100.0,
                            "importance_decay": decay,
                            "loss": result["loss"],
                            "recon": result["recon"],
                            "kl": result["kl"],
                            "linear_offdiag": result["linear_offdiag"],
                            "manifold_offdiag": result["manifold_offdiag"],
                            "orthogonality_delta": result["manifold_offdiag"] - result["linear_offdiag"],
                            "jacobian_condition": result["jacobian_condition"],
                            "factor_mean": result["factor_mean"],
                            "factor_std": result["factor_std"],
                            "weight_norm_min": result["weight_norm_min"],
                            "weight_norm_max": result["weight_norm_max"],
                        })
                        index += 1

            np.savez_compressed(os.path.join(path, "matrices.npz"), **arrays)
            metadata = {
                "experiment_id": experiment_id,
                "created_at": datetime.now().isoformat(),
                "device": str(device),
                "definitions": {
                    "direction": (
                        "v_i is column i of the trained tied mean weight W_mu; each column is projected to unit L2 norm"
                        if config["unit_norm_weights"]
                        else "v_i is column i of the trained tied mean weight W_mu"
                    ),
                    "linear_gram": "L_ij = v_i^T v_j",
                    "linear_normalized": "|L_ij| / (||v_i|| ||v_j||)",
                    "jacobian": "J(z) = d decoder(z) / dz",
                    "factor": "sqrt(det(J(z)^T J(z)))",
                    "metric": "G(z) = J(z)^T J(z), z sampled from q(z|x)",
                    "manifold_gram": "MG_ij = E_x E_{z~q(z|x)}[v_i^T G(z) v_j]",
                    "manifold_normalized": "|MG_ij| / sqrt(MG_ii MG_jj)",
                },
                "config": config,
                "summaries": summaries,
            }
            with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            with np.load(os.path.join(path, "matrices.npz")) as npz:
                initial = _cell_payload(npz, 0, summaries)
            return jsonify({
                "success": True,
                "experiment_id": experiment_id,
                "device": str(device),
                "config": config,
                "definitions": metadata["definitions"],
                "summaries": summaries,
                "cell": initial,
            })
        except Exception as error:
            return jsonify({"success": False, "error": str(error)}), 400

    @app.route("/api/orthogonality/experiments/<experiment_id>/cell", methods=["GET"])
    def get_orthogonality_cell(experiment_id):
        if not experiment_id.startswith("gram_") or any(part in experiment_id for part in ("/", "\\", "..")):
            return jsonify({"success": False, "error": "Invalid experiment id"}), 400
        path = os.path.join(root, experiment_id)
        try:
            with open(os.path.join(path, "metadata.json"), encoding="utf-8") as handle:
                metadata = json.load(handle)
            index = int(request.args.get("index", 0))
            summaries = metadata["summaries"]
            if not 0 <= index < len(summaries):
                raise ValueError("Cell index is outside the experiment grid")
            with np.load(os.path.join(path, "matrices.npz")) as npz:
                cell = _cell_payload(npz, index, summaries)
            return jsonify({"success": True, "cell": cell})
        except (OSError, KeyError, ValueError) as error:
            return jsonify({"success": False, "error": str(error)}), 404
