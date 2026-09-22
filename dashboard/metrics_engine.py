"""Scientific metrics for manifold superposition experiments.

The public entry point, :func:`run_scientific_analysis`, computes a bounded,
seeded subset of local differential/topological metrics and persists the raw
arrays separately from the JSON dashboard summary.  Synthetic factor
Jacobians are exact autograd derivatives of

    parameters -> factor embedding -> mixing matrix -> encoder mean.

The implementation deliberately reports individual metrics.  It does not
invent a combined Manifold Superposition Index before those metrics have been
validated empirically.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper_metrics import run_paper_reproduction_metrics
from paper_requirements import evaluate_paper_requirements


INTRINSIC_DIMS = {
    "linear": 1,
    "circle": 1,
    "sphere": 2,
    "torus": 2,
    "swiss_roll": 2,
    "dsprites": 1,
}


def _torch_factor_embedding(kind, params):
    u, v = params[0], params[1]
    if kind == "linear":
        return u[None]
    if kind == "circle":
        return torch.stack((torch.cos(u), torch.sin(u)))
    if kind == "sphere":
        return torch.stack((torch.sin(u) * torch.cos(v), torch.sin(u) * torch.sin(v), torch.cos(u)))
    if kind == "torus":
        return torch.stack((torch.cos(u), torch.sin(u), torch.cos(v), torch.sin(v))) / math.sqrt(2)
    if kind == "swiss_roll":
        radius = u / (4 * math.pi)
        return torch.stack((radius * torch.cos(u), v, radius * torch.sin(u)))
    return torch.stack((u, torch.sin(math.pi * u), torch.cos(math.pi * u)))


def _sample_indices(total, maximum, seed):
    if total <= maximum:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=maximum, replace=False)).astype(np.int64)


def compute_factor_jacobians(model, data, maximum_samples=256, seed=42):
    """Compute exact local factor and decoder Jacobians with PyTorch autograd."""
    types = [str(x) for x in data["factor_types"].tolist()]
    indices = _sample_indices(len(data["observations"]), maximum_samples, seed)
    latent_dim = int(model.latent_dim)
    observation_dim = int(data["observations"].shape[1])
    factor_jacobians = np.zeros((len(indices), len(types), latent_dim, 2), dtype=np.float32)
    intrinsic_mask = np.zeros((len(types), 2), dtype=np.float32)
    for factor, kind in enumerate(types):
        intrinsic_mask[factor, :INTRINSIC_DIMS[kind]] = 1
    decoder_jacobians = np.zeros((len(indices), observation_dim, latent_dim), dtype=np.float32)
    posterior_precision = np.ones((len(indices), latent_dim), dtype=np.float32)
    latents = np.zeros((len(indices), latent_dim), dtype=np.float32)
    empirical = bool(np.asarray(data.get("empirical", False)).reshape(-1)[0])
    if empirical:
        model.eval()
        encoded_batches = []
        precision_batches = []
        encode_batch_size = max(8, min(512, 1_000_000 // max(1, observation_dim)))
        with torch.no_grad():
            for start in range(0, len(data["observations"]), encode_batch_size):
                batch = torch.from_numpy(data["observations"][start:start + encode_batch_size]).float()
                batch_latents, batch_logvar = model.encode(batch)
                if model.full_covariance:
                    covariance = model._last_cholesky @ model._last_cholesky.transpose(-1, -2)
                    batch_precision = torch.diagonal(torch.linalg.inv(covariance), dim1=-2, dim2=-1)
                elif model.variational:
                    batch_precision = torch.exp(-batch_logvar)
                else:
                    batch_precision = torch.ones_like(batch_latents)
                encoded_batches.append(batch_latents.cpu().numpy())
                precision_batches.append(batch_precision.cpu().numpy())
        all_latents = np.concatenate(encoded_batches)
        latents[:] = all_latents[indices]
        posterior_precision[:] = np.concatenate(precision_batches)[indices]
        scalar_params = data["factor_parameters"][:, :, 0].astype(np.float64)
        scales = scalar_params.std(0) + 1e-8
        normalized = (scalar_params - scalar_params.mean(0)) / scales
        for output_index, sample_index in enumerate(indices):
            delta = normalized - normalized[sample_index]
            for factor, kind in enumerate(types):
                if kind == "circle":
                    raw = scalar_params[:, factor] - scalar_params[sample_index, factor]
                    delta[:, factor] = np.arctan2(np.sin(raw), np.cos(raw)) / scales[factor]
            distances = np.linalg.norm(delta, axis=1)
            neighbors = np.argsort(distances)[1:min(33, len(distances))]
            design = scalar_params[neighbors] - scalar_params[sample_index]
            for factor, kind in enumerate(types):
                if kind == "circle":
                    raw = design[:, factor]
                    design[:, factor] = np.arctan2(np.sin(raw), np.cos(raw))
            response = all_latents[neighbors] - all_latents[sample_index]
            derivative = np.linalg.lstsq(design, response, rcond=None)[0]
            for factor in range(len(types)):
                factor_jacobians[output_index, factor, :, 0] = derivative[factor]
            z = torch.tensor(latents[output_index], dtype=torch.float32, requires_grad=True)
            decoder_jac = torch.autograd.functional.jacobian(lambda value: model.decoder(value[None])[0], z, vectorize=True)
            decoder_jacobians[output_index] = decoder_jac.detach().numpy()
        return {"indices": indices, "factor_jacobians": factor_jacobians, "decoder_jacobians": decoder_jacobians, "posterior_precision": posterior_precision, "posterior_precision_available": bool(model.variational), "intrinsic_mask": intrinsic_mask, "latents": latents, "estimator": "local_linear"}

    mixing = torch.from_numpy(data["mixing_matrix"]).float()
    generator_type = str(np.asarray(data.get("generator_type", ["linear"])).reshape(-1)[0])
    nonlinear_w1 = torch.from_numpy(data["nonlinear_w1"]).float() if "nonlinear_w1" in data else None
    nonlinear_b1 = torch.from_numpy(data["nonlinear_b1"]).float() if "nonlinear_b1" in data else None
    nonlinear_w2 = torch.from_numpy(data["nonlinear_w2"]).float() if "nonlinear_w2" in data else None
    model.eval()

    for output_index, sample_index in enumerate(indices):
        params = torch.tensor(data["factor_parameters"][sample_index], dtype=torch.float32, requires_grad=True)
        active = torch.from_numpy(data["factor_active"][sample_index]).float()

        def encode_parameters(p):
            pieces = [_torch_factor_embedding(kind, p[i]) * active[i] for i, kind in enumerate(types)]
            factors = torch.cat(pieces)
            if generator_type in {"nonlinear", "paper_nonlinear"}:
                observation = torch.tanh(factors @ nonlinear_w1.T + nonlinear_b1) @ nonlinear_w2.T
            else:
                observation = factors @ mixing.T
            mu, _ = model.encode(observation[None])
            return mu[0]

        jacobian = torch.autograd.functional.jacobian(encode_parameters, params, vectorize=True)
        factor_jacobians[output_index] = jacobian.detach().numpy().transpose(1, 0, 2) * intrinsic_mask[:, None, :]
        with torch.no_grad():
            observation = torch.from_numpy(data["observations"][sample_index:sample_index + 1]).float()
            mu, logvar = model.encode(observation)
            latents[output_index] = mu[0].numpy()
            if model.full_covariance:
                covariance = model._last_cholesky @ model._last_cholesky.transpose(-1, -2)
                posterior_precision[output_index] = torch.diagonal(torch.linalg.inv(covariance), dim1=-2, dim2=-1)[0].numpy()
            elif model.variational:
                posterior_precision[output_index] = torch.exp(-logvar)[0].numpy()

        z = torch.tensor(latents[output_index], dtype=torch.float32, requires_grad=True)
        decoder_jac = torch.autograd.functional.jacobian(lambda value: model.decoder(value[None])[0], z, vectorize=True)
        decoder_jacobians[output_index] = decoder_jac.detach().numpy()

    return {
        "indices": indices,
        "factor_jacobians": factor_jacobians,
        "decoder_jacobians": decoder_jacobians,
        "posterior_precision": posterior_precision,
        "posterior_precision_available": bool(model.variational),
        "intrinsic_mask": intrinsic_mask,
        "latents": latents,
        "estimator": "autograd",
    }


def decoder_geometry(jacobian_result, data):
    """Compute local decoder pullback geometry and polarization diagnostics."""
    jacobians = np.asarray(jacobian_result["decoder_jacobians"], dtype=np.float64)
    precision = np.asarray(jacobian_result["posterior_precision"], dtype=np.float64)
    sample_count, _, latent_dim = jacobians.shape
    gram = np.einsum("noi,noj->nij", jacobians, jacobians)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    denominator = np.sqrt(np.maximum(diagonal[:, :, None] * diagonal[:, None, :], 1e-16))
    normalized = gram / denominator
    singular_values = np.zeros((sample_count, latent_dim), dtype=np.float32)
    absolute_v = np.zeros((sample_count, latent_dim, latent_dim), dtype=np.float32)
    dto = np.zeros(sample_count, dtype=np.float32)
    eigenfaces = []
    image_shape = tuple(int(x) for x in np.asarray(data.get("image_shape", [])).reshape(-1))
    for sample, jacobian in enumerate(jacobians):
        u, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
        singular_values[sample, :len(singular)] = singular
        absolute_v[sample, :vh.shape[1], :vh.shape[0]] = np.abs(vh.T)
        off_diagonal = normalized[sample].copy()
        np.fill_diagonal(off_diagonal, 0.0)
        dto[sample] = np.linalg.norm(off_diagonal, ord="fro") / math.sqrt(max(1, latent_dim * (latent_dim - 1)))
        if sample == 0 and len(image_shape) in {2, 3} and int(np.prod(image_shape)) == jacobian.shape[0]:
            eigenfaces = u[:, :min(8, u.shape[1])].T.reshape((-1,) + image_shape).round(6).tolist()
    jacobian_norms = np.sqrt(np.maximum(diagonal, 0))
    safe_precision = np.maximum(precision, 1e-12)
    safe_norms = np.maximum(jacobian_norms, 1e-12)
    precision_threshold = float(np.exp(np.median(np.log(safe_precision))))
    norm_threshold = float(np.exp(np.median(np.log(safe_norms))))
    high_precision = precision >= precision_threshold
    high_norm = jacobian_norms >= norm_threshold
    regimes = {
        "active_committed": int(np.sum(high_precision & high_norm)),
        "uncertain_sensitive": int(np.sum(~high_precision & high_norm)),
        "precise_inactive": int(np.sum(high_precision & ~high_norm)),
        "inactive_prior": int(np.sum(~high_precision & ~high_norm)),
    }
    log_precision, log_norms = np.log(safe_precision).ravel(), np.log(safe_norms).ravel()
    correlation = np.corrcoef(log_precision, log_norms)[0, 1] if log_precision.std() > 1e-12 and log_norms.std() > 1e-12 else 0.0
    return {
        "sample_decoder_jacobian": jacobians[0].round(7).tolist() if sample_count else [],
        "mean_jtj": gram.mean(0).round(7).tolist(),
        "mean_normalized_jtj": normalized.mean(0).round(7).tolist(),
        "mean_singular_values": singular_values.mean(0).round(7).tolist(),
        "mean_absolute_v": absolute_v.mean(0).round(7).tolist(),
        "dto": dto.round(7).tolist(), "mean_dto": float(dto.mean()), "maximum_dto": float(dto.max()),
        "jacobian_column_norms": jacobian_norms.round(7).tolist(),
        "posterior_precision": precision.round(7).tolist(),
        "posterior_precision_available": bool(jacobian_result.get("posterior_precision_available", True)),
        "precision_threshold": precision_threshold, "jacobian_norm_threshold": norm_threshold,
        "precision_jacobian_log_correlation": float(np.nan_to_num(correlation)),
        "polarized_regimes": regimes, "nonlinear_eigenfaces": eigenfaces,
        "image_shape": list(image_shape), "sample_indices": jacobian_result["indices"].astype(int).tolist(),
    }


def _orthonormal_basis(matrix, tolerance=1e-7):
    if not np.any(np.isfinite(matrix)) or np.linalg.norm(matrix) <= tolerance:
        return np.empty((matrix.shape[0], 0), dtype=np.float32)
    clean = np.nan_to_num(matrix)
    rank = int(np.linalg.matrix_rank(clean, tol=tolerance * max(1.0, np.linalg.norm(clean))))
    q, _ = np.linalg.qr(clean, mode="reduced")
    return q[:, :rank].astype(np.float32)


def tangent_analysis(jacobian_result, data):
    jac = jacobian_result["factor_jacobians"]
    indices = jacobian_result["indices"]
    types = [str(x) for x in data["factor_types"].tolist()]
    active = data["factor_active"][indices]
    params = data["factor_parameters"][indices]
    sample_count, factor_count, latent_dim, _ = jac.shape
    norms = np.zeros((sample_count, factor_count), dtype=np.float32)
    factor_ranks = np.zeros((sample_count, factor_count), dtype=np.int16)
    combined_rank = np.zeros(sample_count, dtype=np.int16)
    condition = np.full(sample_count, np.nan, dtype=np.float32)
    singular_values = np.zeros((sample_count, latent_dim), dtype=np.float32)
    tangent_bases = np.zeros_like(jac)
    bases = [[None for _ in range(factor_count)] for _ in range(sample_count)]

    for sample in range(sample_count):
        combined = []
        for factor, kind in enumerate(types):
            dim = INTRINSIC_DIMS[kind]
            tangent = jac[sample, factor, :, :dim]
            norms[sample, factor] = np.linalg.norm(tangent)
            basis = _orthonormal_basis(tangent) if active[sample, factor] else np.empty((latent_dim, 0), dtype=np.float32)
            bases[sample][factor] = basis
            tangent_bases[sample, factor, :, :basis.shape[1]] = basis
            factor_ranks[sample, factor] = basis.shape[1]
            if active[sample, factor]:
                combined.append(tangent)
        combined = np.concatenate(combined, axis=1) if combined else np.empty((latent_dim, 0))
        values = np.linalg.svd(combined, compute_uv=False) if combined.size else np.array([])
        singular_values[sample, :len(values)] = values
        rank = int(np.sum(values > (values[0] * 1e-6 if len(values) else 0)))
        combined_rank[sample] = rank
        nonzero = values[values > 1e-8]
        if len(nonzero):
            condition[sample] = float(nonzero.max() / nonzero.min())

    overlap_samples = np.full((sample_count, factor_count, factor_count), np.nan, dtype=np.float32)
    mean_angles = np.full_like(overlap_samples, np.nan)
    min_angles = np.full_like(overlap_samples, np.nan)
    grassmann = np.full_like(overlap_samples, np.nan)
    pair_summaries = []
    local_profiles = []
    for i in range(factor_count):
        overlap_samples[:, i, i] = 1
        mean_angles[:, i, i] = min_angles[:, i, i] = grassmann[:, i, i] = 0
        for j in range(i + 1, factor_count):
            for sample in range(sample_count):
                u, v = bases[sample][i], bases[sample][j]
                if not (u.shape[1] and v.shape[1]):
                    continue
                cosines = np.clip(np.linalg.svd(u.T @ v, compute_uv=False), 0, 1)
                angles = np.arccos(cosines)
                overlap_samples[sample, i, j] = overlap_samples[sample, j, i] = float(cosines.max())
                mean_angles[sample, i, j] = mean_angles[sample, j, i] = float(angles.mean())
                min_angles[sample, i, j] = min_angles[sample, j, i] = float(angles.min())
                grassmann[sample, i, j] = grassmann[sample, j, i] = float(np.sqrt(np.sum(angles ** 2)))
            valid = np.isfinite(overlap_samples[:, i, j])
            values = overlap_samples[valid, i, j]
            angles = mean_angles[valid, i, j]
            distances = grassmann[valid, i, j]
            pair_summaries.append({
                "factor_i": i,
                "factor_j": j,
                "samples": int(valid.sum()),
                "mean_overlap": float(values.mean()) if len(values) else None,
                "variance_overlap": float(values.var()) if len(values) else None,
                "maximum_overlap": float(values.max()) if len(values) else None,
                "smallest_principal_angle": float(min_angles[valid, i, j].min()) if len(values) else None,
                "mean_principal_angle": float(angles.mean()) if len(values) else None,
                "grassmann_distance": float(distances.mean()) if len(values) else None,
            })
            if len(values):
                selected = np.where(valid)[0]
                local_profiles.append({
                    "factor_i": i,
                    "factor_j": j,
                    "factor_value": params[selected, i, 0].round(6).tolist(),
                    "other_factor_value": params[selected, j, 0].round(6).tolist(),
                    "overlap": values.round(6).tolist(),
                    "angle": angles.round(6).tolist(),
                    "sample_index": indices[selected].astype(int).tolist(),
                })

    mean_overlap = np.eye(factor_count, dtype=np.float32)
    variance_overlap = np.zeros((factor_count, factor_count), dtype=np.float32)
    maximum_overlap = np.eye(factor_count, dtype=np.float32)
    mean_grassmann = np.zeros((factor_count, factor_count), dtype=np.float32)
    for i in range(factor_count):
        for j in range(i + 1, factor_count):
            valid = np.isfinite(overlap_samples[:, i, j])
            if valid.any():
                values = overlap_samples[valid, i, j]
                mean_overlap[i, j] = mean_overlap[j, i] = values.mean()
                variance_overlap[i, j] = variance_overlap[j, i] = values.var()
                maximum_overlap[i, j] = maximum_overlap[j, i] = values.max()
                mean_grassmann[i, j] = mean_grassmann[j, i] = np.nanmean(grassmann[valid, i, j])
            else:
                mean_overlap[i, j] = mean_overlap[j, i] = np.nan
                variance_overlap[i, j] = variance_overlap[j, i] = np.nan
                maximum_overlap[i, j] = maximum_overlap[j, i] = np.nan
                mean_grassmann[i, j] = mean_grassmann[j, i] = np.nan

    off_diagonal = mean_overlap[~np.eye(factor_count, dtype=bool)]
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
    grass_values = mean_grassmann[~np.eye(factor_count, dtype=bool)]
    grass_values = grass_values[np.isfinite(grass_values)]
    return {
        "jacobian_norms": norms,
        "factor_ranks": factor_ranks,
        "combined_rank": combined_rank,
        "singular_values": singular_values,
        "tangent_bases": tangent_bases,
        "condition_number": condition,
        "overlap_samples": overlap_samples,
        "mean_overlap_matrix": mean_overlap,
        "variance_overlap_matrix": variance_overlap,
        "maximum_overlap_matrix": maximum_overlap,
        "grassmann_matrix": mean_grassmann,
        "pair_summaries": pair_summaries,
        "local_profiles": local_profiles,
        "mean_tangent_overlap": float(off_diagonal.mean()) if len(off_diagonal) else 0.0,
        "mean_grassmann_distance": float(grass_values.mean()) if len(grass_values) else 0.0,
    }


def folding_metrics(data, latents, indices, seed=42):
    rng = np.random.default_rng(seed)
    ground = data["factor_embeddings"][indices].astype(np.float64)
    latent = np.asarray(latents, dtype=np.float64)
    n = len(latent)
    if n < 3:
        return {"pairwise_distance_correlation": 0, "neighborhood_preservation": 0, "distortion_cv": 0}
    pair_count = min(12000, n * (n - 1) // 2)
    if n * (n - 1) // 2 <= pair_count:
        pairs = np.asarray(list(itertools.combinations(range(n), 2)), dtype=np.int32)
    else:
        first = rng.integers(0, n, pair_count * 2)
        second = rng.integers(0, n, pair_count * 2)
        valid = first != second
        pairs = np.column_stack((first[valid], second[valid]))[:pair_count]
    gd = np.linalg.norm(ground[pairs[:, 0]] - ground[pairs[:, 1]], axis=1)
    zd = np.linalg.norm(latent[pairs[:, 0]] - latent[pairs[:, 1]], axis=1)
    parameters = data["factor_parameters"][indices]
    activations = data["factor_active"][indices]
    geodesic_sq = np.zeros(len(pairs), dtype=np.float64)
    for factor, kind in enumerate(data["factor_types"].tolist()):
        same_active = activations[pairs[:, 0], factor] * activations[pairs[:, 1], factor]
        activation_change = np.abs(activations[pairs[:, 0], factor] - activations[pairs[:, 1], factor])
        first = parameters[pairs[:, 0], factor]
        second = parameters[pairs[:, 1], factor]
        if kind == "circle":
            delta = np.abs(np.arctan2(np.sin(first[:, 0] - second[:, 0]), np.cos(first[:, 0] - second[:, 0])))
        elif kind == "torus":
            du = np.arctan2(np.sin(first[:, 0] - second[:, 0]), np.cos(first[:, 0] - second[:, 0]))
            dv = np.arctan2(np.sin(first[:, 1] - second[:, 1]), np.cos(first[:, 1] - second[:, 1]))
            delta = np.sqrt(du ** 2 + dv ** 2)
        elif kind == "sphere":
            dot = np.sin(first[:, 0]) * np.sin(second[:, 0]) * np.cos(first[:, 1] - second[:, 1]) + np.cos(first[:, 0]) * np.cos(second[:, 0])
            delta = np.arccos(np.clip(dot, -1, 1))
        else:
            dimension = INTRINSIC_DIMS[str(kind)]
            delta = np.linalg.norm(first[:, :dimension] - second[:, :dimension], axis=1)
        geodesic_sq += (same_active * delta) ** 2 + activation_change
    geodesic = np.sqrt(geodesic_sq)
    valid = (gd > 1e-8) & (zd > 1e-8)
    ratios = zd[valid] / gd[valid]
    correlation = float(np.corrcoef(gd[valid], zd[valid])[0, 1]) if valid.sum() > 2 else 0.0

    distances_ground = np.linalg.norm(ground[:, None, :] - ground[None, :, :], axis=-1)
    distances_latent = np.linalg.norm(latent[:, None, :] - latent[None, :, :], axis=-1)
    neighbors = min(10, n - 1)
    ground_knn = np.argsort(distances_ground, axis=1)[:, 1:neighbors + 1]
    latent_knn = np.argsort(distances_latent, axis=1)[:, 1:neighbors + 1]
    preservation = np.mean([len(set(a).intersection(b)) / neighbors for a, b in zip(ground_knn, latent_knn)])
    scale = np.median(ratios) if len(ratios) else 1
    normalized_distortion = np.abs(ratios / max(scale, 1e-9) - 1) if len(ratios) else np.array([0])
    geodesic_valid = (geodesic > 1e-8) & (zd > 1e-8)
    geodesic_ratio = zd[geodesic_valid] / geodesic[geodesic_valid]
    geodesic_scale = np.median(geodesic_ratio) if len(geodesic_ratio) else 1
    geodesic_distortion = np.abs(geodesic_ratio / max(geodesic_scale, 1e-9) - 1) if len(geodesic_ratio) else np.array([0])
    # Self-intersections: very close in latent space but far in factor space.
    latent_threshold = np.quantile(zd[valid], .05) if valid.any() else 0
    ground_threshold = np.quantile(gd[valid], .75) if valid.any() else 0
    intersections = (zd < latent_threshold) & (gd > ground_threshold)
    return {
        "pairwise_distance_correlation": float(np.nan_to_num(correlation)),
        "neighborhood_preservation": float(preservation),
        "latent_distance_distortion_mean": float(normalized_distortion.mean()),
        "latent_distance_distortion_max": float(normalized_distortion.max()),
        "distortion_cv": float(ratios.std() / max(ratios.mean(), 1e-9)) if len(ratios) else 0.0,
        "geodesic_distance_distortion": float(geodesic_distortion.mean()),
        "geodesic_distance_distortion_max": float(geodesic_distortion.max()),
        "self_intersection_rate": float(intersections.mean()),
        "distance_ratio_sample": ratios[:2000].round(6).tolist(),
        "ground_distance_sample": gd[valid][:2000].round(6).tolist(),
        "latent_distance_sample": zd[valid][:2000].round(6).tolist(),
    }


def topology_metrics(latents, factor_types, maximum_points=300):
    points = np.asarray(latents, dtype=np.float64)
    if len(points) > maximum_points:
        points = points[np.linspace(0, len(points) - 1, maximum_points, dtype=int)]
    expected = None
    types = [str(x) for x in factor_types]
    if len(types) == 1 and types[0] == "circle":
        expected = [1, 1, 0]
    elif len(types) == 1 and types[0] == "torus":
        expected = [1, 2, 1]
    elif len(types) == 1 and types[0] == "sphere":
        expected = [1, 0, 1]
    elif types and all(x == "circle" for x in types):
        expected = [1, len(types), math.comb(len(types), 2)]
    try:
        from ripser import ripser
        result = ripser(points, maxdim=2)
        diagrams = result["dgms"]
        diameter = float(np.linalg.norm(points.max(0) - points.min(0)))
        threshold = max(1e-6, diameter * .08)
        betti = [1]
        encoded = []
        for dimension, diagram in enumerate(diagrams):
            finite = diagram[np.isfinite(diagram[:, 1])]
            lifetimes = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
            if dimension > 0:
                betti.append(int(np.sum(lifetimes > threshold)))
            encoded.append([[float(b), None if not np.isfinite(d) else float(d)] for b, d in diagram[:500]])
        while len(betti) < 3:
            betti.append(0)
        score = None
        if expected is not None:
            score = float(math.exp(-sum(abs(a - b) for a, b in zip(betti, expected))))
        return {"available": True, "betti_numbers": betti[:3], "expected_betti_numbers": expected, "topology_preservation": score, "persistence_threshold": threshold, "persistence_diagrams": encoded}
    except Exception as exc:
        return {"available": False, "error": str(exc), "betti_numbers": None, "expected_betti_numbers": expected, "topology_preservation": None, "persistence_diagrams": []}


def factor_probes(latents, data, indices, seed=42):
    z = np.asarray(latents, dtype=np.float64)
    parameters = data["factor_parameters"][indices]
    active = data["factor_active"][indices]
    types = [str(x) for x in data["factor_types"].tolist()]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(z))
    cut = max(2, int(.8 * len(z)))
    train, test = order[:cut], order[cut:]
    if not len(test):
        test = train
    zmean, zstd = z[train].mean(0), z[train].std(0) + 1e-8
    zn = (z - zmean) / zstd
    design = np.column_stack((zn, np.ones(len(zn))))
    results, coefficient_sets = [], []
    for factor, kind in enumerate(types):
        if kind in {"circle", "torus"}:
            target = np.column_stack((np.cos(parameters[:, factor, 0]), np.sin(parameters[:, factor, 0])))
        elif kind == "sphere":
            u, v = parameters[:, factor, 0], parameters[:, factor, 1]
            target = np.column_stack((np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)))
        else:
            target = parameters[:, factor, :INTRINSIC_DIMS[kind]]
        coefficient = np.linalg.lstsq(design[train], target[train], rcond=None)[0]
        prediction = design[test] @ coefficient
        residual = np.sum((target[test] - prediction) ** 2)
        total = np.sum((target[test] - target[test].mean(0)) ** 2)
        linear_score = float(1 - residual / max(total, 1e-12))
        coefficient_sets.append(coefficient[:-1])
        activation_coefficient = np.linalg.lstsq(design[train], active[train, factor], rcond=None)[0]
        activation_prediction = (design[test] @ activation_coefficient > .5).astype(float)
        activation_accuracy = float(np.mean(activation_prediction == active[test, factor]))
        categorical_accuracy = None
        try:
            from sklearn.linear_model import LogisticRegression
            activation_classes = np.unique(active[train, factor])
            if len(activation_classes) > 1:
                classifier = LogisticRegression(max_iter=300, random_state=seed).fit(zn[train], active[train, factor].astype(int))
                activation_accuracy = float(classifier.score(zn[test], active[test, factor].astype(int)))
            raw_classes = np.unique(parameters[:, factor, 0])
            if kind == "dsprites" and 1 < len(raw_classes) <= 20:
                categorical = LogisticRegression(max_iter=400, random_state=seed).fit(zn[train], parameters[train, factor, 0].astype(int))
                categorical_accuracy = float(categorical.score(zn[test], parameters[test, factor, 0].astype(int)))
        except Exception:
            pass
        nonlinear_score = None
        try:
            from sklearn.neural_network import MLPRegressor
            probe = MLPRegressor(hidden_layer_sizes=(32,), max_iter=250, random_state=seed, early_stopping=True)
            probe.fit(zn[train], target[train])
            nonlinear_score = float(probe.score(zn[test], target[test]))
        except Exception:
            nonlinear_score = linear_score
        results.append({"factor": factor, "type": kind, "linear_r2": linear_score, "activation_accuracy": activation_accuracy, "categorical_accuracy": categorical_accuracy, "nonlinear_mlp_r2": nonlinear_score})
    return results, coefficient_sets, zmean, zstd


def causal_interference(tangent, probe_coefficients, latent_scale=None):
    jac = tangent["factor_jacobians"]
    factor_count = jac.shape[1]
    directions = []
    for factor in range(factor_count):
        direction = np.nanmean(jac[:, factor, :, 0], axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        directions.append(direction)
    matrix = np.zeros((factor_count, factor_count), dtype=np.float32)
    for measured in range(factor_count):
        coefficient = probe_coefficients[measured]
        for intervened in range(factor_count):
            direction = directions[intervened] / latent_scale if latent_scale is not None else directions[intervened]
            matrix[measured, intervened] = np.linalg.norm(direction @ coefficient)
    column_scale = np.maximum(np.diag(matrix), 1e-8)
    normalized = matrix / column_scale[None, :]
    off = normalized[~np.eye(factor_count, dtype=bool)]
    return {"raw_matrix": matrix.tolist(), "normalized_matrix": normalized.tolist(), "mean_off_target": float(off.mean()) if len(off) else 0.0, "maximum_off_target": float(off.max()) if len(off) else 0.0}


def curvature_estimates(latents, tangent):
    z = np.asarray(latents)
    if len(z) < 4:
        return np.zeros(len(z), dtype=np.float32)
    distances = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    neighbors = np.argsort(distances, axis=1)[:, 1:min(9, len(z))]
    ranks = tangent["combined_rank"].astype(float)
    singular = tangent["singular_values"]
    estimates = np.zeros(len(z), dtype=np.float32)
    for i, local in enumerate(neighbors):
        local_scale = max(float(np.mean(distances[i, local])), 1e-8)
        spectral_change = np.mean(np.linalg.norm(singular[local] - singular[i], axis=1))
        rank_change = np.mean(np.abs(ranks[local] - ranks[i]))
        estimates[i] = (spectral_change + rank_change) / local_scale
    return estimates


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else 0.0


def run_scientific_analysis(model, data, training_analysis, output_dir, config=None):
    config = config or {}
    requested_maximum = max(16, min(1024, int(config.get("maximum_samples", 256))))
    local_complexity = int(data["observations"].shape[1]) * int(model.latent_dim)
    adaptive_maximum = max(16, min(1024, 300000 // max(1, local_complexity)))
    maximum = min(requested_maximum, adaptive_maximum)
    seed = int(config.get("seed", 42))
    output_dir = Path(output_dir)
    jacobian = compute_factor_jacobians(model, data, maximum_samples=maximum, seed=seed)
    tangent = tangent_analysis(jacobian, data)
    decoder = decoder_geometry(jacobian, data)
    folding = folding_metrics(data, jacobian["latents"], jacobian["indices"], seed)
    topology = topology_metrics(jacobian["latents"], data["factor_types"].tolist(), int(config.get("topology_points", 300)))
    probes, coefficients, probe_zmean, probe_zstd = factor_probes(jacobian["latents"], data, jacobian["indices"], seed)
    interference = causal_interference(jacobian, coefficients, probe_zstd)
    curvature = curvature_estimates(jacobian["latents"], tangent)
    paper = None
    if bool(config.get("paper_metrics", False)):
        paper = run_paper_reproduction_metrics(model, data, config)
        (output_dir / "paper_metrics.json").write_text(json.dumps(paper), encoding="utf-8")

    last_metric = (training_analysis.get("metrics") or [{}])[-1]
    latent_variance = np.var(jacobian["latents"], axis=0)
    expected_intrinsic = sum(INTRINSIC_DIMS[str(x)] for x in data["factor_types"].tolist())
    load_ratio = float(jacobian["latents"].shape[1] / max(1, expected_intrinsic))
    recoverability = _safe_mean([max(item["linear_r2"], item["nonlinear_mlp_r2"]) for item in probes])
    summary = {
        "schema_version": 1,
        "sample_count": int(len(jacobian["indices"])),
        "factor_count": int(data["factor_active"].shape[1]),
        "latent_dimension": int(jacobian["latents"].shape[1]),
        "required_intrinsic_dimensions": int(expected_intrinsic),
        "load_ratio": load_ratio,
        "tangent_overlap": tangent["mean_tangent_overlap"],
        "grassmann_distance": tangent["mean_grassmann_distance"],
        "reconstruction_error": float(last_metric.get("validation_reconstruction", last_metric.get("reconstruction_loss", 0))),
        "kl_divergence": float(last_metric.get("kl_divergence", 0)),
        "paper_polarized_continuous_percent": training_analysis.get("polarization_summary", {}).get("continuous_percent_to_end"),
        "factor_recoverability": recoverability,
        "manifold_distortion": float(folding["distortion_cv"]),
        "topology_preservation": topology.get("topology_preservation"),
        "causal_interference": float(interference["mean_off_target"]),
        "latent_utilization": int(np.sum(latent_variance > .01)),
        "mean_jacobian_norm": float(tangent["jacobian_norms"].mean()),
        "mean_jacobian_rank": float(tangent["combined_rank"].mean()),
        "mean_condition_number": _safe_mean(tangent["condition_number"]),
        "mean_curvature_estimate": float(curvature.mean()),
        "decoder_dto": decoder["mean_dto"],
        "decoder_maximum_dto": decoder["maximum_dto"],
        "decoder_gram_offdiagonal": decoder["mean_dto"],
        "precision_jacobian_correlation": decoder["precision_jacobian_log_correlation"],
    }
    if paper is not None:
        summary.update({
            "paper_dto": paper["dto"]["mean"],
            "paper_disentanglement": paper["disentanglement"]["score"],
            "paper_test_samples": paper["test_samples"],
            "paper_evaluation_scope": paper["evaluation_scope"],
            "paper_delta_kl": (paper.get("polarization_final") or {}).get("delta_kl"),
        })
    paper_requirements = evaluate_paper_requirements(
        config.get("dataset_source", ""), getattr(model, "kind", ""),
        getattr(model, "latent_dim", 0), summary,
    )
    summary.update({
        "paper_requirement_status": paper_requirements["status"],
        "paper_requirement_pass_fraction": paper_requirements["pass_fraction"],
        "paper_requirement_passed": paper_requirements["passed"],
        "paper_requirement_applicable": paper_requirements["applicable"],
    })
    dashboard = {
        "analysis_plan": [str(item) for item in config.get("analysis_plan", [])],
        "provenance": {
            "analysis_seed": seed,
            "seed_source": str(config.get("seed_source", "effective_training_configuration")),
        },
        "summary": summary,
        "paper_requirements": paper_requirements,
        "jacobian": {
            "estimator": jacobian.get("estimator", "autograd"),
            "norm_by_factor": tangent["jacobian_norms"].mean(0).tolist(),
            "rank_distribution": np.bincount(tangent["combined_rank"].astype(int), minlength=jacobian["latents"].shape[1] + 1).tolist(),
            "condition_numbers": np.nan_to_num(tangent["condition_number"], nan=-1, posinf=-1).round(6).tolist(),
            "mean_singular_values": tangent["singular_values"].mean(0).round(6).tolist(),
            "sample_indices": jacobian["indices"].astype(int).tolist(),
            "latent_points": jacobian["latents"][:, :3].round(6).tolist(),
            "local_directions": jacobian["factor_jacobians"][:, :, :3, 0].round(6).tolist(),
        },
        "tangent": {
            "mean_overlap_matrix": np.nan_to_num(tangent["mean_overlap_matrix"], nan=0).round(6).tolist(),
            "variance_overlap_matrix": np.nan_to_num(tangent["variance_overlap_matrix"], nan=0).round(6).tolist(),
            "maximum_overlap_matrix": np.nan_to_num(tangent["maximum_overlap_matrix"], nan=0).round(6).tolist(),
            "grassmann_matrix": np.nan_to_num(tangent["grassmann_matrix"], nan=0).round(6).tolist(),
            "pairs": tangent["pair_summaries"],
            "local_profiles": tangent["local_profiles"],
        },
        "folding": folding,
        "topology": topology,
        "probes": probes,
        "interference": interference,
        "decoder_geometry": decoder,
        "paper_metrics": paper,
        "curvature": curvature.round(6).tolist(),
    }
    max_target = max((item.shape[1] for item in coefficients), default=1)
    padded_coefficients = np.zeros((len(coefficients), jacobian["latents"].shape[1], max_target), dtype=np.float32)
    target_dims = np.zeros(len(coefficients), dtype=np.int16)
    for index, coefficient in enumerate(coefficients):
        padded_coefficients[index, :, :coefficient.shape[1]] = coefficient
        target_dims[index] = coefficient.shape[1]
    np.savez_compressed(
        output_dir / "jacobians.npz",
        sample_indices=jacobian["indices"], factor_jacobians=jacobian["factor_jacobians"],
        decoder_jacobians=jacobian["decoder_jacobians"], posterior_precision=jacobian["posterior_precision"], intrinsic_mask=jacobian["intrinsic_mask"],
        jacobian_norms=tangent["jacobian_norms"], jacobian_ranks=tangent["combined_rank"],
        singular_values=tangent["singular_values"], tangent_bases=tangent["tangent_bases"], condition_number=tangent["condition_number"],
        overlap_samples=tangent["overlap_samples"], curvature=curvature,
        probe_coefficients=padded_coefficients, probe_target_dims=target_dims,
        probe_zmean=probe_zmean.astype(np.float32), probe_zstd=probe_zstd.astype(np.float32),
    )
    (output_dir / "scientific_metrics.json").write_text(json.dumps(dashboard), encoding="utf-8")
    return dashboard
