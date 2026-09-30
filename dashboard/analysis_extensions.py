"""Architecture-agnostic ablation and declarative custom analyses.

The registry deliberately accepts data, not Python source. This keeps a local
dashboard from turning a saved analysis definition into arbitrary code
execution while still allowing analyses to be reused in experiment plans.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flask import jsonify, request


CUSTOM_TYPES = {
    "activation_statistics": "Activation statistics",
    "parameter_statistics": "Parameter statistics",
    "representation_statistics": "Representation statistics",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _registry_path(root):
    return Path(root) / "custom_analyses.json"


def _load_registry(root):
    path = _registry_path(root)
    if not path.exists():
        return {"schema_version": 1, "analyses": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("analyses"), list):
            return value
    except Exception:
        pass
    return {"schema_version": 1, "analyses": []}


def custom_analysis_ids(root):
    return {f"custom:{item['id']}" for item in _load_registry(root)["analyses"] if "id" in item}


def _validate_definition(raw):
    identifier = str(raw.get("id", "")).strip()
    if not SAFE_ID.fullmatch(identifier):
        raise ValueError("Analysis id must contain only letters, numbers, '-' or '_'")
    name = str(raw.get("name", "")).strip()[:120]
    if not name:
        raise ValueError("Analysis name is required")
    kind = str(raw.get("type", "activation_statistics"))
    if kind not in CUSTOM_TYPES:
        raise ValueError(f"Analysis type must be one of: {', '.join(CUSTOM_TYPES)}")
    module_pattern = str(raw.get("module_pattern", ".*"))[:240]
    try:
        re.compile(module_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid module regular expression: {exc}") from exc
    statistics = raw.get("statistics") or ["mean", "std", "l1", "l2", "sparsity"]
    allowed = {"mean", "std", "min", "max", "l1", "l2", "sparsity", "q05", "median", "q95"}
    if not isinstance(statistics, list) or not statistics or any(str(item) not in allowed for item in statistics):
        raise ValueError(f"statistics must be a non-empty list drawn from {sorted(allowed)}")
    return {
        "id": identifier,
        "name": name,
        "description": str(raw.get("description", "")).strip()[:1000],
        "type": kind,
        "module_pattern": module_pattern,
        "statistics": list(dict.fromkeys(map(str, statistics))),
        "max_samples": max(1, min(10000, int(raw.get("max_samples", 256)))),
        "sparsity_threshold": max(0.0, float(raw.get("sparsity_threshold", 1e-6))),
        "created_at": str(raw.get("created_at") or _now()),
    }


def _stats(tensor, names, threshold):
    values = tensor.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {name: None for name in names}
    result = {}
    for name in names:
        if name == "mean": result[name] = float(values.mean())
        elif name == "std": result[name] = float(values.std(unbiased=False))
        elif name == "min": result[name] = float(values.min())
        elif name == "max": result[name] = float(values.max())
        elif name == "l1": result[name] = float(values.abs().mean())
        elif name == "l2": result[name] = float(values.square().mean().sqrt())
        elif name == "sparsity": result[name] = float((values.abs() <= threshold).float().mean())
        elif name == "q05": result[name] = float(torch.quantile(values, .05))
        elif name == "median": result[name] = float(torch.quantile(values, .5))
        elif name == "q95": result[name] = float(torch.quantile(values, .95))
    return result


def _update_accumulator(accumulator, tensor, threshold):
    values = tensor.detach().float().reshape(-1).cpu()
    if not values.numel():
        return
    accumulator["count"] += int(values.numel())
    accumulator["sum"] += float(values.sum())
    accumulator["sum_squares"] += float(values.square().sum())
    accumulator["absolute_sum"] += float(values.abs().sum())
    accumulator["sparse"] += int((values.abs() <= threshold).sum())
    accumulator["min"] = min(accumulator["min"], float(values.min()))
    accumulator["max"] = max(accumulator["max"], float(values.max()))
    if len(accumulator["sample"]) < 10000:
        stride = max(1, values.numel() // 4096)
        accumulator["sample"].extend(values[::stride][:10000 - len(accumulator["sample"])].tolist())


def _finalize_accumulator(accumulator, names):
    count = accumulator["count"]
    if not count:
        return {name: None for name in names}
    mean = accumulator["sum"] / count
    sample = torch.tensor(accumulator["sample"], dtype=torch.float32)
    values = {
        "mean": mean,
        "std": max(0.0, accumulator["sum_squares"] / count - mean * mean) ** .5,
        "min": accumulator["min"], "max": accumulator["max"],
        "l1": accumulator["absolute_sum"] / count,
        "l2": (accumulator["sum_squares"] / count) ** .5,
        "sparsity": accumulator["sparse"] / count,
        "q05": float(torch.quantile(sample, .05)), "median": float(torch.quantile(sample, .5)),
        "q95": float(torch.quantile(sample, .95)),
    }
    return {name: values[name] for name in names}


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    return None


def run_custom_analyses(model, data, directory, selected_ids, root):
    selected = {item.split(":", 1)[1] for item in selected_ids if str(item).startswith("custom:")}
    definitions = [item for item in _load_registry(root)["analyses"] if item.get("id") in selected]
    if not definitions:
        return {}
    observations = torch.as_tensor(data["observations"], dtype=torch.float32)
    results = {}
    modules = dict(model.named_modules())
    for definition in definitions:
        maximum = min(len(observations), int(definition["max_samples"]))
        batch = observations[:maximum]
        names = definition["statistics"]
        threshold = float(definition["sparsity_threshold"])
        kind = definition["type"]
        output = {"definition": definition, "sample_count": maximum, "modules": {}}
        if kind == "parameter_statistics":
            pattern = re.compile(definition["module_pattern"])
            for module_name, module in modules.items():
                if module_name and pattern.search(module_name):
                    for parameter_name, parameter in module.named_parameters(recurse=False):
                        output["modules"][f"{module_name}.{parameter_name}"] = _stats(parameter, names, threshold)
        elif kind == "representation_statistics":
            with torch.no_grad():
                encoded = [model.encode(batch[start:start + 128]) for start in range(0, len(batch), 128)]
                mu = torch.cat([item[0] for item in encoded])
                logvar = torch.cat([item[1] for item in encoded])
            output["modules"]["latent_mean"] = _stats(mu, names, threshold)
            output["modules"]["latent_log_variance"] = _stats(logvar, names, threshold)
        else:
            pattern = re.compile(definition["module_pattern"])
            handles = []
            captured = {}
            for module_name, module in modules.items():
                if module_name and pattern.search(module_name):
                    def capture(_module, _inputs, value, key=module_name):
                        tensor = _first_tensor(value)
                        if tensor is not None:
                            accumulator = captured.setdefault(key, {
                                "count": 0, "sum": 0.0, "sum_squares": 0.0, "absolute_sum": 0.0,
                                "sparse": 0, "min": float("inf"), "max": float("-inf"), "sample": [],
                            })
                            _update_accumulator(accumulator, tensor, threshold)
                    handles.append(module.register_forward_hook(capture))
            try:
                with torch.no_grad():
                    for start in range(0, len(batch), 32):
                        model(batch[start:start + 32])
            finally:
                for handle in handles:
                    handle.remove()
            output["modules"] = {key: _finalize_accumulator(value, names) for key, value in captured.items()}
        results[definition["id"]] = output
    artifact = {"schema_version": 1, "generated_at": _now(), "analyses": results}
    _atomic_json(Path(directory) / "custom_analysis_results.json", artifact)
    return artifact


def _module_targets(model, sample):
    captured = {}
    handles = []
    candidates = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.RNN, nn.GRU, nn.LSTM, nn.MultiheadAttention, nn.TransformerEncoderLayer)
    for name, module in model.named_modules():
        if name and (isinstance(module, candidates) or module.__class__.__module__.startswith("mamba_ssm")):
            def capture(_module, _inputs, output, key=name):
                tensor = _first_tensor(output)
                if tensor is not None:
                    captured[key] = list(tensor.shape)
            handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    modules = dict(model.named_modules())
    targets = []
    for name, shape in captured.items():
        axis = 1 if len(shape) == 4 else max(1, len(shape) - 1)
        targets.append({
            "name": name, "type": modules[name].__class__.__name__, "output_shape": shape,
            "unit_axis": axis, "unit_count": int(shape[axis]),
            "parameter_count": sum(parameter.numel() for parameter in modules[name].parameters(recurse=False)),
        })
    return targets


def _replace_first_tensor(output, replacement):
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = replacement
                return tuple(values)
    if isinstance(output, list):
        values = list(output)
        for index, value in enumerate(values):
            if torch.is_tensor(value):
                values[index] = replacement
                return values
    return output


def _ablation_hook(indices, axis, mode):
    def hook(_module, _inputs, output):
        tensor = _first_tensor(output)
        if tensor is None:
            return output
        changed = tensor.clone()
        selection = [slice(None)] * changed.ndim
        selection[axis] = indices
        selection = tuple(selection)
        if mode == "zero":
            changed[selection] = 0
        elif mode == "mean":
            reduced = changed.mean(dim=0, keepdim=True)
            changed[selection] = reduced.expand_as(changed)[selection]
        elif mode == "permutation":
            permutation = torch.arange(changed.shape[0] - 1, -1, -1, device=changed.device)
            changed[selection] = changed.index_select(0, permutation)[selection]
        return _replace_first_tensor(output, changed)
    return hook


def _reconstruction_mean(raw, training):
    return torch.sigmoid(raw) if training.get("reconstruction_loss") == "bernoulli_logits" else raw


def _run_manual_ablation(model, observations, training, target, indices, axis, mode, batch_size):
    baseline_errors, ablated_errors, latent_shifts = [], [], []
    module = dict(model.named_modules())[target]
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            batch = observations[start:start + batch_size]
            baseline_mu, _ = model.encode(batch)
            baseline_raw = model.decode(baseline_mu)
            baseline = _reconstruction_mean(baseline_raw, training)
            handle = module.register_forward_hook(_ablation_hook(indices, axis, mode))
            try:
                ablated_mu, _ = model.encode(batch)
                ablated_raw = model.decode(ablated_mu)
            finally:
                handle.remove()
            ablated = _reconstruction_mean(ablated_raw, training)
            baseline_errors.append((baseline - batch).square().flatten(1).mean(1).cpu())
            ablated_errors.append((ablated - batch).square().flatten(1).mean(1).cpu())
            latent_shifts.append((ablated_mu - baseline_mu).square().sum(1).sqrt().cpu())
    baseline = torch.cat(baseline_errors).numpy()
    ablated = torch.cat(ablated_errors).numpy()
    shifts = torch.cat(latent_shifts).numpy()
    delta = ablated - baseline
    return {
        "baseline_reconstruction_mse": float(baseline.mean()),
        "ablated_reconstruction_mse": float(ablated.mean()),
        "absolute_mse_delta": float(delta.mean()),
        "relative_mse_delta": float(delta.mean() / max(1e-12, baseline.mean())),
        "mean_latent_l2_shift": float(shifts.mean()),
        "sample_delta": delta.round(8).tolist(),
    }


def _captum_scores(model, observations, target, axis, max_units):
    try:
        from captum.attr import LayerActivation, LayerFeatureAblation
    except Exception as exc:
        raise RuntimeError("Captum is not installed. Install dashboard requirements to enable layer feature ablation.") from exc
    module = dict(model.named_modules())[target]
    inputs = observations[: min(len(observations), 128)]

    def score(batch):
        mu, _ = model.encode(batch)
        reconstruction = model.decode(mu)
        return -(reconstruction - batch).square().flatten(1).mean(1)

    activation = LayerActivation(score, module).attribute(inputs)
    activation = _first_tensor(activation)
    unit_count = int(activation.shape[axis])
    units = min(unit_count, max(1, int(max_units)))
    selected = torch.linspace(0, unit_count - 1, units, device=activation.device).round().long().unique()
    group_ids = torch.full((unit_count,), len(selected), dtype=torch.long, device=activation.device)
    group_ids[selected] = torch.arange(len(selected), device=activation.device)
    mask_shape = [1] * activation.ndim
    mask_shape[axis] = unit_count
    mask = group_ids.reshape(mask_shape)
    mask = mask.expand(*([inputs.shape[0]] + list(activation.shape[1:])))
    attribution = LayerFeatureAblation(score, module).attribute(
        inputs, layer_baselines=0, layer_mask=mask, perturbations_per_eval=1,
    )
    attribution = _first_tensor(attribution).detach().abs()
    reduction = tuple(index for index in range(attribution.ndim) if index != axis)
    values = attribution.mean(dim=reduction)
    ranked = selected[torch.argsort(values[selected], descending=True)]
    return [{"unit": int(index), "absolute_importance": float(values[index])} for index in ranked]


def register_analysis_extensions(app, root, experiment_directory, load_model, load_data):
    root = Path(root)

    @app.route("/api/geometry/custom-analyses", methods=["GET", "POST"])
    def custom_analyses_api():
        registry = _load_registry(root)
        if request.method == "GET":
            return jsonify({"success": True, "types": CUSTOM_TYPES, **registry})
        try:
            definition = _validate_definition(request.get_json(force=True) or {})
            registry["analyses"] = [item for item in registry["analyses"] if item.get("id") != definition["id"]]
            registry["analyses"].append(definition)
            _atomic_json(_registry_path(root), registry)
            return jsonify({"success": True, "analysis": definition})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/custom-analyses/<analysis_id>", methods=["DELETE"])
    def delete_custom_analysis(analysis_id):
        try:
            if not SAFE_ID.fullmatch(analysis_id):
                raise ValueError("Invalid analysis id")
            registry = _load_registry(root)
            before = len(registry["analyses"])
            registry["analyses"] = [item for item in registry["analyses"] if item.get("id") != analysis_id]
            if len(registry["analyses"]) == before:
                return jsonify({"success": False, "error": "Analysis not found"}), 404
            _atomic_json(_registry_path(root), registry)
            return jsonify({"success": True})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/ablation/capabilities", methods=["GET"])
    def ablation_capabilities():
        try:
            import captum
            captum_version = getattr(captum, "__version__", "installed")
        except Exception:
            captum_version = None
        try:
            import transformer_lens
            transformer_lens_version = getattr(transformer_lens, "__version__", "installed")
        except Exception:
            transformer_lens_version = None
        try:
            import mamba_ssm
            mamba_version = getattr(mamba_ssm, "__version__", "installed")
        except Exception:
            mamba_version = None
        return jsonify({"success": True, "packages": {
            "captum": {"available": captum_version is not None, "version": captum_version, "role": "layer feature ablation"},
            "transformer_lens": {"available": transformer_lens_version is not None, "version": transformer_lens_version, "role": "optional transformer hook adapter"},
            "mamba_ssm": {"available": mamba_version is not None, "version": mamba_version, "role": "optional Mamba model family"},
        }, "manual_hooks": True})

    @app.route("/api/geometry/experiments/<exp_id>/ablation/targets", methods=["GET"])
    def ablation_targets(exp_id):
        try:
            model, _training = load_model(exp_id)
            data = load_data(exp_id)
            sample = torch.as_tensor(data["observations"][:1], dtype=torch.float32)
            return jsonify({"success": True, "targets": _module_targets(model, sample)})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.route("/api/geometry/experiments/<exp_id>/ablations", methods=["GET", "POST"])
    def experiment_ablations(exp_id):
        try:
            directory = experiment_directory(exp_id)
            if request.method == "GET":
                artifacts = []
                for path in sorted(directory.glob("ablation_*.json"), reverse=True):
                    try:
                        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
                    except Exception:
                        continue
                return jsonify({"success": True, "ablations": artifacts})
            raw = request.get_json(force=True) or {}
            model, training = load_model(exp_id)
            data = load_data(exp_id)
            target = str(raw.get("target", ""))
            targets = {item["name"]: item for item in _module_targets(model, torch.as_tensor(data["observations"][:1], dtype=torch.float32))}
            if target not in targets:
                raise ValueError("Select a valid ablation target")
            metadata = targets[target]
            axis = int(raw.get("axis", metadata["unit_axis"]))
            indices = sorted(set(int(item) for item in (raw.get("indices") or [0])))
            if not indices or min(indices) < 0 or max(indices) >= metadata["unit_count"]:
                raise ValueError(f"Neuron/channel indices must be between 0 and {metadata['unit_count'] - 1}")
            mode = str(raw.get("mode", "zero"))
            if mode not in {"zero", "mean", "permutation"}:
                raise ValueError("Ablation mode must be zero, mean, or permutation")
            maximum = max(1, min(len(data["observations"]), int(raw.get("max_samples", 512))))
            observations = torch.as_tensor(data["observations"][:maximum], dtype=torch.float32)
            started = time.time()
            metrics = _run_manual_ablation(
                model, observations, training, target, indices, axis, mode,
                max(1, min(2048, int(raw.get("batch_size", 128)))),
            )
            method = str(raw.get("method", "manual_causal"))
            ranking = _captum_scores(model, observations, target, axis, int(raw.get("max_ranked_units", 64))) if method == "captum_layer_ablation" else []
            artifact = {
                "schema_version": 1, "id": "ablation_" + uuid.uuid4().hex[:12],
                "created_at": _now(), "experiment_id": exp_id, "architecture": getattr(model, "architecture", "unknown"),
                "method": method, "target": metadata, "indices": indices, "mode": mode,
                "sample_count": maximum, "metrics": metrics, "captum_unit_ranking": ranking,
                "elapsed_seconds": round(time.time() - started, 4),
            }
            _atomic_json(directory / f"{artifact['id']}.json", artifact)
            return jsonify({"success": True, "ablation": artifact})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
