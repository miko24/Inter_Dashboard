"""Acceptance calculations for paper-reproduction metrics.

Table 1 values are reproduction targets, not optimization objectives.  A value
that is much better than a published control is therefore a mismatch rather
than an automatic pass.  Per-run checks use the published mean +/- sample SD;
the suite endpoint performs the more meaningful aggregate compatibility test.
"""

from __future__ import annotations

import math


# Values transcribed from Rolinek et al. (2019), Tables 1 and 2.
TABLE1_TARGETS = {
    ("dsprites", "beta_vae"): {"dto": (0.76, 0.08), "disentanglement": (0.33, 0.15)},
    ("dsprites", "vae"): {"dto": (1.08, 0.15), "disentanglement": (0.21, 0.10)},
    ("dsprites", "ae"): {"dto": (1.62, 0.03), "disentanglement": (0.09, 0.04)},
    ("dsprites", "full_covariance"): {"dto": (1.73, 0.14), "disentanglement": (0.12, 0.06)},
    ("dsprites", "random_decoder"): {"dto": (1.86, 0.11)},
    ("paper_linear", "beta_vae"): {"dto": (0.00, 0.00), "disentanglement": (0.99, 0.01)},
    ("paper_linear", "ae"): {"dto": (0.33, 0.18), "disentanglement": (0.71, 0.19)},
    ("paper_linear", "full_covariance"): {"dto": (0.34, 0.35), "disentanglement": (0.71, 0.31)},
    ("paper_linear", "random_decoder"): {"dto": (0.79, 0.21)},
    ("paper_nonlinear", "beta_vae"): {"dto": (0.18, 0.02), "disentanglement": (0.73, 0.16)},
    ("paper_nonlinear", "ae"): {"dto": (0.54, 0.13), "disentanglement": (0.59, 0.30)},
    ("paper_nonlinear", "full_covariance"): {"dto": (0.55, 0.02), "disentanglement": (0.42, 0.24)},
    ("paper_nonlinear", "random_decoder"): {"dto": (0.89, 0.16)},
    ("mnist", "vae"): {"dto": (1.59, 0.08)},
    ("mnist", "ae"): {"dto": (1.83, 0.05)},
    ("mnist", "full_covariance"): {"dto": (1.93, 0.08)},
    ("mnist", "random_decoder"): {"dto": (2.11, 0.11)},
    ("fashion_mnist", "vae"): {"dto": (1.36, 0.05)},
    ("fashion_mnist", "ae"): {"dto": (1.87, 0.03)},
    ("fashion_mnist", "full_covariance"): {"dto": (2.02, 0.08)},
    ("fashion_mnist", "random_decoder"): {"dto": (2.11, 0.11)},
}

TABLE2_TARGETS = {
    "dsprites": {"dataset_dependent_dim": 5, "dataset_dependent": 97.8, "latent_10": 90.6},
    "fashion_mnist": {"dataset_dependent_dim": 6, "dataset_dependent": 99.8, "latent_10": 97.7},
    "mnist": {"dataset_dependent_dim": 6, "dataset_dependent": 99.8, "latent_10": 99.5},
    "paper_linear": {"dataset_dependent_dim": 2, "dataset_dependent": 99.8, "latent_10": 96.7},
    "paper_nonlinear": {"dataset_dependent_dim": 2, "dataset_dependent": 99.9, "latent_10": 98.5},
}


def model_family(model_type: str) -> str:
    if model_type in {"linear_ae", "nonlinear_ae"}:
        return "ae"
    if model_type == "beta_vae_full_cov":
        return "full_covariance"
    if model_type == "conv_beta_vae":
        return "beta_vae"
    return str(model_type)


def paper_reproduction_spec(dataset_source, model_type, latent_dim):
    """Return the published metrics applicable to one experimental condition."""
    dataset = str(dataset_source or "")
    family = model_family(str(model_type or ""))
    latent_dim = int(latent_dim or 0)
    metrics = []
    dataset_dependent_dim = TABLE2_TARGETS.get(dataset, {}).get("dataset_dependent_dim")
    table1 = (
        TABLE1_TARGETS.get((dataset, family), {})
        if dataset_dependent_dim is None or latent_dim == dataset_dependent_dim
        else {}
    )

    table1_fields = (
        ("dto", "paper_dto", "dto", "Paper DtO (Equation 29)"),
        ("disentanglement", "paper_disentanglement", "disentanglement", "Paper Disentanglement Score"),
    )
    for target_key, metric, aggregate_key, label in table1_fields:
        if target_key not in table1:
            continue
        mean, spread = table1[target_key]
        rounding = 0.005 if mean == 0 and spread == 0 else 0.0
        tolerance = max(float(spread), rounding)
        metrics.append({
            "metric": metric,
            "aggregate_key": aggregate_key,
            "label": label,
            "kind": "reported_mean",
            "paper_target": {"mean": mean, "sample_standard_deviation": spread},
            "mean": float(mean),
            "spread": float(spread),
            "per_run_bounds": [max(0.0, float(mean) - tolerance), float(mean) + tolerance],
            "source": "Table 1" + ("; 0.005 is the rounding bound for 0.00" if rounding else ""),
        })

    if family == "beta_vae" and dataset in TABLE2_TARGETS:
        target = TABLE2_TARGETS[dataset]
        polarized_target = None
        target_variant = None
        if latent_dim == target["dataset_dependent_dim"]:
            polarized_target = target["dataset_dependent"]
            target_variant = "dataset-dependent latent dimension"
        elif latent_dim == 10:
            polarized_target = target["latent_10"]
            target_variant = "latent dimension 10"
        if polarized_target is not None:
            metrics.extend([
                {
                    "metric": "paper_polarized_continuous_percent",
                    "aggregate_key": "continuous_percent_to_end",
                    "label": "Polarized duration (%)",
                    "kind": "minimum",
                    "paper_target": {"percent": polarized_target, "variant": target_variant},
                    "threshold": float(polarized_target),
                    "resolution_aware": True,
                    "source": f"Table 2, {target_variant}",
                },
                {
                    "metric": "paper_delta_kl",
                    "aggregate_key": "final_delta_kl",
                    "label": "Final Delta_KL (Equation 30)",
                    "kind": "maximum",
                    "paper_target": {"maximum": 0.03},
                    "threshold": 0.03,
                    "source": "Equation 30 / Table 2",
                },
            ])

    return {
        "dataset_source": dataset,
        "model_family": family,
        "latent_dimension": latent_dim,
        "metrics": metrics,
    }


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check(metric, label, actual, operator, threshold, target, source, rationale):
    value = _finite(actual)
    if value is None:
        status = "missing"
    elif operator == "between":
        lower, upper = threshold
        status = "pass" if lower <= value <= upper else "fail"
    elif operator == "at_most":
        status = "pass" if value <= threshold else "fail"
    else:
        status = "pass" if value >= threshold else "fail"
    return {
        "metric": metric,
        "label": label,
        "actual": value,
        "operator": operator,
        "threshold": [float(item) for item in threshold] if operator == "between" else float(threshold),
        "target": target,
        "source": source,
        "rationale": rationale,
        "status": status,
    }


def evaluate_paper_requirements(dataset_source, model_type, latent_dim, metrics):
    """Evaluate available run metrics against the matching published targets."""
    dataset = str(dataset_source or "")
    family = model_family(str(model_type or ""))
    latent_dim = int(latent_dim or 0)
    metrics = metrics or {}
    checks = []
    spec = paper_reproduction_spec(dataset, family, latent_dim)
    resolution = max(0.0, _finite(metrics.get("paper_polarization_resolution_percent")) or 0.0)
    for definition in spec["metrics"]:
        kind = definition["kind"]
        if kind == "reported_mean":
            operator = "between"
            threshold = definition["per_run_bounds"]
            rationale = "Reproduction agreement requires the run to lie within the published mean +/- sample SD."
        elif kind == "minimum":
            operator = "at_least"
            threshold = max(0.0, definition["threshold"] - resolution)
            rationale = "The paper threshold is reduced by at most one dashboard evaluation interval because duration is sampled on that grid."
        else:
            operator = "at_most"
            threshold = definition["threshold"]
            rationale = "The final diagnostic must be inside the paper's polarized-regime threshold."
        checks.append(_check(
            definition["metric"], definition["label"], metrics.get(definition["metric"]),
            operator, threshold, definition["paper_target"], definition["source"], rationale,
        ))

    passed = sum(item["status"] == "pass" for item in checks)
    failed = sum(item["status"] == "fail" for item in checks)
    missing = sum(item["status"] == "missing" for item in checks)
    if not checks:
        status = "not_applicable"
    elif missing:
        status = "incomplete"
    elif failed:
        status = "fail"
    else:
        status = "pass"
    evaluated = passed + failed
    return {
        "schema_version": 2,
        "status": status,
        "dataset_source": dataset,
        "model_family": family,
        "latent_dimension": latent_dim,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "applicable": len(checks),
        "evaluated": evaluated,
        "pass_fraction": (passed / len(checks)) if checks else None,
        "checks": checks,
        "note": "Per-run diagnostic only. Table 1 uses two-sided reproduction agreement; final acceptance requires aggregation across independent seeds and a 95% confidence interval.",
    }
