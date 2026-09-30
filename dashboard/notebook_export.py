"""Portable, auditable Jupyter notebook exports for Geometry Lab experiments."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NOTEBOOK_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _encoded_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _cell(cell_type: str, source: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    result.update(extra)
    return result


def _source_blocks(path: Path, names: list[str]) -> list[tuple[str, str]]:
    """Extract exact top-level definitions using Python's syntax tree."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    wanted = set(names)
    blocks: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            end = node.end_lineno or node.lineno
            blocks.append((node.name, "\n".join(lines[start:end])))
    order = {name: index for index, name in enumerate(names)}
    return sorted(blocks, key=lambda item: order.get(item[0], len(order)))


def _artifact_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() != ".ipynb":
            manifest[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    return manifest


def build_experiment_notebook(directory: Path, dashboard_root: Path) -> Path:
    """Generate a notebook that audits and can re-run one dashboard experiment."""
    directory = Path(directory).resolve()
    dashboard_root = Path(dashboard_root).resolve()
    required = ("config.json", "training_config.json", "dataset.npz", "model.pt")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ValueError("A reproducibility notebook requires a trained complete experiment; missing: " + ", ".join(missing))
    config = _read_json(directory / "config.json", {}) or {}
    training = _read_json(directory / "training_config.json", {}) or {}
    results = _read_json(directory / "metrics.json", {}) or {}
    scientific = _read_json(directory / "scientific_metrics.json", {}) or {}
    analysis = _read_json(directory / "analysis.json", {}) or {}
    environment = _read_json(directory / "environment.json", {}) or {}
    experiment_id = str(config.get("id") or config.get("experiment_id") or directory.name)
    architecture = str((training.get("custom_model") or {}).get("architecture", training.get("architecture", "auto"))).lower()
    variation = str(training.get("model_type", "beta-vae")).lower()

    source_names = [
        "_normalized_training_config",
        "_resolve_training_device",
        "_reconstruction_mean",
        "_loss_terms",
        "_make_optimizer",
        "GeometryAutoencoder",
        "train_job",
    ]
    architecture_names = {
        "mlp": ["MLP"],
        "auto": ["MLP"],
        "convolutional": ["_conv_output_size", "ConvEncoder", "ConvDecoder"],
        "transformer": ["_sequence_shape", "TransformerVectorEncoder", "TransformerVectorDecoder"],
        "rnn": ["_sequence_shape", "RecurrentVectorEncoder", "RecurrentVectorDecoder"],
        "mamba": ["_sequence_shape", "PortableMamba", "_mamba_class", "MambaVectorNetwork"],
    }
    source_names.extend(architecture_names.get(architecture, []))
    if variation == "factorvae":
        source_names.append("FactorDiscriminator")
    if variation == "pcl":
        source_names.append("PairDiscriminator")

    geometry_path = dashboard_root / "geometry_lab.py"
    blocks = _source_blocks(geometry_path, source_names)
    source_files = [
        dashboard_root / "geometry_lab.py",
        dashboard_root / "metrics_engine.py",
        dashboard_root / "analysis_extensions.py",
        dashboard_root / "paper_metrics.py",
        dashboard_root / "paper_requirements.py",
        dashboard_root / "zietlow_reproduction.py",
    ]
    source_manifest = {
        path.name: _sha256(path) for path in source_files if path.is_file()
    }
    artifacts = _artifact_manifest(directory)
    embedded = {
        "config": config,
        "training": training,
        "results": results,
        "scientific": scientific,
        "analysis": analysis,
        "environment": environment,
        "artifact_manifest": artifacts,
        "source_manifest": source_manifest,
    }

    cells: list[dict[str, Any]] = [
        _cell(
            "markdown",
            f"# Reproducibility notebook — `{experiment_id}`\n\n"
            "Generated by the dashboard from the experiment's immutable configuration and saved artifacts. "
            "The notebook provides provenance checks, independent checkpoint evaluation, exact internal-source "
            "snapshots, and an opt-in route for re-running the same dashboard pipeline.\n\n"
            f"- Architecture: `{architecture}`\n"
            f"- Variation: `{variation}`\n"
            f"- Exported (UTC): `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- Notebook schema: `{NOTEBOOK_VERSION}`",
        ),
        _cell(
            "markdown",
            "## 1. Portable setup\n\n"
            "Keep this notebook beside the exported experiment folder, or set `EXPERIMENT_ROOT` and "
            "`DASHBOARD_ROOT` below. The original absolute locations are only fallbacks.",
        ),
        _cell(
            "code",
            "from pathlib import Path\n"
            "import base64, hashlib, json, os, sys\n"
            "import numpy as np\n"
            "import torch\n"
            "try:\n"
            "    from IPython.display import display\n"
            "except ImportError:\n"
            "    display = lambda value: print(ascii(value))\n\n"
            f"EXPERIMENT_ID = {experiment_id!r}\n"
            f"ORIGINAL_EXPERIMENT_DIR = Path({str(directory)!r})\n"
            f"ORIGINAL_DASHBOARD_ROOT = Path({str(dashboard_root)!r})\n\n"
            "def first_existing(*candidates):\n"
            "    for candidate in candidates:\n"
            "        candidate = Path(candidate).expanduser().resolve()\n"
            "        if candidate.exists():\n"
            "            return candidate\n"
            "    raise FileNotFoundError('None of the candidate paths exists: ' + ', '.join(map(str, candidates)))\n\n"
            "DASHBOARD_ROOT = first_existing(\n"
            "    os.environ.get('DASHBOARD_ROOT', Path.cwd() / 'dashboard'),\n"
            "    Path.cwd(),\n"
            "    ORIGINAL_DASHBOARD_ROOT,\n"
            ")\n"
            "EXPERIMENT_DIR = first_existing(\n"
            "    os.environ.get('EXPERIMENT_ROOT', Path.cwd() / EXPERIMENT_ID),\n"
            "    Path.cwd() / 'experiments' / EXPERIMENT_ID,\n"
            "    ORIGINAL_EXPERIMENT_DIR,\n"
            ")\n"
            "if str(DASHBOARD_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(DASHBOARD_ROOT))\n"
            "print('Dashboard:', DASHBOARD_ROOT)\n"
            "print('Experiment:', EXPERIMENT_DIR)\n"
            "print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| available:', torch.cuda.is_available())",
        ),
        _cell(
            "markdown",
            "## 2. Embedded configuration and provenance\n\n"
            "The JSON below is embedded at export time, so it remains inspectable even if separate report files change.",
        ),
        _cell(
            "code",
            f"EMBEDDED = json.loads(base64.b64decode({_encoded_json(embedded)!r}).decode('utf-8'))\n"
            "CONFIG = EMBEDDED['config']\n"
            "TRAINING = EMBEDDED['training']\n"
            "SAVED_RESULTS = EMBEDDED['results']\n"
            "display({'configuration': CONFIG, 'training': TRAINING, 'environment': EMBEDDED['environment']})",
        ),
        _cell(
            "code",
            "def sha256_file(path):\n"
            "    digest = hashlib.sha256()\n"
            "    with Path(path).open('rb') as stream:\n"
            "        for chunk in iter(lambda: stream.read(1024 * 1024), b''):\n"
            "            digest.update(chunk)\n"
            "    return digest.hexdigest()\n\n"
            "artifact_checks = {}\n"
            "for name, expected in EMBEDDED['artifact_manifest'].items():\n"
            "    path = EXPERIMENT_DIR / name\n"
            "    actual = sha256_file(path) if path.is_file() else None\n"
            "    artifact_checks[name] = {'present': path.is_file(), 'expected': expected['sha256'], 'actual': actual, 'match': actual == expected['sha256']}\n"
            "source_checks = {}\n"
            "for name, expected in EMBEDDED['source_manifest'].items():\n"
            "    path = DASHBOARD_ROOT / name\n"
            "    actual = sha256_file(path) if path.is_file() else None\n"
            "    source_checks[name] = {'present': path.is_file(), 'expected': expected, 'actual': actual, 'match': actual == expected}\n"
            "display({'artifact_integrity': artifact_checks, 'source_integrity': source_checks})\n"
            "assert artifact_checks.get('dataset.npz', {}).get('match'), 'Dataset is missing or differs from the exported artifact.'\n"
            "assert artifact_checks.get('model.pt', {}).get('match'), 'Checkpoint is missing or differs from the exported artifact.'",
        ),
        _cell(
            "markdown",
            "## 3. Load the exact dataset split and checkpoint\n\n"
            "Evaluation defaults to CPU for portability. Set `NOTEBOOK_DEVICE=cuda` to require a CUDA device.",
        ),
        _cell(
            "code",
            "from geometry_lab import GeometryAutoencoder, _normalized_training_config, _reconstruction_mean\n\n"
            "requested_device = os.environ.get('NOTEBOOK_DEVICE', 'cpu').lower()\n"
            "if requested_device == 'cuda' and not torch.cuda.is_available():\n"
            "    raise RuntimeError('NOTEBOOK_DEVICE=cuda was requested, but CUDA is unavailable.')\n"
            "DEVICE = torch.device(requested_device)\n"
            "dataset = np.load(EXPERIMENT_DIR / 'dataset.npz', allow_pickle=False)\n"
            "observations = np.asarray(dataset['observations'], dtype=np.float32)\n"
            "split = np.asarray(dataset['split'], dtype=np.int8)\n"
            "test_label = 2 if np.any(split == 2) else 1\n"
            "test_indices = np.flatnonzero(split == test_label).astype(np.int64)\n"
            "input_dim = int(np.prod(observations.shape[1:]))\n"
            "normalized_training = _normalized_training_config(TRAINING)\n"
            "checkpoint = torch.load(EXPERIMENT_DIR / 'model.pt', map_location='cpu', weights_only=True)\n"
            "checkpoint_training = checkpoint.get('training_config', normalized_training)\n"
            "model = GeometryAutoencoder(input_dim, checkpoint_training).to(DEVICE)\n"
            "model.load_state_dict(checkpoint['model_state_dict'])\n"
            "model.eval()\n"
            "print(f\"Loaded {len(test_indices):,} test samples on {DEVICE}; backend={getattr(model, 'architecture_backend', 'default')}\")",
        ),
        _cell(
            "markdown",
            "## 4. Independent deterministic checkpoint evaluation\n\n"
            "This audit decodes the posterior mean (`z = μ`) rather than sampling latent noise. It therefore checks "
            "the saved model deterministically; stochastic metrics in the original report can differ slightly.",
        ),
        _cell(
            "code",
            "@torch.no_grad()\n"
            "def deterministic_reconstruction_audit(batch_size=1024):\n"
            "    squared_error = 0.0\n"
            "    absolute_error = 0.0\n"
            "    element_count = 0\n"
            "    for start in range(0, len(test_indices), batch_size):\n"
            "        index = test_indices[start:start + batch_size]\n"
            "        batch = torch.from_numpy(observations[index].reshape(len(index), -1)).to(DEVICE)\n"
            "        mu, _ = model.encode(batch)\n"
            "        reconstruction = _reconstruction_mean(model.decode(mu), checkpoint_training)\n"
            "        difference = reconstruction - batch\n"
            "        squared_error += difference.square().sum().item()\n"
            "        absolute_error += difference.abs().sum().item()\n"
            "        element_count += difference.numel()\n"
            "    return {\n"
            "        'deterministic_mse': squared_error / element_count,\n"
            "        'deterministic_rmse': (squared_error / element_count) ** 0.5,\n"
            "        'deterministic_mae': absolute_error / element_count,\n"
            "        'samples': int(len(test_indices)),\n"
            "        'elements': int(element_count),\n"
            "    }\n\n"
            "AUDIT_RESULTS = deterministic_reconstruction_audit()\n"
            "display({'independent_audit': AUDIT_RESULTS, 'saved_training_results': SAVED_RESULTS})",
        ),
        _cell(
            "markdown",
            "## 5. Saved analyses\n\n"
            "These are the dashboard's complete stored scientific and interpretability outputs at export time.",
        ),
        _cell(
            "code",
            "def compact(value, list_limit=20):\n"
            "    if isinstance(value, dict):\n"
            "        return {key: compact(item, list_limit) for key, item in value.items()}\n"
            "    if isinstance(value, list) and len(value) > list_limit:\n"
            "        return {'type': 'list', 'length': len(value), 'preview': value[:3]}\n"
            "    return value\n\n"
            "display({\n"
            "    'scientific_results': compact(EMBEDDED['scientific']),\n"
            "    'analysis_results': compact(EMBEDDED['analysis']),\n"
            "})",
        ),
        _cell(
            "markdown",
            "## 6. Re-run the exact dashboard pipeline (opt in)\n\n"
            "The cell below is intentionally disabled. When the local dashboard is running and still contains this "
            "experiment, setting `RUN_DASHBOARD_RETRAIN = True` clones its immutable dataset and submits the embedded "
            "training configuration to the same internal trainer—including variation-specific objectives. It does not "
            "send data to an external service.",
        ),
        _cell(
            "code",
            "RUN_DASHBOARD_RETRAIN = False\n"
            "DASHBOARD_URL = os.environ.get('DASHBOARD_URL', 'http://127.0.0.1:5000').rstrip('/')\n\n"
            "def request_json(method, path, payload=None):\n"
            "    from urllib.request import Request, urlopen\n"
            "    body = json.dumps(payload).encode('utf-8') if payload is not None else None\n"
            "    request = Request(DASHBOARD_URL + path, data=body, method=method, headers={'Content-Type': 'application/json'})\n"
            "    with urlopen(request) as response:\n"
            "        return json.loads(response.read().decode('utf-8'))\n\n"
            "if RUN_DASHBOARD_RETRAIN:\n"
            "    clone = request_json('POST', f'/api/geometry/experiments/{EXPERIMENT_ID}/clone', {\n"
            "        'name': f\"{CONFIG.get('name', EXPERIMENT_ID)} — notebook rerun\",\n"
            "        'run_label': 'notebook-rerun',\n"
            "        'tags': list(CONFIG.get('tags') or []) + ['notebook-rerun'],\n"
            "        'paper_protocol': CONFIG.get('paper_protocol') or {},\n"
            "    })\n"
            "    rerun_id = clone.get('experiment_id') or clone.get('experiment', {}).get('id')\n"
            "    if not rerun_id:\n"
            "        raise RuntimeError(f'Clone response did not contain an experiment id: {clone}')\n"
            "    rerun_training = dict(TRAINING)\n"
            "    effective_device = str(TRAINING.get('effective_device', 'cpu'))\n"
            "    rerun_training['device'] = 'cuda' if effective_device.startswith('cuda') else 'cpu'\n"
            "    submitted = request_json('POST', f'/api/geometry/experiments/{rerun_id}/train', rerun_training)\n"
            "    display({'cloned_experiment_id': rerun_id, 'submission': submitted})\n"
            "else:\n"
            "    print('Exact dashboard retraining is disabled. Set RUN_DASHBOARD_RETRAIN=True to submit it.')",
        ),
        _cell(
            "markdown",
            "## 7. Internal implementation snapshot\n\n"
            "The following blocks were extracted directly from the dashboard source at export time. The hashes above "
            "detect later source drift. The notebook executes these definitions through the imported dashboard module; "
            "the snapshots make the implementation reviewable without hiding logic behind the UI.",
        ),
    ]

    for name, source in blocks:
        cells.append(_cell("markdown", f"### `{name}`\n\n```python\n{source}\n```"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "geometry_lab": {
                "schema_version": NOTEBOOK_VERSION,
                "experiment_id": experiment_id,
                "artifact_manifest": artifacts,
                "source_manifest": source_manifest,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = directory / f"{experiment_id}_reproducibility.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return destination
