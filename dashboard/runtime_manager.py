"""Local compute detection and constrained PyTorch CUDA runtime installation.

Only fixed, official PyTorch wheel indexes can be selected. CUDA wheels are
installed into an isolated runtime directory and activated on the next server
restart, leaving the existing CPU runtime intact for recovery.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from flask import jsonify, request


CUDA_BUILDS = {
    "cu130": {"label": "CUDA 13.0", "maximum_driver_cuda": 13.0},
    "cu128": {"label": "CUDA 12.8", "maximum_driver_cuda": 12.8},
    "cu126": {"label": "CUDA 12.6", "maximum_driver_cuda": 12.6},
    "cu124": {"label": "CUDA 12.4", "maximum_driver_cuda": 12.4},
    "cu121": {"label": "CUDA 12.1", "maximum_driver_cuda": 12.1},
    "cu118": {"label": "CUDA 11.8", "maximum_driver_cuda": 11.8},
}
CONFIRMATION = "INSTALL CUDA PYTORCH"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _same_origin_request():
    origin = request.headers.get("Origin", "")
    if not origin:
        return True
    return bool(re.fullmatch(r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?", origin, re.IGNORECASE))


def _nvidia_status():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi was not found"}
    try:
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        rows = []
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                rows.append({"name": parts[0], "driver_version": parts[1], "memory_mib": int(float(parts[2])), "compute_capability": parts[3]})
        full = subprocess.run([executable], capture_output=True, text=True, timeout=10, check=True).stdout
        match = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+(?:\.[0-9]+)?)", full)
        maximum_cuda = float(match.group(1)) if match else None
        return {"available": bool(rows), "gpus": rows, "maximum_cuda": maximum_cuda, "executable": executable}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _recommended_build(nvidia):
    maximum = nvidia.get("maximum_cuda")
    if maximum is None:
        return None
    for build, metadata in CUDA_BUILDS.items():
        if metadata["maximum_driver_cuda"] <= maximum:
            return build
    return None


def _public_status(dashboard_root):
    nvidia = _nvidia_status()
    recommended = _recommended_build(nvidia)
    cuda_runtime = Path(dashboard_root) / ".runtime-cuda"
    return {
        "success": True,
        "python": sys.version.split()[0],
        "torch": {
            "version": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        },
        "nvidia": nvidia,
        "builds": [{"id": key, **value, "compatible": nvidia.get("maximum_cuda") is not None and value["maximum_driver_cuda"] <= nvidia["maximum_cuda"]} for key, value in CUDA_BUILDS.items()],
        "recommended_build": recommended,
        "isolated_cuda_runtime_installed": cuda_runtime.exists(),
        "restart_required": cuda_runtime.exists() and not torch.cuda.is_available(),
        "confirmation_phrase": CONFIRMATION,
        "official_index_policy": "https://download.pytorch.org/whl/<approved CUDA build>",
    }


def register_runtime_manager(app):
    dashboard_root = Path(__file__).resolve().parent
    jobs = {}
    lock = threading.Lock()

    def update(job_id, **values):
        with lock:
            jobs[job_id].update(values)

    def install(job_id, build):
        staging = dashboard_root / ".runtime-cuda.staging"
        active = dashboard_root / ".runtime-cuda"
        try:
            update(job_id, status="running", stage="preparing isolated runtime", progress=0.05)
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            index_url = f"https://download.pytorch.org/whl/{build}"
            command = [
                sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps", "--no-cache-dir",
                "--target", str(staging), "--index-url", index_url, "torch",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            lines = []
            for line in process.stdout or []:
                clean = line.strip()
                if clean:
                    lines.append(clean[-1000:])
                    update(job_id, stage=clean[-180:], progress=min(0.75, 0.1 + len(lines) * .01), log=lines[-40:])
            if process.wait() != 0:
                raise RuntimeError("PyTorch CUDA wheel installation failed; inspect the installer log")
            update(job_id, stage="verifying CUDA in isolated subprocess", progress=0.82, log=lines[-40:])
            environment = os.environ.copy()
            base_runtime = dashboard_root / ".runtime"
            environment["PYTHONPATH"] = os.pathsep.join([str(staging), str(base_runtime)])
            verification_code = (
                "import json, torch; "
                "assert torch.version.cuda, 'Installed wheel is not CUDA-enabled'; "
                "assert torch.cuda.is_available(), 'CUDA wheel installed but GPU initialization failed'; "
                "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}))"
            )
            verified = subprocess.run(
                [sys.executable, "-c", verification_code], capture_output=True, text=True,
                timeout=120, env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if verified.returncode:
                raise RuntimeError((verified.stderr or verified.stdout or "CUDA verification failed")[-2000:])
            details = json.loads(verified.stdout.strip().splitlines()[-1])
            backup = None
            if active.exists():
                backup = dashboard_root / f".runtime-cuda.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                active.replace(backup)
            staging.replace(active)
            metadata = {
                "installed_at": _utc_now(), "build": build, "index_url": index_url,
                "verification": details, "python": sys.version,
                "backup": backup.name if backup else None,
            }
            (active / "dashboard_cuda_runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            update(job_id, status="complete", stage="installed; restart dashboard to activate", progress=1.0, result=metadata, log=lines[-40:])
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            update(job_id, status="error", stage="failed", error=str(exc), progress=1.0)

    @app.route("/api/runtime/compute", methods=["GET"])
    def runtime_compute_status():
        if not _same_origin_request():
            return jsonify({"success": False, "error": "Compute details require a same-origin localhost request"}), 403
        return jsonify(_public_status(dashboard_root))

    @app.route("/api/runtime/pytorch-cuda/install", methods=["POST"])
    def runtime_install_cuda():
        if not _same_origin_request():
            return jsonify({"success": False, "error": "Runtime changes require a same-origin localhost request"}), 403
        raw = request.get_json(force=True) or {}
        build = str(raw.get("build", ""))
        if build not in CUDA_BUILDS:
            return jsonify({"success": False, "error": "Unsupported CUDA build"}), 400
        if str(raw.get("confirmation", "")) != CONFIRMATION:
            return jsonify({"success": False, "error": f"Type {CONFIRMATION} to confirm"}), 400
        status = _public_status(dashboard_root)
        selected = next(item for item in status["builds"] if item["id"] == build)
        if not status["nvidia"].get("available"):
            return jsonify({"success": False, "error": "No NVIDIA GPU/driver was detected"}), 400
        if not selected["compatible"]:
            return jsonify({"success": False, "error": "The NVIDIA driver does not support this CUDA wheel build"}), 400
        with lock:
            if any(job.get("status") in {"queued", "running"} for job in jobs.values()):
                return jsonify({"success": False, "error": "A CUDA runtime installation is already running"}), 409
            job_id = "cuda_" + uuid.uuid4().hex[:12]
            jobs[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0, "build": build, "created_at": _utc_now(), "log": []}
        threading.Thread(target=install, args=(job_id, build), daemon=True, name="pytorch-cuda-installer").start()
        return jsonify({"success": True, "job_id": job_id})

    @app.route("/api/runtime/pytorch-cuda/install/<job_id>", methods=["GET"])
    def runtime_install_status(job_id):
        if not _same_origin_request():
            return jsonify({"success": False, "error": "Installer logs require a same-origin localhost request"}), 403
        with lock:
            job = jobs.get(job_id)
            if not job:
                return jsonify({"success": False, "error": "Installation job not found"}), 404
            return jsonify({"success": True, "job": dict(job)})
