"""
Flask backend for the VAE Superposition Dashboard.

Wraps the existing PyTorch training code and serves matplotlib plots
as base64 PNGs to the frontend.
"""

import io
import os
import sys
import json
import base64
import traceback
import warnings
warnings.filterwarnings('ignore', r'All-NaN slice encountered')
import math
from datetime import datetime

# Optional binary dependencies can be installed locally beside the dashboard.
# Keeping this directory on sys.path lets the portable launcher use them without
# modifying the user's global Python environment.
for local_packages_name in (".runtime", ".vendor", f".vendor-py{sys.version_info.major}{sys.version_info.minor}"):
    local_packages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_packages_name)
    if os.path.isdir(local_packages_dir) and local_packages_dir not in sys.path:
        sys.path.insert(0, local_packages_dir)

def safe_float(v):
    try:
        f = float(v)
        return -1.0 if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return -1.0
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Add parent dir to path so we can import existing code ─────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_sweep")
os.makedirs(TEMP_DIR, exist_ok=True)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# The Geometry Lab is kept in a separate module so the existing experiment and
# sweep APIs remain backwards compatible.
from geometry_lab import register_geometry_lab
register_geometry_lab(app, EXPERIMENTS_DIR)

# Dedicated regular-VAE orthogonality sweep (linear vs decoder-manifold Gram).
from orthogonality_experiment import register_orthogonality_experiment
register_orthogonality_experiment(app, EXPERIMENTS_DIR)

# Clear temp folder at startup
try:
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
except Exception as e:
    print(f"Error clearing temp folder: {e}")


# ═══════════════════════════════════════════════════════════════
# Model Definitions (mirrored from existing code)
# ═══════════════════════════════════════════════════════════════

def init_weights(parameter, method="xavier_normal"):
    """Applies weight initialization to a parameter matrix of shape (hidden, n_features)."""
    with torch.no_grad():
        if method == "xavier_normal":
            nn.init.xavier_normal_(parameter)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(parameter)
        elif method == "kaiming_normal":
            nn.init.kaiming_normal_(parameter, mode="fan_in", nonlinearity="relu")
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(parameter, mode="fan_in", nonlinearity="relu")
        elif method == "orthogonal":
            nn.init.orthogonal_(parameter)
        elif method == "zeros":
            nn.init.zeros_(parameter)
        elif method == "random_0.05":
            nn.init.normal_(parameter, mean=0.0, std=0.05)
        else:
            nn.init.xavier_normal_(parameter)


class ToyBottleneck(nn.Module):
    """Standard tied-weight bottleneck (Elhage et al. 2022 style)."""
    def __init__(self, n_features=256, hidden=2, init_method="xavier_normal", normalize_weights=False):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.normalize_weights = normalize_weights
        self.W = nn.Parameter(torch.empty(hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        init_weights(self.W, init_method)

    def forward(self, x):
        W = F.normalize(self.W, p=2, dim=0) if self.normalize_weights else self.W
        h = x @ W.t()
        x_hat = F.relu(h @ W + self.b)
        return x_hat


class ToyVAEBottleneck(nn.Module):
    """VAE-style bottleneck with mu & logvar + reparameterization."""
    def __init__(self, n_features=256, hidden=2, init_method="xavier_normal", normalize_weights=False):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.normalize_weights = normalize_weights
        self.W_mu = nn.Parameter(torch.empty(hidden, n_features))
        self.W_logvar = nn.Parameter(torch.empty(hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        init_weights(self.W_mu, init_method)
        init_weights(self.W_logvar, init_method)

    def encode(self, x):
        W_mu = F.normalize(self.W_mu, p=2, dim=0) if self.normalize_weights else self.W_mu
        W_logvar = F.normalize(self.W_logvar, p=2, dim=0) if self.normalize_weights else self.W_logvar
        mu = x @ W_mu.t()
        logvar = torch.clamp(x @ W_logvar.t(), min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        W_mu = F.normalize(self.W_mu, p=2, dim=0) if self.normalize_weights else self.W_mu
        return F.relu(z @ W_mu + self.b)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


# ═══════════════════════════════════════════════════════════════
# Riemannian Manifold Primitives
# ═══════════════════════════════════════════════════════════════

def poincare_conformal_factor(x, c=1.0):
    return 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)

def mobius_add(x, y, c=1.0):
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y_sq) * x + (1.0 - c * x_sq) * y
    denom = 1.0 + 2.0 * c * xy + c * c * x_sq * y_sq
    return num / denom.clamp(min=1e-6)

def poincare_log_map_origin(x, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.arctanh((sqrt_c * x_norm).clamp(max=1.0 - 1e-5)) * (x / x_norm)

def poincare_exp_map_origin(v, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(sqrt_c * v_norm / 2.0) * (v / v_norm) / sqrt_c

def poincare_geodesic_distance(x, y, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.arctanh((sqrt_c * diff_norm).clamp(max=1.0 - 1e-5))

def project_to_poincare_ball(x, c=1.0, eps=1e-4):
    max_norm = 1.0 / (c ** 0.5) - eps
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cond = norms > max_norm
    return torch.where(cond, x * (max_norm / norms), x)

def sphere_conformal_factor(x, c=1.0):
    return 2.0 / (1.0 + c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)

def sphere_exp_map_origin(v, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tan(sqrt_c * v_norm / 2.0) * (v / v_norm) / sqrt_c

def sphere_log_map_origin(x, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.atan(sqrt_c * x_norm) * (x / x_norm)

def sphere_geodesic_distance(x, y, c=1.0):
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    lam_x = sphere_conformal_factor(x, c)
    lam_y = sphere_conformal_factor(y, c)
    diff_norm = (x - y).norm(dim=-1).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.asin(
        (sqrt_c * lam_x.squeeze(-1) * lam_y.squeeze(-1) * diff_norm / 4.0).clamp(max=1.0 - 1e-5))


class ToyManifoldVAEBottleneck(nn.Module):
    """Manifold VAE bottleneck with Riemannian geometry (Poincaré or Sphere)."""
    def __init__(self, n_features=256, hidden=2, curvature=1.0,
                 manifold_type="poincare", init_method="xavier_normal",
                 normalize_weights=False):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.curvature = curvature
        self.manifold_type = manifold_type
        self.normalize_weights = normalize_weights

        self.W_tangent = nn.Parameter(torch.empty(hidden, n_features))
        self.b_enc = nn.Parameter(torch.zeros(hidden))
        self.b_dec = nn.Parameter(torch.zeros(n_features))
        self.W_mu = nn.Parameter(torch.empty(hidden, n_features))
        self.W_logvar = nn.Parameter(torch.empty(hidden, n_features))
        init_weights(self.W_tangent, init_method)
        init_weights(self.W_mu, init_method)
        init_weights(self.W_logvar, init_method)

    def _get_ops(self):
        if self.manifold_type == "sphere":
            return (sphere_log_map_origin, sphere_exp_map_origin,
                    sphere_geodesic_distance, sphere_conformal_factor)
        return (poincare_log_map_origin, poincare_exp_map_origin,
                poincare_geodesic_distance, poincare_conformal_factor)

    def _get_W(self):
        return F.normalize(self.W_tangent, p=2, dim=0) if self.normalize_weights else self.W_tangent

    def encode(self, x):
        _, exp_map, _, _ = self._get_ops()
        c = self.curvature
        W = self._get_W()
        v = x @ W.t() + self.b_enc
        a = F.relu(v)
        mu_tangent = a
        logvar_tangent = torch.clamp(x @ self.W_logvar.t(), min=-10.0, max=10.0)
        mu = exp_map(mu_tangent, c)
        if self.manifold_type == "poincare":
            mu = project_to_poincare_ball(mu, c)
        return mu, logvar_tangent, a

    def reparameterize(self, mu, logvar_tangent):
        _, exp_map, _, _ = self._get_ops()
        c = self.curvature
        std = torch.exp(0.5 * logvar_tangent)
        eps = torch.randn_like(std)
        v = eps * std
        if self.manifold_type == "poincare":
            z_sample = poincare_exp_map_origin(v, c)
            z = mobius_add(mu, z_sample, c)
            z = project_to_poincare_ball(z, c)
        else:
            z_sample = sphere_exp_map_origin(v, c)
            z = z_sample + mu
        return z

    def decode(self, z):
        log_map, _, _, _ = self._get_ops()
        c = self.curvature
        W = self._get_W()
        v_z = log_map(z, c)
        return F.relu(v_z @ W + self.b_dec)

    def forward(self, x):
        mu, logvar, tangent_act = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z, tangent_act


def manifold_tangent_penalty(tangent_activations, c=1.0):
    return (tangent_activations * tangent_activations).sum(dim=-1).mean()

def manifold_riemannian_kl(mu, logvar, c=1.0, manifold_type="poincare"):
    if manifold_type == "poincare":
        lam = poincare_conformal_factor(mu, c)
    else:
        lam = sphere_conformal_factor(mu, c)
    kl_euclidean = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    hidden = mu.shape[-1]
    log_lam = torch.log(lam.squeeze(-1).clamp(min=1e-8))
    return (kl_euclidean + hidden * log_lam).mean()

def manifold_score_matching(z, mu, logvar, c=1.0, manifold_type="poincare"):
    if manifold_type == "poincare":
        lam = poincare_conformal_factor(z, c)
    else:
        lam = sphere_conformal_factor(z, c)
    std = torch.exp(0.5 * logvar)
    score_q = -(z - mu) / (std.pow(2) + 1e-8)
    score_prior = -z
    diff = score_q - score_prior
    riemann_norm_sq = (diff * diff).sum(dim=-1) / (lam.squeeze(-1).pow(2) + 1e-8)
    return riemann_norm_sq.mean()


# ═══════════════════════════════════════════════════════════════
# Data & Training
# ═══════════════════════════════════════════════════════════════

def generate_batch(batch_size, n_features, density, device="cpu"):
    mask = (torch.rand(batch_size, n_features, device=device) < density).float()
    values = torch.rand(batch_size, n_features, device=device)
    return mask * values

def safe_eval_loss(expr, local_vars):
    """Safely evaluate a custom loss expression using a restricted namespace."""
    # Build a restricted dictionary of available functions
    allowed_globals = {
        "__builtins__": {},
        "torch": torch,  # Allow full torch for math (e.g., torch.abs, torch.sum, torch.mean)
    }
    
    # Execute the expression
    try:
        # Evaluate as an expression
        result = eval(expr, allowed_globals, local_vars)
        return result
    except Exception as e:
        raise ValueError(f"Invalid custom loss expression: {str(e)}")


def compute_manifold_metrics(W, b, importance, z_batch):
    """Compute decoder Jacobian, metric tensor, and loss Hessian analytically.

    Because the decoder is  x_hat = ReLU(z @ W + b)  (piecewise linear),
    all quantities have closed-form expressions:

        J(z) = diag(mask(z)) @ Wᵀ          — Jacobian of decoder
        G(z) = W @ diag(mask(z)) @ Wᵀ      — Metric tensor (Riemannian)
        H(z) = 2·W @ diag(imp⊙mask) @ Wᵀ  — Hessian of recon loss w.r.t. z

    Args:
        W:          (hidden, n_features) numpy array
        b:          (n_features,) numpy array
        importance: (n_features,) numpy array
        z_batch:    (batch, hidden) numpy array — latent samples
    Returns:
        dict with matrices (at final step) and scalar metrics.
    """
    hidden, n_features = W.shape
    batch_size = z_batch.shape[0]

    # Pre-activation and ReLU mask for each sample: (batch, n_features)
    pre = z_batch @ W + b[np.newaxis, :]          # (batch, n_features)
    mask = (pre > 0).astype(np.float64)            # binary mask

    # Mean activation fraction
    active_fraction = float(mask.mean())

    # --- Mean Jacobian ---
    # J[b, i, j] = W[j, i] * mask[b, i]
    # Mean over batch: J_mean[i, j] = W[j, i] * mean_mask[i]
    mean_mask = mask.mean(axis=0)                  # (n_features,)
    J_mean = W.T * mean_mask[:, np.newaxis]        # (n_features, hidden) * broadcast
    # Wait — W.T is (n_features, hidden), mean_mask is (n_features,)
    # J_mean[i, j] = W.T[i, j] * mean_mask[i]
    J_mean = (W.T) * mean_mask[:, np.newaxis]      # (n_features, hidden)

    # SVD of mean Jacobian (robust against NaN/Inf and convergence issues)
    if np.any(np.isnan(J_mean)) or np.any(np.isinf(J_mean)):
        S_j = np.zeros(min(n_features, hidden))
    else:
        try:
            U_j, S_j, Vt_j = np.linalg.svd(J_mean, full_matrices=False)
        except np.linalg.LinAlgError:
            S_j = np.zeros(min(n_features, hidden))
    # S_j has shape (min(n_features, hidden),)

    # Condition number
    cond = float(S_j[0] / (S_j[-1] + 1e-12)) if len(S_j) > 0 else 0.0

    # Effective rank from singular-value entropy
    S_norm = S_j / (S_j.sum() + 1e-12)
    S_norm = S_norm[S_norm > 1e-12]
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12))))

    # --- Mean Metric Tensor  G = Jᵀ J = W diag(mask) Wᵀ ---
    # For batch: G[b] = (W * mask[b]) @ W.T
    # Mean:  G_mean = W @ diag(mean_mask) @ W.T
    G_mean = (W * mean_mask[np.newaxis, :]) @ W.T  # (hidden, hidden)
    if np.any(np.isnan(G_mean)) or np.any(np.isinf(G_mean)):
        G_eigvals = np.zeros(hidden)
        det_G = 0.0
    else:
        try:
            G_eigvals = np.linalg.eigvalsh(G_mean)          # sorted ascending
            G_eigvals = G_eigvals[::-1].copy()              # descending
            det_G = float(np.linalg.det(G_mean))
        except np.linalg.LinAlgError:
            G_eigvals = np.zeros(hidden)
            det_G = 0.0

    # --- Mean Loss Hessian  H = 2 W diag(imp ⊙ mask) Wᵀ ---
    imp_mask_mean = importance * mean_mask           # (n_features,)
    H_mean = 2.0 * (W * imp_mask_mean[np.newaxis, :]) @ W.T  # (hidden, hidden)
    if np.any(np.isnan(H_mean)) or np.any(np.isinf(H_mean)):
        H_eigvals = np.zeros(hidden)
    else:
        try:
            H_eigvals = np.linalg.eigvalsh(H_mean)
            H_eigvals = H_eigvals[::-1].copy()
        except np.linalg.LinAlgError:
            H_eigvals = np.zeros(hidden)

    return {
        # Matrices (for heatmap plots)
        "J_mean": J_mean,                        # (n_features, hidden)
        "G_mean": G_mean,                        # (hidden, hidden)
        "H_mean": H_mean,                        # (hidden, hidden)
        # Spectra
        "J_singular_values": S_j,                # (min(n_feat, hidden),)
        "G_eigenvalues": G_eigvals,              # (hidden,)
        "H_eigenvalues": H_eigvals,              # (hidden,)
        # Scalar metrics
        "condition_number": cond,
        "effective_rank": eff_rank,
        "det_G": det_G,
        "trace_H": float(np.trace(H_mean)),
        "active_fraction": active_fraction,
    }


def train_standard_model(density, n_features, hidden, steps, batch_size, lr,
                         importance, device, init_method="xavier_normal",
                         custom_loss_expr=None, normalize_weights=False,
                         snapshot_interval=0):
    model = ToyBottleneck(n_features, hidden, init_method=init_method, normalize_weights=normalize_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    imp_np = importance.cpu().numpy()
    losses = []
    snapshots = []
    manifold_evolution = []  # per-snapshot scalar metrics for line plots

    for step in range(steps):
        x = generate_batch(batch_size, n_features, density, device=device)
        x_hat = model(x)
        
        if custom_loss_expr:
            local_vars = {"x": x, "x_hat": x_hat, "importance": importance, "torch": torch}
            loss = safe_eval_loss(custom_loss_expr, local_vars)
        else:
            loss = (importance * (x - x_hat) ** 2).sum(dim=1).mean()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % max(steps // 10, 1) == 0:
            losses.append(safe_float(loss.item()))
            
        if snapshot_interval > 0 and (step == 0 or (step + 1) % snapshot_interval == 0):
            if model.normalize_weights:
                W_snap = F.normalize(model.W, p=2, dim=0).detach().cpu().numpy()
            else:
                W_snap = model.W.detach().cpu().numpy()

            # Compute z samples by encoding eval data
            with torch.no_grad():
                x_eval = generate_batch(512, n_features, density, device=device)
                W_eff = F.normalize(model.W, p=2, dim=0) if model.normalize_weights else model.W
                z_eval = (x_eval @ W_eff.t()).cpu().numpy()

            b_np = model.b.detach().cpu().numpy()
            m_metrics = compute_manifold_metrics(W_snap, b_np, imp_np, z_eval)

            snap = {"step": step + 1, "W": W_snap}
            snap["manifold"] = m_metrics
            snapshots.append(snap)

            manifold_evolution.append({
                "step": step + 1,
                "condition_number": m_metrics["condition_number"],
                "effective_rank": m_metrics["effective_rank"],
                "det_G": m_metrics["det_G"],
                "trace_H": m_metrics["trace_H"],
                "active_fraction": m_metrics["active_fraction"],
                "J_singular_values": m_metrics["J_singular_values"].tolist(),
                "G_eigenvalues": m_metrics["G_eigenvalues"].tolist(),
                "H_eigenvalues": m_metrics["H_eigenvalues"].tolist(),
            })

    if model.normalize_weights:
        W = F.normalize(model.W, p=2, dim=0).detach().cpu().numpy()
    else:
        W = model.W.detach().cpu().numpy()
        
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    # Final-step manifold metrics (always computed)
    with torch.no_grad():
        x_eval_final = generate_batch(1024, n_features, density, device=device)
        W_eff_final = F.normalize(model.W, p=2, dim=0) if model.normalize_weights else model.W
        z_eval_final = (x_eval_final @ W_eff_final.t()).cpu().numpy()
        z_eval_final = np.nan_to_num(z_eval_final, nan=0.0, posinf=0.0, neginf=0.0)
        
    b_final = model.b.detach().cpu().numpy()
    manifold_final = compute_manifold_metrics(W, b_final, imp_np, z_eval_final)

    return {
        "W": W,
        "losses": losses,
        "final_loss": safe_float(loss.item()),
        "snapshots": snapshots,
        "manifold": manifold_final,
        "manifold_evolution": manifold_evolution,
    }


def train_vae_model(density, n_features, hidden, steps, batch_size, lr,
                    beta, beta_warmup, importance, device, init_method="xavier_normal",
                    custom_loss_expr=None, normalize_weights=False,
                    snapshot_interval=0):
    model = ToyVAEBottleneck(n_features, hidden, init_method=init_method, normalize_weights=normalize_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    imp_np = importance.cpu().numpy()
    losses = []
    snapshots = []
    manifold_evolution = []

    for step in range(steps):
        beta_t = beta * min(1.0, step / max(beta_warmup, 1))
        x = generate_batch(batch_size, n_features, density, device=device)
        x_hat, mu, logvar, z = model(x)

        recon = (importance * (x - x_hat) ** 2).sum(dim=1).mean()
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        if custom_loss_expr:
            local_vars = {"x": x, "x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z,
                          "importance": importance, "beta_t": beta_t, "recon": recon, "kl": kl, "torch": torch}
            loss = safe_eval_loss(custom_loss_expr, local_vars)
        else:
            loss = recon + beta_t * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % max(steps // 10, 1) == 0:
            losses.append({
                "total": safe_float(loss.item()),
                "recon": safe_float(recon.item()),
                "kl": safe_float(kl.item()),
            })
            
        if snapshot_interval > 0 and (step == 0 or (step + 1) % snapshot_interval == 0):
            if model.normalize_weights:
                W_mu_snap = F.normalize(model.W_mu, p=2, dim=0).detach().cpu().numpy()
                W_logvar_snap = F.normalize(model.W_logvar, p=2, dim=0).detach().cpu().numpy()
            else:
                W_mu_snap = model.W_mu.detach().cpu().numpy()
                W_logvar_snap = model.W_logvar.detach().cpu().numpy()
            
            with torch.no_grad():
                x_eval_snap = generate_batch(1024, n_features, density, device=device)
                _, mu_eval_snap, logvar_eval_snap, z_eval_snap = model(x_eval_snap)

            b_np = model.b.detach().cpu().numpy()
            z_np = z_eval_snap.cpu().numpy()
            m_metrics = compute_manifold_metrics(W_mu_snap, b_np, imp_np, z_np)

            snapshots.append({
                "step": step + 1,
                "W_mu": W_mu_snap,
                "W_logvar": W_logvar_snap,
                "z": z_np,
                "manifold": m_metrics,
            })

            manifold_evolution.append({
                "step": step + 1,
                "condition_number": m_metrics["condition_number"],
                "effective_rank": m_metrics["effective_rank"],
                "det_G": m_metrics["det_G"],
                "trace_H": m_metrics["trace_H"],
                "active_fraction": m_metrics["active_fraction"],
                "J_singular_values": m_metrics["J_singular_values"].tolist(),
                "G_eigenvalues": m_metrics["G_eigenvalues"].tolist(),
                "H_eigenvalues": m_metrics["H_eigenvalues"].tolist(),
            })

    # Collect z samples
    with torch.no_grad():
        x_eval = generate_batch(2048, n_features, density, device=device)
        _, mu_eval, logvar_eval, z_eval = model(x_eval)

    if model.normalize_weights:
        W_mu = F.normalize(model.W_mu, p=2, dim=0).detach().cpu().numpy()
        W_logvar = F.normalize(model.W_logvar, p=2, dim=0).detach().cpu().numpy()
    else:
        W_mu = model.W_mu.detach().cpu().numpy()
        W_logvar = model.W_logvar.detach().cpu().numpy()
        
    W_mu = np.nan_to_num(W_mu, nan=0.0, posinf=0.0, neginf=0.0)
    W_logvar = np.nan_to_num(W_logvar, nan=0.0, posinf=0.0, neginf=0.0)

    # Final-step manifold metrics (always computed, using W_mu)
    b_final = model.b.detach().cpu().numpy()
    z_final = z_eval.cpu().numpy()
    z_final = np.nan_to_num(z_final, nan=0.0, posinf=0.0, neginf=0.0)
    manifold_final = compute_manifold_metrics(W_mu, b_final, imp_np, z_final)

    return {
        "W_mu": W_mu,
        "W_logvar": W_logvar,
        "z": z_final,
        "mu": mu_eval.cpu().numpy(),
        "losses": losses,
        "final_loss": safe_float(loss.item()),
        "final_recon": safe_float(recon.item()),
        "final_kl": safe_float(kl.item()),
        "snapshots": snapshots,
        "manifold": manifold_final,
        "manifold_evolution": manifold_evolution,
    }


def train_manifold_vae_model(density, n_features, hidden, steps, batch_size, lr,
                              beta, beta_warmup, importance, device,
                              curvature=1.0, manifold_type="poincare",
                              manifold_loss_type="riemannian_kl",
                              tangent_weight=0.01,
                              init_method="xavier_normal",
                              custom_loss_expr=None, normalize_weights=False,
                              snapshot_interval=0):
    """Train a manifold VAE bottleneck and return results."""
    model = ToyManifoldVAEBottleneck(
        n_features, hidden, curvature=curvature,
        manifold_type=manifold_type, init_method=init_method,
        normalize_weights=normalize_weights
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    imp_np = importance.cpu().numpy()
    c = curvature
    losses = []
    snapshots = []
    manifold_evolution = []

    for step in range(steps):
        beta_t = beta * min(1.0, step / max(beta_warmup, 1))
        x = generate_batch(batch_size, n_features, density, device=device)
        x_hat, mu, logvar, z, tangent_act = model(x)

        recon = (importance * (x - x_hat) ** 2).sum(dim=1).mean()

        if custom_loss_expr:
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            tangent_pen = manifold_tangent_penalty(tangent_act, c)
            score_loss = manifold_score_matching(z, mu, logvar, c, manifold_type)
            local_vars = {
                "x": x, "x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z,
                "importance": importance, "beta_t": beta_t, "recon": recon,
                "kl": kl, "tangent_pen": tangent_pen, "score_loss": score_loss,
                "torch": torch
            }
            loss = safe_eval_loss(custom_loss_expr, local_vars)
        elif manifold_loss_type == "tangent_penalty":
            tangent_pen = manifold_tangent_penalty(tangent_act, c)
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl + tangent_weight * tangent_pen
        elif manifold_loss_type == "geodesic":
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl
        elif manifold_loss_type == "riemannian_kl":
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl
        elif manifold_loss_type == "score_matching":
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            score_loss = manifold_score_matching(z, mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl + 0.1 * score_loss
        else:
            kl = manifold_riemannian_kl(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if (step + 1) % max(steps // 10, 1) == 0:
            losses.append({
                "total": safe_float(loss.item()),
                "recon": safe_float(recon.item()),
                "kl": safe_float(kl.item()) if 'kl' in dir() else 0.0,
            })

        if snapshot_interval > 0 and (step == 0 or (step + 1) % snapshot_interval == 0):
            if model.normalize_weights:
                W_t_snap = F.normalize(model.W_tangent, p=2, dim=0).detach().cpu().numpy()
            else:
                W_t_snap = model.W_tangent.detach().cpu().numpy()
            W_mu_snap = model.W_mu.detach().cpu().numpy()
            W_lv_snap = model.W_logvar.detach().cpu().numpy()

            with torch.no_grad():
                x_eval_snap = generate_batch(1024, n_features, density, device=device)
                _, mu_snap, lv_snap, z_snap, tang_snap = model(x_eval_snap)

            b_np = model.b_dec.detach().cpu().numpy()
            z_np = z_snap.cpu().numpy()
            m_metrics = compute_manifold_metrics(W_t_snap, b_np, imp_np, z_np)

            snapshots.append({
                "step": step + 1,
                "W_tangent": W_t_snap,
                "W_mu": W_mu_snap,
                "W_logvar": W_lv_snap,
                "z": z_np,
                "manifold": m_metrics,
            })

            manifold_evolution.append({
                "step": step + 1,
                "condition_number": m_metrics["condition_number"],
                "effective_rank": m_metrics["effective_rank"],
                "det_G": m_metrics["det_G"],
                "trace_H": m_metrics["trace_H"],
                "active_fraction": m_metrics["active_fraction"],
                "J_singular_values": m_metrics["J_singular_values"].tolist(),
                "G_eigenvalues": m_metrics["G_eigenvalues"].tolist(),
                "H_eigenvalues": m_metrics["H_eigenvalues"].tolist(),
            })

    # Collect z samples
    with torch.no_grad():
        x_eval = generate_batch(2048, n_features, density, device=device)
        _, mu_eval, logvar_eval, z_eval, tang_eval = model(x_eval)

    if model.normalize_weights:
        W_tangent = F.normalize(model.W_tangent, p=2, dim=0).detach().cpu().numpy()
    else:
        W_tangent = model.W_tangent.detach().cpu().numpy()
    W_mu = model.W_mu.detach().cpu().numpy()
    W_logvar = model.W_logvar.detach().cpu().numpy()
    
    W_tangent = np.nan_to_num(W_tangent, nan=0.0, posinf=0.0, neginf=0.0)
    W_mu = np.nan_to_num(W_mu, nan=0.0, posinf=0.0, neginf=0.0)
    W_logvar = np.nan_to_num(W_logvar, nan=0.0, posinf=0.0, neginf=0.0)

    # Final manifold metrics
    b_final = model.b_dec.detach().cpu().numpy()
    z_final = z_eval.cpu().numpy()
    z_final = np.nan_to_num(z_final, nan=0.0, posinf=0.0, neginf=0.0)
    manifold_final = compute_manifold_metrics(W_tangent, b_final, imp_np, z_final)

    return {
        "W_tangent": W_tangent,
        "W_mu": W_mu,
        "W_logvar": W_logvar,
        "z": z_final,
        "mu": mu_eval.cpu().numpy(),
        "tangent_activations": tang_eval.cpu().numpy(),
        "losses": losses,
        "final_loss": safe_float(loss.item()),
        "final_recon": safe_float(recon.item()),
        "final_kl": safe_float(kl.item()) if 'kl' in dir() else 0.0,
        "snapshots": snapshots,
        "manifold": manifold_final,
        "manifold_evolution": manifold_evolution,
        "curvature": curvature,
        "manifold_type": manifold_type,
    }


# ═══════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════

def fig_to_base64(fig, dpi=150):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def plot_arrows(ax, W, cmap, max_norm=None, label=None):
    """Plot columns of W (hidden×N) as arrows. Works for 2D hidden only."""
    weights = W.T  # (N, 2)
    norms = np.linalg.norm(weights, axis=1)
    if max_norm is None:
        max_norm = norms.max() if norms.max() > 1e-8 else 0.1
    norm_n = norms / (max_norm + 1e-8)
    order = np.argsort(norms)
    for i in order:
        c = cmap(np.clip(norm_n[i], 0, 1))
        ax.annotate("", xy=(weights[i, 0], weights[i, 1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, alpha=0.7))
    if label:
        ax.plot([], [], color=cmap(0.7), lw=2, label=label)


def style_ax_dark(ax):
    """Apply dark theme to axes."""
    ax.set_facecolor("#0c0c1e")
    ax.tick_params(colors="#9892b3", labelsize=8)
    ax.spines["bottom"].set_color("#2a2554")
    ax.spines["top"].set_color("#2a2554")
    ax.spines["left"].set_color("#2a2554")
    ax.spines["right"].set_color("#2a2554")
    ax.xaxis.label.set_color("#9892b3")
    ax.yaxis.label.set_color("#9892b3")
    ax.title.set_color("#e8e6f0")


def create_arrow_plot(W, title, cmap_name="plasma"):
    """Create a single arrow plot for a weight matrix (hidden=2 x N)."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#06060f")
    style_ax_dark(ax)

    cmap = getattr(plt.cm, cmap_name, plt.cm.plasma)
    plot_arrows(ax, W, cmap, label=title)

    weights = W.T
    norms = np.linalg.norm(weights, axis=1)
    max_norm = norms.max() if norms.max() > 1e-6 else 0.1
    lim = max_norm * 1.25

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.3, lw=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)

    n_active = int((norms > 0.05 * norms.max()).sum())
    ax.set_title(f"{title}\n({n_active} active, max ‖w‖ = {norms.max():.4f})",
                 fontsize=11, pad=12)
    ax.set_xlabel("Hidden dim 1", fontsize=10)
    ax.set_ylabel("Hidden dim 2", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap,
        norm=mcolors.Normalize(vmin=norms.min(), vmax=norms.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight-vector norm", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)

    fig.tight_layout()
    return fig


def create_z_scatter(z, title="z samples"):
    """Scatter plot of z samples (N x 2)."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#06060f")
    style_ax_dark(ax)

    try:
        z_abs_max = np.nanmax(np.abs(z))
        if np.isnan(z_abs_max) or np.isinf(z_abs_max):
            z_abs_max = 0.0
    except ValueError:
        z_abs_max = 0.0
    z_max = max(z_abs_max * 1.1, 0.5) if len(z) > 0 else 1.0
    ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.35, c="#a78bfa", edgecolors="none")
    ax.set_xlim(-z_max, z_max)
    ax.set_ylim(-z_max, z_max)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)
    ax.set_title(f"{title}\n(z range = ±{z_max:.2f})", fontsize=11, pad=12)
    ax.set_xlabel("z₁", fontsize=10)
    ax.set_ylabel("z₂", fontsize=10)
    fig.tight_layout()
    return fig


def create_combined_plot(result, density):
    """Combined overlay: W_mu arrows + W_logvar arrows + z scatter."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#06060f")
    style_ax_dark(ax)

    z = result["z"]
    W_mu = result["W_mu"]
    W_lv = result["W_logvar"]

    ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.12, c="#6b7280", edgecolors="none",
               label="z samples", zorder=1)
    plot_arrows(ax, W_mu, plt.cm.plasma, label="W_mu")
    plot_arrows(ax, W_lv, plt.cm.viridis, label="W_logvar")

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.2, lw=0.7)

    def panel_lim(W):
        norms = np.linalg.norm(W, axis=0)
        m = norms.max() if norms.max() > 1e-6 else 0.1
        return m * 1.25

    w_max = max(panel_lim(W_mu), panel_lim(W_lv))
    try:
        z_abs_max = np.nanmax(np.abs(z))
        if np.isnan(z_abs_max) or np.isinf(z_abs_max):
            z_abs_max = 0.0
    except ValueError:
        z_abs_max = 0.0
    z_max = z_abs_max * 1.1 if len(z) > 0 else 0.5
    
    combined_lim = max(w_max, z_max, 1.15)
    if np.isnan(combined_lim) or np.isinf(combined_lim):
        combined_lim = 1.15
    ax.set_xlim(-combined_lim, combined_lim)
    ax.set_ylim(-combined_lim, combined_lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)
    ax.set_title(f"Combined — Density = {density:.0%}", fontsize=11, pad=12)
    ax.legend(fontsize=8, loc="upper right",
              facecolor="#0c0c1e", edgecolor="#2a2554", labelcolor="#e8e6f0")
    fig.tight_layout()
    return fig


def create_interference_plot(W, title="W"):
    """WᵀW interference matrix heatmap."""
    WtW = W.T @ W
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#06060f")

    for ax in axes:
        style_ax_dark(ax)

    # Full matrix
    vmax = np.abs(WtW).max()
    im = axes[0].imshow(WtW, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title(f"{title}ᵀ{title} ({WtW.shape[0]}×{WtW.shape[1]})", fontsize=11)
    axes[0].set_xlabel("Feature j", fontsize=9)
    axes[0].set_ylabel("Feature i", fontsize=9)
    cbar1 = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#9892b3", labelsize=7)

    # Off-diagonal
    WtW_off = WtW.copy()
    np.fill_diagonal(WtW_off, 0)
    vmax2 = np.abs(WtW_off).max() if np.abs(WtW_off).max() > 1e-8 else 1.0
    im2 = axes[1].imshow(WtW_off, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    axes[1].set_title("Off-diagonal (interference)", fontsize=11)
    axes[1].set_xlabel("Feature j", fontsize=9)
    axes[1].set_ylabel("Feature i", fontsize=9)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#9892b3", labelsize=7)

    fig.tight_layout()
    return fig, WtW


def create_norms_plot(W, title="W"):
    """Per-feature weight norms bar chart + histogram."""
    norms = np.linalg.norm(W, axis=0)
    n = len(norms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#06060f")
    for ax in axes:
        style_ax_dark(ax)

    # Sorted bar chart
    sorted_idx = np.argsort(norms)[::-1]
    colors = plt.cm.plasma(norms[sorted_idx] / (norms.max() + 1e-8))
    axes[0].bar(range(n), norms[sorted_idx], color=colors, width=1.0, edgecolor="none")
    axes[0].set_xlabel("Feature (sorted by norm)", fontsize=9)
    axes[0].set_ylabel("‖wᵢ‖", fontsize=9)
    axes[0].set_title(f"{title} norms (sorted)", fontsize=11)
    axes[0].axhline(1.0, color="#5e577a", ls="--", lw=0.6, alpha=0.5)

    # Histogram
    try:
        axes[1].hist(norms, bins=40, color="#7c3aed", edgecolor="#0c0c1e", alpha=0.85)
    except ValueError:
        # Avoid "Too many bins for data range" error for identical norms or tiny variance
        bins = np.linspace(float(norms.min()) - 0.1, float(norms.max()) + 0.1, 10)
        axes[1].hist(norms, bins=bins, color="#7c3aed", edgecolor="#0c0c1e", alpha=0.85)
    axes[1].set_xlabel("‖wᵢ‖", fontsize=9)
    axes[1].set_ylabel("Count", fontsize=9)
    axes[1].set_title("Norm distribution", fontsize=11)
    axes[1].axvline(1.0, color="#5e577a", ls="--", lw=0.6, alpha=0.5)

    fig.tight_layout()
    return fig, norms


def create_heatmap(W, title="Weight Matrix"):
    """For hidden > 2: show the raw weight matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#06060f")
    style_ax_dark(ax)

    vmax = np.abs(W).max()
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title} ({W.shape[0]}×{W.shape[1]})", fontsize=11, pad=12)
    ax.set_xlabel("Feature index", fontsize=9)
    ax.set_ylabel("Hidden dimension", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#9892b3", labelsize=7)
    fig.tight_layout()
    return fig


def create_pca_projection(W, title="PCA Projection"):
    """For hidden > 2: project weight columns onto top-2 PCs and plot as arrows."""
    # W is (hidden, n_features). Columns are feature directions.
    # Compute PCA of W.T (n_features x hidden) → keep 2 components
    cols = W.T  # (n_features, hidden)
    mean = cols.mean(axis=0)
    centered = cols - mean
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto first 2 PCs
    proj = centered @ Vt[:2].T  # (n_features, min(2, K))
    if proj.shape[1] < 2:
        proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])))
        
    # Create a (2, n_features) matrix for plot_arrows
    W_proj = proj.T

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#06060f")
    style_ax_dark(ax)

    plot_arrows(ax, W_proj, plt.cm.plasma, label=title)

    norms = np.linalg.norm(proj, axis=1)
    max_norm = norms.max() if norms.max() > 1e-6 else 0.1
    lim = max_norm * 1.25

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.3, lw=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)

    S_sum = (S ** 2).sum()
    if S_sum < 1e-12:
        explained = [0.0, 0.0]
    else:
        explained = (S[:2] ** 2 / S_sum * 100).tolist()
        while len(explained) < 2:
            explained.append(0.0)
            
    n_active = int((norms > 0.05 * max_norm).sum())
    ax.set_title(f"{title}\n({n_active} active, PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)",
                 fontsize=11, pad=12)
    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
        norm=mcolors.Normalize(vmin=norms.min(), vmax=norms.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Projected norm", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)

    fig.tight_layout()
    return fig


def create_graph_plot(W, title="Weight Graph"):
    """Node-link diagram representing feature directions and their absolute dot products."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#06060f")
    style_ax_dark(ax)

    # Calculate absolute dot product matrix
    WtW = np.abs(W.T @ W)
    np.fill_diagonal(WtW, 0)
    norms = np.linalg.norm(W, axis=0)
    max_norm = norms.max() if norms.max() > 1e-8 else 0.1
    
    # Thresholds
    active_thresh = 0.05 * max_norm
    max_dot = WtW.max() if WtW.max() > 1e-8 else 1.0
    edge_thresh = 0.1 * max_dot

    G = nx.Graph()
    n_features = W.shape[1]
    
    # Add active nodes
    active_nodes = []
    for i in range(n_features):
        if norms[i] > active_thresh:
            G.add_node(i, norm=norms[i])
            active_nodes.append(i)
            
    # Add edges
    for i in range(len(active_nodes)):
        for j in range(i + 1, len(active_nodes)):
            u = active_nodes[i]
            v = active_nodes[j]
            weight = WtW[u, v]
            if weight > edge_thresh:
                G.add_edge(u, v, weight=weight)
                
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No active features to display", 
                ha='center', va='center', color="#9892b3", transform=ax.transAxes)
        ax.set_title(f"{title} (Graph)", fontsize=11, pad=12)
        fig.tight_layout()
        return fig

    # Draw graph
    pos = nx.spring_layout(G, k=1.0, seed=42)
    
    node_norms = [G.nodes[n]['norm'] for n in G.nodes()]
    node_colors = plt.cm.plasma(np.array(node_norms) / max_norm)
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, 
                           node_color=node_colors, edgecolors="#06060f", linewidths=0.5)
    
    # Edges
    if G.number_of_edges() > 0:
        edges = G.edges(data=True)
        edge_weights = np.array([d['weight'] for u, v, d in edges])
        edge_colors = plt.cm.plasma(edge_weights / max_dot)
        edge_widths = 0.5 + 2.5 * (edge_weights / max_dot)
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                               edge_color=edge_colors, alpha=0.6)
                               
    ax.set_title(f"{title} (Graph)\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", 
                 fontsize=11, pad=12)
                 
    # Add colorbar for dot product strength
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                               norm=mcolors.Normalize(vmin=0, vmax=max_dot))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dot Product Strength", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)
    
    # Hide axis markers but keep background
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_coord_graph_plot(W, title="Weight Coord Graph"):
    """Node-link diagram using actual weight coordinates as node positions.

    For hidden=2 the (x, y) of each node is the raw weight vector.
    For hidden>2 the weight columns are projected onto their top-2 PCs.
    Edge intensity encodes the absolute dot product between feature directions.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#06060f")
    style_ax_dark(ax)

    n_features = W.shape[1]
    hidden = W.shape[0]

    # --- Compute 2-D positions for every feature ---
    if hidden == 2:
        coords = W.T  # (n_features, 2)
    else:
        cols = W.T  # (n_features, hidden)
        mean = cols.mean(axis=0)
        centered = cols - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ Vt[:2].T  # (n_features, 2)
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))

    # --- Dot-product / interference matrix ---
    WtW = np.abs(W.T @ W)  # (n_features, n_features)
    np.fill_diagonal(WtW, 0)
    norms = np.linalg.norm(W, axis=0)
    max_norm = norms.max() if norms.max() > 1e-8 else 0.1

    # Thresholds
    active_thresh = 0.05 * max_norm
    max_dot = WtW.max() if WtW.max() > 1e-8 else 1.0
    edge_thresh = 0.1 * max_dot

    # --- Build graph ---
    G = nx.Graph()
    active_nodes = []
    for i in range(n_features):
        if norms[i] > active_thresh:
            G.add_node(i, norm=norms[i])
            active_nodes.append(i)

    for i in range(len(active_nodes)):
        for j in range(i + 1, len(active_nodes)):
            u = active_nodes[i]
            v = active_nodes[j]
            weight = WtW[u, v]
            if weight > edge_thresh:
                G.add_edge(u, v, weight=weight)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No active features to display",
                ha='center', va='center', color="#9892b3", transform=ax.transAxes)
        ax.set_title(f"{title} (Coord Graph)", fontsize=11, pad=12)
        fig.tight_layout()
        return fig

    # --- Position dict from actual coordinates ---
    pos = {i: (coords[i, 0], coords[i, 1]) for i in G.nodes()}

    node_norms = [G.nodes[n]['norm'] for n in G.nodes()]
    node_colors = plt.cm.plasma(np.array(node_norms) / max_norm)

    # Edges
    if G.number_of_edges() > 0:
        edges = G.edges(data=True)
        edge_weights = np.array([d['weight'] for u, v, d in edges])
        edge_colors = plt.cm.plasma(edge_weights / max_dot)
        edge_widths = 0.5 + 2.5 * (edge_weights / max_dot)

        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color=edge_colors, alpha=0.5)

    # Nodes (drawn after edges so they sit on top)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60,
                           node_color=node_colors, edgecolors="#06060f", linewidths=0.5)

    # --- Axes & styling ---
    all_x = [coords[i, 0] for i in G.nodes()]
    all_y = [coords[i, 1] for i in G.nodes()]
    pad = max(max(np.abs(all_x)), max(np.abs(all_y)), 0.1) * 1.25
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)

    # Reference unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.2, lw=0.7)

    dim_label = "Weight" if hidden == 2 else "PC"
    ax.set_xlabel(f"{dim_label} dim 1", fontsize=10)
    ax.set_ylabel(f"{dim_label} dim 2", fontsize=10)
    ax.set_title(f"{title} (Coord Graph)\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                 fontsize=11, pad=12)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=mcolors.Normalize(vmin=0, vmax=max_dot))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dot Product Strength", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)

    fig.tight_layout()
    return fig


def create_jacobian_plot(result, title="Jacobian"):
    """Two-panel plot: Jacobian heatmap at final step, and singular value evolution."""
    manifold = result.get("manifold")
    evolution = result.get("manifold_evolution")
    if not manifold:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#06060f")
        style_ax_dark(ax)
        ax.text(0.5, 0.5, "No manifold data available", ha='center', va='center', color='red')
        return fig

    J_mean = manifold["J_mean"]
    n_features, hidden = J_mean.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#06060f")
    
    # Left: Jacobian Heatmap
    ax = axes[0]
    style_ax_dark(ax)
    vmax = np.abs(J_mean).max() if np.abs(J_mean).max() > 1e-8 else 1.0
    im = ax.imshow(J_mean, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f"Mean Jacobian (Features × Latent)", fontsize=11, pad=12)
    ax.set_xlabel("Latent dimension (hidden)", fontsize=10)
    ax.set_ylabel("Feature dimension (n_features)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)

    # Right: SV Evolution or Bar chart
    ax = axes[1]
    style_ax_dark(ax)
    if evolution and len(evolution) > 0:
        steps = [e["step"] for e in evolution]
        svs = np.array([e["J_singular_values"] for e in evolution])  # (steps, min(n_feat, hidden))
        for i in range(svs.shape[1]):
            ax.plot(steps, svs[:, i], lw=2, label=rf"$\sigma_{{{i+1}}}$")
        ax.set_title("Jacobian Singular Value Evolution", fontsize=11, pad=12)
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel("Singular Value", fontsize=10)
        ax.legend(facecolor="#100f1a", edgecolor="#242136", labelcolor="#9892b3")
    else:
        svs = manifold["J_singular_values"]
        ax.bar(np.arange(len(svs)), svs, color="#ea9a97")
        ax.set_title("Jacobian Singular Values", fontsize=11, pad=12)
        ax.set_xlabel("Index", fontsize=10)
        ax.set_xticks(np.arange(len(svs)))

    fig.suptitle(title, color="#e0def4", fontsize=14, y=1.02)
    fig.tight_layout()
    # Add interpretive text at bottom
    fig.text(0.5, -0.05, "Jacobian shows how each feature responds to perturbations in each latent direction.\nLarge singular values = high magnification in that direction.", 
             ha='center', color="#9892b3", fontsize=10)
    return fig


def create_geometry_plot(result, title="Geometry"):
    """Four-panel plot: Metric tensor (G) and Loss Hessian (H) heatmaps + eigenvalue evolution."""
    manifold = result.get("manifold")
    evolution = result.get("manifold_evolution")
    if not manifold:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#06060f")
        style_ax_dark(ax)
        ax.text(0.5, 0.5, "No manifold data available", ha='center', va='center', color='red')
        return fig

    G = manifold["G_mean"]
    H = manifold["H_mean"]
    hidden = G.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#06060f")
    
    # --- Top Left: G Heatmap ---
    ax = axes[0, 0]
    style_ax_dark(ax)
    vmax = np.abs(G).max() if np.abs(G).max() > 1e-8 else 1.0
    im = ax.imshow(G, cmap='viridis', vmin=0, vmax=vmax)
    for i in range(hidden):
        for j in range(hidden):
            ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", color="white" if G[i,j] < vmax/2 else "black")
    ax.set_title("Metric Tensor G = JᵀJ", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Top Right: G Eigenvalues ---
    ax = axes[0, 1]
    style_ax_dark(ax)
    if evolution and len(evolution) > 0:
        steps = [e["step"] for e in evolution]
        eigs = np.array([e["G_eigenvalues"] for e in evolution])
        for i in range(eigs.shape[1]):
            ax.plot(steps, eigs[:, i], lw=2, label=rf"$\lambda_{{{i+1}}}$")
        ax.set_title("G Eigenvalue Evolution", fontsize=11)
        ax.legend(facecolor="#100f1a", edgecolor="#242136", labelcolor="#9892b3")
    else:
        eigs = manifold["G_eigenvalues"]
        ax.bar(np.arange(len(eigs)), eigs, color="#3e8fb0")
        ax.set_title("G Eigenvalues", fontsize=11)

    # --- Bottom Left: H Heatmap ---
    ax = axes[1, 0]
    style_ax_dark(ax)
    vmax_h = np.abs(H).max() if np.abs(H).max() > 1e-8 else 1.0
    im2 = ax.imshow(H, cmap='RdBu_r', vmin=-vmax_h, vmax=vmax_h)
    for i in range(hidden):
        for j in range(hidden):
            ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", color="white" if abs(H[i,j]) > vmax_h/2 else "black")
    ax.set_title("Loss Hessian H", fontsize=11)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # --- Bottom Right: H Eigenvalues ---
    ax = axes[1, 1]
    style_ax_dark(ax)
    ax.axhline(0, color="#eb6f92", ls="--", lw=1)
    if evolution and len(evolution) > 0:
        steps = [e["step"] for e in evolution]
        eigs_h = np.array([e["H_eigenvalues"] for e in evolution])
        for i in range(eigs_h.shape[1]):
            ax.plot(steps, eigs_h[:, i], lw=2, label=rf"$\lambda_{{{i+1}}}$")
        ax.set_title("H Hessian Eigenvalue Evolution", fontsize=11)
        ax.legend(facecolor="#100f1a", edgecolor="#242136", labelcolor="#9892b3")
    else:
        eigs_h = manifold["H_eigenvalues"]
        ax.bar(np.arange(len(eigs_h)), eigs_h, color="#9ccfd8")
        ax.set_title("H Eigenvalues", fontsize=11)

    fig.suptitle(title, color="#e0def4", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.text(0.5, -0.05, "G (Metric Tensor): local distances in latent space. Equal eigenvalues = isotropic mapping.\nH (Hessian): curvature of loss landscape. Positive eigenvalues = local minimum.", 
             ha='center', color="#9892b3", fontsize=10)
    return fig


def create_manifold_summary_plot(result, title="Manifold Summary"):
    """Four-panel plot tracking condition number, effective rank, det(G), and interpretive text."""
    manifold = result.get("manifold")
    evolution = result.get("manifold_evolution")
    if not manifold:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#06060f")
        style_ax_dark(ax)
        ax.text(0.5, 0.5, "No manifold data available", ha='center', va='center', color='red')
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#06060f")
    
    if evolution and len(evolution) > 0:
        steps = [e["step"] for e in evolution]
        
        # Condition number
        ax = axes[0, 0]
        style_ax_dark(ax)
        cond = [e["condition_number"] for e in evolution]
        ax.plot(steps, cond, color="#ea9a97", lw=2)
        ax.set_yscale('log')
        ax.set_title("Condition Number (Log Scale)", fontsize=11)
        
        # Effective rank
        ax = axes[0, 1]
        style_ax_dark(ax)
        erank = [e["effective_rank"] for e in evolution]
        ax.plot(steps, erank, color="#c4a7e7", lw=2)
        ax.set_title("Effective Rank", fontsize=11)
        
        # det(G)
        ax = axes[1, 0]
        style_ax_dark(ax)
        det = [e["det_G"] for e in evolution]
        ax.plot(steps, det, color="#f6c177", lw=2)
        ax.set_title("det(G) [Area Magnification]", fontsize=11)
    else:
        # Static fallback
        axes[0,0].text(0.5, 0.5, f"Cond: {manifold['condition_number']:.2f}", ha='center', color='white', transform=axes[0,0].transAxes)
        axes[0,1].text(0.5, 0.5, f"Eff Rank: {manifold['effective_rank']:.2f}", ha='center', color='white', transform=axes[0,1].transAxes)
        axes[1,0].text(0.5, 0.5, f"det(G): {manifold['det_G']:.4f}", ha='center', color='white', transform=axes[1,0].transAxes)
        for ax in [axes[0,0], axes[0,1], axes[1,0]]: style_ax_dark(ax)

    # Text summary panel
    ax = axes[1, 1]
    ax.axis('off')
    
    c = manifold["condition_number"]
    cond_str = "well-conditioned" if c < 10 else ("moderately conditioned" if c < 100 else "ill-conditioned")
    
    h_eigs = manifold["H_eigenvalues"]
    pos = sum(h_eigs > 1e-6)
    neg = sum(h_eigs < -1e-6)
    if neg == 0:
        h_str = "Locally convex (minimum)"
    elif pos == 0:
        h_str = "Locally concave (maximum)"
    else:
        h_str = f"Saddle region ({pos} pos, {neg} neg)"

    text = f"""Final Manifold Metrics:

• Condition Number: {c:.2f} ({cond_str})
  (Ratio of max to min magnification)

• Effective Rank: {manifold["effective_rank"]:.2f}
  (Dimensionality effectively used)

• det(G): {manifold["det_G"]:.4f}
  (Local area magnification factor)

• Active ReLU Fraction: {manifold["active_fraction"]:.1%}
  (Mean % of active features per sample)

• Loss Hessian: {h_str}
  (Trace / Mean Curvature: {manifold["trace_H"]:.4f})
"""
    ax.text(0.1, 0.5, text, va='center', ha='left', color="#e0def4", fontsize=12,
            bbox=dict(facecolor='#191724', edgecolor='#26233a', boxstyle='round,pad=1'))

    fig.suptitle(title, color="#e0def4", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def create_manifold_disk_scatter(z, title="Manifold Samples", manifold_type="poincare", curvature=1.0):
    """Scatter plot of z samples on the Poincaré disk or stereographic sphere."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#06060f")
    style_ax_dark(ax)

    c = curvature
    theta = np.linspace(0, 2 * np.pi, 200)

    if manifold_type == "poincare":
        r = 1.0 / (c ** 0.5)
        ax.fill(r * np.cos(theta), r * np.sin(theta), color="#0d0820", alpha=0.5)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="#a78bfa", alpha=0.4, lw=1.5, label=f"Ball boundary (r={r:.2f})")
        # Geodesic grid circles
        for d_geo in [0.5, 1.0, 1.5, 2.0, 3.0]:
            r_eucl = np.tanh(d_geo * c**0.5 / 2.0) / c**0.5
            if r_eucl < r:
                ax.plot(r_eucl * np.cos(theta), r_eucl * np.sin(theta),
                        color="#5e577a", ls=":", lw=0.5, alpha=0.3)
    else:
        ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.3, lw=1.0)

    ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.35, c="#a78bfa", edgecolors="none", zorder=3)

    try:
        z_abs_max = np.nanmax(np.abs(z))
        if np.isnan(z_abs_max) or np.isinf(z_abs_max):
            z_abs_max = 0.0
    except ValueError:
        z_abs_max = 0.0
        
    z_max = max(z_abs_max * 1.15, 0.5)
    if manifold_type == "poincare":
        z_max = max(z_max, 1.15 / c**0.5)
    
    # Final check just to be absolutely safe
    if np.isnan(z_max) or np.isinf(z_max):
        z_max = 1.0
        
    ax.set_xlim(-z_max, z_max)
    ax.set_ylim(-z_max, z_max)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)
    disk_label = "Poincaré Disk" if manifold_type == "poincare" else "Stereographic Sphere"
    ax.set_title(f"{title}\n({disk_label}, c={c:.2f})", fontsize=11, pad=12)
    ax.set_xlabel("z₁", fontsize=10)
    ax.set_ylabel("z₂", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", facecolor="#0c0c1e", edgecolor="#2a2554", labelcolor="#e8e6f0")
    fig.tight_layout()
    return fig


def create_geodesic_distance_matrix(z, title="Geodesic Distance", manifold_type="poincare", curvature=1.0, n_samples=100):
    """Pairwise geodesic distance heatmap for a subset of z samples."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#06060f")
    style_ax_dark(ax)

    c = curvature
    # Subsample for speed
    n = min(n_samples, len(z))
    z_sub = z[:n]

    z_t = torch.tensor(z_sub, dtype=torch.float32)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        zi = z_t[i:i+1].expand(n, -1)
        if manifold_type == "poincare":
            dists = poincare_geodesic_distance(zi, z_t, c)
        else:
            dists = sphere_geodesic_distance(zi, z_t, c)
        dist_matrix[i] = dists.numpy()

    im = ax.imshow(dist_matrix, cmap="magma", aspect="auto")
    ax.set_title(f"{title}\n(n={n} samples, c={c:.2f})", fontsize=11, pad=12)
    ax.set_xlabel("Sample j", fontsize=9)
    ax.set_ylabel("Sample i", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Geodesic Distance", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)
    fig.tight_layout()
    return fig


def create_curvature_heatmap(z, title="Conformal Factor", manifold_type="poincare", curvature=1.0):
    """Visualize the conformal factor λ(x) across the latent space."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#06060f")
    style_ax_dark(ax)

    c = curvature
    if manifold_type == "poincare":
        r_max = 1.0 / (c ** 0.5) * 0.99
    else:
        r_max = 2.0

    n_grid = 200
    x_lin = np.linspace(-r_max, r_max, n_grid)
    y_lin = np.linspace(-r_max, r_max, n_grid)
    xx, yy = np.meshgrid(x_lin, y_lin)
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)

    if manifold_type == "poincare":
        lam = poincare_conformal_factor(pts, c).squeeze(-1).numpy()
        # Mask outside ball
        norms = np.sqrt(xx.ravel()**2 + yy.ravel()**2)
        lam[norms >= 1.0 / c**0.5] = np.nan
    else:
        lam = sphere_conformal_factor(pts, c).squeeze(-1).numpy()

    lam_grid = lam.reshape(n_grid, n_grid)

    im = ax.imshow(lam_grid, extent=[-r_max, r_max, -r_max, r_max],
                   origin="lower", cmap="inferno", aspect="equal")
    ax.scatter(z[:, 0], z[:, 1], s=1, alpha=0.2, c="white", edgecolors="none", zorder=3)

    theta = np.linspace(0, 2 * np.pi, 200)
    if manifold_type == "poincare":
        r = 1.0 / c**0.5
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#a78bfa", alpha=0.5, lw=1.0)

    ax.set_title(f"{title}\nλ(x) = conformal factor ({manifold_type}, c={c:.2f})", fontsize=11, pad=12)
    ax.set_xlabel("z₁", fontsize=10)
    ax.set_ylabel("z₂", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("λ(x)", fontsize=9, color="#9892b3")
    cbar.ax.tick_params(colors="#9892b3", labelsize=8)
    fig.tight_layout()
    return fig


def create_manifold_combined(result, density, manifold_type="poincare", curvature=1.0):
    """Combined overlay: W_tangent arrows + z scatter on manifold disk."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#06060f")
    style_ax_dark(ax)

    z = result["z"]
    W_t = result["W_tangent"]
    c = curvature
    theta = np.linspace(0, 2 * np.pi, 200)

    # Draw manifold background
    if manifold_type == "poincare":
        r = 1.0 / (c ** 0.5)
        ax.fill(r * np.cos(theta), r * np.sin(theta), color="#0d0820", alpha=0.3)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#a78bfa", alpha=0.3, lw=1.0)
    else:
        ax.plot(np.cos(theta), np.sin(theta), color="#5e577a", ls="--", alpha=0.2, lw=0.7)

    # z scatter (background)
    ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.12, c="#6b7280", edgecolors="none",
               label="z samples", zorder=1)

    # W_tangent arrows (plasma)
    plot_arrows(ax, W_t, plt.cm.plasma, label="W_tangent")

    def panel_lim(W):
        try:
            norms = np.linalg.norm(W, axis=0)
            m = np.nanmax(norms)
            if np.isnan(m) or np.isinf(m):
                m = 0.1
        except ValueError:
            m = 0.1
        return max(m, 0.1) * 1.25

    w_max = panel_lim(W_t)
    
    try:
        z_abs_max = np.nanmax(np.abs(z))
        if np.isnan(z_abs_max) or np.isinf(z_abs_max):
            z_abs_max = 0.0
    except ValueError:
        z_abs_max = 0.0
        
    z_max = max(z_abs_max * 1.1, 0.5)
    r_ref = 1.15 / c**0.5 if manifold_type == "poincare" else 1.15
    combined_lim = max(w_max, z_max, r_ref)
    
    if np.isnan(combined_lim) or np.isinf(combined_lim):
        combined_lim = 1.0
        
    ax.set_xlim(-combined_lim, combined_lim)
    ax.set_ylim(-combined_lim, combined_lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="#5e577a", lw=0.3)
    ax.axvline(0, color="#5e577a", lw=0.3)
    disk_label = "Poincaré" if manifold_type == "poincare" else "Sphere"
    ax.set_title(f"Combined — Density = {density:.0%} ({disk_label}, c={c:.2f})", fontsize=11, pad=12)
    ax.legend(fontsize=8, loc="upper right",
              facecolor="#0c0c1e", edgecolor="#2a2554", labelcolor="#e8e6f0")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/api/train", methods=["POST"])
def api_train():
    """Train a model and return results + available plots."""
    try:
        config = request.json
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = config.get("model_type", "vae")
        n_features = int(config.get("n_features", 256))
        hidden = int(config.get("hidden", 2))
        density = float(config.get("density", 0.5))
        steps = int(config.get("steps", 5000))
        batch_size = int(config.get("batch_size", 512))
        lr = float(config.get("lr", 1e-3))
        beta = float(config.get("beta", 0.05))
        beta_warmup = int(config.get("beta_warmup", 2000))
        importance_decay = float(config.get("importance_decay", 0.5))
        use_importance = config.get("use_importance", True)
        init_method = config.get("init_method", "xavier_normal")
        normalize_weights = config.get("normalize_weights", False)
        custom_loss_expr = config.get("custom_loss_expr", None)
        snapshot_interval = int(config.get("snapshot_interval", 0))

        # Build importance vector
        if use_importance and importance_decay > 0:
            importance = torch.tensor(
                [(1.0 - i / n_features) ** importance_decay for i in range(n_features)],
                device=device
            )
        else:
            importance = torch.ones(n_features, device=device)

        # Train
        if model_type == "standard":
            result = train_standard_model(
                density, n_features, hidden, steps, batch_size, lr,
                importance, device, init_method=init_method,
                custom_loss_expr=custom_loss_expr,
                normalize_weights=normalize_weights,
                snapshot_interval=snapshot_interval
            )
            available_matrices = ["W"]
            available_plots = ["arrows_W", "interference_W", "norms_W", "graph_W", "coord_graph_W",
                               "manifold_jacobian_W", "manifold_geometry_W", "manifold_summary_W"]
            if hidden > 2:
                available_plots = ["heatmap_W", "pca_W", "interference_W", "norms_W", "graph_W", "coord_graph_W",
                                   "manifold_jacobian_W", "manifold_geometry_W", "manifold_summary_W"]

            # Compute stats
            W = result["W"]
            norms = np.linalg.norm(W, axis=0)
            stats = {
                "max_norm_W": float(norms.max()),
                "mean_norm_W": float(norms.mean()),
                "active_features_W": int((norms > 0.05 * norms.max()).sum()),
                "final_loss": result["final_loss"],
            }
        elif model_type == "manifold_vae":
            curvature = float(config.get("curvature", 1.0))
            manifold_type = config.get("manifold_type", "poincare")
            manifold_loss_type = config.get("manifold_loss_type", "riemannian_kl")
            tangent_weight = float(config.get("tangent_weight", 0.01))

            result = train_manifold_vae_model(
                density, n_features, hidden, steps, batch_size, lr,
                beta, beta_warmup, importance, device,
                curvature=curvature, manifold_type=manifold_type,
                manifold_loss_type=manifold_loss_type,
                tangent_weight=tangent_weight,
                init_method=init_method,
                custom_loss_expr=custom_loss_expr,
                normalize_weights=normalize_weights,
                snapshot_interval=snapshot_interval
            )
            available_matrices = ["W_tangent", "W_mu", "W_logvar"]
            if hidden == 2:
                available_plots = [
                    "arrows_W_tangent", "arrows_W_mu", "arrows_W_logvar",
                    "manifold_disk_scatter", "manifold_combined",
                    "geodesic_matrix", "curvature_heatmap",
                    "interference_W_tangent", "norms_W_tangent",
                    "norms_W_mu", "norms_W_logvar",
                    "graph_W_tangent", "coord_graph_W_tangent",
                    "manifold_jacobian_W_tangent", "manifold_geometry_W_tangent",
                    "manifold_summary_W_tangent",
                ]
            else:
                available_plots = [
                    "heatmap_W_tangent", "pca_W_tangent",
                    "heatmap_W_mu", "pca_W_mu",
                    "manifold_disk_scatter", "geodesic_matrix",
                    "interference_W_tangent", "norms_W_tangent",
                    "norms_W_mu", "norms_W_logvar",
                    "graph_W_tangent", "coord_graph_W_tangent",
                    "manifold_jacobian_W_tangent", "manifold_geometry_W_tangent",
                    "manifold_summary_W_tangent",
                ]

            W_t = result["W_tangent"]
            W_mu = result["W_mu"]
            W_lv = result["W_logvar"]
            norms_t = np.linalg.norm(W_t, axis=0)
            norms_mu = np.linalg.norm(W_mu, axis=0)
            norms_lv = np.linalg.norm(W_lv, axis=0)
            stats = {
                "max_norm_W_tangent": float(norms_t.max()),
                "mean_norm_W_tangent": float(norms_t.mean()),
                "active_features_W_tangent": int((norms_t > 0.05 * norms_t.max()).sum()),
                "max_norm_W_mu": float(norms_mu.max()),
                "mean_norm_W_mu": float(norms_mu.mean()),
                "max_norm_W_logvar": float(norms_lv.max()),
                "final_loss": result["final_loss"],
                "final_recon": result.get("final_recon", 0),
                "final_kl": result.get("final_kl", 0),
                "z_range": float(np.abs(result["z"]).max()) if "z" in result else 0,
                "curvature": curvature,
                "manifold_type": manifold_type,
            }
        else:
            result = train_vae_model(
                density, n_features, hidden, steps, batch_size, lr,
                beta, beta_warmup, importance, device, init_method=init_method,
                custom_loss_expr=custom_loss_expr,
                normalize_weights=normalize_weights,
                snapshot_interval=snapshot_interval
            )
            available_matrices = ["W_mu", "W_logvar"]
            available_plots = ["arrows_W_mu", "arrows_W_logvar", "z_samples",
                              "combined", "interference_W_mu", "interference_W_logvar",
                              "norms_W_mu", "norms_W_logvar", "graph_W_mu", "graph_W_logvar",
                              "coord_graph_W_mu", "coord_graph_W_logvar",
                              "manifold_jacobian_W_mu", "manifold_geometry_W_mu", "manifold_summary_W_mu"]
            if hidden > 2:
                available_plots = ["heatmap_W_mu", "heatmap_W_logvar",
                                   "pca_W_mu", "pca_W_logvar",
                                   "interference_W_mu", "interference_W_logvar",
                                   "norms_W_mu", "norms_W_logvar", "graph_W_mu", "graph_W_logvar",
                                   "coord_graph_W_mu", "coord_graph_W_logvar",
                                   "manifold_jacobian_W_mu", "manifold_geometry_W_mu", "manifold_summary_W_mu"]

            # Compute stats
            W_mu = result["W_mu"]
            W_lv = result["W_logvar"]
            norms_mu = np.linalg.norm(W_mu, axis=0)
            norms_lv = np.linalg.norm(W_lv, axis=0)
            stats = {
                "max_norm_W_mu": float(norms_mu.max()),
                "mean_norm_W_mu": float(norms_mu.mean()),
                "active_features_W_mu": int((norms_mu > 0.05 * norms_mu.max()).sum()),
                "max_norm_W_logvar": float(norms_lv.max()),
                "mean_norm_W_logvar": float(norms_lv.mean()),
                "active_features_W_logvar": int((norms_lv > 0.05 * norms_lv.max()).sum()),
                "final_loss": result["final_loss"],
                "final_recon": result.get("final_recon", 0),
                "final_kl": result.get("final_kl", 0),
                "z_range": float(np.abs(result["z"]).max()) if "z" in result else 0,
            }

        # Store result in app context for plotting
        app.config["LAST_RESULT"] = result
        app.config["LAST_CONFIG"] = config

        # Save experiment locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"exp_{timestamp}"
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_id)
        os.makedirs(exp_path, exist_ok=True)

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "exp_id": exp_id,
            "config": config,
            "stats": stats,
            "available_plots": available_plots,
            "available_matrices": available_matrices,
            "model_type": model_type,
            "hidden": hidden,
            "losses": result.get("losses", [])
        }
        with open(os.path.join(exp_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save numpy arrays and lists in result.npz
        result_arrays = {k: v for k, v in result.items() if k != "losses"}
        np.savez_compressed(os.path.join(exp_path, "result.npz"), **result_arrays)

        # Save graphs to the experiment folder
        for ptype in available_plots:
            filepath = os.path.join(exp_path, f"{ptype}.png")
            save_plot_to_file(ptype, result, density, hidden, filepath)

        return jsonify({
            "success": True,
            "model_type": model_type,
            "hidden": hidden,
            "available_matrices": available_matrices,
            "available_plots": available_plots,
            "stats": stats,
            "losses": result.get("losses", []),
            "device": str(device),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/plot", methods=["POST"])
def api_plot():
    """Generate a specific plot and return as base64 PNG."""
    try:
        plot_type = request.json.get("plot_type", "")
        result = app.config.get("LAST_RESULT")
        config = app.config.get("LAST_CONFIG", {})

        if result is None:
            return jsonify({"success": False, "error": "No model trained yet."}), 400

        density = float(config.get("density", 0.5))
        hidden = int(config.get("hidden", 2))
        is_2d = (hidden == 2)

        def _get_fig(res):
            if plot_type == "arrows_W" and "W" in res and is_2d:
                return create_arrow_plot(res["W"], "W (tied weights)", "plasma")
            elif plot_type == "arrows_W_mu" and "W_mu" in res and is_2d:
                return create_arrow_plot(res["W_mu"], "W_mu (mean weights)", "plasma")
            elif plot_type == "arrows_W_logvar" and "W_logvar" in res and is_2d:
                return create_arrow_plot(res["W_logvar"], "W_logvar (log-var weights)", "viridis")
            elif plot_type == "z_samples" and "z" in res and is_2d:
                return create_z_scatter(res["z"], "Reparameterized z samples")
            elif plot_type == "combined" and "W_mu" in res and is_2d:
                return create_combined_plot(res, density)
            elif plot_type == "interference_W" and "W" in res:
                return create_interference_plot(res["W"], "W")[0]
            elif plot_type == "interference_W_mu" and "W_mu" in res:
                return create_interference_plot(res["W_mu"], "W_mu")[0]
            elif plot_type == "interference_W_logvar" and "W_logvar" in res:
                return create_interference_plot(res["W_logvar"], "W_logvar")[0]
            elif plot_type == "norms_W" and "W" in res:
                return create_norms_plot(res["W"], "W")[0]
            elif plot_type == "norms_W_mu" and "W_mu" in res:
                return create_norms_plot(res["W_mu"], "W_mu")[0]
            elif plot_type == "norms_W_logvar" and "W_logvar" in res:
                return create_norms_plot(res["W_logvar"], "W_logvar")[0]
            elif plot_type == "heatmap_W" and "W" in res:
                return create_heatmap(res["W"], "W")
            elif plot_type == "heatmap_W_mu" and "W_mu" in res:
                return create_heatmap(res["W_mu"], "W_mu")
            elif plot_type == "heatmap_W_logvar" and "W_logvar" in res:
                return create_heatmap(res["W_logvar"], "W_logvar")
            elif plot_type == "pca_W" and "W" in res:
                return create_pca_projection(res["W"], "W (PCA)")
            elif plot_type == "pca_W_mu" and "W_mu" in res:
                return create_pca_projection(res["W_mu"], "W_mu (PCA)")
            elif plot_type == "pca_W_logvar" and "W_logvar" in res:
                return create_pca_projection(res["W_logvar"], "W_logvar (PCA)")
            elif plot_type == "graph_W" and "W" in res:
                return create_graph_plot(res["W"], "W")
            elif plot_type == "graph_W_mu" and "W_mu" in res:
                return create_graph_plot(res["W_mu"], "W_mu")
            elif plot_type == "graph_W_logvar" and "W_logvar" in res:
                return create_graph_plot(res["W_logvar"], "W_logvar")
            elif plot_type == "coord_graph_W" and "W" in res:
                return create_coord_graph_plot(res["W"], "W")
            elif plot_type == "coord_graph_W_mu" and "W_mu" in res:
                return create_coord_graph_plot(res["W_mu"], "W_mu")
            elif plot_type == "coord_graph_W_logvar" and "W_logvar" in res:
                return create_coord_graph_plot(res["W_logvar"], "W_logvar")
            elif plot_type == "manifold_jacobian_W":
                return create_jacobian_plot(res, "W Jacobian")
            elif plot_type == "manifold_jacobian_W_mu":
                return create_jacobian_plot(res, "W_mu Jacobian")
            elif plot_type == "manifold_geometry_W":
                return create_geometry_plot(res, "W Geometry")
            elif plot_type == "manifold_geometry_W_mu":
                return create_geometry_plot(res, "W_mu Geometry")
            elif plot_type == "manifold_summary_W":
                return create_manifold_summary_plot(res, "W Manifold Summary")
            elif plot_type == "manifold_summary_W_mu":
                return create_manifold_summary_plot(res, "W_mu Manifold Summary")
            
            # Manifold VAE Specific Plots
            elif plot_type == "arrows_W_tangent" and "W_tangent" in res and is_2d:
                return create_arrow_plot(res["W_tangent"], "W_tangent (Tangent Space)", "plasma")
            elif plot_type == "manifold_disk_scatter" and "z" in res:
                return create_manifold_disk_scatter(res["z"], manifold_type=config.get("manifold_type", "poincare"), curvature=float(config.get("curvature", 1.0)))
            elif plot_type == "manifold_combined" and "W_tangent" in res and is_2d:
                return create_manifold_combined(res, density, manifold_type=config.get("manifold_type", "poincare"), curvature=float(config.get("curvature", 1.0)))
            elif plot_type == "geodesic_matrix" and "z" in res:
                return create_geodesic_distance_matrix(res["z"], manifold_type=config.get("manifold_type", "poincare"), curvature=float(config.get("curvature", 1.0)))
            elif plot_type == "curvature_heatmap" and "z" in res:
                return create_curvature_heatmap(res["z"], manifold_type=config.get("manifold_type", "poincare"), curvature=float(config.get("curvature", 1.0)))
            elif plot_type == "heatmap_W_tangent" and "W_tangent" in res:
                return create_heatmap(res["W_tangent"], "W_tangent")
            elif plot_type == "pca_W_tangent" and "W_tangent" in res:
                return create_pca_projection(res["W_tangent"], "W_tangent (PCA)")
            elif plot_type == "interference_W_tangent" and "W_tangent" in res:
                return create_interference_plot(res["W_tangent"], "W_tangent")[0]
            elif plot_type == "norms_W_tangent" and "W_tangent" in res:
                return create_norms_plot(res["W_tangent"], "W_tangent")[0]
            elif plot_type == "graph_W_tangent" and "W_tangent" in res:
                return create_graph_plot(res["W_tangent"], "W_tangent")
            elif plot_type == "coord_graph_W_tangent" and "W_tangent" in res:
                return create_coord_graph_plot(res["W_tangent"], "W_tangent")
            elif plot_type == "manifold_jacobian_W_tangent":
                return create_jacobian_plot(res, "W_tangent Jacobian")
            elif plot_type == "manifold_geometry_W_tangent":
                return create_geometry_plot(res, "W_tangent Geometry")
            elif plot_type == "manifold_summary_W_tangent":
                return create_manifold_summary_plot(res, "W_tangent Manifold Summary")
            return None

        fig = _get_fig(result)
        if fig is None:
            return jsonify({"success": False, "error": f"Unknown or unsupported plot type: {plot_type}"}), 400

        b64 = fig_to_base64(fig)
        
        return jsonify({
            "success": True, 
            "image": b64, 
            "plot_type": plot_type
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/plot_gif", methods=["POST"])
def api_plot_gif():
    """Generate a GIF of the plot across snapshots and return as base64."""
    try:
        plot_type = request.json.get("plot_type", "")
        result = app.config.get("LAST_RESULT")
        config = app.config.get("LAST_CONFIG", {})

        if result is None:
            return jsonify({"success": False, "error": "No model trained yet."}), 400
            
        if "snapshots" not in result or not result["snapshots"]:
            return jsonify({"success": False, "error": "No snapshots available for GIF generation."}), 400

        density = float(config.get("density", 0.5))
        hidden = int(config.get("hidden", 2))
        is_2d = (hidden == 2)

        def _get_fig(res):
            if plot_type == "arrows_W" and "W" in res and is_2d:
                return create_arrow_plot(res["W"], "W (tied weights)", "plasma")
            elif plot_type == "arrows_W_mu" and "W_mu" in res and is_2d:
                return create_arrow_plot(res["W_mu"], "W_mu (mean weights)", "plasma")
            elif plot_type == "arrows_W_logvar" and "W_logvar" in res and is_2d:
                return create_arrow_plot(res["W_logvar"], "W_logvar (log-var weights)", "viridis")
            elif plot_type == "z_samples" and "z" in res and is_2d:
                return create_z_scatter(res["z"], "Reparameterized z samples")
            elif plot_type == "combined" and "W_mu" in res and is_2d:
                return create_combined_plot(res, density)
            elif plot_type == "interference_W" and "W" in res:
                return create_interference_plot(res["W"], "W")[0]
            elif plot_type == "interference_W_mu" and "W_mu" in res:
                return create_interference_plot(res["W_mu"], "W_mu")[0]
            elif plot_type == "interference_W_logvar" and "W_logvar" in res:
                return create_interference_plot(res["W_logvar"], "W_logvar")[0]
            elif plot_type == "norms_W" and "W" in res:
                return create_norms_plot(res["W"], "W")[0]
            elif plot_type == "norms_W_mu" and "W_mu" in res:
                return create_norms_plot(res["W_mu"], "W_mu")[0]
            elif plot_type == "norms_W_logvar" and "W_logvar" in res:
                return create_norms_plot(res["W_logvar"], "W_logvar")[0]
            elif plot_type == "heatmap_W" and "W" in res:
                return create_heatmap(res["W"], "W")
            elif plot_type == "heatmap_W_mu" and "W_mu" in res:
                return create_heatmap(res["W_mu"], "W_mu")
            elif plot_type == "heatmap_W_logvar" and "W_logvar" in res:
                return create_heatmap(res["W_logvar"], "W_logvar")
            elif plot_type == "pca_W" and "W" in res:
                return create_pca_projection(res["W"], "W (PCA)")
            elif plot_type == "pca_W_mu" and "W_mu" in res:
                return create_pca_projection(res["W_mu"], "W_mu (PCA)")
            elif plot_type == "pca_W_logvar" and "W_logvar" in res:
                return create_pca_projection(res["W_logvar"], "W_logvar (PCA)")
            elif plot_type == "graph_W" and "W" in res:
                return create_graph_plot(res["W"], "W")
            elif plot_type == "graph_W_mu" and "W_mu" in res:
                return create_graph_plot(res["W_mu"], "W_mu")
            elif plot_type == "graph_W_logvar" and "W_logvar" in res:
                return create_graph_plot(res["W_logvar"], "W_logvar")
            elif plot_type == "coord_graph_W" and "W" in res:
                return create_coord_graph_plot(res["W"], "W")
            elif plot_type == "coord_graph_W_mu" and "W_mu" in res:
                return create_coord_graph_plot(res["W_mu"], "W_mu")
            elif plot_type == "coord_graph_W_logvar" and "W_logvar" in res:
                return create_coord_graph_plot(res["W_logvar"], "W_logvar")
            elif plot_type == "manifold_jacobian_W":
                return create_jacobian_plot(res, "W Jacobian")
            elif plot_type == "manifold_jacobian_W_mu":
                return create_jacobian_plot(res, "W_mu Jacobian")
            elif plot_type == "manifold_geometry_W":
                return create_geometry_plot(res, "W Geometry")
            elif plot_type == "manifold_geometry_W_mu":
                return create_geometry_plot(res, "W_mu Geometry")
            elif plot_type == "manifold_summary_W":
                return create_manifold_summary_plot(res, "W Manifold Summary")
            elif plot_type == "manifold_summary_W_mu":
                return create_manifold_summary_plot(res, "W_mu Manifold Summary")
            return None

        images = []
        frames_to_process = result["snapshots"] + [result]
        
        for snap in frames_to_process:
            fig = _get_fig(snap)
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
                buf.seek(0)
                img = Image.open(buf)
                images.append(img.copy())
                plt.close(fig)
        
        if not images:
            return jsonify({"success": False, "error": f"Could not generate frames for plot type: {plot_type}"}), 400
            
        out_buf = io.BytesIO()
        images[0].save(out_buf, format="GIF", save_all=True, append_images=images[1:], duration=600)
        out_buf.seek(0)
        b64 = base64.b64encode(out_buf.read()).decode("utf-8")
        
        return jsonify({
            "success": True, 
            "image": b64, 
            "plot_type": plot_type,
            "is_gif": True
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def api_report():
    """Generate a JSON report of the experiment."""
    try:
        result = app.config.get("LAST_RESULT")
        config = app.config.get("LAST_CONFIG", {})

        if result is None:
            return jsonify({"success": False, "error": "No model trained yet."}), 400

        model_type = config.get("model_type", "vae")
        n_features = int(config.get("n_features", 256))
        hidden = int(config.get("hidden", 2))
        density = float(config.get("density", 0.5))

        report = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "Model Type": model_type.upper(),
                "Hidden Dimension": hidden,
                "Number of Features (n)": n_features,
                "Density": f"{density:.0%}",
                "Weight Initialization": config.get("init_method", "xavier_normal").replace("_", " ").title(),
                "Training Steps": int(config.get("steps", 5000)),
                "Batch Size": int(config.get("batch_size", 512)),
                "Learning Rate": float(config.get("lr", 1e-3)),
                "Importance Decay": float(config.get("importance_decay", 0.5)),
            },
            "results": {},
        }

        if model_type == "vae":
            report["parameters"]["β (KL Weight)"] = float(config.get("beta", 0.05))
            report["parameters"]["β Warmup Steps"] = int(config.get("beta_warmup", 2000))

            W_mu = result["W_mu"]
            W_lv = result["W_logvar"]
            norms_mu = np.linalg.norm(W_mu, axis=0)
            norms_lv = np.linalg.norm(W_lv, axis=0)

            report["results"] = {
                "Final Loss (total)": f"{result['final_loss']:.6f}",
                "Final Reconstruction Loss": f"{result.get('final_recon', 0):.6f}",
                "Final KL Divergence": f"{result.get('final_kl', 0):.6f}",
                "W_mu — Max ‖w‖": f"{norms_mu.max():.6f}",
                "W_mu — Mean ‖w‖": f"{norms_mu.mean():.6f}",
                "W_mu — Active Features": int((norms_mu > 0.05 * norms_mu.max()).sum()),
                "W_logvar — Max ‖w‖": f"{norms_lv.max():.6f}",
                "W_logvar — Mean ‖w‖": f"{norms_lv.mean():.6f}",
                "W_logvar — Active Features": int((norms_lv > 0.05 * norms_lv.max()).sum()),
                "z — Max |z|": f"{np.abs(result['z']).max():.6f}",
                "z — Mean |z|": f"{np.abs(result['z']).mean():.6f}",
            }

            # Interference stats
            WtW_mu = W_mu.T @ W_mu
            offdiag_mu = WtW_mu[~np.eye(WtW_mu.shape[0], dtype=bool)]
            report["results"]["W_mu — Interference (off-diag mean)"] = f"{offdiag_mu.mean():.6f}"
            report["results"]["W_mu — Interference (off-diag max |val|)"] = f"{np.abs(offdiag_mu).max():.6f}"

        else:
            W = result["W"]
            norms = np.linalg.norm(W, axis=0)
            report["results"] = {
                "Final Loss": f"{result['final_loss']:.6f}",
                "W — Max ‖w‖": f"{norms.max():.6f}",
                "W — Mean ‖w‖": f"{norms.mean():.6f}",
                "W — Active Features": int((norms > 0.05 * norms.max()).sum()),
            }
            WtW = W.T @ W
            offdiag = WtW[~np.eye(WtW.shape[0], dtype=bool)]
            report["results"]["W — Interference (off-diag mean)"] = f"{offdiag.mean():.6f}"
            report["results"]["W — Interference (off-diag max |val|)"] = f"{np.abs(offdiag).max():.6f}"

        report["results"]["Device"] = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        return jsonify({"success": True, "report": report})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/validate-loss", methods=["POST"])
def api_validate_loss():
    """Validate a custom loss expression using dummy tensors."""
    try:
        data = request.json
        expr = data.get("expr", "").strip()
        model_type = data.get("model_type", "vae")

        if not expr:
            return jsonify({"success": False, "error": "Expression is empty."}), 400

        # Create dummy tensors to match shape (batch=2, features=4)
        device = "cpu"
        x = torch.rand(2, 4, device=device)
        x_hat = torch.rand(2, 4, device=device)
        importance = torch.ones(4, device=device)

        local_vars = {"x": x, "x_hat": x_hat, "importance": importance, "torch": torch}

        if model_type == "vae" or model_type == "manifold_vae":
            mu = torch.rand(2, 2, device=device)
            logvar = torch.rand(2, 2, device=device)
            z = torch.rand(2, 2, device=device)
            beta_t = 0.05
            recon = (importance * (x - x_hat) ** 2).sum(dim=1).mean()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            local_vars.update({
                "mu": mu, "logvar": logvar, "z": z, 
                "beta_t": beta_t, "recon": recon, "kl": kl
            })
            
            if model_type == "manifold_vae":
                tangent_pen = torch.tensor(0.1, device=device)
                score_loss = torch.tensor(0.1, device=device)
                local_vars.update({"tangent_pen": tangent_pen, "score_loss": score_loss})

        # Test evaluation
        loss = safe_eval_loss(expr, local_vars)

        # Check if it returns a scalar tensor
        if not isinstance(loss, torch.Tensor):
            return jsonify({"success": False, "error": "Expression must return a PyTorch tensor."}), 400
        if loss.numel() != 1:
            return jsonify({"success": False, "error": f"Expression must return a scalar, got shape {list(loss.shape)}."}), 400

        return jsonify({"success": True, "message": "Expression is valid!"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/weights", methods=["GET"])
def api_weights():
    """Return raw weight matrices from the last trained model as JSON."""
    try:
        result = app.config.get("LAST_RESULT")
        config = app.config.get("LAST_CONFIG", {})

        if result is None:
            return jsonify({"success": False, "error": "No model trained yet."}), 400

        model_type = config.get("model_type", "vae")
        matrices = {}

        def _matrix_info(W, name):
            """Build a serializable dict for a weight matrix."""
            norms = np.linalg.norm(W, axis=0).tolist()
            return {
                "name": name,
                "shape": list(W.shape),
                "values": np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
                "norms": [round(n, 6) for n in norms],
                "max_norm": round(float(max(norms)), 6),
                "mean_norm": round(float(np.mean(norms)), 6),
            }

        if model_type == "standard":
            if "W" in result:
                matrices["W"] = _matrix_info(result["W"], "W (tied weights)")
        elif model_type == "manifold_vae":
            if "W_tangent" in result:
                matrices["W_tangent"] = _matrix_info(result["W_tangent"], "W_tangent")
            if "W_mu" in result:
                matrices["W_mu"] = _matrix_info(result["W_mu"], "W_μ (mean)")
            if "W_logvar" in result:
                matrices["W_logvar"] = _matrix_info(result["W_logvar"], "W_logvar")
        else:  # vae
            if "W_mu" in result:
                matrices["W_mu"] = _matrix_info(result["W_mu"], "W_μ (mean)")
            if "W_logvar" in result:
                matrices["W_logvar"] = _matrix_info(result["W_logvar"], "W_logvar")

        bias = None
        if "b" in result:
            b = result["b"]
            if hasattr(b, 'tolist'):
                bias = np.nan_to_num(b, nan=0.0).tolist()

        return jsonify({
            "success": True,
            "model_type": model_type,
            "matrices": matrices,
            "bias": bias,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def generate_plot_b64(plot_type, res, density, hidden):
    is_2d = (hidden == 2)
    if plot_type == "arrows_W" and "W" in res and is_2d:
        fig = create_arrow_plot(res["W"], f"W (d={density:.2f})", "plasma")
    elif plot_type == "arrows_W_mu" and "W_mu" in res and is_2d:
        fig = create_arrow_plot(res["W_mu"], f"W_mu (d={density:.2f})", "plasma")
    elif plot_type == "arrows_W_logvar" and "W_logvar" in res and is_2d:
        fig = create_arrow_plot(res["W_logvar"], f"W_lv (d={density:.2f})", "viridis")
    elif plot_type == "z_samples" and "z" in res and is_2d:
        fig = create_z_scatter(res["z"], f"z (d={density:.2f})")
    elif plot_type == "combined" and "W_mu" in res and is_2d:
        fig = create_combined_plot(res, density)
    elif plot_type == "interference_W" and "W" in res:
        fig, _ = create_interference_plot(res["W"], "W")
    elif plot_type == "interference_W_mu" and "W_mu" in res:
        fig, _ = create_interference_plot(res["W_mu"], "W_mu")
    elif plot_type == "interference_W_logvar" and "W_logvar" in res:
        fig, _ = create_interference_plot(res["W_logvar"], "W_logvar")
    elif plot_type == "norms_W" and "W" in res:
        fig, _ = create_norms_plot(res["W"], "W")
    elif plot_type == "norms_W_mu" and "W_mu" in res:
        fig, _ = create_norms_plot(res["W_mu"], "W_mu")
    elif plot_type == "norms_W_logvar" and "W_logvar" in res:
        fig, _ = create_norms_plot(res["W_logvar"], "W_logvar")
    elif plot_type == "heatmap_W" and "W" in res:
        fig = create_heatmap(res["W"], "W")
    elif plot_type == "heatmap_W_mu" and "W_mu" in res:
        fig = create_heatmap(res["W_mu"], "W_mu")
    elif plot_type == "heatmap_W_logvar" and "W_logvar" in res:
        fig = create_heatmap(res["W_logvar"], "W_logvar")
    elif plot_type == "pca_W" and "W" in res:
        fig = create_pca_projection(res["W"], "W (PCA)")
    elif plot_type == "pca_W_mu" and "W_mu" in res:
        fig = create_pca_projection(res["W_mu"], "W_mu (PCA)")
    elif plot_type == "pca_W_logvar" and "W_logvar" in res:
        fig = create_pca_projection(res["W_logvar"], "W_logvar (PCA)")
    
    # Manifold VAE Specific Plots
    elif plot_type == "arrows_W_tangent" and "W_tangent" in res and is_2d:
        fig = create_arrow_plot(res["W_tangent"], f"W_tangent (d={density:.2f})", "plasma")
    elif plot_type == "manifold_disk_scatter" and "z" in res:
        fig = create_manifold_disk_scatter(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "manifold_combined" and "W_tangent" in res and is_2d:
        fig = create_manifold_combined(res, density, manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "geodesic_matrix" and "z" in res:
        fig = create_geodesic_distance_matrix(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "curvature_heatmap" and "z" in res:
        fig = create_curvature_heatmap(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "heatmap_W_tangent" and "W_tangent" in res:
        fig = create_heatmap(res["W_tangent"], "W_tangent")
    elif plot_type == "pca_W_tangent" and "W_tangent" in res:
        fig = create_pca_projection(res["W_tangent"], "W_tangent (PCA)")
    elif plot_type == "interference_W_tangent" and "W_tangent" in res:
        fig, _ = create_interference_plot(res["W_tangent"], "W_tangent")
    elif plot_type == "norms_W_tangent" and "W_tangent" in res:
        fig, _ = create_norms_plot(res["W_tangent"], "W_tangent")
    elif plot_type == "graph_W_tangent" and "W_tangent" in res:
        fig = create_graph_plot(res["W_tangent"], f"W_tangent Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W_tangent" and "W_tangent" in res:
        fig = create_coord_graph_plot(res["W_tangent"], f"W_tangent Coord Graph (d={density:.2f})")
    elif plot_type == "manifold_jacobian_W_tangent":
        fig = create_jacobian_plot(res, "W_tangent Jacobian")
    elif plot_type == "manifold_geometry_W_tangent":
        fig = create_geometry_plot(res, "W_tangent Geometry")
    elif plot_type == "manifold_summary_W_tangent":
        fig = create_manifold_summary_plot(res, "W_tangent Manifold Summary")

    else:
        # Fallback based on models
        if "W_tangent" in res:
            if is_2d:
                fig = create_manifold_combined(res, density, manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
            else:
                fig = create_pca_projection(res["W_tangent"], "W_tangent (PCA)")
        elif "W_mu" in res:
            if is_2d:
                fig = create_combined_plot(res, density)
            else:
                fig = create_pca_projection(res["W_mu"], "W_mu (PCA)")
        else:
            if is_2d:
                fig = create_arrow_plot(res["W"], f"W (d={density:.2f})", "plasma")
            else:
                fig = create_pca_projection(res["W"], "W (PCA)")
    return fig_to_base64(fig)


def save_plot_to_file(plot_type, res, density, hidden, filepath):
    """Generates the plot matching plot_type and saves it directly to filepath."""
    is_2d = (hidden == 2)
    if plot_type == "arrows_W" and "W" in res and is_2d:
        fig = create_arrow_plot(res["W"], f"W (d={density:.2f})", "plasma")
    elif plot_type == "arrows_W_mu" and "W_mu" in res and is_2d:
        fig = create_arrow_plot(res["W_mu"], f"W_mu (d={density:.2f})", "plasma")
    elif plot_type == "arrows_W_logvar" and "W_logvar" in res and is_2d:
        fig = create_arrow_plot(res["W_logvar"], f"W_lv (d={density:.2f})", "viridis")
    elif plot_type == "z_samples" and "z" in res and is_2d:
        fig = create_z_scatter(res["z"], f"z (d={density:.2f})")
    elif plot_type == "combined" and "W_mu" in res and is_2d:
        fig = create_combined_plot(res, density)
    elif plot_type == "interference_W" and "W" in res:
        fig, _ = create_interference_plot(res["W"], "W")
    elif plot_type == "interference_W_mu" and "W_mu" in res:
        fig, _ = create_interference_plot(res["W_mu"], "W_mu")
    elif plot_type == "interference_W_logvar" and "W_logvar" in res:
        fig, _ = create_interference_plot(res["W_logvar"], "W_logvar")
    elif plot_type == "norms_W" and "W" in res:
        fig, _ = create_norms_plot(res["W"], "W")
    elif plot_type == "norms_W_mu" and "W_mu" in res:
        fig, _ = create_norms_plot(res["W_mu"], "W_mu")
    elif plot_type == "norms_W_logvar" and "W_logvar" in res:
        fig, _ = create_norms_plot(res["W_logvar"], "W_logvar")
    elif plot_type == "heatmap_W" and "W" in res:
        fig = create_heatmap(res["W"], "W")
    elif plot_type == "heatmap_W_mu" and "W_mu" in res:
        fig = create_heatmap(res["W_mu"], "W_mu")
    elif plot_type == "heatmap_W_logvar" and "W_logvar" in res:
        fig = create_heatmap(res["W_logvar"], "W_logvar")
    elif plot_type == "pca_W" and "W" in res:
        fig = create_pca_projection(res["W"], "W (PCA)")
    elif plot_type == "pca_W_mu" and "W_mu" in res:
        fig = create_pca_projection(res["W_mu"], "W_mu (PCA)")
    elif plot_type == "pca_W_logvar" and "W_logvar" in res:
        fig = create_pca_projection(res["W_logvar"], "W_logvar (PCA)")
    elif plot_type == "graph_W" and "W" in res:
        fig = create_graph_plot(res["W"], f"W Graph (d={density:.2f})")
    elif plot_type == "graph_W_mu" and "W_mu" in res:
        fig = create_graph_plot(res["W_mu"], f"W_mu Graph (d={density:.2f})")
    elif plot_type == "graph_W_logvar" and "W_logvar" in res:
        fig = create_graph_plot(res["W_logvar"], f"W_logvar Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W" and "W" in res:
        fig = create_coord_graph_plot(res["W"], f"W Coord Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W_mu" and "W_mu" in res:
        fig = create_coord_graph_plot(res["W_mu"], f"W_mu Coord Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W_logvar" and "W_logvar" in res:
        fig = create_coord_graph_plot(res["W_logvar"], f"W_logvar Coord Graph (d={density:.2f})")
    elif plot_type == "manifold_jacobian_W":
        fig = create_jacobian_plot(res, "W Jacobian")
    elif plot_type == "manifold_jacobian_W_mu":
        fig = create_jacobian_plot(res, "W_mu Jacobian")
    elif plot_type == "manifold_geometry_W":
        fig = create_geometry_plot(res, "W Geometry")
    elif plot_type == "manifold_geometry_W_mu":
        fig = create_geometry_plot(res, "W_mu Geometry")
    elif plot_type == "manifold_summary_W":
        fig = create_manifold_summary_plot(res, "W Manifold Summary")
    elif plot_type == "manifold_summary_W_mu":
        fig = create_manifold_summary_plot(res, "W_mu Manifold Summary")
        
    # Manifold VAE Specific Plots
    elif plot_type == "arrows_W_tangent" and "W_tangent" in res and is_2d:
        fig = create_arrow_plot(res["W_tangent"], f"W_tangent (d={density:.2f})", "plasma")
    elif plot_type == "manifold_disk_scatter" and "z" in res:
        fig = create_manifold_disk_scatter(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "manifold_combined" and "W_tangent" in res and is_2d:
        fig = create_manifold_combined(res, density, manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "geodesic_matrix" and "z" in res:
        fig = create_geodesic_distance_matrix(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "curvature_heatmap" and "z" in res:
        fig = create_curvature_heatmap(res["z"], manifold_type=res.get("manifold_type", "poincare"), curvature=float(res.get("curvature", 1.0)))
    elif plot_type == "heatmap_W_tangent" and "W_tangent" in res:
        fig = create_heatmap(res["W_tangent"], "W_tangent")
    elif plot_type == "pca_W_tangent" and "W_tangent" in res:
        fig = create_pca_projection(res["W_tangent"], "W_tangent (PCA)")
    elif plot_type == "interference_W_tangent" and "W_tangent" in res:
        fig, _ = create_interference_plot(res["W_tangent"], "W_tangent")
    elif plot_type == "norms_W_tangent" and "W_tangent" in res:
        fig, _ = create_norms_plot(res["W_tangent"], "W_tangent")
    elif plot_type == "graph_W_tangent" and "W_tangent" in res:
        fig = create_graph_plot(res["W_tangent"], f"W_tangent Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W_tangent" and "W_tangent" in res:
        fig = create_coord_graph_plot(res["W_tangent"], f"W_tangent Coord Graph (d={density:.2f})")
    elif plot_type == "manifold_jacobian_W_tangent":
        fig = create_jacobian_plot(res, "W_tangent Jacobian")
    elif plot_type == "manifold_geometry_W_tangent":
        fig = create_geometry_plot(res, "W_tangent Geometry")
    elif plot_type == "manifold_summary_W_tangent":
        fig = create_manifold_summary_plot(res, "W_tangent Manifold Summary")

    if fig:
        fig.savefig(filepath, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
    elif plot_type == "graph_W_mu" and "W_mu" in res:
        fig = create_graph_plot(res["W_mu"], f"W_mu Graph (d={density:.2f})")
    elif plot_type == "graph_W_logvar" and "W_logvar" in res:
        fig = create_graph_plot(res["W_logvar"], f"W_logvar Graph (d={density:.2f})")
    elif plot_type == "coord_graph_W" and "W" in res:
        fig = create_coord_graph_plot(res["W"], f"W Coord (d={density:.2f})")
    elif plot_type == "coord_graph_W_mu" and "W_mu" in res:
        fig = create_coord_graph_plot(res["W_mu"], f"W_mu Coord (d={density:.2f})")
    elif plot_type == "coord_graph_W_logvar" and "W_logvar" in res:
        fig = create_coord_graph_plot(res["W_logvar"], f"W_logvar Coord (d={density:.2f})")
    elif plot_type == "manifold_jacobian_W":
        fig = create_jacobian_plot(res, f"W Jacobian (d={density:.2f})")
    elif plot_type == "manifold_jacobian_W_mu":
        fig = create_jacobian_plot(res, f"W_mu Jacobian (d={density:.2f})")
    elif plot_type == "manifold_geometry_W":
        fig = create_geometry_plot(res, f"W Geometry (d={density:.2f})")
    elif plot_type == "manifold_geometry_W_mu":
        fig = create_geometry_plot(res, f"W_mu Geometry (d={density:.2f})")
    elif plot_type == "manifold_summary_W":
        fig = create_manifold_summary_plot(res, f"W Manifold (d={density:.2f})")
    elif plot_type == "manifold_summary_W_mu":
        fig = create_manifold_summary_plot(res, f"W_mu Manifold (d={density:.2f})")
    else:
        # Fallback based on models
        if "W_mu" in res:
            if is_2d:
                fig = create_combined_plot(res, density)
            else:
                fig = create_pca_projection(res["W_mu"], "W_mu (PCA)")
        else:
            if is_2d:
                fig = create_arrow_plot(res["W"], f"W (d={density:.2f})", "plasma")
            else:
                fig = create_pca_projection(res["W"], "W (PCA)")
    
    fig.savefig(filepath, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    """Run a grid sweep training over multiple density and importance decay values."""
    try:
        config = request.json
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_type = config.get("model_type", "vae")
        n_features = int(config.get("n_features", 256))
        hidden = int(config.get("hidden", 2))
        init_method = config.get("init_method", "xavier_normal")
        steps = int(config.get("steps", 1000))
        batch_size = int(config.get("batch_size", 512))
        lr = float(config.get("lr", 1e-3))
        beta = float(config.get("beta", 0.05))
        beta_warmup = int(config.get("beta_warmup", 2000))
        normalize_weights = config.get("normalize_weights", False)
        custom_loss_expr = config.get("custom_loss_expr", None)
        
        # Read list of plot types
        plot_types = config.get("plot_types", [])
        if not plot_types:
            single_type = config.get("plot_type", "combined")
            plot_types = [single_type]

        # Range variables
        density_start = float(config.get("density_start", 0.1))
        density_end = float(config.get("density_end", 1.0))
        density_step = float(config.get("density_step", 0.3))

        decay_start = float(config.get("importance_decay_start", 0.0))
        decay_end = float(config.get("importance_decay_end", 2.0))
        decay_step = float(config.get("importance_decay_step", 1.0))

        # Build list of densities
        densities = []
        d = density_start
        while d <= density_end + 1e-5:
            densities.append(round(d, 4))
            if density_step <= 0:
                break
            d += density_step

        # Build list of decay values
        decays = []
        dec = decay_start
        while dec <= decay_end + 1e-5:
            decays.append(round(dec, 4))
            if decay_step <= 0:
                break
            dec += decay_step

        # Build list of sweep configurations
        custom_sweeps = config.get("custom_sweeps", None)
        sweep_configs = []
        if custom_sweeps:
            for item in custom_sweeps:
                feats = int(item.get("features", 256))
                hid = int(item.get("hidden", 2))
                runs = int(item.get("runs", 1))
                for run_idx in range(1, runs + 1):
                    sweep_configs.append({
                        "features": feats,
                        "hidden": hid,
                        "run_id": run_idx
                    })
        else:
            sweep_features = config.get("sweep_features", False)
            hidden = int(config.get("hidden", 2))
            if sweep_features:
                features_start = int(config.get("features_start", 64))
                features_end = int(config.get("features_end", 256))
                features_step = int(config.get("features_step", 64))
                
                f = features_start
                while f <= features_end:
                    sweep_configs.append({
                        "features": f,
                        "hidden": hidden,
                        "run_id": 1
                    })
                    if features_step <= 0:
                        break
                    f += features_step
            else:
                sweep_configs.append({
                    "features": n_features,
                    "hidden": hidden,
                    "run_id": 1
                })

        # Create experiment directory for this sweep
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"sweep_{timestamp}"
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_id)
        os.makedirs(exp_path, exist_ok=True)

        total_runs = len(sweep_configs) * len(densities) * len(decays)
        if total_runs > 500:
            return jsonify({
                "success": False,
                "error": f"Sweep grid is too large ({len(sweep_configs)} configs x {len(densities)} densities x {len(decays)} decays = {total_runs} runs). Max allowed is 500 to prevent CPU lockups."
            }), 400

        grid_results = []

        for s_cfg in sweep_configs:
            f_val = s_cfg["features"]
            h_val = s_cfg["hidden"]
            r_id = s_cfg["run_id"]

            for d_val in densities:
                for dec_val in decays:
                    # Build importance decay vector for this run
                    importance = torch.tensor(
                        [(1.0 - i / f_val) ** dec_val for i in range(f_val)],
                        device=device
                    )

                    if model_type == "standard":
                        res = train_standard_model(
                            d_val, f_val, h_val, steps, batch_size, lr,
                            importance, device, init_method=init_method,
                            custom_loss_expr=custom_loss_expr,
                            normalize_weights=normalize_weights
                        )
                    elif model_type == "manifold_vae":
                        curvature = float(config.get("curvature", 1.0))
                        manifold_type = config.get("manifold_type", "poincare")
                        manifold_loss_type = config.get("manifold_loss_type", "riemannian_kl")
                        tangent_weight = float(config.get("tangent_weight", 0.01))
                        res = train_manifold_vae_model(
                            d_val, f_val, h_val, steps, batch_size, lr,
                            beta, beta_warmup, importance, device,
                            curvature=curvature, manifold_type=manifold_type,
                            manifold_loss_type=manifold_loss_type,
                            tangent_weight=tangent_weight,
                            init_method=init_method,
                            custom_loss_expr=custom_loss_expr,
                            normalize_weights=normalize_weights
                        )
                    else:
                        res = train_vae_model(
                            d_val, f_val, h_val, steps, batch_size, lr,
                            beta, beta_warmup, importance, device, init_method=init_method,
                            custom_loss_expr=custom_loss_expr,
                            normalize_weights=normalize_weights
                        )

                    # Save plots for all requested plot types to experiment folder
                    plot_paths = {}
                    for ptype in plot_types:
                        filename = f"feats_{f_val}_hid_{h_val}_run_{r_id}_density_{d_val:.4f}_decay_{dec_val:.4f}_{ptype}.png"
                        filepath = os.path.join(exp_path, filename)
                        save_plot_to_file(ptype, res, d_val, h_val, filepath)
                        plot_paths[ptype] = f"experiments/{exp_id}/{filename}"

                    # Compute weight norms for filtering
                    if model_type == "standard":
                        W_arr = res["W"]
                    elif model_type == "manifold_vae":
                        W_arr = res["W_tangent"]
                    else:
                        W_arr = res["W_mu"]
                    
                    norms = np.linalg.norm(W_arr, axis=0)
                    mean_weight_norm = float(norms.mean())
                    max_weight_norm = float(norms.max())

                    # Set backwards-compatible single plot path
                    default_plot_path = plot_paths.get(plot_types[0], f"experiments/{exp_id}/feats_{f_val}_hid_{h_val}_run_{r_id}_density_{d_val:.4f}_decay_{dec_val:.4f}_{plot_types[0]}.png")

                    grid_results.append({
                        "features": f_val,
                        "hidden": h_val,
                        "run_id": r_id,
                        "density": d_val,
                        "importance_decay": dec_val,
                        "plot_path": default_plot_path,
                        "plot_paths": plot_paths,
                        "final_loss": res["final_loss"],
                        "recon": res.get("final_recon", 0) if model_type == "vae" else None,
                        "kl": res.get("final_kl", 0) if model_type == "vae" else None,
                        "mean_weight_norm": mean_weight_norm,
                        "max_weight_norm": max_weight_norm
                    })

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "exp_id": exp_id,
            "is_sweep": True,
            "config": config,
            "grid": grid_results,
            "density_values": densities,
            "importance_decay_values": decays,
            "sweep_configs": sweep_configs,
            "model_type": model_type,
            "hidden": hidden
        }
        with open(os.path.join(exp_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            "success": True,
            "is_sweep": True,
            "grid": grid_results,
            "density_values": densities,
            "importance_decay_values": decays,
            "sweep_configs": sweep_configs,
            "device": str(device)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments", methods=["GET"])
def api_experiments():
    """List saved experiments."""
    try:
        experiments = []
        if os.path.exists(EXPERIMENTS_DIR):
            for exp_id in os.listdir(EXPERIMENTS_DIR):
                exp_path = os.path.join(EXPERIMENTS_DIR, exp_id)
                metadata_path = os.path.join(exp_path, "metadata.json")
                if os.path.isdir(exp_path) and os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
        # Sort descending by timestamp
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jsonify({"success": True, "experiments": experiments})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments/load", methods=["POST"])
def api_experiments_load():
    """Load a specific experiment."""
    try:
        exp_id = request.json.get("exp_id")
        if not exp_id:
            return jsonify({"success": False, "error": "Missing exp_id"}), 400

        exp_path = os.path.join(EXPERIMENTS_DIR, exp_id)
        metadata_path = os.path.join(exp_path, "metadata.json")
        result_path = os.path.join(exp_path, "result.npz")

        if not os.path.exists(metadata_path) or not os.path.exists(result_path):
            return jsonify({"success": False, "error": "Experiment files not found."}), 404

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if metadata.get("is_sweep"):
            return jsonify({
                "success": True,
                "is_sweep": True,
                "config": metadata.get("config", {}),
                "grid": metadata.get("grid", []),
                "density_values": metadata.get("density_values", []),
                "importance_decay_values": metadata.get("importance_decay_values", []),
                "sweep_configs": metadata.get("sweep_configs", []),
                "model_type": metadata.get("model_type"),
                "hidden": metadata.get("hidden"),
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            })

        # Load result
        result_arrays = np.load(result_path, allow_pickle=True)
        result = {k: result_arrays[k] for k in result_arrays.files}
        if "losses" in metadata:
            result["losses"] = metadata["losses"]
        else:
            result["losses"] = []

        app.config["LAST_RESULT"] = result
        app.config["LAST_CONFIG"] = metadata.get("config", {})

        return jsonify({
            "success": True,
            "model_type": metadata.get("model_type"),
            "hidden": metadata.get("hidden"),
            "available_matrices": metadata.get("available_matrices", []),
            "available_plots": metadata.get("available_plots", []),
            "stats": metadata.get("stats", {}),
            "losses": result["losses"],
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  VAE Superposition Dashboard — Backend")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
