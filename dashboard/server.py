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
from datetime import datetime
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
        logvar = x @ W_logvar.t()
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


def train_standard_model(density, n_features, hidden, steps, batch_size, lr,
                         importance, device, init_method="xavier_normal",
                         custom_loss_expr=None, normalize_weights=False,
                         snapshot_interval=0):
    model = ToyBottleneck(n_features, hidden, init_method=init_method, normalize_weights=normalize_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    snapshots = []
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
            losses.append(float(loss.item()))
            
        if snapshot_interval > 0 and (step == 0 or (step + 1) % snapshot_interval == 0):
            if model.normalize_weights:
                W_snap = F.normalize(model.W, p=2, dim=0).detach().cpu().numpy()
            else:
                W_snap = model.W.detach().cpu().numpy()
            snapshots.append({"step": step + 1, "W": W_snap})

    if model.normalize_weights:
        W = F.normalize(model.W, p=2, dim=0).detach().cpu().numpy()
    else:
        W = model.W.detach().cpu().numpy()

    return {
        "W": W,
        "losses": losses,
        "final_loss": float(loss.item()),
        "snapshots": snapshots,
    }


def train_vae_model(density, n_features, hidden, steps, batch_size, lr,
                    beta, beta_warmup, importance, device, init_method="xavier_normal",
                    custom_loss_expr=None, normalize_weights=False,
                    snapshot_interval=0):
    model = ToyVAEBottleneck(n_features, hidden, init_method=init_method, normalize_weights=normalize_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    snapshots = []
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
                "total": float(loss.item()),
                "recon": float(recon.item()),
                "kl": float(kl.item()),
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
                
            snapshots.append({
                "step": step + 1,
                "W_mu": W_mu_snap,
                "W_logvar": W_logvar_snap,
                "z": z_eval_snap.cpu().numpy()
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

    return {
        "W_mu": W_mu,
        "W_logvar": W_logvar,
        "z": z_eval.cpu().numpy(),
        "mu": mu_eval.cpu().numpy(),
        "losses": losses,
        "final_loss": float(loss.item()),
        "final_recon": float(recon.item()),
        "final_kl": float(kl.item()),
        "snapshots": snapshots,
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

    z_max = max(np.abs(z).max() * 1.1, 0.5) if len(z) > 0 else 1.0
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
    z_max = np.abs(z).max() * 1.1 if len(z) > 0 else 0.5
    combined_lim = max(w_max, z_max, 1.15)
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
            available_plots = ["arrows_W", "interference_W", "norms_W", "graph_W"]
            if hidden > 2:
                available_plots = ["heatmap_W", "pca_W", "interference_W", "norms_W", "graph_W"]

            # Compute stats
            W = result["W"]
            norms = np.linalg.norm(W, axis=0)
            stats = {
                "max_norm_W": float(norms.max()),
                "mean_norm_W": float(norms.mean()),
                "active_features_W": int((norms > 0.05 * norms.max()).sum()),
                "final_loss": result["final_loss"],
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
                              "norms_W_mu", "norms_W_logvar", "graph_W_mu", "graph_W_logvar"]
            if hidden > 2:
                available_plots = ["heatmap_W_mu", "heatmap_W_logvar",
                                   "pca_W_mu", "pca_W_logvar",
                                   "interference_W_mu", "interference_W_logvar",
                                   "norms_W_mu", "norms_W_logvar", "graph_W_mu", "graph_W_logvar"]

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

        if model_type == "vae":
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
