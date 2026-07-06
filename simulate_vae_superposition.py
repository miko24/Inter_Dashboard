"""
VAE-style Toy Bottleneck with mu & logvar weights + reparameterization.

For each feature-density level (5% to 100%), train a bottleneck VAE:

    mu     = W_mu x
    logvar = W_logvar x
    z      = mu + eps * exp(0.5 * logvar)    (reparameterization)
    x_hat  = ReLU(W_mu^T z + b)             (tied weights with mu)

Then plot three things per density:
  1. W_mu   columns in 2D  (mean directions)
  2. W_logvar columns in 2D  (variance directions)
  3. Sampled z points in 2D  (the reparameterized latent codes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── VAE-style toy bottleneck (tied weights via W_mu) ─────────────
class ToyVAEBottleneck(nn.Module):
    def __init__(self, n_features=256, hidden=2, normalize_weights=False):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.normalize_weights = normalize_weights
        self.W_mu = nn.Parameter(torch.randn(hidden, n_features) * 0.05)
        self.W_logvar = nn.Parameter(torch.randn(hidden, n_features) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_features))

    def encode(self, x):
        W_mu = F.normalize(self.W_mu, p=2, dim=0) if self.normalize_weights else self.W_mu
        W_logvar = F.normalize(self.W_logvar, p=2, dim=0) if self.normalize_weights else self.W_logvar
        mu = x @ W_mu.t()          # (batch, hidden)
        logvar = x @ W_logvar.t()   # (batch, hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Tied weights: use W_mu^T
        W_mu = F.normalize(self.W_mu, p=2, dim=0) if self.normalize_weights else self.W_mu
        return F.relu(z @ W_mu + self.b)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


# ── Synthetic data generator ────────────────────────────────────
def generate_batch(batch_size, n_features, density, device="cpu"):
    mask = (torch.rand(batch_size, n_features, device=device) < density).float()
    values = torch.rand(batch_size, n_features, device=device)
    return mask * values


# ── VAE loss (reconstruction + KL) ──────────────────────────────
def vae_bottleneck_loss(x, x_hat, mu, logvar, importance, beta=1.0):
    recon = (importance * (x - x_hat) ** 2).sum(dim=1).mean()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon + beta * kl, recon, kl


# ── Training loop ───────────────────────────────────────────────
def train_vae_model(density, n_features=256, hidden=2, steps=8000,
                    batch_size=512, lr=1e-3, beta=0.05,
                    beta_warmup=2000,
                    importance=None, device="cpu", normalize_weights=False):
    model = ToyVAEBottleneck(n_features, hidden, normalize_weights=normalize_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if importance is None:
        importance = torch.ones(n_features, device=device)

    for step in range(steps):
        # KL warmup: linearly anneal beta from 0 to target
        beta_t = beta * min(1.0, step / max(beta_warmup, 1))
        x = generate_batch(batch_size, n_features, density, device=device)
        x_hat, mu, logvar, z = model(x)
        loss, recon, kl = vae_bottleneck_loss(x, x_hat, mu, logvar, importance, beta_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 2000 == 0:
            print(f"  density={density:.0%}  step={step+1}/{steps}  "
                  f"loss={loss.item():.4f}  recon={recon.item():.4f}  kl={kl.item():.4f}")

    # Collect z samples for plotting
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
        "logvar": logvar_eval.cpu().numpy(),
        "model": model,
    }


# ── Plotting helpers ────────────────────────────────────────────
def plot_arrows(ax, W, cmap, max_norm=None, label=None):
    """Plot columns of W (2×N) as arrows. If max_norm=None, uses per-panel max."""
    weights = W.T  # (N, 2)
    norms = np.linalg.norm(weights, axis=1)
    if max_norm is None:
        max_norm = norms.max()
    norm_n = norms / (max_norm + 1e-8)
    order = np.argsort(norms)
    for i in order:
        c = cmap(np.clip(norm_n[i], 0, 1))
        ax.annotate("", xy=(weights[i, 0], weights[i, 1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, alpha=0.7))
    # For legend
    if label:
        ax.plot([], [], color=cmap(0.7), lw=2, label=label)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_FEATURES = 256
    HIDDEN = 2
    STEPS = 8000
    BATCH_SIZE = 512
    BETA = 0.05  # KL weight (lower = more reconstruction-focused)

    densities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]

    importance_decay = 0.5
    importance = torch.tensor(
        [(1.0 - i / N_FEATURES) ** importance_decay for i in range(N_FEATURES)],
        device=device,
    )

    # ── Train for each density ───────────────────────────────────
    results = {}
    for d in densities:
        print(f"\n--- Training density = {d:.0%} ---")
        results[d] = train_vae_model(
            density=d, n_features=N_FEATURES, hidden=HIDDEN,
            steps=STEPS, batch_size=BATCH_SIZE, lr=1e-3,
            beta=BETA, beta_warmup=2000,
            importance=None, device=device,
        )
        max_mu = np.linalg.norm(results[d]["W_mu"], axis=0).max()
        max_lv = np.linalg.norm(results[d]["W_logvar"], axis=0).max()
        print(f"  Max ||w_mu|| = {max_mu:.4f},  Max ||w_logvar|| = {max_lv:.4f}")

    # ── Use per-panel scaling so every density is visible ─────────
    n = len(densities)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    def panel_lim(W):
        """Compute a nice axis limit for one panel's weight matrix."""
        norms = np.linalg.norm(W, axis=0)
        m = norms.max() if norms.max() > 1e-6 else 0.1
        return m * 1.25

    # ═════════════════════════════════════════════════════════════
    # Figure 1 — W_mu  weight arrows per density
    # ═════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes1 = axes1.flatten()
    for idx, d in enumerate(densities):
        ax = axes1[idx]
        W_mu = results[d]["W_mu"]
        plot_arrows(ax, W_mu, plt.cm.plasma, label="W_mu")  # per-panel norm
        lim = panel_lim(W_mu)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, lw=0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        norms = np.linalg.norm(W_mu, axis=0)
        n_active = (norms > 0.05 * norms.max()).sum()
        ax.set_title(f"Density = {d:.0%}  ({n_active} active, max={norms.max():.3f})",
                     fontsize=9)
    for idx in range(n, len(axes1)):
        axes1[idx].set_visible(False)
    fig1.suptitle("W_mu columns in 2D hidden space (mean weights)\n(per-panel scaling)",
                  fontsize=14, y=1.02)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════
    # Figure 2 — W_logvar  weight arrows per density
    # ═════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes2 = axes2.flatten()
    for idx, d in enumerate(densities):
        ax = axes2[idx]
        W_lv = results[d]["W_logvar"]
        plot_arrows(ax, W_lv, plt.cm.viridis, label="W_logvar")  # per-panel norm
        lim = panel_lim(W_lv)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, lw=0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        norms = np.linalg.norm(W_lv, axis=0)
        n_active = (norms > 0.05 * norms.max()).sum()
        ax.set_title(f"Density = {d:.0%}  ({n_active} active, max={norms.max():.3f})",
                     fontsize=9)
    for idx in range(n, len(axes2)):
        axes2[idx].set_visible(False)
    fig2.suptitle("W_logvar columns in 2D hidden space (log-variance weights)\n(per-panel scaling)",
                  fontsize=14, y=1.02)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════
    # Figure 3 — Sampled z in 2D hidden space per density
    # ═════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes3 = axes3.flatten()
    for idx, d in enumerate(densities):
        ax = axes3[idx]
        z = results[d]["z"]
        # Per-panel z limits
        z_max = max(np.abs(z).max() * 1.1, 0.5)
        ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.3, c="#6a0dad", edgecolors="none")
        ax.set_xlim(-z_max, z_max); ax.set_ylim(-z_max, z_max)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        ax.set_title(f"Density = {d:.0%}  (z range={z_max:.2f})", fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("z_2")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("z_1")
    for idx in range(n, len(axes3)):
        axes3[idx].set_visible(False)
    fig3.suptitle("Reparameterized z samples in 2D hidden space\n(per-panel scaling)",
                  fontsize=14, y=1.02)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════
    # Figure 4 — Combined overlay: W_mu arrows + W_logvar arrows + z scatter
    # ═════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows))
    axes4 = axes4.flatten()
    for idx, d in enumerate(densities):
        ax = axes4[idx]
        z = results[d]["z"]
        W_mu = results[d]["W_mu"]
        W_lv = results[d]["W_logvar"]

        # z scatter (background)
        ax.scatter(z[:, 0], z[:, 1], s=2, alpha=0.15, c="grey", edgecolors="none",
                   label="z samples", zorder=1)

        # W_mu arrows (plasma) — per-panel norm
        plot_arrows(ax, W_mu, plt.cm.plasma, label="W_mu")

        # W_logvar arrows (viridis) — per-panel norm
        plot_arrows(ax, W_lv, plt.cm.viridis, label="W_logvar")

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.15, lw=0.7)

        # Per-panel limits: encompass both weight arrows and z cloud
        w_max = max(panel_lim(W_mu), panel_lim(W_lv))
        z_max = np.abs(z).max() * 1.1 if len(z) > 0 else 0.5
        combined_lim = max(w_max, z_max, 1.15)
        ax.set_xlim(-combined_lim, combined_lim)
        ax.set_ylim(-combined_lim, combined_lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        ax.set_title(f"Density = {d:.0%}", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")
    for idx in range(n, len(axes4)):
        axes4[idx].set_visible(False)
    fig4.suptitle("Combined: W_mu (plasma) + W_logvar (viridis) + z samples (grey)",
                  fontsize=14, y=1.02)
    plt.tight_layout()

    SAVE_DIR = r"d:\Interpretability\plots"
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig1.savefig(os.path.join(SAVE_DIR, "exp2_w_mu_arrows.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(SAVE_DIR, "exp2_w_logvar_arrows.png"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(SAVE_DIR, "exp2_z_samples.png"), dpi=150, bbox_inches="tight")
    fig4.savefig(os.path.join(SAVE_DIR, "exp2_combined.png"), dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {SAVE_DIR}")

    plt.show()
