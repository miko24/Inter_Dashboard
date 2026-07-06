"""
Visualise the tied-weight VAE in the style of the Toy Model of Superposition.

The encoder uses W (shape 2×256) to compress 256 → 2, and the decoder
uses Wᵀ to decompress 2 → 256.  This lets us inspect:

  1. The 256 weight columns of W as 2-D arrows (feature directions)
  2. Wᵀ W  (256×256) — the interference / overlap matrix
  3. ‖wᵢ‖  — the norm of each feature's weight vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Model definition (must match the trained model) ──────────────
class VAE(nn.Module):
    def __init__(self, hidden=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, hidden)
        self.fc_logvar = nn.Linear(256, hidden)
        # Tied weights — decoder reuses fc_mu.weight transposed
        self.dec_bias = nn.Parameter(torch.zeros(256))
        self.fc3 = nn.Linear(256, 784)

    def encoder(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(F.linear(z, self.fc_mu.weight.t(), self.dec_bias))
        x_hat = torch.sigmoid(self.fc3(h))
        return x_hat.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


# ═══════════════════════════════════════════════════════════════════
# Plot 1 — Weight vectors as 2-D arrows
# ═══════════════════════════════════════════════════════════════════
def plot_superposition_weights(weights, title="Weight columns in 2D hidden space"):
    """
    Args:
        weights: (N, 2) array — each row is a feature's direction in
                 the 2-D hidden space.
    """
    norms = np.linalg.norm(weights, axis=1)
    norm_normalised = norms / (norms.max() + 1e-8)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(8, 8))

    order = np.argsort(norms)
    for i in order:
        color = cmap(norm_normalised[i])
        ax.annotate(
            "",
            xy=(weights[i, 0], weights[i, 1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=0.7),
        )

    # Reference circle
    theta = np.linspace(0, 2 * np.pi, 200)
    max_norm = norms.max()
    ax.plot(
        max_norm * np.cos(theta), max_norm * np.sin(theta),
        "k--", alpha=0.25, lw=1, label=f"r = {max_norm:.2f}",
    )

    ax.set_aspect("equal")
    lim = max_norm * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.4)
    ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel("Latent dim 1", fontsize=12)
    ax.set_ylabel("Latent dim 2", fontsize=12)
    ax.set_title(title, fontsize=14, pad=12)
    ax.legend(fontsize=10, loc="upper right")

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=norms.min(), vmax=norms.max()),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight-vector norm", fontsize=11)

    plt.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# Plot 2 — Wᵀ W  interference matrix
# ═══════════════════════════════════════════════════════════════════
def plot_interference_matrix(W, title="Wᵀ W  interference matrix"):
    """
    Args:
        W: (hidden, n_features) numpy array, e.g. shape (2, 256).
           Computes  Wᵀ W  of shape (n_features, n_features).
    """
    WtW = W.T @ W  # (256, 256)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1]})

    # ── Full matrix ──────────────────────────────────────────────
    ax = axes[0]
    vmax = np.abs(WtW).max()
    im = ax.imshow(WtW, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{title}\n(full {WtW.shape[0]}×{WtW.shape[1]})", fontsize=13)
    ax.set_xlabel("Feature j")
    ax.set_ylabel("Feature i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Off-diagonal only (interference) ─────────────────────────
    ax2 = axes[1]
    WtW_offdiag = WtW.copy()
    np.fill_diagonal(WtW_offdiag, 0)
    vmax2 = np.abs(WtW_offdiag).max()
    im2 = ax2.imshow(WtW_offdiag, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    ax2.set_title("Off-diagonal only (interference)", fontsize=13)
    ax2.set_xlabel("Feature j")
    ax2.set_ylabel("Feature i")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig, axes, WtW


# ═══════════════════════════════════════════════════════════════════
# Plot 3 — Per-feature norms  ‖wᵢ‖
# ═══════════════════════════════════════════════════════════════════
def plot_feature_norms(W, title="Per-feature weight norms ‖wᵢ‖"):
    """
    Args:
        W: (hidden, n_features) numpy array, e.g. shape (2, 256).
    """
    norms = np.linalg.norm(W, axis=0)  # one norm per feature column
    n = len(norms)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Bar chart (sorted) ───────────────────────────────────────
    ax = axes[0]
    sorted_idx = np.argsort(norms)[::-1]
    colors = plt.cm.plasma(norms[sorted_idx] / (norms.max() + 1e-8))
    ax.bar(range(n), norms[sorted_idx], color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Feature index (sorted by norm)", fontsize=12)
    ax.set_ylabel("‖wᵢ‖", fontsize=12)
    ax.set_title(f"{title}\n(sorted descending)", fontsize=13)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5, label="‖w‖ = 1")
    ax.legend()

    # ── Histogram ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(norms, bins=40, color="#6a0dad", edgecolor="white", alpha=0.85)
    ax2.set_xlabel("‖wᵢ‖", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of weight norms", fontsize=13)
    ax2.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.5, label="‖w‖ = 1")
    ax2.legend()

    plt.tight_layout()
    return fig, axes, norms


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(hidden=2).to(device)
    # vae.load_state_dict(torch.load("vae_checkpoint.pt", map_location=device))

    # ── Extract the shared weight matrix W ───────────────────────
    # fc_mu.weight has shape (hidden, 256) = (2, 256)
    # Each column wᵢ is a feature's 2-D direction
    W = vae.fc_mu.weight.detach().cpu().numpy()  # (2, 256)
    print(f"W shape (hidden × features): {W.shape}")
    print(f"Encoder:  h = W x      -- compresses 256 -> 2")
    print(f"Decoder:  x_hat = W^T h + b -- decompresses 2 -> 256  (tied weights)")

    # ── Plot 1: feature directions in 2-D ────────────────────────
    # Columns of W transposed → rows of shape (256, 2)
    W_cols = W.T  # (256, 2)
    fig1, ax1 = plot_superposition_weights(
        W_cols,
        title="W columns in 2D hidden space\n(256 features × 2D bottleneck)",
    )

    # ── Plot 2: Wᵀ W interference matrix ────────────────────────
    fig2, axes2, WtW = plot_interference_matrix(
        W,
        title="Wᵀ W  interference matrix",
    )
    print(f"\nWᵀW shape: {WtW.shape}")
    print(f"Diagonal  (self-overlap) — mean: {np.diag(WtW).mean():.4f}, "
          f"std: {np.diag(WtW).std():.4f}")
    offdiag = WtW[~np.eye(WtW.shape[0], dtype=bool)]
    print(f"Off-diag  (interference) — mean: {offdiag.mean():.4f}, "
          f"std: {offdiag.std():.4f}, max |val|: {np.abs(offdiag).max():.4f}")

    # ── Plot 3: per-feature norms ────────────────────────────────
    fig3, axes3, norms = plot_feature_norms(
        W,
        title="Per-feature weight norms ‖wᵢ‖",
    )
    print(f"\nNorms — min: {norms.min():.4f}, max: {norms.max():.4f}, "
          f"mean: {norms.mean():.4f}")

    plt.show()
