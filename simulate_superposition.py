"""
Toy Model of Superposition simulation for the 256 -> 2 bottleneck.

For each feature-density level (5% to 100%), we train a simple
tied-weight linear model:

    x_hat = ReLU(W^T W x + b)

where W has shape (2, 256), and plot the 256 weight columns as
2-D arrows to see how superposition emerges as density increases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Toy bottleneck model (tied weights) ──────────────────────────
class ToyBottleneck(nn.Module):
    """
    256 features compressed to `hidden` dims and decompressed with
    tied weights, exactly as in Elhage et al. (2022).
    """

    def __init__(self, n_features=256, hidden=2):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.W = nn.Parameter(torch.randn(hidden, n_features) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        # Encode:  h = W x          (batch, hidden)
        h = x @ self.W.t()
        # Decode:  x_hat = ReLU(W^T h + b)   (batch, n_features)
        x_hat = F.relu(h @ self.W + self.b)
        return x_hat


# ── Synthetic data generator ────────────────────────────────────
def generate_batch(batch_size, n_features, density, importance=None, device="cpu"):
    """
    Generate a batch of sparse feature vectors.

    Args:
        density:    fraction of features active per sample (0..1)
        importance: (n_features,) tensor of per-feature importance weights.
                    If None, all features are equally important.
    """
    # Each feature is independently active with probability `density`
    mask = (torch.rand(batch_size, n_features, device=device) < density).float()
    # Active features have values ~ U(0, 1)
    values = torch.rand(batch_size, n_features, device=device)
    x = mask * values
    return x


# ── Training loop for one density level ─────────────────────────
def train_model(density, n_features=256, hidden=2, steps=5000,
                batch_size=512, lr=1e-3, importance=None, device="cpu"):
    """Train and return the weight matrix W for a given density."""
    model = ToyBottleneck(n_features, hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if importance is None:
        # Decaying importance like in the paper: importance_i = (1 - i/n) ^ decay
        # Using uniform importance is also valid
        importance = torch.ones(n_features, device=device)

    for step in range(steps):
        x = generate_batch(batch_size, n_features, density,
                           importance=importance, device=device)
        x_hat = model(x)
        # Weighted MSE loss
        loss = (importance * (x - x_hat) ** 2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"  density={density:.0%}  step={step+1}/{steps}  loss={loss.item():.6f}")

    W = model.W.detach().cpu().numpy()
    return W, model


# ── Plot a single panel of weight arrows ─────────────────────────
def plot_weights_on_ax(ax, W, density, max_global_norm=None):
    """
    Plot columns of W (shape 2×N) as 2-D arrows on a given axis.
    """
    weights = W.T  # (N, 2)
    norms = np.linalg.norm(weights, axis=1)
    cmap = plt.cm.plasma

    if max_global_norm is None:
        max_global_norm = norms.max()

    norm_normalised = norms / (max_global_norm + 1e-8)

    order = np.argsort(norms)
    for i in order:
        color = cmap(np.clip(norm_normalised[i], 0, 1))
        ax.annotate(
            "",
            xy=(weights[i, 0], weights[i, 1]),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, alpha=0.7),
        )

    # Unit circle reference
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, lw=0.7)

    ax.set_aspect("equal")
    lim = max(max_global_norm * 1.15, 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", lw=0.3)
    ax.axvline(0, color="grey", lw=0.3)

    # Count "active" features (norm > 0.1)
    n_active = (norms > 0.1).sum()
    ax.set_title(f"Density = {density:.0%}\n({n_active} active features)",
                 fontsize=10, pad=6)


# ═══════════════════════════════════════════════════════════════════
# Main — sweep over density levels
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_FEATURES = 256
    HIDDEN = 2
    STEPS = 5000
    BATCH_SIZE = 512

    # Density levels: 5% to 100% in steps of ~10%
    densities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]

    # Optional: decaying feature importance (like the paper)
    # Set to None for uniform importance
    importance_decay = 0.5
    importance = torch.tensor(
        [(1.0 - i / N_FEATURES) ** importance_decay for i in range(N_FEATURES)],
        device=device,
    )

    # ── Train for each density ───────────────────────────────────
    results = {}
    for d in densities:
        print(f"\n--- Training density = {d:.0%} ---")
        W, model = train_model(
            density=d,
            n_features=N_FEATURES,
            hidden=HIDDEN,
            steps=STEPS,
            batch_size=BATCH_SIZE,
            lr=1e-3,
            importance=importance,
            device=device,
        )
        results[d] = W
        print(f"  Done. Max norm = {np.linalg.norm(W, axis=0).max():.4f}")

    # ── Compute global max norm for consistent scaling ───────────
    global_max = max(np.linalg.norm(W, axis=0).max() for W in results.values())

    # ── Plot grid ────────────────────────────────────────────────
    n = len(densities)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, d in enumerate(densities):
        plot_weights_on_ax(axes[idx], results[d], d, max_global_norm=global_max)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Superposition in a {N_FEATURES} -> {HIDDEN} -> {N_FEATURES} bottleneck\n"
        f"Weight columns in 2D hidden space across feature densities",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    # ── Also plot norms across densities ─────────────────────────
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes2 = axes2.flatten()

    for idx, d in enumerate(densities):
        W = results[d]
        norms = np.linalg.norm(W, axis=0)
        sorted_norms = np.sort(norms)[::-1]
        colors = plt.cm.plasma(sorted_norms / (global_max + 1e-8))
        axes2[idx].bar(range(N_FEATURES), sorted_norms, color=colors,
                       width=1.0, edgecolor="none")
        axes2[idx].set_ylim(0, global_max * 1.1)
        axes2[idx].set_title(f"Density = {d:.0%}", fontsize=10)
        axes2[idx].axhline(1.0, color="grey", ls="--", lw=0.6, alpha=0.5)
        if idx % ncols == 0:
            axes2[idx].set_ylabel("||w_i||")

    for idx in range(n, len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle(
        "Per-feature weight norms (sorted) across densities",
        fontsize=14, y=1.02,
    )
    SAVE_DIR = r"d:\Interpretability\plots"
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig.savefig(os.path.join(SAVE_DIR, "exp1_weight_arrows.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(SAVE_DIR, "exp1_weight_norms.png"), dpi=150, bbox_inches="tight")
    print(f"\nPlots saved to {SAVE_DIR}")

    plt.show()
