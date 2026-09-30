"""
Manifold VAE Toy Bottleneck with Riemannian latent space.

For each feature-density level (5% to 100%), train a manifold-aware VAE:

    v      = log_p(x)                — logarithmic map to tangent space
    a      = ReLU(W_tangent · v + b) — linear + activation in tangent space
    z      = exp_p(a)                — exponential map back to manifold
    x_hat  = ReLU(W_tangent^T · log_p(z) + b_dec)  — decode via tied weights

Supports two manifold geometries:
  - Poincaré ball (negative curvature, κ = -1/c²)
  - Stereographic sphere (positive curvature, κ = 1/c²)

Loss functions:
  1. Tangent space penalty
  2. Geodesic distance loss
  3. Riemannian KL divergence
  4. Riemannian score matching

Then plot per density:
  1. W_tangent columns in 2D (tangent-space weight directions)
  2. Manifold z samples on the Poincaré disk / sphere projection
  3. Pairwise geodesic distance matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore', r'All-NaN slice encountered')


# ═══════════════════════════════════════════════════════════════════
# Riemannian Primitives — Poincaré Ball Model
# ═══════════════════════════════════════════════════════════════════

def poincare_conformal_factor(x, c=1.0):
    """λ_x = 2 / (1 - c·‖x‖²).  Shape-preserving."""
    return 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)


def mobius_add(x, y, c=1.0):
    """Möbius addition x ⊕_c y in the Poincaré ball."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * y_sq) * x + (1.0 - c * x_sq) * y
    denom = 1.0 + 2.0 * c * xy + c * c * x_sq * y_sq
    return num / denom.clamp(min=1e-6)


def poincare_log_map(x, p, c=1.0):
    """Logarithmic map: project point x onto tangent space at p."""
    neg_p = -p
    add_result = mobius_add(neg_p, x, c)
    norm = add_result.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    lam = poincare_conformal_factor(p, c)
    return (2.0 / (torch.sqrt(torch.tensor(c, device=x.device)) * lam)) * torch.arctanh(
        (torch.sqrt(torch.tensor(c, device=x.device)) * norm).clamp(max=1.0 - 1e-5)
    ) * (add_result / norm)


def poincare_exp_map(v, p, c=1.0):
    """Exponential map: map tangent vector v at p back to manifold."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    lam = poincare_conformal_factor(p, c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    second = torch.tanh(sqrt_c * lam * v_norm / 2.0) * (v / v_norm) / sqrt_c
    return mobius_add(p, second, c)


def poincare_log_map_origin(x, c=1.0):
    """Logarithmic map at origin (simplified): log_0(x)."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.arctanh(
        (sqrt_c * x_norm).clamp(max=1.0 - 1e-5)
    ) * (x / x_norm)


def poincare_exp_map_origin(v, c=1.0):
    """Exponential map at origin (simplified): exp_0(v)."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(sqrt_c * v_norm / 2.0) * (v / v_norm) / sqrt_c


def poincare_geodesic_distance(x, y, c=1.0):
    """Geodesic distance d(x, y) on the Poincaré ball."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.arctanh(
        (sqrt_c * diff_norm).clamp(max=1.0 - 1e-5)
    )


def project_to_poincare_ball(x, c=1.0, eps=1e-4):
    """Clip points to stay inside the Poincaré ball ‖x‖ < 1/√c - eps."""
    max_norm = 1.0 / (c ** 0.5) - eps
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    cond = norms > max_norm
    return torch.where(cond, x * (max_norm / norms), x)


# ═══════════════════════════════════════════════════════════════════
# Riemannian Primitives — Stereographic Sphere Model
# ═══════════════════════════════════════════════════════════════════

def sphere_conformal_factor(x, c=1.0):
    """λ_x = 2 / (1 + c·‖x‖²) for the stereographic sphere."""
    return 2.0 / (1.0 + c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-6)


def sphere_exp_map_origin(v, c=1.0):
    """Exponential map at origin for stereographic sphere."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=v.device))
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tan(sqrt_c * v_norm / 2.0) * (v / v_norm) / sqrt_c


def sphere_log_map_origin(x, c=1.0):
    """Logarithmic map at origin for stereographic sphere."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.atan(sqrt_c * x_norm) * (x / x_norm)


def sphere_geodesic_distance(x, y, c=1.0):
    """Geodesic distance on the stereographic sphere."""
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    lam_x = sphere_conformal_factor(x, c)
    lam_y = sphere_conformal_factor(y, c)
    diff_norm = (x - y).norm(dim=-1).clamp(min=1e-7)
    return (2.0 / sqrt_c) * torch.asin(
        (sqrt_c * lam_x.squeeze(-1) * lam_y.squeeze(-1) * diff_norm / 4.0).clamp(max=1.0 - 1e-5)
    )


# ═══════════════════════════════════════════════════════════════════
# Manifold VAE Bottleneck Model
# ═══════════════════════════════════════════════════════════════════

class ToyManifoldVAEBottleneck(nn.Module):
    """
    Manifold VAE bottleneck with Riemannian geometry.

    Pipeline:
        x → log_map(x) → v (tangent space)
        v → W_tangent · v + b → ReLU → a (activations in tangent space)
        a → exp_map(a) → z (curved manifold point)
        z → log_map(z) → W_tangent^T · (...) + b_dec → x_hat

    Supports Poincaré ball (negative curvature) and stereographic sphere
    (positive curvature).
    """

    def __init__(self, n_features=256, hidden=2, curvature=1.0,
                 manifold_type="poincare", normalize_weights=False):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.curvature = curvature
        self.manifold_type = manifold_type  # "poincare" or "sphere"
        self.normalize_weights = normalize_weights

        self.W_tangent = nn.Parameter(torch.randn(hidden, n_features) * 0.05)
        self.b_enc = nn.Parameter(torch.zeros(hidden))
        self.b_dec = nn.Parameter(torch.zeros(n_features))

        # For VAE: mu and logvar heads in tangent space
        self.W_mu = nn.Parameter(torch.randn(hidden, n_features) * 0.05)
        self.W_logvar = nn.Parameter(torch.randn(hidden, n_features) * 0.05)

    def _get_ops(self):
        """Return the appropriate manifold operations."""
        if self.manifold_type == "sphere":
            return (sphere_log_map_origin, sphere_exp_map_origin,
                    sphere_geodesic_distance, sphere_conformal_factor)
        else:  # poincare
            return (poincare_log_map_origin, poincare_exp_map_origin,
                    poincare_geodesic_distance, poincare_conformal_factor)

    def _get_W(self):
        W = F.normalize(self.W_tangent, p=2, dim=0) if self.normalize_weights else self.W_tangent
        return W

    def encode(self, x):
        """Encode input to tangent space, then to manifold for mu/logvar."""
        log_map, exp_map, _, _ = self._get_ops()
        c = self.curvature
        W = self._get_W()

        # Project input into tangent space at origin
        # x is in R^n, we treat it as flat and project via W
        # Tangent space activations
        v = x @ W.t() + self.b_enc  # (batch, hidden) — tangent vector

        # ReLU activation in tangent space
        a = F.relu(v)

        # Compute mu in tangent space, then map to manifold
        mu_tangent = a
        logvar_tangent = torch.clamp(x @ self.W_logvar.t(), min=-10.0, max=10.0)

        # Map mu to manifold
        mu = exp_map(mu_tangent, c)
        if self.manifold_type == "poincare":
            mu = project_to_poincare_ball(mu, c)

        return mu, logvar_tangent, a

    def reparameterize(self, mu, logvar_tangent):
        """Reparameterize on the manifold using wrapped normal."""
        log_map, exp_map, _, _ = self._get_ops()
        c = self.curvature

        std = torch.exp(0.5 * logvar_tangent)
        eps = torch.randn_like(std)
        # Sample in tangent space at mu, then map to manifold
        v = eps * std  # tangent vector at origin

        # Transport to manifold via exp map at origin, then Möbius add with mu
        if self.manifold_type == "poincare":
            z_sample = poincare_exp_map_origin(v, c)
            z = mobius_add(mu, z_sample, c)
            z = project_to_poincare_ball(z, c)
        else:
            z_sample = sphere_exp_map_origin(v, c)
            z = z_sample + mu  # simplified for sphere
        return z

    def decode(self, z):
        """Decode from manifold point back to input space."""
        log_map, exp_map, _, _ = self._get_ops()
        c = self.curvature
        W = self._get_W()

        # Log map to get tangent vector at origin
        v_z = log_map(z, c)

        # Linear decode with tied weights
        x_hat = F.relu(v_z @ W + self.b_dec)
        return x_hat

    def forward(self, x):
        mu, logvar, tangent_activations = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z, tangent_activations


# ═══════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════

def tangent_space_penalty(tangent_activations, c=1.0):
    """Penalize large tangent vectors (keeps points near origin on manifold)."""
    norms_sq = (tangent_activations * tangent_activations).sum(dim=-1)
    return norms_sq.mean()


def geodesic_distance_loss(x_hat, x, W_tangent, importance, c=1.0,
                           manifold_type="poincare"):
    """Reconstruction loss using geodesic distance in input space."""
    # Standard reconstruction (we use Euclidean in input space, geodesic in latent)
    return (importance * (x - x_hat) ** 2).sum(dim=1).mean()


def riemannian_kl_divergence(mu, logvar, c=1.0, manifold_type="poincare"):
    """
    KL divergence for wrapped normal on manifold.
    Correction: KL_riemann = KL_euclidean + log(λ_μ) correction.
    """
    if manifold_type == "poincare":
        lam = poincare_conformal_factor(mu, c)
    else:
        lam = sphere_conformal_factor(mu, c)

    # Standard KL
    kl_euclidean = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    # Riemannian correction: account for volume element change
    hidden = mu.shape[-1]
    log_lam = torch.log(lam.squeeze(-1).clamp(min=1e-8))
    correction = hidden * log_lam

    return (kl_euclidean + correction).mean()


def riemannian_score_matching(z, mu, logvar, c=1.0, manifold_type="poincare"):
    """
    Implicit Riemannian score matching.
    Approximation: ‖∇ log p(z) - score‖²_g
    Uses the conformal factor to weight the score difference.
    """
    if manifold_type == "poincare":
        lam = poincare_conformal_factor(z, c)
    else:
        lam = sphere_conformal_factor(z, c)

    # Score of the approximate posterior q(z|x) = N(mu, diag(exp(logvar)))
    std = torch.exp(0.5 * logvar)
    score_q = -(z - mu) / (std.pow(2) + 1e-8)

    # Score of the prior p(z) = N(0, I) at origin
    score_prior = -z

    # Riemannian norm: ‖v‖²_g = ‖v‖² / λ²
    diff = score_q - score_prior
    riemann_norm_sq = (diff * diff).sum(dim=-1) / (lam.squeeze(-1).pow(2) + 1e-8)

    return riemann_norm_sq.mean()


# ═══════════════════════════════════════════════════════════════════
# Synthetic Data & Training
# ═══════════════════════════════════════════════════════════════════

def generate_batch(batch_size, n_features, density, device="cpu"):
    mask = (torch.rand(batch_size, n_features, device=device) < density).float()
    values = torch.rand(batch_size, n_features, device=device)
    return mask * values


def train_manifold_model(density, n_features=256, hidden=2, steps=8000,
                         batch_size=512, lr=1e-3, curvature=1.0,
                         manifold_type="poincare",
                         loss_type="geodesic",
                         beta=0.05, beta_warmup=2000,
                         tangent_weight=0.01,
                         importance=None, device="cpu",
                         normalize_weights=False):
    model = ToyManifoldVAEBottleneck(
        n_features, hidden, curvature=curvature,
        manifold_type=manifold_type,
        normalize_weights=normalize_weights,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if importance is None:
        importance = torch.ones(n_features, device=device)

    c = curvature

    for step in range(steps):
        beta_t = beta * min(1.0, step / max(beta_warmup, 1))
        x = generate_batch(batch_size, n_features, density, device=device)
        x_hat, mu, logvar, z, tangent_act = model(x)

        # Reconstruction
        recon = geodesic_distance_loss(x_hat, x, model.W_tangent, importance,
                                       c, manifold_type)

        # Select manifold loss
        if loss_type == "tangent_penalty":
            manifold_loss = tangent_space_penalty(tangent_act, c)
            loss = recon + tangent_weight * manifold_loss
        elif loss_type == "geodesic":
            kl = riemannian_kl_divergence(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl
        elif loss_type == "riemannian_kl":
            kl = riemannian_kl_divergence(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl
        elif loss_type == "score_matching":
            score_loss = riemannian_score_matching(z, mu, logvar, c, manifold_type)
            kl = riemannian_kl_divergence(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl + 0.1 * score_loss
        else:
            kl = riemannian_kl_divergence(mu, logvar, c, manifold_type)
            loss = recon + beta_t * kl

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability with manifold ops
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if (step + 1) % 2000 == 0:
            print(f"  density={density:.0%}  step={step+1}/{steps}  "
                  f"loss={loss.item():.4f}  recon={recon.item():.4f}")

    # Collect eval samples
    with torch.no_grad():
        x_eval = generate_batch(2048, n_features, density, device=device)
        x_hat_eval, mu_eval, logvar_eval, z_eval, tang_eval = model(x_eval)

    if model.normalize_weights:
        W_tangent = F.normalize(model.W_tangent, p=2, dim=0).detach().cpu().numpy()
    else:
        W_tangent = model.W_tangent.detach().cpu().numpy()

    W_mu = model.W_mu.detach().cpu().numpy()
    W_logvar = model.W_logvar.detach().cpu().numpy()

    return {
        "W_tangent": W_tangent,
        "W_mu": W_mu,
        "W_logvar": W_logvar,
        "z": z_eval.cpu().numpy(),
        "mu": mu_eval.cpu().numpy(),
        "logvar": logvar_eval.cpu().numpy(),
        "tangent_activations": tang_eval.cpu().numpy(),
        "model": model,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════════

def plot_arrows(ax, W, cmap, max_norm=None, label=None):
    """Plot columns of W (2×N) as arrows."""
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
    if label:
        ax.plot([], [], color=cmap(0.7), lw=2, label=label)


def plot_manifold_disk(ax, z, manifold_type="poincare", c=1.0):
    """Plot z samples on the Poincaré disk or stereographic sphere."""
    theta = np.linspace(0, 2 * np.pi, 200)

    if manifold_type == "poincare":
        # Boundary is at radius 1/√c
        r = 1.0 / (c ** 0.5)
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                "w-", alpha=0.3, lw=1.5, label=f"Ball boundary (r={r:.2f})")
        ax.fill(r * np.cos(theta), r * np.sin(theta),
                color="#1a0a30", alpha=0.3)
    else:
        # Sphere: no strict boundary, but show unit circle reference
        ax.plot(np.cos(theta), np.sin(theta),
                "w--", alpha=0.3, lw=1.0, label="Unit circle")

    # Plot z samples
    ax.scatter(z[:, 0], z[:, 1], s=3, alpha=0.4, c="#a78bfa", edgecolors="none",
               zorder=3)

    # Geodesic grid lines (concentric circles at equal geodesic distance)
    if manifold_type == "poincare":
        for d_geo in [0.5, 1.0, 1.5, 2.0]:
            r_eucl = np.tanh(d_geo * c**0.5 / 2.0) / c**0.5
            if r_eucl < r:
                ax.plot(r_eucl * np.cos(theta), r_eucl * np.sin(theta),
                        color="#5e577a", ls=":", lw=0.5, alpha=0.4)


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
    BETA = 0.05
    CURVATURE = 1.0
    MANIFOLD_TYPE = "poincare"  # "poincare" or "sphere"
    LOSS_TYPE = "riemannian_kl"  # tangent_penalty, geodesic, riemannian_kl, score_matching

    densities = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00]

    importance_decay = 0.5
    importance = torch.tensor(
        [(1.0 - i / N_FEATURES) ** importance_decay for i in range(N_FEATURES)],
        device=device,
    )

    # ── Train for each density ───────────────────────────────────
    results = {}
    for d in densities:
        print(f"\n--- Training density = {d:.0%} ({MANIFOLD_TYPE}, {LOSS_TYPE}) ---")
        results[d] = train_manifold_model(
            density=d, n_features=N_FEATURES, hidden=HIDDEN,
            steps=STEPS, batch_size=BATCH_SIZE, lr=1e-3,
            curvature=CURVATURE, manifold_type=MANIFOLD_TYPE,
            loss_type=LOSS_TYPE,
            beta=BETA, beta_warmup=2000,
            importance=None, device=device,
        )
        max_wt = np.linalg.norm(results[d]["W_tangent"], axis=0).max()
        print(f"  Max ||w_tangent|| = {max_wt:.4f}")

    # ── Per-panel scaling ────────────────────────────────────────
    n = len(densities)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    def panel_lim(W):
        try:
            norms = np.linalg.norm(W, axis=0)
            m = np.nanmax(norms)
            if np.isnan(m) or np.isinf(m):
                m = 0.1
        except ValueError:
            m = 0.1
        return max(m, 0.1) * 1.25

    # ═════════════════════════════════════════════════════════════
    # Figure 1 — W_tangent weight arrows per density
    # ═════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes1 = axes1.flatten()
    for idx, d in enumerate(densities):
        ax = axes1[idx]
        W_t = results[d]["W_tangent"]
        plot_arrows(ax, W_t, plt.cm.plasma, label="W_tangent")
        lim = panel_lim(W_t)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2, lw=0.7)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        norms = np.linalg.norm(W_t, axis=0)
        n_active = (norms > 0.05 * norms.max()).sum()
        ax.set_title(f"Density = {d:.0%}  ({n_active} active, max={norms.max():.3f})",
                     fontsize=9)
    for idx in range(n, len(axes1)):
        axes1[idx].set_visible(False)
    fig1.suptitle(f"W_tangent columns in 2D tangent space ({MANIFOLD_TYPE}, {LOSS_TYPE})",
                  fontsize=14, y=1.02)
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════
    # Figure 2 — Manifold z samples on Poincaré disk / sphere
    # ═════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                                facecolor="#0a0a18")
    axes2 = axes2.flatten()
    for idx, d in enumerate(densities):
        ax = axes2[idx]
        ax.set_facecolor("#0a0a18")
        z = results[d]["z"]
        plot_manifold_disk(ax, z, MANIFOLD_TYPE, CURVATURE)
        try:
            z_abs_max = np.nanmax(np.abs(z))
            if np.isnan(z_abs_max) or np.isinf(z_abs_max):
                z_abs_max = 0.0
        except ValueError:
            z_abs_max = 0.0
            
        z_max = max(z_abs_max * 1.15, 1.15 / CURVATURE**0.5)
        if np.isnan(z_max) or np.isinf(z_max):
            z_max = 1.0
            
        ax.set_xlim(-z_max, z_max); ax.set_ylim(-z_max, z_max)
        ax.set_aspect("equal")
        ax.axhline(0, color="#5e577a", lw=0.3); ax.axvline(0, color="#5e577a", lw=0.3)
        ax.set_title(f"Density = {d:.0%}", fontsize=9, color="white")
        ax.tick_params(colors="#9892b3")
        for spine in ax.spines.values():
            spine.set_color("#2a2554")
    for idx in range(n, len(axes2)):
        axes2[idx].set_visible(False)
    manifold_label = "Poincaré Disk" if MANIFOLD_TYPE == "poincare" else "Stereographic Sphere"
    fig2.suptitle(f"z samples on {manifold_label} (c={CURVATURE})",
                  fontsize=14, y=1.02, color="white")
    plt.tight_layout()

    # ═════════════════════════════════════════════════════════════
    # Figure 3 — Combined: W_tangent arrows + z scatter on manifold
    # ═════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows),
                                facecolor="#0a0a18")
    axes3 = axes3.flatten()
    for idx, d in enumerate(densities):
        ax = axes3[idx]
        ax.set_facecolor("#0a0a18")
        z = results[d]["z"]
        W_t = results[d]["W_tangent"]

        plot_manifold_disk(ax, z, MANIFOLD_TYPE, CURVATURE)
        plot_arrows(ax, W_t, plt.cm.plasma, label="W_tangent")

        w_max = panel_lim(W_t)
        
        try:
            z_abs_max = np.nanmax(np.abs(z))
            if np.isnan(z_abs_max) or np.isinf(z_abs_max):
                z_abs_max = 0.0
        except ValueError:
            z_abs_max = 0.0
            
        z_max = max(z_abs_max * 1.1, 0.5)
        combined_lim = max(w_max, z_max, 1.15 / CURVATURE**0.5)
        if np.isnan(combined_lim) or np.isinf(combined_lim):
            combined_lim = 1.0
            
        ax.set_xlim(-combined_lim, combined_lim)
        ax.set_ylim(-combined_lim, combined_lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="#5e577a", lw=0.3); ax.axvline(0, color="#5e577a", lw=0.3)
        ax.set_title(f"Density = {d:.0%}", fontsize=10, color="white")
        ax.tick_params(colors="#9892b3")
        for spine in ax.spines.values():
            spine.set_color("#2a2554")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right",
                      facecolor="#0a0a18", edgecolor="#2a2554", labelcolor="white")
    for idx in range(n, len(axes3)):
        axes3[idx].set_visible(False)
    fig3.suptitle(f"Combined: W_tangent (plasma) + z samples ({manifold_label})",
                  fontsize=14, y=1.02, color="white")
    plt.tight_layout()

    # ── Save plots ──────────────────────────────────────────────
    SAVE_DIR = r"d:\Interpretability\plots"
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig1.savefig(os.path.join(SAVE_DIR, f"exp3_manifold_w_tangent_{MANIFOLD_TYPE}.png"),
                 dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(SAVE_DIR, f"exp3_manifold_z_samples_{MANIFOLD_TYPE}.png"),
                 dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    fig3.savefig(os.path.join(SAVE_DIR, f"exp3_manifold_combined_{MANIFOLD_TYPE}.png"),
                 dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
    print(f"\nPlots saved to {SAVE_DIR}")

    plt.show()
