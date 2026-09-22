"""Mathematical explanations shown beside Geometry Lab analyses."""

CONCEPTS = {
    "experiment_identity": {
        "title": "Reproducible experiment identity",
        "formula": "run = (paper, citation, data, model, optimizer, seed, code version)",
        "meaning": "A result is reproducible only when its scientific provenance and every stochastic/configuration choice are recorded together.",
        "interpretation": "Use the paper and run labels to distinguish an attempted reproduction from a new ablation. Compare matching seeds before pooling runs.",
    },
    "full_covariance": {
        "title": "Full-covariance Gaussian posterior",
        "formula": "q(z|x)=N(mu(x), L(x)L(x)^T),  KL=1/2[tr(Sigma)+mu^Tmu-d-log det Sigma]",
        "meaning": "Unlike a diagonal posterior, a learned Cholesky factor L can represent conditional correlations between latent coordinates.",
        "interpretation": "Off-diagonal covariance can model locally rotated uncertainty, but may also hide coordinate entanglement. Inspect precision together with decoder geometry.",
    },
    "decoder_jacobian": {
        "title": "Decoder Jacobian",
        "formula": "J(z)=partial D(z)/partial z",
        "meaning": "Column j is the instantaneous change in observation space caused by moving latent coordinate z_j.",
        "interpretation": "Large column norms indicate locally influential latent directions; near-zero columns indicate locally inactive dimensions.",
    },
    "pullback_metric": {
        "title": "Decoder pullback metric",
        "formula": "G(z)=J(z)^T J(z)",
        "meaning": "G transfers Euclidean observation-space lengths back to latent space: ds_x^2 = dz^T G dz.",
        "interpretation": "Diagonal entries are squared decoder sensitivities. Off-diagonal entries show directions whose decoded effects are not orthogonal.",
    },
    "normalized_metric": {
        "title": "Normalized pullback metric",
        "formula": "Gbar_ij = G_ij / sqrt(G_ii G_jj)",
        "meaning": "Normalization removes scale so entries measure the cosine between decoder-Jacobian columns.",
        "interpretation": "An identity-like matrix indicates locally orthogonal decoder directions. Strong off-diagonals indicate local geometric interference.",
    },
    "decoder_svd": {
        "title": "Decoder Jacobian SVD",
        "formula": "J = U diag(sigma) V^T",
        "meaning": "V gives orthogonal latent directions, U gives the corresponding observation-space change patterns, and sigma gives their magnification.",
        "interpretation": "A rapidly decaying spectrum suggests an effectively low-rank decoder. Stable, nonzero singular values indicate locally utilized directions.",
    },
    "absolute_v": {
        "title": "Absolute right-singular vectors",
        "formula": "|V|_ij = absolute contribution of latent coordinate i to singular direction j",
        "meaning": "The heatmap shows whether geometric directions align with individual coordinates or are distributed across them.",
        "interpretation": "Permutation-like structure is coordinate-aligned; diffuse columns indicate rotated or shared latent directions. Rotation alone is not necessarily harmful.",
    },
    "dto": {
        "title": "Paper Distance to Orthogonality (DtO)",
        "formula": "DtO=(1/N) sum_i ||V_i-P(V_i)||_F,  J_i=U_i Sigma_i V_i^T",
        "meaning": "The paper's Equation 29 compares every decoder right-singular-vector matrix with its nearest signed permutation matrix.",
        "interpretation": "Zero means the decoder singular directions are axis-aligned up to sign and permutation. The separate Gram off-diagonal statistic is a dashboard extension, not this metric.",
    },
    "precision_jacobian": {
        "title": "Posterior precision vs decoder Jacobian norm",
        "formula": "precision_j = (Sigma(x)^-1)_jj,  sensitivity_j = ||J_:j(z)||_2",
        "meaning": "Precision measures encoder certainty while sensitivity measures how strongly the decoder uses the same latent direction.",
        "interpretation": "High precision/high sensitivity dimensions are active and committed. Low precision/low sensitivity dimensions are inactive. Mixed quadrants reveal mismatch or transition regimes; posterior precision is not defined for deterministic autoencoders.",
    },
    "polarized_regime": {
        "title": "Polarized regime",
        "formula": "active_j := precision_j > tau_p and ||J_:j|| > tau_J",
        "meaning": "A VAE can polarize latent dimensions into active, informative coordinates and inactive coordinates close to the prior.",
        "interpretation": "Look for two separated populations, not merely a favorable mean. Thresholds are displayed and derived robustly from the current run.",
    },
    "degeneracy": {
        "title": "Synthetic degeneracy experiment",
        "formula": "A_epsilon = U diag(1,...,epsilon) V^T",
        "meaning": "The generator deliberately collapses one or more observation directions while holding factors and training protocol fixed.",
        "interpretation": "Track whether decoder rank, condition number, alignment metrics, reconstruction, and factor recovery fail smoothly or undergo a sharp transition as epsilon approaches zero.",
    },
    "disentanglement": {
        "title": "Representation disentanglement",
        "formula": "factor recoverability + low cross-factor tangent/causal overlap",
        "meaning": "No single scalar fully establishes disentanglement; recovery, geometry and interventions test complementary claims.",
        "interpretation": "High recoverability with high tangent overlap is a plausible successful-superposition regime, not classical axis alignment.",
    },
    "active_dimensions": {
        "title": "Active latent dimensions",
        "formula": "active_j := Var_x[mu_j(x)] > tau",
        "meaning": "A latent coordinate is active when its posterior mean varies materially across the dataset.",
        "interpretation": "Compare active dimensions with intrinsic dimension and decoder rank. The numerical threshold should be treated as a diagnostic convention.",
    },
    "latent_traversal": {
        "title": "Latent traversal",
        "formula": "x(alpha)=D(z+alpha e_j) or D(z+alpha v_i)",
        "meaning": "A traversal visualizes the decoded effect of a controlled latent intervention.",
        "interpretation": "Smooth semantic change is useful evidence; causal leakage into unrelated factors is measured separately by the interference matrix.",
    },
    "nonlinear_eigenfaces": {
        "title": "CelebA nonlinear eigenfaces",
        "formula": "u_j(z) = left singular vector j of J_D(z)",
        "meaning": "PCA eigenfaces are global linear directions; decoder-Jacobian left singular vectors are local nonlinear image-space directions around one face.",
        "interpretation": "The sign is arbitrary. Compare magnitude patterns and stability across nearby faces; do not interpret a single local direction as a globally consistent attribute.",
    },
}
