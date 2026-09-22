/**
 * VAE Superposition Dashboard — Client-Side Logic
 *
 * Handles: parameter collection, API calls to Flask backend,
 * plot rendering, tab management, and report generation.
 */

const API_BASE = "";

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════
const state = {
  modelType: "vae",
  sweepMode: "single",
  isTraining: false,
  trainedOnce: false,
  availablePlots: [],
  currentPlot: null,
  stats: {},
  losses: [],
  plotCache: {},   // cache base64 images by plot_type
  customLossExpr: null, // active custom loss expression, if any
};

// ═══════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  initModelTypeToggle();
  initRangeInputs();
  initImportancePreview();
  setupTrainButton();
  setupReportButton();
  setupPlotTabs();
  updateConditionalFields();
  initModeTabs();
  updateSweepPlotOptions();
  initLossEditor();

  fetchPastExperiments();
  $("#btn-refresh-experiments")?.addEventListener("click", fetchPastExperiments);

  // Export sweep action listener
  $("#btn-export-sweep")?.addEventListener("click", () => {
    if (state.lastSweepData) {
      const activePlot = state.activeSweepPlot;
      
      const parameters = {
        "Model Type": state.modelType.toUpperCase(),
        "Weight Initialization": $("#init-method")?.value.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) || "Xavier Normal",
        "Density Sweep Range": `${parseFloat($("#sweep-density-start")?.value)} to ${parseFloat($("#sweep-density-end")?.value)} (step ${parseFloat($("#sweep-density-step")?.value)})`,
        "Decay Sweep Range": `${parseFloat($("#sweep-decay-start")?.value)} to ${parseFloat($("#sweep-decay-end")?.value)} (step ${parseFloat($("#sweep-decay-step")?.value)})`,
        "Training Steps per Model": parseInt($("#sweep-steps")?.value ?? 1000),
        "Learning Rate": parseFloat($("#learning-rate")?.value ?? 0.001),
        "Batch Size": parseInt($("#batch-size")?.value ?? 512),
        "Grid Plot Type": PLOT_LABELS[activePlot] || activePlot
      };
      
      const grid = state.lastSweepData.grid.map(cell => ({
          features: cell.features,
          hidden: cell.hidden,
          run_id: cell.run_id,
          density: cell.density,
          importance_decay: cell.importance_decay,
          final_loss: cell.final_loss,
          recon: cell.recon,
          kl: cell.kl,
          plot_path: cell.plot_paths ? (cell.plot_paths[activePlot] || cell.plot_path) : cell.plot_path,
          plot_paths: cell.plot_paths
        }));
        
      window._lastSweepReport = {
        timestamp: new Date().toISOString(),
        parameters: parameters,
        grid: grid
      };
      
      showSweepReportModal(window._lastSweepReport);
    }
  });

  // Synchronize global checkboxes
  const sidebarCb = $("#sweep-show-plots-global");
  const headerCb = $("#sweep-toggle-all-plots");
  
  if (sidebarCb && headerCb) {
    sidebarCb.addEventListener("change", () => {
      if (headerCb.checked !== sidebarCb.checked) {
        headerCb.checked = sidebarCb.checked;
        headerCb.dispatchEvent(new Event("change"));
      }
    });
    headerCb.addEventListener("change", () => {
      if (sidebarCb.checked !== headerCb.checked) {
        sidebarCb.checked = headerCb.checked;
      }
    });
  }

  // Toggle All plots event listener
  headerCb?.addEventListener("change", () => {
    const isChecked = headerCb.checked;
    document.querySelectorAll(".sweep-cell-cb").forEach(cb => {
      if (cb.checked !== isChecked) {
        cb.checked = isChecked;
        cb.dispatchEvent(new Event("change"));
      }
    });
  });

  // Filter input listeners
  const filterInputs = [
    "#filter-min-density",
    "#filter-min-decay",
    "#filter-max-loss",
    "#filter-min-norm"
  ];
  filterInputs.forEach(sel => {
    $(sel)?.addEventListener("input", applySweepFilters);
  });

  $("#btn-reset-filters")?.addEventListener("click", () => {
    filterInputs.forEach(sel => {
      const el = $(sel);
      if (el) el.value = "";
    });
    applySweepFilters();
  });

  // Listeners for sweep options
  $("#hidden-dim")?.addEventListener("input", updateSweepPlotOptions);
  $("#hidden-dim")?.addEventListener("change", updateSweepPlotOptions);

  // Toggle sweep architecture inputs based on mode
  const sweepArchMode = $("#sweep-arch-mode");
  const sweepFeaturesRangeGroup = $("#sweep-features-range-group");
  const sweepCustomListGroup = $("#sweep-custom-list-group");

  sweepArchMode?.addEventListener("change", () => {
    const val = sweepArchMode.value;
    if (val === "range") {
      sweepFeaturesRangeGroup?.classList.remove("hidden");
      sweepCustomListGroup?.classList.add("hidden");
    } else if (val === "custom") {
      sweepFeaturesRangeGroup?.classList.add("hidden");
      sweepCustomListGroup?.classList.remove("hidden");
    } else {
      sweepFeaturesRangeGroup?.classList.add("hidden");
      sweepCustomListGroup?.classList.add("hidden");
    }
  });

  // Timelapse toggle
  const enableTimelapse = $("#enable-timelapse");
  const timelapseSettings = $("#timelapse-settings");
  enableTimelapse?.addEventListener("change", () => {
    if (enableTimelapse.checked) {
      timelapseSettings?.classList.remove("hidden");
    } else {
      timelapseSettings?.classList.add("hidden");
    }
  });
});

// ── Model Type Toggle ─────────────────────────────────────────
function initModelTypeToggle() {
  const buttons = $$(".toggle-option[data-model]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.modelType = btn.dataset.model;
      updateConditionalFields();
      updateModeBadge();
      updateSweepPlotOptions();
      updateLossEditorForModel();
    });
  });
}

function updateConditionalFields() {
  const vaeFields = $$(".vae-only");
  vaeFields.forEach((el) => {
    if (state.modelType === "vae" || state.modelType === "manifold_vae") {
      el.classList.remove("disabled");
    } else {
      el.classList.add("disabled");
    }
  });

  const manifoldFields = $$(".manifold-vae-only");
  manifoldFields.forEach((el) => {
    if (state.modelType === "manifold_vae") {
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  });

  if (state.modelType === "manifold_vae") {
    document.body.classList.add("manifold-vae-active");
  } else {
    document.body.classList.remove("manifold-vae-active");
  }
}

function updateModeBadge() {
  const badge = $(".mode-badge");
  if (badge) {
    if (state.modelType === "vae") badge.textContent = "VAE";
    else if (state.modelType === "manifold_vae") badge.textContent = "MANIFOLD";
    else badge.textContent = "STD";
  }
}

// ── Range Input Live Values ───────────────────────────────────
function initRangeInputs() {
  const ranges = $$(".form-range[data-display]");
  ranges.forEach((range) => {
    const display = $(`#${range.dataset.display}`);
    if (display) {
      const updateDisplay = () => {
        let val = parseFloat(range.value);
        if (range.dataset.format === "percent") {
          display.textContent = `${(val * 100).toFixed(0)}%`;
        } else if (range.dataset.format === "fixed2") {
          display.textContent = val.toFixed(2);
        } else if (range.dataset.format === "fixed3") {
          display.textContent = val.toFixed(3);
        } else {
          display.textContent = val;
        }
      };
      range.addEventListener("input", updateDisplay);
      updateDisplay();
    }
  });

  // Also update importance preview when decay changes
  const decayRange = $("#importance-decay");
  if (decayRange) {
    decayRange.addEventListener("input", () => drawImportancePreview());
  }

  // Update n_features display and importance preview
  const nFeaturesInput = $("#n-features");
  if (nFeaturesInput) {
    nFeaturesInput.addEventListener("input", () => drawImportancePreview());
  }
}

// ── Importance Curve Preview ──────────────────────────────────
function initImportancePreview() {
  drawImportancePreview();
}

function drawImportancePreview() {
  const canvas = $("#importance-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * 2;
  canvas.height = rect.height * 2;
  ctx.scale(2, 2);
  const w = rect.width;
  const h = rect.height;

  const decay = parseFloat($("#importance-decay")?.value ?? 0.5);
  const n = parseInt($("#n-features")?.value ?? 256);

  ctx.clearRect(0, 0, w, h);

  // Draw curve
  ctx.beginPath();
  ctx.strokeStyle = "#8b5cf6";
  ctx.lineWidth = 1.5;
  for (let i = 0; i < w; i++) {
    const feat_idx = (i / w) * n;
    const imp = Math.pow(1.0 - feat_idx / n, decay);
    const y = h - imp * (h - 4) - 2;
    if (i === 0) ctx.moveTo(i, y);
    else ctx.lineTo(i, y);
  }
  ctx.stroke();

  // Fill under
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = "rgba(139, 92, 246, 0.1)";
  ctx.fill();
}

// ── Train Button ──────────────────────────────────────────────
function setupTrainButton() {
  const btn = $("#btn-train");
  if (btn) {
    btn.addEventListener("click", () => {
      if (state.sweepMode === "sweep") {
        runSweep();
      } else {
        trainModel();
      }
    });
  }
}

function getConfig() {
  const config = {
    model_type: state.modelType,
    n_features: parseInt($("#n-features")?.value ?? 256),
    hidden: parseInt($("#hidden-dim")?.value ?? 2),
    density: parseFloat($("#density")?.value ?? 0.5),
    init_method: $("#init-method")?.value ?? "xavier_normal",
    normalize_weights: $("#normalize-weights")?.checked ?? false,
    steps: parseInt($("#steps")?.value ?? 5000),
    batch_size: parseInt($("#batch-size")?.value ?? 512),
    lr: parseFloat($("#learning-rate")?.value ?? 0.001),
    beta: parseFloat($("#beta")?.value ?? 0.05),
    beta_warmup: parseInt($("#beta-warmup")?.value ?? 2000),
    importance_decay: parseFloat($("#importance-decay")?.value ?? 0.5),
    use_importance: true,
  };
  
  if (state.modelType === "manifold_vae") {
    config.manifold_type = $("#manifold-type")?.value ?? "poincare";
    config.curvature = parseFloat($("#curvature")?.value ?? 1.0);
    config.tangent_weight = parseFloat($("#tangent-weight")?.value ?? 0.01);
    
    // Attempt to extract loss preset
    const preset = $("#loss-preset");
    if (preset && preset.value && preset.value !== "custom") {
        config.manifold_loss_type = preset.value;
    } else {
        config.manifold_loss_type = "riemannian_kl"; // fallback
    }
  }
  
  if ($("#enable-timelapse")?.checked) {
    config.snapshot_interval = parseInt($("#snapshot-interval")?.value ?? 500);
  }
  if (state.customLossExpr) {
    config.custom_loss_expr = state.customLossExpr;
  }
  return config;
}

async function trainModel() {
  if (state.isTraining) return;
  state.isTraining = true;
  state.plotCache = {};

  const btn = $("#btn-train");
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:18px;height:18px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px;"></span> Training...`;

  showLoading("Training model...");
  clearStats();
  clearPlotTabs();

  const config = getConfig();

  try {
    const resp = await fetch(`${API_BASE}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Training failed", "error");
      return;
    }

    state.trainedOnce = true;
    state.availablePlots = data.available_plots;
    state.stats = data.stats;
    state.losses = data.losses;

    renderPlotTabs(data.available_plots);
    renderStats(data.stats, data);
    showToast("Model trained successfully", "success");

    // Auto-load first plot
    if (data.available_plots.length > 0) {
      loadPlot(data.available_plots[0]);
    }

    fetchPastExperiments();
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
  } finally {
    state.isTraining = false;
    btn.disabled = false;
    btn.innerHTML = originalText;
    hideLoading();
  }
}

// ── Plot Tabs ─────────────────────────────────────────────────
const PLOT_LABELS = {
  arrows_W: "W Arrows",
  arrows_W_mu: "W_μ Arrows",
  arrows_W_logvar: "W_logvar Arrows",
  z_samples: "z Samples",
  combined: "Combined",
  interference_W: "W Interference",
  interference_W_mu: "W_μ Interference",
  interference_W_logvar: "W_logvar Interference",
  norms_W: "W Norms",
  norms_W_mu: "W_μ Norms",
  norms_W_logvar: "W_logvar Norms",
  heatmap_W: "W Heatmap",
  heatmap_W_mu: "W_μ Heatmap",
  heatmap_W_logvar: "W_logvar Heatmap",
  pca_W: "W PCA",
  pca_W_mu: "W_μ PCA",
  pca_W_logvar: "W_logvar PCA",
  graph_W: "W Graph",
  graph_W_mu: "W_μ Graph",
  graph_W_logvar: "W_logvar Graph",
  coord_graph_W: "W Coord Graph",
  coord_graph_W_mu: "W_μ Coord Graph",
  coord_graph_W_logvar: "W_logvar Coord Graph",
  manifold_jacobian_W: "W Jacobian",
  manifold_jacobian_W_mu: "W_μ Jacobian",
  manifold_geometry_W: "W Geometry",
  manifold_geometry_W_mu: "W_μ Geometry",
  manifold_summary_W: "W Manifold Summary",
  manifold_summary_W_mu: "W_μ Manifold Summary",
  
  // Manifold VAE Specific
  arrows_W_tangent: "W_tangent Arrows",
  manifold_disk_scatter: "Manifold Disk",
  manifold_combined: "Combined Manifold",
  geodesic_matrix: "Geodesic Matrix",
  curvature_heatmap: "Curvature λ(x)",
  heatmap_W_tangent: "W_tangent Heatmap",
  pca_W_tangent: "W_tangent PCA",
  interference_W_tangent: "W_tangent Interference",
  norms_W_tangent: "W_tangent Norms",
  graph_W_tangent: "W_tangent Graph",
  coord_graph_W_tangent: "W_tangent Coord Graph",
  manifold_jacobian_W_tangent: "W_tangent Jacobian",
  manifold_geometry_W_tangent: "W_tangent Geometry",
  manifold_summary_W_tangent: "W_tangent Manifold",
};

// ═══════════════════════════════════════════════════════════════
// Plot Explanations — Theory + Interpretation Guide
// ═══════════════════════════════════════════════════════════════
const PLOT_EXPLANATIONS = {
  // ── Arrow plots (2D hidden) ──
  arrows_W: {
    title: "Weight Arrows — W (tied weights)",
    theory: "Each arrow represents one feature's weight vector <code>w_i ∈ ℝ²</code> in the bottleneck. In the Elhage et al. (2022) toy model, the network learns to encode <strong>n features into just 2 hidden dimensions</strong>. The direction of each arrow shows where that feature is represented in the latent space; the length (norm) indicates how strongly the model commits to encoding it.",
    intuition: "<strong>Look for:</strong> If arrows spread evenly around the circle, the model is using <strong>superposition</strong> — packing more features than dimensions allow by using non-orthogonal directions. If only 2 arrows are large (aligned with axes), the model uses a <strong>dedicated basis</strong> with no superposition. Short arrows = features the model has chosen to ignore (low-importance features). Arrows that nearly overlap indicate <strong>interference</strong> between those features."
  },
  arrows_W_mu: {
    title: "Weight Arrows — W_μ (mean encoder)",
    theory: "The <code>W_μ</code> matrix maps inputs to the <strong>mean of the approximate posterior</strong> q(z|x). Each column is a feature's encoding direction in the VAE's latent space. Unlike the standard bottleneck, the VAE has a separate variance pathway, so W_μ captures only the 'where to encode' decision, not 'how certain.'",
    intuition: "<strong>Interpretation is similar to W arrows</strong>, but now the arrows represent the <em>deterministic component</em> of the encoding. Evenly spread arrows = superposition in the mean space. Compare with <strong>W_logvar arrows</strong> to see if the model encodes uncertainty differently for different features. Features with large W_μ norms but small W_logvar norms are encoded with <strong>high confidence</strong>."
  },
  arrows_W_logvar: {
    title: "Weight Arrows — W_logvar (variance encoder)",
    theory: "The <code>W_logvar</code> matrix maps inputs to the <strong>log-variance of the approximate posterior</strong>. This determines how much noise is injected per latent dimension for each feature. It controls the <strong>information bottleneck</strong> — large variance = more noise = less information transmitted.",
    intuition: "<strong>Key insight:</strong> Features with large W_logvar norms have <strong>high variance</strong> (more noise), meaning the model is less certain about them. If W_logvar arrows are small/near zero, the model acts almost deterministically (like a standard autoencoder). If W_logvar arrows point in <em>different directions</em> than W_μ, the model applies anisotropic noise — critical for understanding superposition under uncertainty."
  },
  arrows_W_tangent: {
    title: "Weight Arrows — W_tangent (tangent space)",
    theory: "In the manifold VAE, <code>W_tangent</code> maps inputs into the <strong>tangent space at the origin</strong> of the chosen manifold (Poincaré ball or sphere). This is then projected onto the manifold via the exponential map. The tangent space is Euclidean, so these arrows behave like standard weight vectors but are interpreted as velocities on the manifold.",
    intuition: "<strong>Compare with standard arrows:</strong> Similar spread patterns indicate superposition, but the actual latent representations will be distorted by the manifold's curvature. Arrows near the origin produce latent points near the manifold center (low curvature region); large-norm arrows push points toward the boundary where geodesic distances diverge."
  },

  // ── z Samples ──
  z_samples: {
    title: "Latent Space Samples — z",
    theory: "This scatter plot shows samples from the <strong>reparameterized posterior</strong> z = μ + ε·σ. Each point is a latent representation of one input. The distribution of z reveals how the model uses its latent space — whether it's concentrated, spread out, or has structure.",
    intuition: "<strong>Look for:</strong> A <strong>circular cloud</strong> centered at origin suggests the KL term is regularizing well toward the N(0,I) prior. An <strong>elongated or multi-modal</strong> distribution means certain latent directions carry more information. If z samples form <strong>rays or clusters</strong>, the model has learned distinct encoding modes. Compare with the W_μ arrows — z samples should cluster along the directions defined by the large weight vectors."
  },

  // ── Combined ──
  combined: {
    title: "Combined View — Weights + Latent Samples",
    theory: "Overlays <code>W_μ</code> (plasma colormap), <code>W_logvar</code> (viridis colormap), and <strong>z samples</strong> (gray dots) in one plot. This gives the complete picture of how features are encoded and where latent samples actually land.",
    intuition: "<strong>The key plot for understanding superposition:</strong> Do the z samples align with the weight arrow directions? If yes, the model successfully encodes those features. If z samples avoid certain directions, those features may be <strong>suppressed by the KL term</strong>. The gap between W_μ arrows and z cloud extent shows how much the variance (W_logvar) is blurring the encoding."
  },
  manifold_combined: {
    title: "Combined Manifold View",
    theory: "Overlays <code>W_tangent</code> arrows with <strong>z samples on the manifold disk</strong>. For Poincaré manifolds, the disk boundary represents infinity — samples near the edge are geodesically far from the origin.",
    intuition: "<strong>Look for:</strong> How weight arrows relate to the distribution of z on the manifold. Points clustered near the center live in a nearly-Euclidean region. Points near the boundary exploit the manifold's curvature for <strong>hierarchical</strong> or <strong>tree-like</strong> representations. The unit circle (Poincaré boundary) should not be crossed by z samples."
  },

  // ── Interference ──
  interference_W: {
    title: "Interference Matrix — WᵀW",
    theory: "The <strong>Gram matrix</strong> <code>WᵀW</code> encodes pairwise dot products between all feature weight vectors. Diagonal entries are squared norms (how strongly each feature is represented). Off-diagonal entries measure <strong>interference</strong> — how much reconstructing one feature corrupts another.",
    intuition: "<strong>Ideal case:</strong> A diagonal matrix (no interference, orthogonal features). In superposition, off-diagonal entries are nonzero — this is the <strong>cost of packing more features than dimensions</strong>. The right panel (off-diagonal only) isolates interference: bright spots indicate feature pairs that strongly interfere. <strong>Blue = negative interference</strong> (anti-correlated reconstruction errors), <strong>red = positive interference</strong>."
  },
  interference_W_mu: {
    title: "Interference Matrix — W_μᵀW_μ",
    theory: "Same as W interference but for the <strong>mean encoder only</strong>. Measures how much the mean encoding directions of different features overlap. High off-diagonal values mean the VAE's posterior means for two features are <strong>entangled</strong>.",
    intuition: "<strong>Compare with W interference in standard models.</strong> The VAE's KL regularization may <em>reduce</em> interference by pushing features toward orthogonality, or <em>increase</em> it by compressing the representation. A block-diagonal structure suggests <strong>feature grouping</strong> — the model clusters related features."
  },
  interference_W_logvar: {
    title: "Interference Matrix — W_logvarᵀW_logvar",
    theory: "Gram matrix for the variance encoder. Shows whether features share <strong>noise directions</strong>. If two features have similar W_logvar vectors, the VAE adds correlated noise to both — they share the same uncertainty structure.",
    intuition: "<strong>Often more uniform than W_μ interference</strong> because the model may use a simpler variance structure. A near-uniform matrix means the model applies similar noise everywhere. Strong off-diagonal structure means the model has learned <strong>feature-specific uncertainty patterns</strong>."
  },
  interference_W_tangent: {
    title: "Interference Matrix — W_tangentᵀW_tangent",
    theory: "Gram matrix for the tangent-space encoder of the manifold VAE. In tangent space, dot products are Euclidean, so this measures geometric overlap before the exponential map projects to the manifold.",
    intuition: "<strong>Note:</strong> Tangent-space interference does not directly translate to interference on the manifold due to curvature effects. Two features with similar tangent vectors may end up geodesically far apart near the boundary. Use alongside the <strong>Geodesic Matrix</strong> for the full picture."
  },

  // ── Norms ──
  norms_W: {
    title: "Weight Norms — ‖wᵢ‖",
    theory: "Bar chart of per-feature weight vector norms, sorted by magnitude. The norm <code>‖wᵢ‖</code> determines how strongly feature i is represented in the bottleneck. In the Elhage et al. framework, the model makes a <strong>binary decision</strong> per feature: represent it (large norm) or discard it (zero norm).",
    intuition: "<strong>Look for a sharp cutoff</strong> between 'active' features (norm > 0) and 'dead' features (norm ≈ 0). The transition point reveals the model's <strong>effective capacity</strong>. The histogram (right panel) shows the distribution — a bimodal distribution (peaks at 0 and ~1) indicates clean superposition. A smooth gradient suggests the model is hedging on which features to keep."
  },
  norms_W_mu: {
    title: "Weight Norms — ‖w_μᵢ‖",
    theory: "Per-feature norms of the mean encoder. Determines how far from the origin each feature pushes the posterior mean — equivalently, how much <strong>signal</strong> each feature contributes to the latent code.",
    intuition: "<strong>Same interpretation as W norms</strong>, but remember that features with large W_μ norms but also large W_logvar norms may still be <strong>noisy</strong> (high signal-to-noise requires large μ / small σ). The histogram shape tells you if the VAE has learned a cleaner or messier feature selection than a standard bottleneck."
  },
  norms_W_logvar: {
    title: "Weight Norms — ‖w_logvarᵢ‖",
    theory: "Per-feature norms of the log-variance encoder. Controls how much <strong>posterior noise</strong> each feature induces. Larger norms = the model's posterior variance is more feature-dependent.",
    intuition: "<strong>If all norms are similar</strong>, the model uses isotropic noise (all features equally uncertain). <strong>If some norms are much larger</strong>, those features drive high-variance latent directions — the model is <em>uncertain</em> about them or intentionally injecting noise to prevent overfitting. Compare with W_μ norms: the ratio ‖w_μ‖/‖w_logvar‖ approximates per-feature SNR."
  },
  norms_W_tangent: {
    title: "Weight Norms — ‖w_tangentᵢ‖",
    theory: "Per-feature norms in the tangent space encoder. Determines how far from the origin each feature's tangent vector reaches before being mapped to the manifold.",
    intuition: "<strong>For Poincaré manifolds</strong>, large tangent norms push the exponential map output toward the disk boundary, where distances grow exponentially. This means high-norm features are encoded in a region with <strong>much more representational capacity</strong> — the manifold provides an implicit feature importance weighting."
  },

  // ── Heatmaps (hidden > 2) ──
  heatmap_W: {
    title: "Weight Heatmap — W",
    theory: "Visualizes the raw <code>W</code> matrix as a 2D heatmap (hidden dimensions × features). Each cell shows the weight connecting one hidden unit to one feature. Red = positive, blue = negative, white = zero.",
    intuition: "<strong>For high-dimensional hidden spaces:</strong> Look for <strong>column patterns</strong> — features with similar color profiles across hidden dims are encoded similarly (potential interference). <strong>Sparse columns</strong> (mostly white) indicate features the model barely represents. <strong>Row patterns</strong> show which hidden dimensions specialize in which features."
  },
  heatmap_W_mu: {
    title: "Weight Heatmap — W_μ",
    theory: "Raw heatmap of the mean encoder matrix for hidden > 2. Each row is a hidden dimension, each column is a feature.",
    intuition: "<strong>Same as W heatmap interpretation.</strong> Structured patterns (stripes, blocks) indicate learned organization. Noisy-looking rows may indicate dimensions the model hasn't fully utilized."
  },
  heatmap_W_logvar: {
    title: "Weight Heatmap — W_logvar",
    theory: "Raw heatmap of the log-variance encoder matrix. Shows how each feature contributes to the posterior variance in each latent dimension.",
    intuition: "<strong>Often less structured than W_μ.</strong> Uniform values = isotropic noise. Structured patterns reveal feature-dependent noise shaping."
  },
  heatmap_W_tangent: {
    title: "Weight Heatmap — W_tangent",
    theory: "Raw heatmap of the tangent-space projection matrix for manifold VAE with hidden > 2.",
    intuition: "<strong>Interpretation combines standard heatmap reading with manifold awareness.</strong> The actual latent representations will be nonlinearly transformed by the exponential map."
  },

  // ── PCA (hidden > 2) ──
  pca_W: {
    title: "PCA Projection — W",
    theory: "Projects the high-dimensional weight vectors onto their <strong>top 2 principal components</strong>. This gives the best 2D view of how features are arranged in the hidden space. The percentage labels show how much variance each PC captures.",
    intuition: "<strong>Interpret like the 2D arrow plot</strong>, but remember this is a projection — features that appear close may actually be separated in other dimensions. If PC1+PC2 capture >90% of variance, the model is effectively using only 2 dimensions despite having more. Low explained variance means the features are spread across many dimensions (rich representation)."
  },
  pca_W_mu: {
    title: "PCA Projection — W_μ",
    theory: "PCA projection of the high-dimensional mean encoder weight vectors onto their top 2 PCs.",
    intuition: "<strong>Same as W PCA.</strong> Compare explained variance percentages between W_μ and W_logvar to understand if the model uses different effective dimensionalities for mean vs. variance encoding."
  },
  pca_W_logvar: {
    title: "PCA Projection — W_logvar",
    theory: "PCA projection of the log-variance encoder weight vectors.",
    intuition: "<strong>If explained variance is very high (>95%)</strong>, the variance structure is essentially low-dimensional even if the hidden space is large — the model is not fully utilizing variance capacity."
  },
  pca_W_tangent: {
    title: "PCA Projection — W_tangent",
    theory: "PCA projection of the tangent-space weight vectors for manifold VAE with hidden > 2.",
    intuition: "<strong>This is a Euclidean projection of tangent vectors.</strong> Remember that the manifold's curvature will distort the actual latent geometry non-linearly."
  },

  // ── Graph plots ──
  graph_W: {
    title: "Feature Graph — W",
    theory: "A <strong>node-link diagram</strong> where each node is an active feature and edges connect features with high absolute dot product <code>|wᵢᵀwⱼ|</code>. Node color = weight norm, edge intensity = interference strength. Layout uses spring forces (features with higher interference are pulled closer).",
    intuition: "<strong>Clusters of densely connected nodes</strong> indicate feature groups that are encoded in similar directions and strongly interfere with each other. <strong>Isolated nodes</strong> are cleanly represented features with little interference. The graph topology reveals the <strong>interference structure</strong> at a glance — a densely connected graph means heavy superposition."
  },
  graph_W_mu: {
    title: "Feature Graph — W_μ",
    theory: "Same as W graph but for the mean encoder. Edges represent interference in the mean encoding only.",
    intuition: "<strong>Compare with W graph:</strong> The VAE's KL regularization may change the interference topology — features that interfere in a standard model might be separated by the variational objective."
  },
  graph_W_logvar: {
    title: "Feature Graph — W_logvar",
    theory: "Graph of variance encoder interference patterns.",
    intuition: "<strong>Features connected here share noise directions.</strong> This reveals which features the model treats as having correlated uncertainty."
  },
  graph_W_tangent: {
    title: "Feature Graph — W_tangent",
    theory: "Feature interference graph for the tangent-space encoder.",
    intuition: "<strong>Tangent-space interference ≠ manifold interference.</strong> Use alongside the geodesic distance matrix for accurate manifold geometry."
  },

  // ── Coordinate Graphs ──
  coord_graph_W: {
    title: "Coordinate Graph — W",
    theory: "Like the feature graph, but node positions are the <strong>actual weight vector coordinates</strong> (or PCA projections for hidden>2). This shows the true geometric relationship between features in weight space, with edges showing interference.",
    intuition: "<strong>The most geometrically faithful view:</strong> Nearby nodes truly have similar encoding directions. Clusters reveal superposition groups directly. The unit circle shows the natural scale — features inside it are sub-unit-norm, features outside are amplified."
  },
  coord_graph_W_mu: {
    title: "Coordinate Graph — W_μ",
    theory: "Coordinate graph using actual W_μ vector positions.",
    intuition: "<strong>Shows where features actually sit in the mean latent space.</strong> Compare with z samples to verify that the encoding is being used as intended."
  },
  coord_graph_W_logvar: {
    title: "Coordinate Graph — W_logvar",
    theory: "Coordinate graph using W_logvar positions.",
    intuition: "<strong>Shows the geometry of the noise structure.</strong> Features that cluster here will have correlated posterior noise."
  },
  coord_graph_W_tangent: {
    title: "Coordinate Graph — W_tangent",
    theory: "Coordinate graph using tangent-space weight positions.",
    intuition: "<strong>Shows tangent-space geometry.</strong> Remember to mentally apply the exponential map when reasoning about the actual manifold positions."
  },

  // ── Manifold-specific ──
  manifold_disk_scatter: {
    title: "Manifold Disk — z Samples",
    theory: "Scatter plot of reparameterized z samples on the <strong>Poincaré disk</strong> (or stereographic sphere). The disk boundary represents geodesic infinity — distances grow exponentially near the edge. Concentric dashed circles show constant geodesic distances from the origin.",
    intuition: "<strong>Key insight:</strong> Points near the center are in a nearly-Euclidean region. Points near the boundary are in a <strong>high-curvature region</strong> with exponentially growing distances — ideal for encoding hierarchical structure. If all z samples cluster near the center, the model isn't using the manifold's curvature advantage."
  },
  geodesic_matrix: {
    title: "Geodesic Distance Matrix",
    theory: "Pairwise <strong>geodesic distances</strong> (not Euclidean!) between z samples. On the Poincaré disk, the geodesic metric is <code>d(x,y) = (2/√c) · arctanh(√c · ‖-x ⊕ y‖)</code>. This shows the true manifold-aware distances.",
    intuition: "<strong>Look for block structure</strong> — groups of samples that are geodesically close form clusters. The color scale matters: dark = close, bright = far. Compare with the Euclidean z scatter — pairs that look close in Euclidean space near the boundary may actually be very far in geodesic distance."
  },
  curvature_heatmap: {
    title: "Conformal Factor λ(x) Heatmap",
    theory: "Visualizes the <strong>conformal factor</strong> λ(x) = 2/(1 - c·‖x‖²) across the latent space, with z samples overlaid. The conformal factor measures how much the Riemannian metric stretches/compresses distances relative to Euclidean space at each point.",
    intuition: "<strong>Bright regions</strong> (high λ) have enormous metric stretching — small Euclidean moves correspond to large geodesic distances. This is where the manifold provides the most <strong>representational capacity per unit of Euclidean space</strong>. If z samples concentrate in high-λ regions, the model is exploiting curvature for efficient encoding."
  },

  // ── Jacobian ──
  manifold_jacobian_W: {
    title: "Decoder Jacobian — W",
    theory: "The <strong>mean Jacobian</strong> J = ∂x̂/∂z of the decoder, averaged over a batch of latent samples. Since the decoder is ReLU(z·W+b), the Jacobian is J = diag(mask)·Wᵀ where mask is the ReLU activation pattern. <strong>Singular values</strong> of J reveal the decoder's magnification per latent direction.",
    intuition: "<strong>Left panel (heatmap):</strong> Shows how each feature responds to each latent direction. Rows with all near-zero values = features that are never activated (dead). <strong>Right panel (singular values):</strong> If σ₁ ≫ σ₂, the decoder compresses one direction — a sign of <strong>anisotropic representation</strong>. Equal singular values = isotropic decoding."
  },
  manifold_jacobian_W_mu: {
    title: "Decoder Jacobian — W_μ",
    theory: "Jacobian analysis using the mean encoder weights. Identical theory to W Jacobian since the VAE decoder uses W_μ.",
    intuition: "<strong>Track singular value evolution</strong> (right panel, if time-lapse enabled) to see how the decoder's magnification changes during training. Convergence of singular values suggests the model has found a stable representation."
  },
  manifold_jacobian_W_tangent: {
    title: "Decoder Jacobian — W_tangent",
    theory: "Jacobian for the manifold VAE decoder, using the tangent-space weights.",
    intuition: "<strong>Note:</strong> This is the tangent-space Jacobian. The full manifold Jacobian also includes the log map's derivative, which amplifies sensitivities near the boundary."
  },

  // ── Geometry ──
  manifold_geometry_W: {
    title: "Geometry — Metric Tensor & Hessian",
    theory: "<strong>G = JᵀJ</strong> is the <strong>metric tensor</strong> (pull-back of the feature-space Euclidean metric). It measures local distances in latent space. <strong>H = 2W·diag(imp⊙mask)·Wᵀ</strong> is the loss <strong>Hessian</strong> w.r.t. z, measuring curvature of the loss landscape.",
    intuition: "<strong>G (top row):</strong> Diagonal = magnification per hidden dim. Off-diagonal = coupling between dims. Equal eigenvalues = isotropic metric (distances are preserved). <strong>H (bottom row):</strong> All positive eigenvalues → local minimum. Mixed signs → saddle point. Large eigenvalues → sharp curvature (sensitive to small z perturbations)."
  },
  manifold_geometry_W_mu: {
    title: "Geometry — W_μ Metric & Hessian",
    theory: "Same geometry analysis but using the VAE's mean encoder weights.",
    intuition: "<strong>Same interpretation.</strong> Compare G eigenvalue ratios across model types to see if the VAE's regularization produces more isotropic metrics."
  },
  manifold_geometry_W_tangent: {
    title: "Geometry — W_tangent Metric & Hessian",
    theory: "Geometry analysis for the manifold VAE's tangent-space encoder.",
    intuition: "<strong>The metric tensor here describes tangent-space geometry.</strong> For the full Riemannian picture, multiply by the conformal factor squared."
  },

  // ── Manifold Summary ──
  manifold_summary_W: {
    title: "Manifold Summary — W",
    theory: "Dashboard of key manifold diagnostics: <strong>condition number</strong> (ratio of max/min Jacobian singular values), <strong>effective rank</strong> (entropy of normalized singular values), <strong>det(G)</strong> (volume element / area magnification), and <strong>Hessian analysis</strong>.",
    intuition: "<strong>Condition number:</strong> < 10 = well-conditioned, > 100 = ill-conditioned (some directions are compressed vs. stretched). <strong>Effective rank:</strong> close to hidden dim = all directions used equally. Close to 1 = model only uses one direction. <strong>det(G):</strong> local area magnification. Near zero = the mapping collapses volume (degenerate). <strong>Active fraction:</strong> % of ReLU neurons active on average."
  },
  manifold_summary_W_mu: {
    title: "Manifold Summary — W_μ",
    theory: "Same manifold summary for the VAE mean encoder.",
    intuition: "<strong>Compare with standard model summary</strong> to quantify the effect of variational regularization on the representation geometry."
  },
  manifold_summary_W_tangent: {
    title: "Manifold Summary — W_tangent",
    theory: "Manifold summary for the tangent-space encoder.",
    intuition: "<strong>In manifold VAEs, the condition number reflects tangent-space conditioning.</strong> The actual manifold geometry also depends on curvature — a well-conditioned tangent map can still produce poor manifold representations at high curvature."
  },
};

// ── State for explanation panel ──
let explanationCollapsed = false;

function renderExplanationPanel(plotType) {
  const container = $("#plot-explanation-container");
  if (!container) return;

  // No explanation for raw_weights (it has its own UI)
  if (plotType === "raw_weights" || !PLOT_EXPLANATIONS[plotType]) {
    container.innerHTML = "";
    return;
  }

  const ex = PLOT_EXPLANATIONS[plotType];
  const bodyClass = explanationCollapsed ? "collapsed" : "";
  const toggleText = explanationCollapsed ? "▸ Show" : "▾ Hide";

  container.innerHTML = `
    <div class="plot-explanation" id="plot-explanation">
      <div class="plot-explanation__header" onclick="toggleExplanation()">
        <div class="plot-explanation__title">
          <span>📖</span> ${ex.title}
        </div>
        <button class="plot-explanation__toggle" id="explanation-toggle">${toggleText}</button>
      </div>
      <div class="plot-explanation__body ${bodyClass}" id="explanation-body">
        <div class="plot-explanation__section">
          <div class="plot-explanation__section-label theory">🔬 Theory</div>
          <div class="plot-explanation__text">${ex.theory}</div>
        </div>
        <div class="plot-explanation__section">
          <div class="plot-explanation__section-label intuition">💡 Intuition — How to Read</div>
          <div class="plot-explanation__text">${ex.intuition}</div>
        </div>
      </div>
    </div>
  `;
}

function toggleExplanation() {
  explanationCollapsed = !explanationCollapsed;
  const body = $("#explanation-body");
  const toggle = $("#explanation-toggle");
  if (body) {
    body.classList.toggle("collapsed", explanationCollapsed);
  }
  if (toggle) {
    toggle.textContent = explanationCollapsed ? "▸ Show" : "▾ Hide";
  }
}

// ═══════════════════════════════════════════════════════════════
// Raw Weight Vector Viewer
// ═══════════════════════════════════════════════════════════════
let weightViewerActiveMatrix = null;
let weightViewerData = null;

async function loadWeightViewer() {
  const canvas = $("#plot-canvas");
  if (!canvas) return;

  // Show loading
  showLoading("Loading weight data...");
  renderExplanationPanel("raw_weights");

  try {
    const resp = await fetch(`${API_BASE}/api/weights`);
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Failed to load weights", "error");
      hideLoading();
      return;
    }

    weightViewerData = data;
    const matrixKeys = Object.keys(data.matrices);
    if (matrixKeys.length === 0) {
      showToast("No weight matrices available", "warning");
      hideLoading();
      return;
    }

    // Default to first matrix
    if (!weightViewerActiveMatrix || !data.matrices[weightViewerActiveMatrix]) {
      weightViewerActiveMatrix = matrixKeys[0];
    }

    renderWeightViewer(data, weightViewerActiveMatrix);
    hideLoading();

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
    hideLoading();
  }
}

function renderWeightViewer(data, activeKey) {
  const canvas = $("#plot-canvas");
  if (!canvas) return;

  // Remove existing content
  const oldImg = canvas.querySelector("img");
  if (oldImg) oldImg.remove();
  const emptyState = canvas.querySelector(".plot-canvas__empty");
  if (emptyState) emptyState.remove();
  const oldViewer = canvas.querySelector(".weight-viewer");
  if (oldViewer) oldViewer.remove();

  const mat = data.matrices[activeKey];
  if (!mat) return;

  const matrixKeys = Object.keys(data.matrices);

  // Build matrix tabs
  let tabsHtml = "";
  matrixKeys.forEach(key => {
    const info = data.matrices[key];
    const cls = key === activeKey ? "active" : "";
    tabsHtml += `<button class="weight-viewer__matrix-tab ${cls}" onclick="switchWeightMatrix('${key}')">${info.name}</button>`;
  });

  // Find global max absolute value for color scaling
  const allValues = mat.values.flat();
  const absMax = Math.max(...allValues.map(Math.abs), 1e-8);

  // Build table HTML
  const [hidden, nFeatures] = mat.shape;
  const headerCells = [`<th class="row-header"></th>`];
  for (let j = 0; j < nFeatures; j++) {
    headerCells.push(`<th>f${j}</th>`);
  }

  let rowsHtml = "";
  for (let i = 0; i < hidden; i++) {
    let cells = `<td class="row-label">h${i}</td>`;
    for (let j = 0; j < nFeatures; j++) {
      const v = mat.values[i][j];
      const color = weightCellColor(v, absMax);
      const formatted = formatWeightValue(v);
      cells += `<td class="weight-cell" style="background:${color}" title="h${i}, f${j}: ${v.toFixed(8)}">${formatted}</td>`;
    }
    rowsHtml += `<tr>${cells}</tr>`;
  }

  // Norms row
  let normCells = `<td class="row-label">‖wᵢ‖</td>`;
  const normMax = mat.max_norm || 1;
  for (let j = 0; j < nFeatures; j++) {
    const n = mat.norms[j];
    const intensity = Math.min(n / normMax, 1);
    const r = Math.round(139 + intensity * 80);
    const g = Math.round(92 - intensity * 40);
    const b = Math.round(246);
    const bg = `rgba(${r}, ${g}, ${b}, ${(0.08 + intensity * 0.25).toFixed(2)})`;
    normCells += `<td class="weight-cell" style="background:${bg}" title="‖w_${j}‖ = ${n.toFixed(6)}">${n.toFixed(4)}</td>`;
  }
  rowsHtml += `<tr class="norm-row">${normCells}</tr>`;

  const viewer = document.createElement("div");
  viewer.className = "weight-viewer";
  viewer.innerHTML = `
    <div class="weight-viewer__header">
      <div class="weight-viewer__title">📊 Raw Weight Vectors</div>
      <div class="weight-viewer__matrix-tabs">${tabsHtml}</div>
    </div>
    <div class="weight-viewer__info">
      <div class="weight-viewer__info-item">
        Shape: <span class="weight-viewer__info-value">${hidden} × ${nFeatures}</span>
      </div>
      <div class="weight-viewer__info-item">
        Max Norm: <span class="weight-viewer__info-value">${mat.max_norm.toFixed(4)}</span>
      </div>
      <div class="weight-viewer__info-item">
        Mean Norm: <span class="weight-viewer__info-value">${mat.mean_norm.toFixed(4)}</span>
      </div>
      <div class="weight-viewer__legend">
        <span>−${absMax.toFixed(2)}</span>
        <div class="weight-viewer__legend-bar"></div>
        <span>+${absMax.toFixed(2)}</span>
      </div>
    </div>
    <div class="weight-table-container">
      <table class="weight-table">
        <thead><tr>${headerCells.join("")}</tr></thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    </div>
  `;

  canvas.insertBefore(viewer, canvas.firstChild);
}

function switchWeightMatrix(key) {
  weightViewerActiveMatrix = key;
  if (weightViewerData) {
    renderWeightViewer(weightViewerData, key);
  }
}

function weightCellColor(value, absMax) {
  const t = value / absMax; // [-1, 1]
  if (t > 0) {
    // Red tones
    const intensity = Math.min(t, 1);
    return `rgba(239, 68, 68, ${(intensity * 0.5).toFixed(3)})`;
  } else if (t < 0) {
    // Blue tones
    const intensity = Math.min(-t, 1);
    return `rgba(59, 130, 246, ${(intensity * 0.5).toFixed(3)})`;
  }
  return "transparent";
}

function formatWeightValue(v) {
  if (Math.abs(v) < 0.0001) return v.toExponential(1);
  if (Math.abs(v) < 1) return v.toFixed(4);
  return v.toFixed(3);
}

function setupPlotTabs() {
  // Placeholder — tabs are rendered dynamically after training
}

function clearPlotTabs() {
  const container = $("#plot-tabs");
  if (container) container.innerHTML = "";
}

function renderPlotTabs(plots) {
  const container = $("#plot-tabs");
  if (!container) return;
  container.innerHTML = "";

  plots.forEach((plotType, idx) => {
    const tab = document.createElement("button");
    tab.className = `plot-tab${idx === 0 ? " active" : ""}`;
    tab.textContent = PLOT_LABELS[plotType] || plotType;
    tab.dataset.plot = plotType;
    tab.id = `tab-${plotType}`;
    tab.addEventListener("click", () => {
      $$("#plot-tabs .plot-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      loadPlot(plotType);
    });
    container.appendChild(tab);
  });

  // Append the "Raw Weights" pseudo-tab
  const rawTab = document.createElement("button");
  rawTab.className = "plot-tab";
  rawTab.textContent = "📊 Raw Weights";
  rawTab.dataset.plot = "raw_weights";
  rawTab.id = "tab-raw_weights";
  rawTab.style.borderColor = "var(--accent-3)";
  rawTab.addEventListener("click", () => {
    $$("#plot-tabs .plot-tab").forEach((t) => t.classList.remove("active"));
    rawTab.classList.add("active");
    loadPlot("raw_weights");
  });
  container.appendChild(rawTab);
}

async function loadPlot(plotType) {
  state.currentPlot = plotType;

  // Handle raw weights specially
  if (plotType === "raw_weights") {
    loadWeightViewer();
    return;
  }

  // Show explanation panel
  renderExplanationPanel(plotType);

  // Check cache
  if (state.plotCache[plotType]) {
    renderPlotImage(state.plotCache[plotType], plotType);
    setupTimelineControls(plotType);
    return;
  }

  showLoading("Generating plot...");

  try {
    const resp = await fetch(`${API_BASE}/api/plot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plot_type: plotType }),
    });
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Plot generation failed", "error");
      showEmptyCanvas();
      hideLoading();
      return;
    }

    state.plotCache[plotType] = data.image;
    renderPlotImage(data.image, plotType);
    setupTimelineControls(plotType);
    
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
    showEmptyCanvas();
    hideLoading();
  }
}

function renderPlotImage(base64, plotType) {
  const canvas = $("#plot-canvas");
  if (!canvas) return;

  // Remove any weight viewer if present
  const oldViewer = canvas.querySelector(".weight-viewer");
  if (oldViewer) oldViewer.remove();

  const img = new Image();
  img.src = `data:image/png;base64,${base64}`;
  img.alt = PLOT_LABELS[plotType] || plotType;
  img.style.animation = "fadeIn 0.3s ease";
  
  img.onload = () => {
    const oldImg = canvas.querySelector("img");
    if (oldImg) oldImg.remove();
    const emptyState = canvas.querySelector(".plot-canvas__empty");
    if (emptyState) emptyState.remove();
    
    canvas.insertBefore(img, canvas.firstChild);
    hideLoading();
  };
}

function showEmptyCanvas() {
  const canvas = $("#plot-canvas");
  if (!canvas) return;
  
  const oldImg = canvas.querySelector("img");
  if (oldImg) oldImg.remove();

  if (!canvas.querySelector(".plot-canvas__empty")) {
    const emptyDiv = document.createElement("div");
    emptyDiv.className = "plot-canvas__empty";
    emptyDiv.innerHTML = `
      <div class="plot-canvas__empty-icon">📊</div>
      <div class="plot-canvas__empty-text">No plot loaded</div>
      <div class="plot-canvas__empty-hint">Train a model to see visualizations</div>
    `;
    canvas.insertBefore(emptyDiv, canvas.firstChild);
  }
}

// ── Timeline Controls ─────────────────────────────────────────
function setupTimelineControls(plotType) {
  const controls = $("#timeline-controls");
  const btnGif = $("#btn-generate-gif");
  
  if (!controls || !btnGif) return;
  
  // Show button only if timelapse is enabled
  const enableTimelapse = $("#enable-timelapse")?.checked;
  if (!enableTimelapse || !state.trainedOnce) {
    controls.classList.add("hidden");
    return;
  }
  
  controls.classList.remove("hidden");
  
  const newBtnGif = btnGif.cloneNode(true);
  btnGif.parentNode.replaceChild(newBtnGif, btnGif);
  
  if (state.plotCache[`${plotType}_gif`]) {
    newBtnGif.innerHTML = "<span>🔄</span> Replay GIF";
  } else {
    newBtnGif.innerHTML = "<span>🎬</span> Generate Time-Lapse GIF";
  }
  
  newBtnGif.addEventListener("click", () => {
    if (state.plotCache[`${plotType}_gif`]) {
      renderPlotImage(state.plotCache[`${plotType}_gif`], plotType);
    } else {
      generatePlotGif(plotType);
    }
  });
}

async function generatePlotGif(plotType) {
  showLoading("Generating GIF...");
  const btnGif = $("#btn-generate-gif");
  if (btnGif) btnGif.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/api/plot_gif`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plot_type: plotType }),
    });
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "GIF generation failed", "error");
      hideLoading();
      return;
    }

    state.plotCache[`${plotType}_gif`] = data.image;
    renderPlotImage(data.image, plotType);
    
    if (btnGif) {
      btnGif.innerHTML = "<span>🔄</span> Replay GIF";
    }
    
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
    hideLoading();
  } finally {
    if (btnGif) btnGif.disabled = false;
  }
}

// ── Stats Panel ───────────────────────────────────────────────
function clearStats() {
  const container = $("#stats-content");
  if (container) {
    container.innerHTML = `
      <div class="stat-row">
        <span class="stat-label">No model trained yet</span>
      </div>
    `;
  }
}

function renderStats(stats, data) {
  const container = $("#stats-content");
  if (!container) return;

  let html = "";
  const entries = Object.entries(stats);

  for (const [key, val] of entries) {
    const label = formatStatLabel(key);
    const formatted = typeof val === "number" ? formatNumber(val) : val;
    const cls = getStatClass(key, val);
    html += `
      <div class="stat-row">
        <span class="stat-label">${label}</span>
        <span class="stat-value ${cls}">${formatted}</span>
      </div>
    `;
  }

  // Add device info
  if (data.device) {
    html += `
      <div class="stat-row">
        <span class="stat-label">Device</span>
        <span class="stat-value">${data.device}</span>
      </div>
    `;
  }

  container.innerHTML = html;
}

function formatStatLabel(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace("W Mu", "W_μ")
    .replace("W Logvar", "W_logvar");
}

function formatNumber(n) {
  if (Number.isInteger(n)) return n.toString();
  if (Math.abs(n) < 0.001) return n.toExponential(3);
  return n.toFixed(6);
}

function getStatClass(key, val) {
  if (key.includes("active")) return val > 100 ? "success" : val > 10 ? "" : "warning";
  return "";
}

// ── Loading Overlay ───────────────────────────────────────────
function showLoading(text) {
  const overlay = $("#loading-overlay");
  if (overlay) {
    const textEl = overlay.querySelector(".loading-overlay__text");
    if (textEl) textEl.textContent = text || "Loading...";
    overlay.classList.remove("hidden");
  }
}

function hideLoading() {
  const overlay = $("#loading-overlay");
  if (overlay) {
    overlay.classList.add("hidden");
  }
}

// ── Toast Notifications ───────────────────────────────────────
function showToast(message, type = "info") {
  let container = $(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const icons = { success: "✓", error: "✕", info: "ℹ" };
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || "ℹ"}</span> ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(40px)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Report Generation ─────────────────────────────────────────
function setupReportButton() {
  const btn = $("#btn-report");
  if (btn) {
    btn.addEventListener("click", generateReport);
  }
}

async function generateReport() {
  if (!state.trainedOnce) {
    showToast("Train a model first", "warning");
    return;
  }

  try {
    const resp = await fetch(`${API_BASE}/api/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Report generation failed", "error");
      return;
    }

    showReportModal(data.report);
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

function showReportModal(report) {
  // Remove existing modal
  const existing = $(".modal-backdrop");
  if (existing) existing.remove();

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) backdrop.remove();
  });

  let paramsHtml = "";
  for (const [key, val] of Object.entries(report.parameters)) {
    paramsHtml += `<tr><th>${key}</th><td>${val}</td></tr>`;
  }

  let resultsHtml = "";
  for (const [key, val] of Object.entries(report.results)) {
    resultsHtml += `<tr><th>${key}</th><td>${val}</td></tr>`;
  }

  backdrop.innerHTML = `
    <div class="modal" id="report-modal">
      <div class="modal__header">
        <h2 class="modal__title">Experiment Report</h2>
        <button class="modal__close" onclick="this.closest('.modal-backdrop').remove()">✕</button>
      </div>

      <div class="modal__section">
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:var(--space-md);">
          Generated: ${new Date(report.timestamp).toLocaleString()}
        </div>
      </div>

      <div class="modal__section">
        <h3 class="modal__section-title">Parameters</h3>
        <table class="report-table">
          <tbody>${paramsHtml}</tbody>
        </table>
      </div>

      <div class="modal__section">
        <h3 class="modal__section-title">Results</h3>
        <table class="report-table">
          <tbody>${resultsHtml}</tbody>
        </table>
      </div>

      <div class="modal__actions">
        <button class="btn-secondary" onclick="downloadReportJSON()">⬇ Download JSON</button>
        <button class="btn-primary" onclick="printReport()">🖨 Print / Save PDF</button>
      </div>
    </div>
  `;

  document.body.appendChild(backdrop);

  // Store for download
  window._lastReport = report;
}

function downloadReportJSON() {
  if (!window._lastReport) return;
  const blob = new Blob([JSON.stringify(window._lastReport, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `vae_superposition_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Report downloaded", "success");
}

function printReport() {
  const modal = document.getElementById("report-modal");
  if (!modal) return;

  const printWindow = window.open("", "_blank");
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>VAE Superposition Report</title>
      <style>
        body { font-family: 'Inter', sans-serif; padding: 40px; color: #1a1a2e; }
        h1 { font-size: 1.5rem; margin-bottom: 8px; }
        h2 { font-size: 1.1rem; margin-top: 24px; margin-bottom: 12px; color: #6d28d9; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 0.9rem; }
        th { color: #6b7280; font-weight: 500; }
        td { font-family: 'JetBrains Mono', monospace; }
        .timestamp { font-size: 0.8rem; color: #9ca3af; }
      </style>
    </head>
    <body>
      <h1>VAE Superposition Experiment Report</h1>
      <p class="timestamp">Generated: ${new Date().toLocaleString()}</p>
      ${modal.querySelector(".modal__section:nth-child(2)")?.outerHTML || ""}
      ${modal.querySelector(".modal__section:nth-child(3)")?.outerHTML || ""}
    </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.print();
}


// ═══════════════════════════════════════════════════════════════
// Parameter Sweep Grid Logic
// ═══════════════════════════════════════════════════════════════

async function runSweep() {
  if (state.isTraining) return;
  state.isTraining = true;

  const btn = $("#btn-train");
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:18px;height:18px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px;"></span> Sweeping...`;

  const btnExport = $("#btn-export-sweep");
  if (btnExport) btnExport.disabled = true;

  $("#sweep-empty")?.classList.add("hidden");
  $("#sweep-grid-wrapper")?.classList.add("hidden");
  
  const progressContainer = $("#sweep-progress-container");
  const progressFill = $("#sweep-progress-fill");
  const progressText = $("#sweep-progress-text");
  
  if (progressContainer) progressContainer.classList.remove("hidden");
  if (progressFill) progressFill.style.width = "0%";
  if (progressText) progressText.textContent = "Initializing sweep grid...";

  // Collect checked sweep plot types
  const selectedCbs = $$("#sweep-plot-checkboxes input[type='checkbox']:checked");
  const plotTypes = Array.from(selectedCbs).map(cb => cb.value);

  if (plotTypes.length === 0) {
    showToast("Please select at least one Grid Output Plot type.", "warning");
    state.isTraining = false;
    btn.disabled = false;
    btn.innerHTML = originalText;
    if (progressContainer) progressContainer.classList.add("hidden");
    $("#sweep-empty")?.classList.remove("hidden");
    return;
  }

  // Parse architecture sweep mode and custom list
  const sweepArchModeVal = $("#sweep-arch-mode")?.value ?? "single";
  const customListVal = $("#sweep-custom-list")?.value.trim() ?? "";
  const customSweeps = [];

  if (sweepArchModeVal === "custom" && customListVal) {
    const lines = customListVal.split("\n");
    for (let line of lines) {
      line = line.trim();
      if (!line || line.startsWith("#")) continue;
      const parts = line.split(",").map(p => parseInt(p.trim()));
      if (parts.length >= 2) {
        customSweeps.push({
          features: parts[0],
          hidden: parts[1],
          runs: parts[2] || 1
        });
      }
    }
  }

  if (sweepArchModeVal === "custom" && customSweeps.length === 0) {
    showToast("Please enter at least one valid sweep configuration (format: features, hidden, runs).", "warning");
    state.isTraining = false;
    btn.disabled = false;
    btn.innerHTML = originalText;
    if (progressContainer) progressContainer.classList.add("hidden");
    $("#sweep-empty")?.classList.remove("hidden");
    return;
  }

  // Collect sweep params
  const config = {
    model_type: state.modelType,
    n_features: parseInt($("#n-features")?.value ?? 256),
    hidden: parseInt($("#hidden-dim")?.value ?? 2),
    init_method: $("#init-method")?.value ?? "xavier_normal",
    normalize_weights: $("#normalize-weights")?.checked ?? false,
    steps: parseInt($("#sweep-steps")?.value ?? 1000),
    batch_size: parseInt($("#batch-size")?.value ?? 512),
    lr: parseFloat($("#learning-rate")?.value ?? 0.001),
    beta: parseFloat($("#beta")?.value ?? 0.05),
    beta_warmup: parseInt($("#beta-warmup")?.value ?? 2000),
    
    // Sweep range config
    density_start: parseFloat($("#sweep-density-start")?.value ?? 0.1),
    density_step: parseFloat($("#sweep-density-step")?.value ?? 0.3),
    density_end: parseFloat($("#sweep-density-end")?.value ?? 1.0),
    importance_decay_start: parseFloat($("#sweep-decay-start")?.value ?? 0.0),
    importance_decay_step: parseFloat($("#sweep-decay-step")?.value ?? 1.0),
    importance_decay_end: parseFloat($("#sweep-decay-end")?.value ?? 2.0),

    // Features sweep configs
    sweep_arch_mode: sweepArchModeVal,
    sweep_features: sweepArchModeVal === "range",
    features_start: parseInt($("#sweep-features-start")?.value ?? 64),
    features_step: parseInt($("#sweep-features-step")?.value ?? 64),
    features_end: parseInt($("#sweep-features-end")?.value ?? 256),
    custom_sweeps: customSweeps.length > 0 ? customSweeps : null,
    
    plot_types: plotTypes,
    plot_type: plotTypes[0], // fallback for backwards compatibility
  };

  if (state.customLossExpr) {
    config.custom_loss_expr = state.customLossExpr;
  }

  // Determine grid dimensions for progress calculation
  const densities = [];
  for (let d = config.density_start; d <= config.density_end + 1e-5; d += config.density_step) {
    densities.push(d);
    if (config.density_step <= 0) break;
  }
  const decays = [];
  for (let dec = config.importance_decay_start; dec <= config.importance_decay_end + 1e-5; dec += config.importance_decay_step) {
    decays.push(dec);
    if (config.importance_decay_step <= 0) break;
  }

  const sweepConfigs = [];
  if (config.sweep_arch_mode === "custom") {
    customSweeps.forEach(item => {
      for (let r = 1; r <= item.runs; r++) {
        sweepConfigs.push({
          features: item.features,
          hidden: item.hidden,
          run_id: r
        });
      }
    });
  } else if (config.sweep_arch_mode === "range") {
    for (let f = config.features_start; f <= config.features_end; f += config.features_step) {
      sweepConfigs.push({
        features: f,
        hidden: config.hidden,
        run_id: 1
      });
      if (config.features_step <= 0) break;
    }
  } else {
    sweepConfigs.push({
      features: config.n_features,
      hidden: config.hidden,
      run_id: 1
    });
  }
  
  const totalRuns = densities.length * decays.length * sweepConfigs.length;
  if (totalRuns > 500) {
    showToast(`Sweep grid is too large (${totalRuns} runs). Max allowed is 500 to prevent CPU lockups.`, "error");
    state.isTraining = false;
    btn.disabled = false;
    btn.innerHTML = originalText;
    if (progressContainer) progressContainer.classList.add("hidden");
    $("#sweep-empty")?.classList.remove("hidden");
    return;
  }

  // Update label
  $("#sweep-active-plot-label").textContent = PLOT_LABELS[config.plot_type] || config.plot_type;

  if (progressText) {
    progressText.textContent = `Training ${totalRuns} models in grid sweep... (This may take up to ${Math.round(totalRuns * 0.9)}s)`;
  }
  if (progressFill) {
    progressFill.style.width = "40%";
  }

  try {
    const resp = await fetch(`${API_BASE}/api/sweep`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    
    if (progressFill) progressFill.style.width = "85%";
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Sweep failed", "error");
      if (progressContainer) progressContainer.classList.add("hidden");
      $("#sweep-empty")?.classList.remove("hidden");
      return;
    }

    if (progressFill) progressFill.style.width = "100%";
    
    // Initialize default active plot tab and feature/hidden/run in state
    const firstCfg = data.sweep_configs[0];
    state.activeSweepPlot = plotTypes[0];
    state.activeSweepFeatures = firstCfg.features;
    state.activeSweepHidden = firstCfg.hidden;
    state.activeSweepRun = firstCfg.run_id;
    state.lastSweepData = data;
    state.lastSweepPlotTypes = plotTypes;
    
    renderSweepFeatureTabs(data.sweep_configs);
    renderSweepGrid(data, plotTypes);

    // Save initial report data for active feature count
    window._lastSweepReport = {
      timestamp: new Date().toISOString(),
      parameters: {
        "Model Type": state.modelType.toUpperCase(),
        "Hidden Dimension": firstCfg.hidden,
        "Number of Features (n)": firstCfg.features,
        "Sweep Run ID": `#${firstCfg.run_id}`,
        "Weight Initialization": $("#init-method")?.value.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) || "Xavier Normal",
        "Density Sweep Range": `${config.density_start} to ${config.density_end} (step ${config.density_step})`,
        "Decay Sweep Range": `${config.importance_decay_start} to ${config.importance_decay_end} (step ${config.importance_decay_step})`,
        "Training Steps per Model": config.steps,
        "Learning Rate": config.lr,
        "Batch Size": config.batch_size,
        "Grid Plot Type": PLOT_LABELS[state.activeSweepPlot] || state.activeSweepPlot
      },
      grid: data.grid.filter(cell => cell.features === firstCfg.features && cell.hidden === firstCfg.hidden && cell.run_id === firstCfg.run_id).map(cell => ({
        density: cell.density,
        importance_decay: cell.importance_decay,
        final_loss: cell.final_loss,
        recon: cell.recon,
        kl: cell.kl,
        plot_path: cell.plot_paths ? (cell.plot_paths[state.activeSweepPlot] || cell.plot_path) : cell.plot_path,
        plot_paths: cell.plot_paths
      }))
    };
    if (btnExport) btnExport.disabled = false;

    showToast("Sweep completed successfully", "success");
    
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
    $("#sweep-empty")?.classList.remove("hidden");
  } finally {
    state.isTraining = false;
    btn.disabled = false;
    btn.innerHTML = originalText;
    if (progressContainer) {
      setTimeout(() => progressContainer.classList.add("hidden"), 500);
    }
  }
}

function renderSweepGrid(data, plotTypes) {
  const container = $("#sweep-grid-wrapper");
  if (!container) return;

  container.innerHTML = "";
  container.classList.remove("hidden");

  // Show filter controls card and reset inputs
  const filterCard = $("#sweep-filter-card");
  if (filterCard) {
    filterCard.classList.remove("hidden");
  }
  const filterInputs = [
    "#filter-min-density",
    "#filter-min-decay",
    "#filter-max-loss",
    "#filter-min-norm"
  ];
  filterInputs.forEach(sel => {
    const el = $(sel);
    if (el) el.value = "";
  });

  // Render Sweep tabs dynamically
  renderSweepPlotTabs(plotTypes);

  // Set active label in header
  $("#sweep-active-plot-label").textContent = PLOT_LABELS[state.activeSweepPlot] || state.activeSweepPlot;

  const densities = data.density_values;
  const decays = data.importance_decay_values;
  
  // Filter grid data to only include the active sweep configuration
  const activeFeatures = state.activeSweepFeatures || data.sweep_configs[0].features;
  const activeHidden = state.activeSweepHidden || data.sweep_configs[0].hidden;
  const activeRun = state.activeSweepRun || data.sweep_configs[0].run_id;
  
  const grid = data.grid.filter(item => 
    item.features === activeFeatures && 
    item.hidden === activeHidden && 
    item.run_id === activeRun
  );

  // Create sweep matrix container
  const matrixContainer = document.createElement("div");
  matrixContainer.className = "sweep-matrix-container";

  // Build grid styled matrix
  const matrix = document.createElement("div");
  matrix.className = "sweep-matrix";
  
  // Custom columns styling: y-axis label col (100px) + flex cols for each decay value
  matrix.style.gridTemplateColumns = `100px repeat(${decays.length}, minmax(180px, 1fr))`;

  // 1. Header row
  // empty spacer top-left cell
  const spacer = document.createElement("div");
  spacer.className = "sweep-matrix-spacer";
  spacer.innerHTML = `<span style="font-size:0.7rem;color:var(--text-muted);font-weight:bold;display:flex;align-items:center;justify-content:center;height:100%;">Density \\ Decay</span>`;
  matrix.appendChild(spacer);

  // X-axis label elements
  decays.forEach(dec => {
    const lbl = document.createElement("div");
    lbl.className = "sweep-matrix-x-label";
    lbl.innerHTML = `Decay: ${dec.toFixed(2)}`;
    matrix.appendChild(lbl);
  });

  // 2. Rows for each density
  // Sort densities descending so high density is at top
  const sortedDensities = [...densities].sort((a,b) => b - a);
  const cellCheckboxes = [];

  sortedDensities.forEach(d_val => {
    // Y-axis label
    const yLbl = document.createElement("div");
    yLbl.className = "sweep-matrix-y-label";
    yLbl.innerHTML = `Density: ${(d_val * 100).toFixed(0)}%`;
    matrix.appendChild(yLbl);

    // Cells for this density
    decays.forEach(dec_val => {
      // Find matching grid item
      const cellData = grid.find(item => Math.abs(item.density - d_val) < 1e-4 && Math.abs(item.importance_decay - dec_val) < 1e-4);
      const cell = document.createElement("div");
      cell.className = "sweep-matrix-cell";

      if (cellData) {
        // Set dataset attributes for filtering
        cell.dataset.density = d_val;
        cell.dataset.decay = dec_val;
        cell.dataset.loss = cellData.final_loss;
        cell.dataset.norm = cellData.mean_weight_norm || 0;

        // Checkbox container
        const cbContainer = document.createElement("div");
        cbContainer.className = "sweep-cell-cb-container";
        cbContainer.innerHTML = `
          <label class="sweep-cell-cb-label">
            <input type="checkbox" class="sweep-cell-cb" />
            <span>Plot</span>
          </label>
        `;
        const cb = cbContainer.querySelector(".sweep-cell-cb");
        cellCheckboxes.push(cb);

        // Create visual view plot hint card
        const placeholder = document.createElement("div");
        placeholder.className = "sweep-matrix-cell__placeholder";
        placeholder.innerHTML = `
          <span class="sweep-matrix-cell__icon">🔍</span>
          <span>View Plot</span>
        `;

        // Image Container (initially hidden, lazy loaded)
        const imgContainer = document.createElement("div");
        imgContainer.className = "sweep-matrix-cell__img-container hidden";
        const img = document.createElement("img");
        img.className = "sweep-matrix-cell__img";
        imgContainer.appendChild(img);
        
        cell.appendChild(cbContainer);
        cell.appendChild(placeholder);
        cell.appendChild(imgContainer);

        // Stop propagation so clicking checkbox doesn't trigger lightbox zoom
        cbContainer.addEventListener("click", (e) => {
          e.stopPropagation();
        });

        // Toggle visibility and lazy-load image on check
        cb.addEventListener("change", () => {
          if (cb.checked) {
            cell.classList.add("expanded");
            imgContainer.classList.remove("hidden");
            placeholder.classList.add("hidden");
            if (!img.src) {
              const activePath = cellData.plot_paths ? (cellData.plot_paths[state.activeSweepPlot] || cellData.plot_path) : cellData.plot_path;
              img.src = activePath;
            }
          } else {
            cell.classList.remove("expanded");
            imgContainer.classList.add("hidden");
            placeholder.classList.remove("hidden");
            
            // Sync Toggle All state if unchecked
            const toggleAllEl = document.getElementById("sweep-toggle-all-plots");
            if (toggleAllEl && toggleAllEl.checked) {
              toggleAllEl.checked = false;
              // Synchronize with sidebar checkbox
              const sidebarCb = document.getElementById("sweep-show-plots-global");
              if (sidebarCb) sidebarCb.checked = false;
            }
          }
        });
        
        // Add zoom action (loads from local temp file)
        cell.addEventListener("click", (e) => {
          if (e.target.closest(".sweep-cell-cb-container")) return;
          const activePath = cellData.plot_paths ? (cellData.plot_paths[state.activeSweepPlot] || cellData.plot_path) : cellData.plot_path;
          openLightbox(activePath, `n: ${activeFeatures} | h: ${activeHidden} | Run: ${activeRun} | Density: ${(d_val*100).toFixed(0)}% | Decay: ${dec_val.toFixed(2)} | Loss: ${cellData.final_loss.toFixed(6)} | Plot: ${PLOT_LABELS[state.activeSweepPlot] || state.activeSweepPlot}`);
        });

        // Footer info (final loss)
        const info = document.createElement("div");
        info.className = "sweep-matrix-cell__info";
        info.innerHTML = `<span>Loss</span><span class="sweep-loss-value">${cellData.final_loss.toFixed(5)}</span>`;
        cell.appendChild(info);
      } else {
        cell.innerHTML = `<div style="padding:15px;text-align:center;font-size:0.7rem;color:var(--text-muted);">Error</div>`;
      }
      
      matrix.appendChild(cell);
    });
  });

  matrixContainer.appendChild(matrix);
  container.appendChild(matrixContainer);

  // Auto-expand if show-plots checkbox is checked
  const isGlobalChecked = $("#sweep-show-plots-global")?.checked || $("#sweep-toggle-all-plots")?.checked;
  if (isGlobalChecked) {
    setTimeout(() => {
      const toggleAllEl = document.getElementById("sweep-toggle-all-plots");
      if (toggleAllEl) {
        toggleAllEl.checked = true;
        toggleAllEl.dispatchEvent(new Event("change"));
      }
    }, 0);
  }

  // Add Axis titles
  const xTitle = document.createElement("div");
  xTitle.className = "sweep-matrix-axis-title-x";
  xTitle.innerHTML = `Importance Decay Sweep (→ X Axis)`;
  container.appendChild(xTitle);
}

function renderSweepPlotTabs(plotTypes) {
  const container = $("#sweep-plot-tabs");
  if (!container) return;

  container.innerHTML = "";
  if (!plotTypes || plotTypes.length <= 1) {
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");

  plotTypes.forEach((ptype) => {
    const tab = document.createElement("button");
    tab.className = `plot-tab${ptype === state.activeSweepPlot ? " active" : ""}`;
    tab.textContent = PLOT_LABELS[ptype] || ptype;
    tab.dataset.plot = ptype;
    tab.addEventListener("click", () => {
      container.querySelectorAll(".plot-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      state.activeSweepPlot = ptype;
      $("#sweep-active-plot-label").textContent = PLOT_LABELS[ptype] || ptype;
      
      updateSweepGridImages();
    });
    container.appendChild(tab);
  });
}

function renderSweepFeatureTabs(sweepConfigs) {
  const container = $("#sweep-feature-tabs");
  if (!container) return;

  container.innerHTML = "";
  if (!sweepConfigs || sweepConfigs.length <= 1) {
    container.classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");

  // Create a label prefix span
  const labelPrefix = document.createElement("span");
  labelPrefix.style.fontSize = "0.75rem";
  labelPrefix.style.color = "var(--text-muted)";
  labelPrefix.style.marginRight = "8px";
  labelPrefix.style.alignSelf = "center";
  labelPrefix.style.fontWeight = "bold";
  labelPrefix.textContent = "Sweep Configuration:";
  container.appendChild(labelPrefix);

  sweepConfigs.forEach((cfg) => {
    const tab = document.createElement("button");
    const active = state.activeSweepFeatures === cfg.features && 
                   state.activeSweepHidden === cfg.hidden && 
                   state.activeSweepRun === cfg.run_id;
    tab.className = `plot-tab${active ? " active" : ""}`;
    
    // Label format: n=4, h=2 (#1)
    tab.textContent = `n=${cfg.features}, h=${cfg.hidden} (#${cfg.run_id})`;
    
    tab.addEventListener("click", () => {
      container.querySelectorAll(".plot-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      state.activeSweepFeatures = cfg.features;
      state.activeSweepHidden = cfg.hidden;
      state.activeSweepRun = cfg.run_id;
      
      // Re-render the grid (filtering by active configuration)
      if (state.lastSweepData) {
        renderSweepGrid(state.lastSweepData, state.lastSweepPlotTypes);
      }
    });
    container.appendChild(tab);
  });
}

function updateSweepGridImages() {
  const cells = $$(".sweep-matrix-cell");
  const activeFeatures = state.activeSweepFeatures || state.lastSweepData?.sweep_configs[0].features;
  const activeHidden = state.activeSweepHidden || state.lastSweepData?.sweep_configs[0].hidden;
  const activeRun = state.activeSweepRun || state.lastSweepData?.sweep_configs[0].run_id;

  cells.forEach(cell => {
    const density = parseFloat(cell.dataset.density);
    const decay = parseFloat(cell.dataset.decay);

    // Find the cell data from state.lastSweepData
    const cellData = state.lastSweepData?.grid.find(item => 
      item.features === activeFeatures &&
      item.hidden === activeHidden &&
      item.run_id === activeRun &&
      Math.abs(item.density - density) < 1e-4 && 
      Math.abs(item.importance_decay - decay) < 1e-4
    );

    if (cellData && cellData.plot_paths) {
      const activePath = cellData.plot_paths[state.activeSweepPlot];
      const img = cell.querySelector(".sweep-matrix-cell__img");
      if (img && cell.classList.contains("expanded")) {
        img.src = activePath;
      }
    }
  });
}

function applySweepFilters() {
  const minDensity = parseFloat($("#filter-min-density")?.value);
  const minDecay = parseFloat($("#filter-min-decay")?.value);
  const maxLoss = parseFloat($("#filter-max-loss")?.value);
  const minNorm = parseFloat($("#filter-min-norm")?.value);

  const cells = $$(".sweep-matrix-cell");
  cells.forEach(cell => {
    const d = parseFloat(cell.dataset.density);
    const dec = parseFloat(cell.dataset.decay);
    const loss = parseFloat(cell.dataset.loss);
    const norm = parseFloat(cell.dataset.norm);

    let match = true;
    if (!isNaN(minDensity) && d < minDensity) match = false;
    if (!isNaN(minDecay) && dec < minDecay) match = false;
    if (!isNaN(maxLoss) && loss > maxLoss) match = false;
    if (!isNaN(minNorm) && norm < minNorm) match = false;

    if (match) {
      cell.classList.remove("filtered-out");
    } else {
      cell.classList.add("filtered-out");
      // Uncheck if filtered out
      const cb = cell.querySelector(".sweep-cell-cb");
      if (cb && cb.checked) {
        cb.checked = false;
        cb.dispatchEvent(new Event("change"));
      }
    }
  });
}

function openLightbox(imageSrc, caption) {
  const lightbox = $("#lightbox");
  const lightboxImg = $("#lightbox-img");
  const lightboxCap = $("#lightbox-caption");

  if (lightbox && lightboxImg && lightboxCap) {
    lightboxImg.src = imageSrc; // Static URL path served from flask temp_sweep/
    lightboxCap.textContent = caption;
    lightbox.classList.remove("hidden");
  }
}

function initModeTabs() {
  const workspaceSingle = $("#workspace-single");
  const workspaceSweep = $("#workspace-sweep");
  const workspaceGeometry = $("#workspace-geometry");
  const workspaceOrthogonality = $("#workspace-orthogonality");
  const cardSweep = $("#card-sweep");
  const cardData = document.querySelectorAll(".card")[2];
  const btnTrain = $("#btn-train");
  const navigation = [...document.querySelectorAll(".unified-nav__item")];

  function activateWorkspace(workspace, geometryView = null) {
    navigation.forEach(item => {const active=item.dataset.workspace === workspace && (workspace !== "geometry" || item.dataset.geometryView === geometryView);item.classList.toggle("active",active);if(active)item.setAttribute("aria-current","page");else item.removeAttribute("aria-current");});
    [workspaceSingle, workspaceSweep, workspaceGeometry, workspaceOrthogonality].forEach(panel=>panel?.classList.add("hidden"));
    document.body.classList.remove("geometry-mode", "orthogonality-mode");

    if (workspace === "geometry") {
      workspaceGeometry?.classList.remove("hidden");
      document.body.classList.add("geometry-mode");
      state.sweepMode = "geometry";
      window._pendingGeometryView = geometryView || "data";
      window.geometryLabSetView?.(window._pendingGeometryView);
      const badge=$(".mode-badge");if(badge)badge.textContent="LAB";
    } else if (workspace === "single") {
      workspaceSingle?.classList.remove("hidden");
      cardSweep?.classList.add("hidden");
      if(cardData){cardData.style.opacity="1";cardData.style.pointerEvents="auto";}
      if(btnTrain)btnTrain.innerHTML="▶ Train Model";
      state.sweepMode="single";updateModeBadge();
    } else if (workspace === "sweep") {
      workspaceSweep?.classList.remove("hidden");
      cardSweep?.classList.remove("hidden");
      if(cardData){cardData.style.opacity="0.25";cardData.style.pointerEvents="none";}
      if(btnTrain)btnTrain.innerHTML="▶ Run Sweep Analysis";
      state.sweepMode="sweep";const badge=$(".mode-badge");if(badge)badge.textContent="SWEEP";updateSweepPlotOptions();
    } else if (workspace === "orthogonality") {
      workspaceOrthogonality?.classList.remove("hidden");
      document.body.classList.add("orthogonality-mode");
      state.sweepMode="orthogonality";const badge=$(".mode-badge");if(badge)badge.textContent="GRAM";
    }
    window.dispatchEvent(new Event("resize"));
  }

  window.activateUnifiedWorkspace=activateWorkspace;
  navigation.forEach(item=>item.addEventListener("click",()=>activateWorkspace(item.dataset.workspace,item.dataset.geometryView||null)));
  const initial=navigation.find(item=>item.classList.contains("active"))||navigation[0];
  if(initial)activateWorkspace(initial.dataset.workspace,initial.dataset.geometryView||null);
}

function updateSweepPlotOptions() {
  const container = $("#sweep-plot-checkboxes");
  if (!container) return;

  const hidden = parseInt($("#hidden-dim")?.value ?? 2);
  const is2d = (hidden === 2);
  const isVae = (state.modelType === "vae");
  const isManifold = (state.modelType === "manifold_vae");

  let options = [];

  if (isManifold) {
    if (is2d) {
      options = [
        { value: "manifold_combined", label: "Manifold Combined (Default)" },
        { value: "arrows_W_tangent", label: "W_tangent Arrows" },
        { value: "manifold_disk_scatter", label: "Manifold Disk" },
        { value: "geodesic_matrix", label: "Geodesic Matrix" },
        { value: "curvature_heatmap", label: "Curvature λ(x)" },
        { value: "interference_W_tangent", label: "W_tangent Interference" },
        { value: "norms_W_tangent", label: "W_tangent Norms" },
        { value: "graph_W_tangent", label: "W_tangent Graph" },
        { value: "coord_graph_W_tangent", label: "W_tangent Coord Graph" },
        { value: "manifold_jacobian_W_tangent", label: "W_tangent Jacobian" },
        { value: "manifold_geometry_W_tangent", label: "W_tangent Geometry" },
        { value: "manifold_summary_W_tangent", label: "W_tangent Manifold Summary" }
      ];
    } else {
      options = [
        { value: "pca_W_tangent", label: "W_tangent PCA (Default)" },
        { value: "manifold_disk_scatter", label: "Manifold Disk" },
        { value: "geodesic_matrix", label: "Geodesic Matrix" },
        { value: "curvature_heatmap", label: "Curvature λ(x)" },
        { value: "heatmap_W_tangent", label: "W_tangent Heatmap" },
        { value: "interference_W_tangent", label: "W_tangent Interference" },
        { value: "norms_W_tangent", label: "W_tangent Norms" },
        { value: "graph_W_tangent", label: "W_tangent Graph" },
        { value: "coord_graph_W_tangent", label: "W_tangent Coord Graph" },
        { value: "manifold_jacobian_W_tangent", label: "W_tangent Jacobian" },
        { value: "manifold_geometry_W_tangent", label: "W_tangent Geometry" },
        { value: "manifold_summary_W_tangent", label: "W_tangent Manifold Summary" }
      ];
    }
  } else if (isVae) {
    if (is2d) {
      options = [
        { value: "combined", label: "Combined Overlay (Default)" },
        { value: "arrows_W_mu", label: "W_μ Arrows" },
        { value: "arrows_W_logvar", label: "W_logvar Arrows" },
        { value: "z_samples", label: "z Samples" },
        { value: "interference_W_mu", label: "W_μ Interference" },
        { value: "interference_W_logvar", label: "W_logvar Interference" },
        { value: "norms_W_mu", label: "W_μ Norms" },
        { value: "norms_W_logvar", label: "W_logvar Norms" },
        { value: "graph_W_mu", label: "W_μ Graph" },
        { value: "graph_W_logvar", label: "W_logvar Graph" },
        { value: "coord_graph_W_mu", label: "W_μ Coord Graph" },
        { value: "coord_graph_W_logvar", label: "W_logvar Coord Graph" },
        { value: "manifold_jacobian_W_mu", label: "W_μ Jacobian" },
        { value: "manifold_geometry_W_mu", label: "W_μ Geometry" },
        { value: "manifold_summary_W_mu", label: "W_μ Manifold Summary" }
      ];
    } else {
      options = [
        { value: "pca_W_mu", label: "W_μ PCA (Default)" },
        { value: "pca_W_logvar", label: "W_logvar PCA" },
        { value: "heatmap_W_mu", label: "W_μ Heatmap" },
        { value: "heatmap_W_logvar", label: "W_logvar Heatmap" },
        { value: "interference_W_mu", label: "W_μ Interference" },
        { value: "interference_W_logvar", label: "W_logvar Interference" },
        { value: "norms_W_mu", label: "W_μ Norms" },
        { value: "norms_W_logvar", label: "W_logvar Norms" },
        { value: "graph_W_mu", label: "W_μ Graph" },
        { value: "graph_W_logvar", label: "W_logvar Graph" },
        { value: "coord_graph_W_mu", label: "W_μ Coord Graph" },
        { value: "coord_graph_W_logvar", label: "W_logvar Coord Graph" },
        { value: "manifold_jacobian_W_mu", label: "W_μ Jacobian" },
        { value: "manifold_geometry_W_mu", label: "W_μ Geometry" },
        { value: "manifold_summary_W_mu", label: "W_μ Manifold Summary" }
      ];
    }
  } else {
    if (is2d) {
      options = [
        { value: "arrows_W", label: "W Arrows (Default)" },
        { value: "interference_W", label: "W Interference" },
        { value: "norms_W", label: "W Norms" },
        { value: "graph_W", label: "W Graph" },
        { value: "coord_graph_W", label: "W Coord Graph" },
        { value: "manifold_jacobian_W", label: "W Jacobian" },
        { value: "manifold_geometry_W", label: "W Geometry" },
        { value: "manifold_summary_W", label: "W Manifold Summary" }
      ];
    } else {
      options = [
        { value: "pca_W", label: "W PCA (Default)" },
        { value: "heatmap_W", label: "W Heatmap" },
        { value: "interference_W", label: "W Interference" },
        { value: "norms_W", label: "W Norms" },
        { value: "graph_W", label: "W Graph" },
        { value: "coord_graph_W", label: "W Coord Graph" },
        { value: "manifold_jacobian_W", label: "W Jacobian" },
        { value: "manifold_geometry_W", label: "W Geometry" },
        { value: "manifold_summary_W", label: "W Manifold Summary" }
      ];
    }
  }

  // Preserve previous checked values
  const checkedValues = new Set();
  container.querySelectorAll("input[type='checkbox']:checked").forEach(cb => {
    checkedValues.add(cb.value);
  });

  container.innerHTML = "";
  options.forEach((opt, idx) => {
    const item = document.createElement("div");
    item.style.display = "flex";
    item.style.alignItems = "center";
    item.style.gap = "8px";
    item.style.padding = "2px 0";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = opt.value;
    cb.id = `sweep-cb-plot-${opt.value}`;
    cb.style.accentColor = "var(--accent-2)";
    cb.style.cursor = "pointer";
    cb.style.margin = "0";
    cb.style.width = "14px";
    cb.style.height = "14px";

    if (checkedValues.has(opt.value) || (checkedValues.size === 0 && idx === 0)) {
      cb.checked = true;
    }

    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = opt.label;
    lbl.style.fontSize = "0.76rem";
    lbl.style.color = "var(--text-secondary)";
    lbl.style.cursor = "pointer";
    lbl.style.userSelect = "none";
    lbl.style.margin = "0";

    item.appendChild(cb);
    item.appendChild(lbl);
    container.appendChild(item);
  });
}

// ── Sweep Export & Print Layout ────────────────────────────────

function showSweepReportModal(report) {
  const existing = $(".modal-backdrop");
  if (existing) existing.remove();

  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop";
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) backdrop.remove();
  });

  let paramsHtml = "";
  for (const [key, val] of Object.entries(report.parameters)) {
    paramsHtml += `<tr><th>${key}</th><td>${val}</td></tr>`;
  }

  let tableRows = "";
  report.grid.forEach(row => {
    const reconVal = row.recon !== null ? row.recon.toFixed(6) : "-";
    const klVal = row.kl !== null ? row.kl.toFixed(6) : "-";
    tableRows += `
      <tr>
        <td>${row.features}</td>
        <td>${row.hidden}</td>
        <td>#${row.run_id}</td>
        <td>${(row.density * 100).toFixed(0)}%</td>
        <td>${row.importance_decay.toFixed(2)}</td>
        <td>${row.final_loss.toFixed(6)}</td>
        <td>${reconVal}</td>
        <td>${klVal}</td>
      </tr>
    `;
  });

  backdrop.innerHTML = `
    <div class="modal" id="sweep-report-modal" style="max-width: 900px;">
      <div class="modal__header">
        <h2 class="modal__title">Sweep Experiment Report</h2>
        <button class="modal__close" onclick="this.closest('.modal-backdrop').remove()">✕</button>
      </div>

      <div class="modal__section">
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:var(--space-md);">
          Generated: ${new Date(report.timestamp).toLocaleString()}
        </div>
      </div>

      <div class="modal__section">
        <h3 class="modal__section-title">Sweep Parameters</h3>
        <table class="report-table">
          <tbody>${paramsHtml}</tbody>
        </table>
      </div>

      <div class="modal__section">
        <h3 class="modal__section-title">Grid Sweep Results</h3>
        <div style="max-height: 300px; overflow-y: auto; border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
          <table class="report-table" style="position: relative;">
            <thead style="position: sticky; top: 0; background: var(--bg-secondary); box-shadow: 0 1px 0 var(--border-subtle); z-index: 1;">
              <tr>
                <th>Features</th>
                <th>Hidden</th>
                <th>Run</th>
                <th>Density</th>
                <th>Importance Decay</th>
                <th>Final Loss</th>
                <th>Recon Loss</th>
                <th>KL Div</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      </div>

      <div class="modal__actions">
        <button class="btn-secondary" onclick="downloadSweepReportJSON()">⬇ Download JSON</button>
        <button class="btn-primary" onclick="printSweepReport()">🖨 Print / Save PDF</button>
      </div>
    </div>
  `;

  document.body.appendChild(backdrop);
}

function downloadSweepReportJSON() {
  if (!window._lastSweepReport) return;
  const blob = new Blob([JSON.stringify(window._lastSweepReport, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `vae_superposition_sweep_report_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("Sweep JSON report downloaded", "success");
}

function printSweepReport() {
  if (!window._lastSweepReport) return;
  const report = window._lastSweepReport;
  const modal = document.getElementById("sweep-report-modal");
  if (!modal) return;

  const printWindow = window.open("", "_blank");

  // Group by features, hidden, run_id
  const groups = {};
  report.grid.forEach(cell => {
    const key = `Features: ${cell.features} | Hidden Dim: ${cell.hidden} | Run: #${cell.run_id}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(cell);
  });

  let visualSweepsHtml = "";
  for (const [groupName, cells] of Object.entries(groups)) {
      // Sort cells by density descending, then decay ascending
      cells.sort((a, b) => {
          if (Math.abs(a.density - b.density) > 1e-5) return b.density - a.density;
          return a.importance_decay - b.importance_decay;
      });
      
      let gridCellsHtml = "";
      cells.forEach(cell => {
          gridCellsHtml += `
            <div class="sweep-cell">
              <img src="${window.location.origin}/${cell.plot_path}">
              <div class="sweep-cell__info">
                <span>Den: <b>${(cell.density * 100).toFixed(0)}%</b> | Dec: <b>${cell.importance_decay.toFixed(2)}</b></span>
                <span>Loss: <span class="sweep-loss-value">${cell.final_loss.toFixed(5)}</span></span>
              </div>
            </div>
          `;
      });
      
      visualSweepsHtml += `
        <div class="sweep-group">
          <h3>${groupName}</h3>
          <div class="sweep-grid">
            ${gridCellsHtml}
          </div>
        </div>
      `;
  }

  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>Sweep Experiment Report</title>
      <style>
        body { font-family: 'Inter', sans-serif; padding: 40px; background: #06060f; color: #e8e6f0; }
        h1 { font-size: 1.5rem; margin-bottom: 8px; color: #ffffff; }
        h2 { font-size: 1.1rem; margin-top: 24px; margin-bottom: 12px; color: #a78bfa; page-break-after: avoid; }
        h3 { font-size: 1.0rem; margin-top: 30px; margin-bottom: 16px; color: #e8e6f0; border-bottom: 1px solid #2a2554; padding-bottom: 6px; page-break-after: avoid; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; background: #0c0c1e; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2a2554; font-size: 0.85rem; }
        th { color: #a78bfa; font-weight: 500; background: #13122c; }
        td { font-family: 'JetBrains Mono', monospace; color: #e8e6f0; }
        .timestamp { font-size: 0.8rem; color: #9ca3af; margin-bottom: 20px; }
        
        .sweep-group { margin-bottom: 40px; page-break-inside: avoid; }
        .sweep-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; width: 100%; }
        .sweep-cell { border: 1px solid #2a2554; border-radius: 8px; overflow: hidden; background: #0c0c1e; display: flex; flex-direction: column; page-break-inside: avoid; }
        .sweep-cell img { width: 100%; display: block; object-fit: contain; }
        .sweep-cell__info { display: flex; justify-content: space-between; padding: 10px 14px; font-size: 0.85rem; border-top: 1px solid #2a2554; background: #13122c; color: #e8e6f0; font-family: 'JetBrains Mono', monospace; }
        .sweep-loss-value { font-weight: bold; color: #a78bfa; }
      </style>
    </head>
    <body>
      <h1>VAE Superposition Grid Sweep Report</h1>
      <p class="timestamp">Generated: ${new Date().toLocaleString()}</p>
      
      <h2>Parameters</h2>
      ${modal.querySelector(".modal__section:nth-child(2) table")?.outerHTML || ""}
      
      <h2>Grid Sweep Results</h2>
      ${modal.querySelector(".modal__section:nth-child(3) table")?.outerHTML || ""}
      
      <div style="page-break-before: always;">
        <h2>Visual Sweeps</h2>
        <div id="print-matrix-target">${visualSweepsHtml}</div>
      </div>
    </body>
    </html>
  `);
  printWindow.document.close();

  printWindow.onload = () => {
    setTimeout(() => {
      printWindow.print();
    }, 1000);
  };
}

// Expose report helpers globally
window.downloadSweepReportJSON = downloadSweepReportJSON;
window.printSweepReport = printSweepReport;
window.showSweepReportModal = showSweepReportModal;


// ═══════════════════════════════════════════════════════════════
// Custom Loss Function Editor — Redesigned
// ═══════════════════════════════════════════════════════════════

const LOSS_PRESETS = {
  standard: [
    { id: "default", label: "Default (MSE)", expr: "(importance * (x - x_hat) ** 2).sum(dim=1).mean()" },
    { id: "mse", label: "Pure MSE", expr: "((x - x_hat) ** 2).sum(dim=1).mean()" },
    { id: "l1", label: "L1 Loss", expr: "(importance * torch.abs(x - x_hat)).sum(dim=1).mean()" },
    { id: "huber", label: "Huber / Smooth L1", expr: "torch.nn.functional.smooth_l1_loss(x * importance, x_hat * importance)" },
  ],
  vae: [
    { id: "default", label: "Default (Recon + KL)", expr: "recon + beta_t * kl" },
    { id: "recon_only", label: "Recon Only", expr: "recon" },
    { id: "kl_only", label: "KL Only", expr: "kl" },
    { id: "heavy_kl", label: "Heavy KL (5x)", expr: "recon + 5.0 * beta_t * kl" },
  ],
  manifold_vae: [
    { id: "default", label: "Default (Riemannian KL)", expr: "recon + beta_t * kl" },
    { id: "riemannian_kl", label: "Riemannian KL Divergence", expr: "recon + beta_t * kl" },
    { id: "tangent_penalty", label: "Tangent Space Penalty", expr: "tangent_pen" },
    { id: "geodesic", label: "Geodesic Distance", expr: "geodesic_dist" },
    { id: "score_matching", label: "Riemannian Score Matching", expr: "score_matching" },
  ]
};

const LOSS_SNIPPETS = {
  standard: [
    { label: ".sum(dim=1).mean()", code: ".sum(dim=1).mean()", icon: "Σ" },
    { label: "(x - x_hat) ** 2", code: "(x - x_hat) ** 2", icon: "²" },
    { label: "torch.abs(...)", code: "torch.abs()", icon: "| |" },
    { label: "torch.clamp(..., min=0)", code: "torch.clamp(, min=0)", icon: "⌐" },
    { label: "importance * ...", code: "importance * ", icon: "w" },
    { label: ".mean()", code: ".mean()", icon: "μ" },
  ],
  vae: [
    { label: "recon + β·kl", code: "recon + beta_t * kl", icon: "β" },
    { label: ".sum(dim=1).mean()", code: ".sum(dim=1).mean()", icon: "Σ" },
    { label: "torch.abs(...)", code: "torch.abs()", icon: "| |" },
    { label: "mu.pow(2)", code: "mu.pow(2)", icon: "μ²" },
    { label: "logvar.exp()", code: "logvar.exp()", icon: "eˣ" },
    { label: "torch.clamp(..., min=0)", code: "torch.clamp(, min=0)", icon: "⌐" },
    { label: "importance * ...", code: "importance * ", icon: "w" },
    { label: ".mean()", code: ".mean()", icon: "μ" },
  ],
  manifold_vae: [
    { label: "recon + β·kl + pen", code: "recon + beta_t * kl + tangent_weight * tangent_pen", icon: "Ⓜ" },
    { label: "geodesic_dist", code: "geodesic_dist", icon: "G" },
    { label: "score_matching", code: "score_matching", icon: "S" },
    { label: "tangent_pen", code: "tangent_pen", icon: "T" },
  ]
};

const LOSS_VARIABLES = {
  standard: [
    { name: "x", kind: "tensor", shape: "[B, n]", desc: "Input batch" },
    { name: "x_hat", kind: "tensor", shape: "[B, n]", desc: "Reconstruction" },
    { name: "importance", kind: "tensor", shape: "[n]", desc: "Per-feature weight" },
    { name: "torch", kind: "module", shape: "", desc: "PyTorch namespace" }
  ],
  vae: [
    { name: "x", kind: "tensor", shape: "[B, n]", desc: "Input batch" },
    { name: "x_hat", kind: "tensor", shape: "[B, n]", desc: "Reconstruction" },
    { name: "mu", kind: "tensor", shape: "[B, h]", desc: "Encoder mean" },
    { name: "logvar", kind: "tensor", shape: "[B, h]", desc: "Log variance" },
    { name: "z", kind: "tensor", shape: "[B, h]", desc: "Latent sample" },
    { name: "importance", kind: "tensor", shape: "[n]", desc: "Per-feature weight" },
    { name: "beta_t", kind: "scalar", shape: "float", desc: "Current β (warmed)" },
    { name: "recon", kind: "scalar", shape: "scalar", desc: "Recon loss term" },
    { name: "kl", kind: "scalar", shape: "scalar", desc: "KL divergence" },
    { name: "torch", kind: "module", shape: "", desc: "PyTorch namespace" }
  ],
  manifold_vae: [
    { name: "x", kind: "tensor", shape: "[B, n]", desc: "Input batch" },
    { name: "x_hat", kind: "tensor", shape: "[B, n]", desc: "Reconstruction" },
    { name: "mu", kind: "tensor", shape: "[B, h]", desc: "Encoder mean" },
    { name: "logvar", kind: "tensor", shape: "[B, h]", desc: "Log variance" },
    { name: "z", kind: "tensor", shape: "[B, h]", desc: "Latent sample" },
    { name: "importance", kind: "tensor", shape: "[n]", desc: "Per-feature weight" },
    { name: "beta_t", kind: "scalar", shape: "float", desc: "Current β (warmed)" },
    { name: "tangent_weight", kind: "scalar", shape: "float", desc: "Tangent weight" },
    { name: "recon", kind: "scalar", shape: "scalar", desc: "Recon loss term" },
    { name: "kl", kind: "scalar", shape: "scalar", desc: "Riemannian KL term" },
    { name: "tangent_pen", kind: "scalar", shape: "scalar", desc: "Tangent penalty term" },
    { name: "geodesic_dist", kind: "scalar", shape: "scalar", desc: "Geodesic dist loss" },
    { name: "score_matching", kind: "scalar", shape: "scalar", desc: "Score matching loss" },
    { name: "torch", kind: "module", shape: "", desc: "PyTorch namespace" }
  ]
};

// Track what's currently applied vs what's in the editor
let _lossEditorAppliedExpr = null; // the expr that's actually active

function initLossEditor() {
  const select = $("#loss-preset");
  const btnApply = $("#btn-loss-apply");
  const btnReset = $("#btn-loss-reset");
  const textarea = $("#loss-textarea");

  if (!select) return;

  // Preset change
  select.addEventListener("change", (e) => {
    handleLossPresetChange(e.target.value);
  });

  // Apply button
  if (btnApply) {
    btnApply.addEventListener("click", validateAndApplyCustomLoss);
  }

  // Reset/Clear button
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      if (textarea) textarea.value = "";
      updateLossValidation("Info", "");
      syncHighlightLayer();
      updateEditorStatus();
    });
  }

  // Live highlighting as user types
  if (textarea) {
    textarea.addEventListener("input", () => {
      syncHighlightLayer();
      updateEditorStatus();
    });
    textarea.addEventListener("scroll", syncHighlightScroll);
  }

  // Initialize for current model
  updateLossEditorForModel();
}

function updateLossEditorForModel() {
  const modelType = state.modelType;
  const presets = LOSS_PRESETS[modelType];
  const select = $("#loss-preset");

  // Populate presets
  if (select) {
    select.innerHTML = "";
    presets.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      select.appendChild(opt);
    });
    select.value = "default";
  }

  // Render snippets
  renderSnippets(modelType);

  // Render variables with docs
  renderLossVariables(modelType);

  // Apply default
  handleLossPresetChange("default");
}

function handleLossPresetChange(presetId) {
  const modelType = state.modelType;
  const preset = LOSS_PRESETS[modelType].find(p => p.id === presetId);
  const textarea = $("#loss-textarea");

  if (!preset) return;

  // Always populate the editor with the preset expression
  if (textarea) {
    textarea.value = preset.expr;
    syncHighlightLayer();
  }

  // Default preset = use hardcoded backend logic (no custom expr sent)
  if (presetId === "default") {
    state.customLossExpr = null;
    _lossEditorAppliedExpr = null;
  } else {
    state.customLossExpr = preset.expr;
    _lossEditorAppliedExpr = preset.expr;
  }

  // Update live preview
  renderLossPreview(preset.expr);
  updateLossValidation("Info", "");
  updateEditorStatus();
}

// ── Live Syntax-Highlighted Preview ──────────────────────────

function renderLossPreview(expr) {
  const container = $("#loss-preview-code");
  if (!container) return;

  if (!expr || expr.trim() === "") {
    container.innerHTML = `<span style="color:var(--text-muted);font-style:italic;">No expression defined</span>`;
    return;
  }

  container.innerHTML = highlightExpression(expr);
}

function highlightExpression(expr) {
  // Escape HTML first
  let s = expr
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Order matters: highlight specific → general
  // 1. Method calls like .sum() .mean() .pow() .exp()
  s = s.replace(/\.(\b(?:sum|mean|pow|exp|item|abs|clamp|backward|float|unsqueeze|squeeze|view|reshape|dim|size|numel|detach|clone)\b)/g,
    '.<span class="loss-eq-method">$1</span>');

  // 2. Torch functions
  s = s.replace(/(\btorch(?:\.nn(?:\.functional)?)?(?:\.\w+)*)/g, '<span class="loss-eq-func">$1</span>');

  // 3. Known variables
  s = s.replace(/(\b(?:x_hat|importance|beta_t|logvar|recon|mu|kl|x|z)\b)/g, '<span class="loss-eq-var">$1</span>');

  // 4. Numbers
  s = s.replace(/(\b\d+(?:\.\d+)?\b)/g, '<span class="loss-eq-num">$1</span>');

  // 5. Operators
  s = s.replace(/([\+\-\*\/\=\%]+|\*\*)/g, '<span class="loss-eq-op">$1</span>');

  // 6. Parentheses
  s = s.replace(/([()])/g, '<span class="loss-eq-paren">$1</span>');

  return s;
}

// ── Live Highlight Overlay for Textarea ──────────────────────

function syncHighlightLayer() {
  const textarea = $("#loss-textarea");
  const highlight = $("#loss-highlight-layer");
  if (!textarea || !highlight) return;

  const text = textarea.value;
  if (text.trim() === "") {
    highlight.innerHTML = "";
    return;
  }

  highlight.innerHTML = highlightExpression(text);
}

function syncHighlightScroll() {
  const textarea = $("#loss-textarea");
  const highlight = $("#loss-highlight-layer");
  if (!textarea || !highlight) return;
  highlight.scrollTop = textarea.scrollTop;
  highlight.scrollLeft = textarea.scrollLeft;
}

// ── Editor Status Badge ──────────────────────────────────────

function updateEditorStatus() {
  const statusEl = $("#loss-editor-status");
  const textarea = $("#loss-textarea");
  if (!statusEl || !textarea) return;

  const currentText = textarea.value.trim();
  const defaultPreset = LOSS_PRESETS[state.modelType].find(p => p.id === "default");
  const isDefault = (currentText === defaultPreset?.expr);

  // Determine status
  if (isDefault && state.customLossExpr === null) {
    statusEl.textContent = "Default";
    statusEl.className = "loss-code-editor__status status--default";
  } else if (currentText === _lossEditorAppliedExpr) {
    statusEl.textContent = "Applied ✓";
    statusEl.className = "loss-code-editor__status status--applied";
  } else {
    statusEl.textContent = "Modified";
    statusEl.className = "loss-code-editor__status status--modified";
  }
}

// ── Snippet Templates ────────────────────────────────────────

function renderSnippets(modelType) {
  const container = $("#loss-snippet-list");
  if (!container) return;

  const snippets = LOSS_SNIPPETS[modelType] || [];
  container.innerHTML = "";

  snippets.forEach(snip => {
    const pill = document.createElement("button");
    pill.className = "loss-snippet-pill";
    pill.title = `Insert: ${snip.code}`;
    pill.innerHTML = `<span class="loss-snippet-pill__icon">${snip.icon}</span> ${snip.label}`;

    pill.addEventListener("click", () => {
      insertAtCursor(snip.code);
    });

    container.appendChild(pill);
  });
}

function insertAtCursor(text) {
  const textarea = $("#loss-textarea");
  if (!textarea) return;

  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const before = textarea.value.substring(0, start);
  const after = textarea.value.substring(end);

  textarea.value = before + text + after;
  textarea.focus();

  // Position cursor intelligently: if snippet has empty parens (), place cursor inside
  const parenMatch = text.indexOf("()");
  if (parenMatch !== -1) {
    textarea.selectionStart = textarea.selectionEnd = start + parenMatch + 1;
  } else {
    textarea.selectionStart = textarea.selectionEnd = start + text.length;
  }

  syncHighlightLayer();
  updateEditorStatus();

  // Update the preset selector to show we're editing custom
  const select = $("#loss-preset");
  if (select) {
    // Check if current text matches any preset
    const currentText = textarea.value.trim();
    const matchingPreset = LOSS_PRESETS[state.modelType].find(p => p.expr === currentText);
    if (matchingPreset) {
      select.value = matchingPreset.id;
    }
  }
}

// ── Variables Reference with Inline Docs ─────────────────────

function renderLossVariables(modelType) {
  const container = $("#loss-var-list");
  if (!container) return;

  const vars = LOSS_VARIABLES[modelType];
  container.innerHTML = "";

  vars.forEach(v => {
    const item = document.createElement("div");
    item.className = `loss-var-item${v.kind === 'module' ? ' loss-var-item--func' : ''}`;
    item.title = `Click to insert "${v.kind === 'module' ? v.name + '.' : v.name}"`;

    // Name
    const nameEl = document.createElement("span");
    nameEl.className = "loss-var-item__name";
    nameEl.textContent = v.name;

    // Type badge
    const badge = document.createElement("span");
    const badgeClass = v.kind === 'tensor' ? 'tensor' : v.kind === 'scalar' ? 'scalar' : 'module';
    badge.className = `loss-var-item__badge loss-var-item__badge--${badgeClass}`;
    badge.textContent = v.kind;

    // Shape
    const shape = document.createElement("span");
    shape.className = "loss-var-item__shape";
    shape.textContent = v.shape;

    // Description
    const desc = document.createElement("span");
    desc.className = "loss-var-item__desc";
    desc.textContent = v.desc;

    item.appendChild(nameEl);
    item.appendChild(badge);
    if (v.shape) item.appendChild(shape);
    item.appendChild(desc);

    // Click to insert
    item.addEventListener("click", () => {
      const insertText = v.kind === 'module' ? `${v.name}.` : v.name;
      insertAtCursor(insertText);
    });

    container.appendChild(item);
  });
}

// ── Validate & Apply ─────────────────────────────────────────

async function validateAndApplyCustomLoss() {
  const textarea = $("#loss-textarea");
  const btnApply = $("#btn-loss-apply");
  if (!textarea || !btnApply) return;

  const expr = textarea.value.trim();
  if (!expr) {
    updateLossValidation("Error", "Expression cannot be empty.");
    return;
  }

  btnApply.disabled = true;
  btnApply.textContent = "Validating...";

  try {
    const resp = await fetch(`${API_BASE}/api/validate-loss`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expr: expr,
        model_type: state.modelType
      }),
    });
    const data = await resp.json();

    if (!data.success) {
      updateLossValidation("Error", data.error || "Invalid expression.");
      state.customLossExpr = null;
      _lossEditorAppliedExpr = null;
    } else {
      // Check if it matches the default preset
      const defaultPreset = LOSS_PRESETS[state.modelType].find(p => p.id === "default");
      if (expr === defaultPreset?.expr) {
        state.customLossExpr = null;
        _lossEditorAppliedExpr = null;
      } else {
        state.customLossExpr = expr;
        _lossEditorAppliedExpr = expr;
      }

      updateLossValidation("Success", "✓ Expression validated and applied!");
      renderLossPreview(expr);
      showToast("Custom loss applied", "success");

      // Sync preset dropdown
      const select = $("#loss-preset");
      if (select) {
        const matchingPreset = LOSS_PRESETS[state.modelType].find(p => p.expr === expr);
        if (matchingPreset) {
          select.value = matchingPreset.id;
        }
      }
    }
  } catch (err) {
    updateLossValidation("Error", "Network error during validation.");
    state.customLossExpr = null;
    _lossEditorAppliedExpr = null;
  } finally {
    btnApply.disabled = false;
    btnApply.textContent = "✓ Apply";
    updateEditorStatus();
  }
}

function updateLossValidation(type, msg) {
  const el = $("#loss-validation");
  if (!el) return;
  
  el.className = "loss-validation";
  if (type === "Error") el.classList.add("loss-validation--error");
  else if (type === "Success") el.classList.add("loss-validation--ok");
  else el.classList.add("loss-validation--info");
  
  el.textContent = msg;
}

// ── Past Experiments ──────────────────────────────────────────
async function fetchPastExperiments() {
  const container = $("#past-experiments-list");
  if (!container) return;
  
  try {
    const resp = await fetch(`${API_BASE}/api/experiments`);
    const data = await resp.json();
    
    if (!data.success) {
      console.error("Failed to fetch experiments:", data.error);
      return;
    }
    
    if (data.experiments.length === 0) {
      container.innerHTML = `
        <div class="stat-row">
          <span class="stat-label">No past experiments found.</span>
        </div>
      `;
      return;
    }
    
    let html = "";
    data.experiments.forEach(exp => {
      const date = new Date(exp.timestamp);
      const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
      const type = exp.model_type ? exp.model_type.toUpperCase() : "UNKNOWN";
      
      let labelHTML = "";
      if (exp.is_sweep) {
        const numRuns = exp.grid ? exp.grid.length : 0;
        labelHTML = `SWEEP • ${type} • ${numRuns} runs`;
      } else {
        const density = exp.config && exp.config.density ? (exp.config.density * 100).toFixed(0) : "?";
        const loss = exp.stats && exp.stats.final_loss ? parseFloat(exp.stats.final_loss).toFixed(4) : "N/A";
        labelHTML = `${type} • d=${density}% • Loss=${loss}`;
      }
      
      html += `
        <div class="stat-row" style="cursor: pointer; border-bottom: 1px solid var(--border-subtle); padding: 4px; border-radius: 4px;" 
             onclick="loadExperiment('${exp.exp_id}')"
             onmouseover="this.style.backgroundColor='var(--bg-input)'"
             onmouseout="this.style.backgroundColor='transparent'">
          <div style="display: flex; flex-direction: column;">
            <span class="stat-label" style="color: var(--text-accent); font-weight: 600;">${labelHTML}</span>
            <span class="stat-value" style="font-size: 0.7rem; color: var(--text-muted);">${formattedDate}</span>
          </div>
        </div>
      `;
    });
    
    container.innerHTML = html;
  } catch (err) {
    console.error("Error fetching experiments:", err);
  }
}

async function loadExperiment(expId) {
  if (state.isTraining) return;
  state.isTraining = true;
  state.plotCache = {};

  showLoading("Loading experiment...");
  clearStats();
  clearPlotTabs();

  try {
    const resp = await fetch(`${API_BASE}/api/experiments/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exp_id: expId }),
    });
    const data = await resp.json();

    if (!data.success) {
      showToast(data.error || "Loading failed", "error");
      return;
    }

    if (data.is_sweep) {
      if (data.model_type) {
        state.modelType = data.model_type;
        const btn = document.querySelector(`.toggle-option[data-model="${data.model_type}"]`);
        if (btn) {
          document.querySelectorAll(".toggle-option[data-model]").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          updateConditionalFields();
          updateModeBadge();
        }
      }

      const tabSweep = document.getElementById("tab-mode-sweep");
      if (tabSweep) tabSweep.click();

      const firstCfg = data.sweep_configs && data.sweep_configs.length > 0 ? data.sweep_configs[0] : { features: data.config?.n_features || 256, hidden: data.hidden || 2, run_id: 1 };
      const savedPlotTypes = data.config?.plot_types || [];
      const defaultPlot = data.config?.plot_type || (data.model_type === "vae" && data.hidden === 2 ? "combined" : "pca_W");
      const plotTypesToUse = savedPlotTypes.length > 0 ? savedPlotTypes : [defaultPlot];

      state.activeSweepPlot = plotTypesToUse[0];
      state.activeSweepFeatures = firstCfg.features;
      state.activeSweepHidden = firstCfg.hidden;
      state.activeSweepRun = firstCfg.run_id;
      state.lastSweepData = data;
      state.lastSweepPlotTypes = plotTypesToUse;

      renderSweepFeatureTabs(data.sweep_configs || [firstCfg]);
      renderSweepGrid(data, plotTypesToUse);

      window._lastSweepReport = {
        timestamp: data.timestamp || new Date().toISOString(),
        parameters: {
          "Model Type": (data.model_type || "vae").toUpperCase(),
          "Loaded Sweep": "Yes",
          "Density Sweep Range": `${data.config?.density_start} to ${data.config?.density_end} (step ${data.config?.density_step})`,
          "Decay Sweep Range": `${data.config?.importance_decay_start} to ${data.config?.importance_decay_end} (step ${data.config?.importance_decay_step})`,
          "Training Steps per Model": data.config?.steps || 1000,
          "Grid Plot Type": PLOT_LABELS[state.activeSweepPlot] || state.activeSweepPlot
        },
        grid: data.grid.filter(cell => cell.features === firstCfg.features && cell.hidden === firstCfg.hidden && cell.run_id === firstCfg.run_id).map(cell => ({
          density: cell.density,
          importance_decay: cell.importance_decay,
          final_loss: cell.final_loss,
          recon: cell.recon,
          kl: cell.kl,
          plot_path: cell.plot_paths ? (cell.plot_paths[state.activeSweepPlot] || cell.plot_path) : cell.plot_path,
          plot_paths: cell.plot_paths
        }))
      };

      const btnExport = document.getElementById("btn-export-sweep");
      if (btnExport) btnExport.disabled = false;

      showToast("Sweep experiment loaded successfully", "success");
    } else {
      state.trainedOnce = true;
      state.availablePlots = data.available_plots;
      state.stats = data.stats;
      state.losses = data.losses;
      
      // Update state to match loaded config
      if (data.model_type) {
          state.modelType = data.model_type;
          const btn = document.querySelector(`.toggle-option[data-model="${data.model_type}"]`);
          if (btn) {
              document.querySelectorAll(".toggle-option[data-model]").forEach(b => b.classList.remove("active"));
              btn.classList.add("active");
              updateConditionalFields();
              updateModeBadge();
          }
      }
      
      const tabSingle = document.getElementById("tab-mode-single");
      if (tabSingle) tabSingle.click();
      
      renderPlotTabs(data.available_plots);
      renderStats(data.stats, data);
      showToast("Experiment loaded successfully", "success");

      // Auto-load first plot
      if (data.available_plots.length > 0) {
        loadPlot(data.available_plots[0]);
      }
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    console.error(err);
  } finally {
    state.isTraining = false;
    hideLoading();
  }
}
