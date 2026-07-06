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
    if (state.modelType === "vae") {
      el.classList.remove("disabled");
    } else {
      el.classList.add("disabled");
    }
  });
}

function updateModeBadge() {
  const badge = $(".mode-badge");
  if (badge) {
    badge.textContent = state.modelType === "vae" ? "VAE" : "STD";
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
};

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
      $$(".plot-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      loadPlot(plotType);
    });
    container.appendChild(tab);
  });
}

async function loadPlot(plotType) {
  state.currentPlot = plotType;

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
  const tabSingle = $("#tab-mode-single");
  const tabSweep = $("#tab-mode-sweep");
  const workspaceSingle = $("#workspace-single");
  const workspaceSweep = $("#workspace-sweep");
  const cardSweep = $("#card-sweep");
  const cardData = document.querySelectorAll(".card")[2];
  const btnTrain = $("#btn-train");

  if (tabSingle && tabSweep) {
    tabSingle.addEventListener("click", () => {
      tabSingle.classList.add("active");
      tabSweep.classList.remove("active");
      workspaceSingle.classList.remove("hidden");
      workspaceSweep.classList.add("hidden");
      
      cardSweep?.classList.add("hidden");
      if (cardData) {
        cardData.style.opacity = "1";
        cardData.style.pointerEvents = "auto";
      }
      
      if (btnTrain) {
        btnTrain.innerHTML = "▶ Train Model";
      }
      state.sweepMode = "single";
    });

    tabSweep.addEventListener("click", () => {
      tabSingle.classList.remove("active");
      tabSweep.classList.add("active");
      workspaceSingle.classList.add("hidden");
      workspaceSweep.classList.remove("hidden");
      
      cardSweep?.classList.remove("hidden");
      if (cardData) {
        cardData.style.opacity = "0.25";
        cardData.style.pointerEvents = "none";
      }
      
      if (btnTrain) {
        btnTrain.innerHTML = "▶ Run Sweep Analysis";
      }
      state.sweepMode = "sweep";
      updateSweepPlotOptions();
    });
  }
}

function updateSweepPlotOptions() {
  const container = $("#sweep-plot-checkboxes");
  if (!container) return;

  const hidden = parseInt($("#hidden-dim")?.value ?? 2);
  const is2d = (hidden === 2);
  const isVae = (state.modelType === "vae");

  let options = [];

  if (isVae) {
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
        { value: "graph_W_logvar", label: "W_logvar Graph" }
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
        { value: "graph_W_logvar", label: "W_logvar Graph" }
      ];
    }
  } else {
    if (is2d) {
      options = [
        { value: "arrows_W", label: "W Arrows (Default)" },
        { value: "interference_W", label: "W Interference" },
        { value: "norms_W", label: "W Norms" },
        { value: "graph_W", label: "W Graph" }
      ];
    } else {
      options = [
        { value: "pca_W", label: "W PCA (Default)" },
        { value: "heatmap_W", label: "W Heatmap" },
        { value: "interference_W", label: "W Interference" },
        { value: "norms_W", label: "W Norms" },
        { value: "graph_W", label: "W Graph" }
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
