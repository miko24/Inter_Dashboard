/* Contextual metric, concept, and graph documentation. */
(() => {
  "use strict";

  const TOPICS = {
    mig: {title:"Mutual Information Gap (MIG)",quick:"Measures whether each known factor is captured mainly by one latent coordinate.",definition:"For each factor, MIG subtracts the second-largest latent-factor mutual information from the largest and divides by the factor entropy. The final score is the mean across factors.",example:"If orientation has normalized mutual information 0.80 with z₂ and 0.20 with z₅, its contribution is 0.60. A value near 1 indicates one-coordinate selectivity.",interpretation:"Higher is better, but MIG depends on discretization and known ground-truth factors.",caveat:"Never compare MIG values produced with different binning or evaluation samples.",source:"https://proceedings.neurips.cc/paper_files/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html"},
    dci: {title:"DCI Disentanglement",quick:"Uses supervised probe importances to quantify how exclusively each latent coordinate represents factors.",definition:"A predictor is trained for each factor. Entropy of its feature-importance distribution gives coordinate-wise disentanglement, weighted by each coordinate's total importance.",example:"If z₁ is important only for scale it receives a high score; if it predicts scale, hue, and orientation equally it receives a lower score.",interpretation:"Higher is better. Report probe accuracy beside the score so a useless representation cannot appear disentangled.",caveat:"The result depends on the probe family and train/test split.",source:"https://openreview.net/forum?id=By-7dz-AZ"},
    factorvae_score: {title:"FactorVAE Score",quick:"Tests which latent coordinate becomes least variable when one data factor is held fixed.",definition:"Batches are sampled with one factor fixed. The coordinate with the smallest normalized variance votes for that factor; a majority-vote classifier is evaluated over many batches.",example:"When shape is fixed, z₄ repeatedly has the smallest variance, so z₄ votes for shape.",interpretation:"Higher is better. The metric requires ground-truth factors and enough samples for every fixed-factor group.",caveat:"It can favor axis-aligned but incomplete representations.",source:"https://proceedings.mlr.press/v80/kim18b.html"},
    sap: {title:"Separated Attribute Predictability (SAP)",quick:"Measures the gap between the best and second-best single-coordinate factor predictors.",definition:"For each factor, a separate probe is fitted from every latent coordinate. SAP averages the difference between the two highest prediction scores.",example:"If z₃ predicts object hue with 92% accuracy and the next coordinate reaches 55%, that factor contributes 0.37.",interpretation:"Higher gaps mean factors are concentrated in individual coordinates.",caveat:"A large gap is meaningful only when the best predictor itself performs well.",source:"https://openreview.net/forum?id=H1kG7GZAW"},
    active_units: {title:"Active Latent Units",quick:"Counts coordinates whose posterior variance indicates that they carry information.",definition:"For this reproduction, coordinate i is active when Eₓ[σᵢ²(x)] < 0.8, matching the paper's robustness analysis.",example:"A Shapes3D model requires six factors. If only four coordinates satisfy the rule, the run is marked over-pruned.",interpretation:"The active count should be at least the number of ground-truth factors for robustness comparisons.",caveat:"The rule is paper-specific and differs from variance-of-posterior-mean definitions."},
    over_pruning: {title:"Over-pruning",quick:"A model has fewer active latent units than known generating factors.",definition:"Strong KL or total-correlation pressure can make some posterior coordinates equal the prior and therefore unavailable for encoding factors.",example:"Five active units for six Shapes3D factors cannot represent all six factors independently.",interpretation:"Runs remain stored but are excluded from the paper's hyperparameter-robustness aggregate.",caveat:"Exclusion must be rule-based and declared before inspecting scores."},
    beta_vae: {title:"β-VAE",quick:"A VAE whose KL term is multiplied by β to adjust information capacity and factorization pressure.",definition:"The objective combines reconstruction loss with β·KL(q(z|x)||p(z)). Larger β generally restricts the latent channel more strongly.",example:"The paper uses β=8 on dSprites and β=32 on Shapes3D.",interpretation:"β is dataset-specific; higher values can disentangle or over-prune.",caveat:"β values are not comparable when reconstruction scaling differs.",source:"https://openreview.net/forum?id=Sy2fzU9gl"},
    factor_vae: {title:"FactorVAE",quick:"Penalizes total correlation using a discriminator trained on joint and coordinate-permuted latent samples.",definition:"The discriminator estimates the density ratio q(z)/∏ⱼq(zⱼ). Its logit difference supplies the total-correlation penalty while a second optimizer trains the discriminator.",example:"The paper uses γ=35 for dSprites and γ=7 for Shapes3D.",interpretation:"It targets dependence between aggregated latent coordinates without multiplying every KL component.",caveat:"An off-diagonal covariance penalty is not FactorVAE.",source:"https://proceedings.mlr.press/v80/kim18b.html"},
    beta_tcvae: {title:"β-TCVAE",quick:"Reweights only the total-correlation component of the VAE KL decomposition.",definition:"KL is decomposed into index-code mutual information, total correlation, and dimension-wise KL. β is applied to total correlation using a minibatch-weighted estimator.",example:"On Shapes3D the paper's primary TC weight is 32.",interpretation:"It separates independence pressure from channel capacity more cleanly than β-VAE.",caveat:"Plain covariance regularization is not the published estimator.",source:"https://proceedings.neurips.cc/paper_files/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html"},
    slow_vae: {title:"SlowVAE",quick:"Uses pairs with sparse factor changes and a temporal sparse prior to identify latent axes.",definition:"Adjacent observations are encoded independently, then a conditional sparse prior penalizes their latent transition while standard VAE reconstruction and prior terms remain.",example:"A pair may keep hue, scale, shape and position fixed while changing only orientation.",interpretation:"It introduces weak temporal structure instead of relying only on i.i.d. images.",caveat:"Pair construction and the sparse-prior exponent are part of the protocol.",source:"https://openreview.net/forum?id=EbIDjBynYJ8"},
    pcl: {title:"Permutation Contrastive Learning (PCL)",quick:"Learns nonlinear independent components by distinguishing true adjacent pairs from time-permuted pairs.",definition:"A feature extractor and logistic classifier receive real pairs (xₜ,xₜ₋₁) and negative pairs (xₜ,xₜ*) formed by permutation.",example:"A real pair changes one generating factor; a negative pair combines unrelated observations.",interpretation:"PCL is a non-variational control expected to remain stable under the paper's manipulation.",caveat:"Its identifiability depends on temporal dependence assumptions.",source:"https://proceedings.mlr.press/v54/hyvarinen17a.html"},
    weak_gan: {title:"Weakly Supervised GAN — full sharing",quick:"Uses paired observations sharing factors as supervision for an adversarial representation.",definition:"The full-sharing condition uses pair structure rather than independent-image VAE regularization to identify factors.",example:"Two images known to share object identity but differ in pose constrain which representation components may change.",interpretation:"It is a non-variational control in the paper.",caveat:"Its pairing policy and adversarial optimization must be reported.",source:"https://openreview.net/forum?id=HJgSwyBKvr"},
    manipulation: {title:"Learned Dataset Manipulation",quick:"Adds a small factor-conditioned image perturbation designed to favor an entangled VAE alignment.",definition:"The paper forms x′=x+εmψ(w), constrains ||mψ(w)||∞≤1, and trains mψ to lower entangled-reference reconstruction loss while raising disentangled-reference loss.",example:"For dSprites ε=0.1; factor labels and sample order remain unchanged.",interpretation:"A metric drop indicates sensitivity to local variance structure, not loss of the semantic factors.",caveat:"A generic augmentation is not the paper manipulation.",source:"https://proceedings.mlr.press/v139/zietlow21a.html"},
    noise_control: {title:"Matched Uniform-Noise Control",quick:"Adds independent U[-ε,ε] pixel noise with the same amplitude as the learned manipulation.",definition:"This tests whether score changes come merely from perturbation magnitude rather than the manipulation's structure.",example:"Shapes3D uses ε=0.175 for both learned manipulation and uniform noise.",interpretation:"The structured manipulation should damage VAE methods more systematically than matched noise.",caveat:"Clipping changes the realized noise distribution and must be recorded."},
    robustness: {title:"Hyperparameter Robustness Curve",quick:"Plots performance while scaling each model's primary regularization strength.",definition:"The dashboard multiplies the paper value by 0.75, 1, 1.25, 1.5, 1.75, or 2 and repeats ten seeds.",example:"Shapes3D β-VAE uses β={24,32,40,48,56,64}.",interpretation:"A persistent original-to-manipulated gap supports robustness of the finding.",caveat:"Over-pruned runs are shown but excluded according to the declared rule."},
    latent_traversal: {title:"Latent Traversal",quick:"Varies one latent coordinate while holding the others fixed and decodes each point.",definition:"Traversals provide a qualitative picture of which visible factors a coordinate controls.",example:"A clean coordinate may rotate an object without changing its color or size.",interpretation:"Single-factor changes suggest axis alignment; coupled changes suggest entanglement.",caveat:"Traversal appearance depends on the chosen reference code and traversal range."},
    reconstruction: {title:"Reconstruction Loss",quick:"Measures how closely decoded observations match their inputs.",definition:"This dashboard reports per-sample summed squared pixel error unless a paper preset states another likelihood.",example:"Two models can have equal reconstruction error while using rotated latent coordinate systems.",interpretation:"Lower is better for fidelity, but it does not establish interpretability.",caveat:"Always report whether dimensions are summed or averaged."},
    kl: {title:"KL Divergence",quick:"Measures how far the approximate posterior is from the chosen prior.",definition:"For a diagonal Gaussian it is ½Σⱼ(μⱼ²+σⱼ²−log σⱼ²−1). It regularizes latent capacity.",example:"An inactive coordinate with μ≈0 and σ²≈1 contributes almost no KL.",interpretation:"KL is a capacity cost, not a disentanglement score.",caveat:"Scaling, averaging, and β must be reported."},
    jacobian: {title:"Decoder Jacobian",quick:"Describes how each output pixel changes for a small movement in each latent direction.",definition:"J(z)=∂D(z)/∂z. Columns are observation-space tangent directions associated with latent coordinates.",example:"If two columns point in similar directions, changing either coordinate produces overlapping image changes.",interpretation:"Column norms measure magnification; angles measure local coupling.",caveat:"It is local and can vary across the manifold."},
    pullback: {title:"Pullback Metric JᵀJ",quick:"The latent-space metric induced by Euclidean distances after decoding.",definition:"G(z)=J(z)ᵀJ(z). Diagonal terms measure local magnification and off-diagonal terms measure coordinate coupling.",example:"A diagonal G means locally orthogonal decoder directions.",interpretation:"Eigenvalues are squared decoder singular values.",caveat:"Orthogonality alone does not identify semantic factors."},
    dto: {title:"Distance to Orthogonality (DtO)",quick:"Measures how far decoder right-singular vectors are from a signed permutation.",definition:"For each sample, the paper compares V from the decoder-Jacobian SVD with the nearest signed permutation and averages the Frobenius distance.",example:"DtO=0 means local axes align up to sign and coordinate order.",interpretation:"Lower is better for the Rolinek 2019 criterion.",caveat:"DtO is not reconstruction error and is not a Zietlow 2021 verdict metric."},
    polarization: {title:"Posterior Polarization",quick:"Separates informative active coordinates from prior-like passive coordinates.",definition:"Active dimensions vary across observations and usually have reduced posterior variance; passive dimensions stay near μ=0, σ²=1.",example:"An active position coordinate may have variable means and small uncertainty.",interpretation:"Polarization justifies an approximate KL expression.",caveat:"It does not guarantee axis alignment or disentanglement."},
    delta_kl: {title:"ΔKL",quick:"Relative error between exact and polarized-regime approximate KL.",definition:"ΔKL=|KL−KLapprox|/KL. The Rolinek reproduction treats values at most 0.03 as sufficiently polarized.",example:"ΔKL=0.004 means the approximation differs by about 0.4%.",interpretation:"Lower is better for approximation validity.",caveat:"A small value does not prove semantic factor recovery."},
    gram: {title:"Normalized Gram Off-diagonal",quick:"Summarizes overlap between normalized representation or Jacobian directions.",definition:"A Gram matrix contains pairwise inner products. The off-diagonal statistic aggregates absolute non-diagonal entries after normalization.",example:"The identity matrix has zero off-diagonal overlap.",interpretation:"Lower values indicate more orthogonal directions.",caveat:"The dashboard Gram statistic is not Equation 29 DtO."},
    tangent: {title:"Tangent-Space Overlap",quick:"Measures local overlap between learned and ground-truth manifold directions.",definition:"Jacobians span tangent subspaces; principal angles or normalized inner products compare those subspaces.",example:"Two identical two-dimensional tangent planes have zero principal angles.",interpretation:"Better local recovery means lower distance or higher declared overlap, depending on the displayed form.",caveat:"Check the directionality shown beside the number."},
    grassmann: {title:"Grassmann Distance",quick:"Distance between two subspaces computed from their principal angles.",definition:"It ignores the particular bases used inside each subspace and compares only the spanned tangent spaces.",example:"Rotating a basis within the same plane leaves Grassmann distance zero.",interpretation:"Lower is better for subspace recovery.",caveat:"It does not measure coordinate-wise alignment inside the subspace."},
    topology: {title:"Persistent-Homology and Topology Graphs",quick:"Track connected components, loops, and voids across distance scales.",definition:"A filtration adds edges and simplices as a radius grows; persistent features survive across a broad range of radii.",example:"A sampled circle has one persistent one-dimensional loop.",interpretation:"Similar barcodes suggest topology preservation.",caveat:"Results depend on sample count, distance, and filtration settings."},
    factor_probe: {title:"Factor Probe",quick:"A supervised diagnostic measuring how easily known factors can be predicted from the representation.",definition:"Simple and nonlinear predictors are trained on latent codes and evaluated on a held-out split.",example:"A linear probe with high orientation accuracy indicates linearly accessible orientation information.",interpretation:"High accuracy measures availability, not exclusivity to one coordinate.",caveat:"Keep probe class and split constant across runs."},
    causal: {title:"Causal Interference",quick:"Measures unintended changes in other factors after intervening on one latent coordinate.",definition:"The dashboard moves one latent coordinate, decodes, re-estimates factors, and compares target versus off-target responses.",example:"Changing a position coordinate should not also change color.",interpretation:"Lower off-target change indicates more selective control.",caveat:"It depends on the accuracy of factor estimators."},
    pca: {title:"PCA Projection",quick:"Projects high-dimensional values onto orthogonal directions of greatest sample variance.",definition:"The displayed axes are eigenvectors of the sample covariance matrix and are chosen for visualization, not semantic meaning.",example:"A ten-dimensional latent code can be displayed using its first three principal components.",interpretation:"Nearby projected points may be similar along high-variance directions.",caveat:"Projection can hide structure in omitted components."},
    degeneracy: {title:"Degeneracy ε",quick:"Controls how similar leading singular values or factor scales are.",definition:"When singular values are equal, rotations within their subspace are equivalent and a unique axis alignment is not identifiable.",example:"ε near 1 makes two factor scales nearly equal; smaller values separate their variances.",interpretation:"Higher degeneracy makes PCA-like orientation less unique.",caveat:"This synthetic control is separate from the Zietlow pixel perturbation ε."},
    split: {title:"Train / Validation / Test Split",quick:"Separates optimizer data, model-selection data, and final held-out evaluation data.",definition:"A seeded permutation fixes exact integer membership and is stored with the dataset artifact.",example:"50,000 samples at 80/10/10 produce exactly 40,000/5,000/5,000.",interpretation:"Paper metrics must use the held-out test partition.",caveat:"Do not tune hyperparameters on test results."},
    bootstrap: {title:"Bootstrap 95% Confidence Interval",quick:"Estimates uncertainty of a suite mean by resampling completed seeds with replacement.",definition:"Each bootstrap sample draws n run values from the n observed runs and recomputes the mean; the 2.5th and 97.5th percentiles form the interval.",example:"A CI [0.08,0.14] summarizes uncertainty around mean MIG across ten seeds.",interpretation:"Narrow intervals indicate stable aggregate estimates.",caveat:"Bootstrap uncertainty cannot replace missing variation sources such as dataset-generator seeds."},
    queue: {title:"Persistent Experiment Queue",quick:"Executes saved untrained experiments in FIFO order and preserves state across refreshes.",definition:"Each queue record stores the experiment, effective training configuration, analysis plan, progress, and terminal state on disk.",example:"A 420-run suite remains visible after the browser or server restarts.",interpretation:"Queue order is execution order unless an item is removed before running.",caveat:"Clearing history preserves queued and running work."}
  };

  const ALIASES = [
    ["mutual information gap","mig"],["mig","mig"],["dci","dci"],["factorvae score","factorvae_score"],
    ["factorvae-score","factorvae_score"],["sap","sap"],["active latent","active_units"],["active unit","active_units"],
    ["over-prun","over_pruning"],["β-vae","beta_vae"],["beta-vae","beta_vae"],["factorvae","factor_vae"],
    ["factor-vae","factor_vae"],["β-tcvae","beta_tcvae"],["beta-tcvae","beta_tcvae"],["tc-β-vae","beta_tcvae"],
    ["slowvae","slow_vae"],["slow-vae","slow_vae"],["permutation contrastive","pcl"],["pcl","pcl"],
    ["weakly supervised gan","weak_gan"],["learned manipulation","manipulation"],["manipulated","manipulation"],
    ["uniform noise","noise_control"],["robustness","robustness"],["hyperparameter","robustness"],
    ["latent traversal","latent_traversal"],["reconstruction","reconstruction"],["kl divergence","kl"],["delta_kl","delta_kl"],
    ["delta kl","delta_kl"],["decoder jacobian","jacobian"],["jacobian","jacobian"],["pullback","pullback"],
    ["jᵀj","pullback"],["distance to orthogonality","dto"],["dto","dto"],["polarization","polarization"],
    ["polarized","polarization"],["gram","gram"],["tangent","tangent"],["grassmann","grassmann"],
    ["topology","topology"],["persistence","topology"],["factor probe","factor_probe"],["causal interference","causal"],
    ["pca","pca"],["degeneracy","degeneracy"],["data split","split"],["bootstrap","bootstrap"],["queue","queue"]
  ].sort((a,b) => b[0].length-a[0].length);

  const escape = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  let overlay;

  function topicFor(element) {
    const explicit = element.dataset.help;
    if (explicit && TOPICS[explicit]) return explicit;
    const text = (element.textContent || "").trim().toLowerCase();
    const known = (ALIASES.find(([alias]) => text.includes(alias)) || [])[1];
    if (known) return known;
    if (!text) return null;
    const id = `dashboard_${text.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 56)}`;
    if (!TOPICS[id]) {
      const card = element.closest(".geo-card, .card");
      const context = card?.querySelector("p")?.textContent?.trim();
      const title = (element.childNodes[0]?.textContent || element.textContent || "Dashboard concept").trim();
      TOPICS[id] = {
        title,
        quick: context || `Explains how ${title} is used in this experiment dashboard.`,
        definition: `${title} is a dashboard result or control whose exact meaning depends on the active experiment configuration and saved provenance.`,
        example: `Compare ${title} for two runs only after checking that their dataset, split, model family, seed policy, and evaluation configuration match.`,
        interpretation: "Use the value together with its axis labels, units, evaluation split, and uncertainty—not as an isolated quality score.",
        caveat: "A dashboard visualization is descriptive unless its report declares a paper-defined target or a pre-registered acceptance rule."
      };
    }
    return id;
  }

  function ensureOverlay() {
    if (overlay) return overlay;
    overlay = document.createElement("section");
    overlay.className = "help-documentation";
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML = '<div class="help-documentation__scrim" data-help-close></div><article class="help-documentation__article" role="dialog" aria-modal="true" aria-labelledby="help-doc-title"><button class="help-documentation__close" data-help-close aria-label="Close documentation">×</button><div class="help-documentation__eyebrow">CONCEPT DOCUMENTATION</div><h1 id="help-doc-title"></h1><p class="help-documentation__lead" id="help-doc-quick"></p><h2>Definition</h2><p id="help-doc-definition"></p><h2>Worked example</h2><p id="help-doc-example"></p><h2>How to interpret it</h2><p id="help-doc-interpretation"></p><div class="help-documentation__caveat"><b>Important limitation</b><p id="help-doc-caveat"></p></div><a id="help-doc-source" target="_blank" rel="noreferrer">Open the primary reference ↗</a></article>';
    document.body.appendChild(overlay);
    overlay.querySelectorAll("[data-help-close]").forEach(button => button.addEventListener("click", closeDocs));
    document.addEventListener("keydown", event => { if (event.key === "Escape") closeDocs(); });
    return overlay;
  }

  function openDocs(id, updateHistory=true) {
    const topic = TOPICS[id];
    if (!topic) return;
    ensureOverlay();
    overlay.querySelector("#help-doc-title").textContent = topic.title;
    overlay.querySelector("#help-doc-quick").textContent = topic.quick;
    overlay.querySelector("#help-doc-definition").textContent = topic.definition;
    overlay.querySelector("#help-doc-example").textContent = topic.example;
    overlay.querySelector("#help-doc-interpretation").textContent = topic.interpretation;
    overlay.querySelector("#help-doc-caveat").textContent = topic.caveat;
    const source = overlay.querySelector("#help-doc-source");
    source.hidden = !topic.source; source.href = topic.source || "#";
    overlay.classList.add("is-open"); overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("help-documentation-open");
    if (updateHistory) history.pushState({help:id}, "", `#documentation/${id}`);
    requestAnimationFrame(() => overlay.querySelector(".help-documentation__close").focus());
  }

  function closeDocs(updateHistory=true) {
    if (!overlay?.classList.contains("is-open")) return;
    overlay.classList.remove("is-open"); overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("help-documentation-open");
    if (updateHistory && location.hash.startsWith("#documentation/")) history.pushState({}, "", location.pathname + location.search);
  }

  function enhance(root=document) {
    const selector = "[data-help], .geo-card h2, .geo-card h3, .geo-concept h4, .geo-canvas-label, .geo-analysis-option b, .geo-stat > span, .metric-label, .card__title, .plot-tab";
    const elements = [
      ...(root.matches?.(selector) ? [root] : []),
      ...root.querySelectorAll(selector),
    ];
    elements.forEach(element => {
      if (element.dataset.helpEnhanced === "true" || element.closest(".help-documentation")) return;
      const id = topicFor(element);
      if (!id) return;
      element.dataset.helpEnhanced = "true";
      const topic = TOPICS[id];
      const link = document.createElement("a");
      link.className = "context-help"; link.href = `#documentation/${id}`; link.textContent = "?";
      link.dataset.tooltip = topic.quick; link.setAttribute("aria-label", `Explain ${topic.title}`);
      link.addEventListener("click", event => { event.preventDefault(); openDocs(id); });
      element.appendChild(link);
    });
  }

  function followHash() {
    if (location.hash.startsWith("#documentation/")) openDocs(location.hash.split("/")[1], false);
    else closeDocs(false);
  }

  document.addEventListener("DOMContentLoaded", () => {
    ensureOverlay(); enhance(); followHash();
    const observer = new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(node => {
      if (node.nodeType === 1) enhance(node);
    })));
    observer.observe(document.body, {childList:true, subtree:true});
  });
  window.addEventListener("popstate", followHash);
  window.dashboardHelp = {open:openDocs, enhance, topics:TOPICS};
})();
