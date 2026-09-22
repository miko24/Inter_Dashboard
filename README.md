# Interpretability Dashboard

The dashboard is a Flask application for superposition experiments and sparse
manifold representation learning.

## Run locally

```powershell
cd dashboard
python -m pip install -r requirements.txt
python server.py
```

Open `http://localhost:5000`. The unified workspace opens directly on
**Experiments & Data**; model building, training, geometry, representation,
sweeps, imports, classic feature simulation, and Gram experiments share one
persistent navigation bar.

The research workspace follows **Plan → Run → Results**. Planning captures the
paper and dataset, applies a paper-derived model/training protocol, and records
the analyses that should be produced. The Run step trains once, automatically
executes the selected scientific bundle, and opens the first planned result.
Experiment names, run labels, paper citations/URLs, hypotheses, tags, seeds,
analysis plans, and dataset versions are stored with every run so a paper
result can be reconstructed without relying on a dashboard screenshot.

The workspace stores each reproducible run under
`dashboard/experiments/geometry_lab/<experiment-id>/`. Dataset creation writes
JSON and YAML configs plus a compressed NPZ dataset. Training additionally
writes the model checkpoint, training config (including seed and dataset
version), paper provenance, declared paper protocol, analysis plan, per-epoch
metrics, latent analysis, and projection data.

## Persistent training queue

On **Train & Analyze**, select an untrained experiment, configure its model and
analysis plan, and press **Add untrained experiment to queue**. The server runs
queued experiments one at a time in FIFO order and automatically performs the
saved scientific analysis plan after training. The live queue displays order,
stage, epoch, progress, configuration, and failures, and can reopen any listed
experiment. Queue state is written atomically to
`dashboard/experiments/geometry_lab/training_queue.json`, so it remains visible
after a page refresh. Pending work interrupted by a server restart is recovered
when the dashboard reconnects.

The Model Library includes deterministic and random-decoder controls, standard and β-VAEs,
sparse/TC/Factor/hyperspherical/Riemannian variants, and a β-VAE with a full
covariance Gaussian posterior parameterized by a stable Cholesky factor.
It also includes the paper's four-block convolution/deconvolution β-VAE for
64×64 CelebA experiments. Encoder and decoder activations are independently
configurable, and Adam, AdamW, SGD, and AdaGrad are available.
Declarative custom encoder/decoder definitions can be validated and saved under
`dashboard/experiments/geometry_lab/model_library/`; they never execute user
code.

Convolutional definitions may independently specify encoder channels, kernels,
strides, paddings and dense bridge layers, together with decoder base shape,
dense bridge, transpose-convolution channels, kernels, strides and paddings.
Training can stop by epoch count or an exact optimizer-step budget. The run form
also exposes Bernoulli-logits, summed/mean squared-error and summed absolute-error
reconstruction objectives, plus Adam betas, optimizer epsilon and weight decay.

For Zietlow et al. (2021), the preset uses the Disentanglement Library defaults:
the `32/32/64/64` ReLU encoder with `4/4/2/2` kernels and a 256-unit dense bridge,
the 256-to-1024 decoder bridge with four 4x4 transpose convolutions, Bernoulli
logits, Adam at `1e-4`, batch size 64, and exactly 300,000 optimizer updates.
MIG, DCI, SAP and FactorVAE evaluation use evaluation seed 0 and the reference
10,000/5,000 sample budgets. A preset records whether its attached dataset is
the complete published dataset; subset runs remain explicitly marked as pilots.

Data can be generated with the general linear/nonlinear factor renderer or the
paper-specific two-factor linear and nonlinear generators,
uploaded as NPZ/CSV, or imported from dSprites, MNIST, Fashion-MNIST, and
CelebA. Dataset downloads are explicit and cached under
`dashboard/experiments/geometry_lab/_datasets/`; nothing large is fetched when
the dashboard merely opens. dSprites is retained in its publisher's native NPZ
format. MNIST and Fashion-MNIST are fetched as their published IDX gzip files,
and CelebA as its image/annotation archives, before the selected seeded subset
is converted to the lab's standardized `dataset.npz`. Every derived NPZ gets a
SHA-256 digest, while the experiment JSON/YAML records the source URL, source
format, split, conversion policy, and checksum-verification method.
The paper-linear generator also persists its full construction protocol: the
factor-2 stretch of the first factor, the embedding from two into three
dimensions, and the 45-degree rotation about the `(1, -1, 1)` axis. Split
counts use largest-remainder allocation, so a 50,000-sample 80/10/10 run has
exactly 40,000 training, 5,000 validation, and 5,000 test samples.

## Portable experiment bundles

The **Portable experiment bundles** card exports and imports `.mslab` archives.
An optimized results bundle contains configurations, summaries, training curves,
scientific metrics, paper metrics, reports, and environment provenance while
omitting large binary artifacts. A complete bundle additionally includes the
dataset, checkpoint, latent arrays, raw Jacobians, and other NPZ artifacts, so a
run can be resumed or reanalyzed on another installation.

Each bundle has a versioned manifest, artifact sizes, SHA-256 checksums, a
content fingerprint, and capability flags. Imports reject undeclared files,
unsafe paths, symbolic links, checksum mismatches, malformed manifests, and
oversized archives; installation is atomic and identical bundles are
deduplicated. Binary NPZ/checkpoint files use ZIP storage because they are
already compressed or largely incompressible, while JSON/YAML/Markdown use
deflate compression. Imported checkpoints are loaded with PyTorch's
`weights_only` mode.

## Scientific analysis

After training, open **Scientific Metrics** and run the bounded analysis job.
Synthetic experiments use automatic differentiation through the complete
factor-to-encoder map. The resulting `jacobians.npz` contains factor and decoder
Jacobians, local ranks, singular values, condition numbers, tangent-overlap
samples and curvature estimates. `scientific_metrics.json` contains the
dashboard summaries for tangent/Grassmann overlap, folding, persistent
homology, probes and causal interference. The Decoder Geometry screen also
reports the local decoder Jacobian `J(z)`, pullback metric `JᵀJ`, normalized
metric, singular values, `|V|`, the normalized Gram off-diagonal statistic, posterior
precision versus Jacobian norm, polarized regimes, and image-space nonlinear
eigenfaces. Its historical normalized off-diagonal Gram statistic is labeled
`Gram off-diagonal`; it is no longer presented as the paper's DtO. Every
analysis concept has an in-dashboard mathematical definition
and interpretation guide.

## Rolinek et al. (2019) reproduction mode

Use **Model & Protocol → Paper preset** to load one of the synthetic, dSprites,
MNIST, Fashion-MNIST, or CelebA configurations. Press **Apply protocol** before
training. This also enables paper-defined evaluation on the held-out test split.

For a final paper run:

1. Use the matching paper generator or official dataset importer. For dSprites,
   enable **Paper fidelity**; for CelebA, select 64×64.
2. Import/generate the complete intended dataset, then apply the matching model
   preset. The dashboard uses a seeded 80/10/10 train/validation/test split and
   reports the exact integer counts as well as the nominal fractions.
3. Keep **Paper reproduction diagnostics** enabled to save the exact-versus-
   polarized KL trace every 500 optimizer steps.
4. Leave **Paper test samples** at `0` so evaluation uses the complete test
   split. A positive value is recorded as a seeded pilot subsample.
5. Export `paper_metrics.json`, `polarization_trace.json`, the checkpoint,
   training config, environment manifest, and dataset checksum for each seed.

For the synthetic-linear primary study, the **Strict synthetic-linear
reproduction** panel on **Train & Analyze** automates these steps. Choose the
declared batch-size assumption, then press **Create & queue seeds 1-20**. The
dashboard creates correctly named beta-VAE runs, verifies an exact
40,000/5,000/5,000 allocation before queueing each one, uses effective training
seeds 1 through 20 exactly once, and requests paper metrics on the complete test
split. Queue state remains persistent across refreshes.

When the queue is complete, open **Compare**, select the generated suite under
**20-seed paper reproduction**, and press **Aggregate & compare**. The study
summary excludes duplicate seeds, non-exact splits, and incomplete runs; reports
the arithmetic mean, sample standard deviation, and a deterministic 95%
bootstrap confidence interval; and compares DtO, disentanglement, polarized
duration, and final Delta_KL with the paper-facing gates.

The paper does not specify a minibatch size. The preset value `256` is therefore
stored and displayed as a dashboard reproduction assumption, not attributed to
the paper, and should be included in sensitivity analysis. Layer biases are an
explicit Enabled/Disabled setting (Enabled by default). The saved protocol is
resolved against the effective training configuration, so the protocol,
checkpoint, paper metrics, and report all use the same training seed.
Equation 30 active coordinates use the paper's stated
`sqrt(var(mu_j(x_i))) > 0.5` rule and are evaluated every 500 optimizer batches.

`paper_metrics.json` implements Equation 29 using the decoder Jacobian's full
right-singular-vector matrix and the nearest signed permutation, plus the
one-coordinate five-nearest-neighbor disentanglement score from Equations
65–70. These exact metrics are kept alongside—not substituted for—the broader
dashboard probes and geometry statistics. The **Paper reproduction
requirements** card evaluates the matching Table 1 and Table 2 targets for the
selected dataset, model family, and latent dimension. It reports pass, fail,
missing, or not-applicable for DtO, disentanglement, final Delta_KL, and the
continuous polarized-training percentage. This is a per-run diagnostic; final
acceptance still requires aggregation across independent seeds and a 95%
confidence interval. See
[`PAPER_REPRODUCTION_PLAN.md`](PAPER_REPRODUCTION_PLAN.md) for the experiment
matrix, targets, acceptance gates, and paper omissions that must be disclosed.

The **Sweeps** screen accepts YAML Cartesian-product specifications and includes
capacity, sparsity and beta presets. Sweep artifacts are stored under
`dashboard/experiments/geometry_lab/_sweeps/`. Expensive autograd and topology
steps are deliberately sample-capped independently of dataset size.

The **Dataset Imports** screen can fetch and cache the official dSprites `.npz`
directly or ingest a local copy. It never silently substitutes a synthetic
dataset. Because the benchmark is a discrete image grid rather than a
differentiable renderer, continuous factor directions are estimated with seeded
local linear regression and marked as such in the saved metrics.
