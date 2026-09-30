# Interpretability Dashboard

The dashboard is a Flask application for superposition experiments and sparse
manifold representation learning.

## Run locally

```powershell
cd dashboard
python -m pip install -r requirements.txt
.\start_dashboard.ps1
```

On the Codex desktop workspace, `start_dashboard.ps1` automatically selects
the bundled Python runtime. The launcher disables Flask's debug reloader so the
persistent training-queue worker is started only once. If port 5000 is already
in use by the dashboard, it reports the existing server and exits cleanly.

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

## LLM-assisted paper reproduction and reporting

The **AI reproduction assistant** in **Paper & Data** supports OpenAI,
Anthropic, Google Gemini, Groq, and configurable OpenAI-compatible/local endpoints.
Provider settings and API keys are handled by the Flask server; a key is never
returned to browser JavaScript. Keys can remain in server memory, come from the
provider's environment variable, or be persisted on Windows with current-user
DPAPI protection. The local credential file is excluded from Git.
OpenAI, Anthropic, Gemini, and Groq are locked to their official HTTPS origins.
OpenAI-compatible endpoints may use plain HTTP only on localhost/loopback;
remote custom endpoints require HTTPS. Provider calls never follow redirects,
so authentication headers cannot be replayed to a redirected host.

Upload a PDF/text/Markdown paper, paste its text, or provide a public URL. The
assistant returns a normalized, review-required protocol draft with explicit
reported, inferred, assumed, and missing evidence. **Apply reviewed draft**
fills the ordinary paper, dataset, model protocol, training, seed, and analysis
controls. It deliberately does not create a dataset, register an experiment,
queue work, or start training. The user remains responsible for resolving
missing values, importing empirical data, and approving the final controls.

After results are saved, **Experiment Comparison → AI arXiv report** can draft
a research report for the active experiment, selected experiments, a shared run
label, or all completed runs. The prompt receives compact saved artifacts and
calculated aggregates—not arbitrary dashboard prose—and instructs the model to
mark absent evidence rather than fabricate it. Every generation produces
reviewable `paper.tex`, `paper.md`, and `report.json` files under
`dashboard/ai_reports/`. If `pdflatex` is available, a PDF is also compiled;
otherwise the arXiv-compatible TeX source remains downloadable.

The workspace stores each reproducible run under
`dashboard/experiments/geometry_lab/<experiment-id>/`. Dataset creation writes
JSON and YAML configs plus a compressed NPZ dataset. Training additionally
writes the model checkpoint, training config (including seed and dataset
version), paper provenance, declared paper protocol, analysis plan, per-epoch
metrics, latent analysis, and projection data.

## Jupyter reproducibility notebooks

For any trained experiment, use **Jupyter reproducibility notebook** in the
individual artifacts card or **Export Jupyter notebook** on **Train & Analyze**.
The generated `.ipynb` embeds the resolved experiment/training configuration,
saved metrics, environment provenance, SHA-256 manifests for source and binary
artifacts, and an exact source snapshot of the relevant model, loss, optimizer,
device, and training blocks. Runnable cells locate the portable experiment,
verify that its dataset/checkpoint and dashboard source have not drifted, load
the checkpoint with `weights_only=True`, and independently recompute a
deterministic held-out reconstruction audit. A disabled-by-default cell can
clone the dataset and submit the same configuration to the running local
dashboard, preserving specialized FactorVAE, PCL, and other internal training
behavior. Complete `.mslab` bundles generate and include this notebook
automatically.

## CPU and NVIDIA GPU training

**Train & Analyze** exposes `Auto`, required `NVIDIA CUDA`, and `CPU` device
policies. The effective device, CUDA build, GPU name, and compute capability are
saved with each run instead of reporting a hard-coded CPU value. The local
compute card detects `nvidia-smi`, reads the installed driver capability, and
offers only allowlisted official PyTorch CUDA wheel indexes compatible with
that driver. Installation requires the exact confirmation phrase, rejects
cross-site browser requests, verifies GPU initialization in an isolated
subprocess, and activates a separate `dashboard/.runtime-cuda/` directory only
after verification. The existing CPU runtime remains available for recovery;
restart the dashboard after installation.

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
The same experiment workflow also supports CNN, Transformer, RNN/GRU/LSTM, and
Mamba-family autoencoder and variational-autoencoder models. Encoder and
decoder activations are independently configurable, and Adam, AdamW, SGD, and
AdaGrad are available.
Declarative custom encoder/decoder definitions can be validated and saved under
`dashboard/experiments/geometry_lab/model_library/`; they never execute user
code.

Convolutional definitions may independently specify encoder channels, kernels,
strides, paddings and dense bridge layers, together with decoder base shape,
dense bridge, transpose-convolution channels, kernels, strides and paddings.
Training can stop by epoch count or an exact optimizer-step budget. The run form
also exposes Bernoulli-logits, summed/mean squared-error and summed absolute-error
reconstruction objectives, plus Adam betas, optimizer epsilon and weight decay.

Sequence definitions expose sequence length, hidden width, layer count, heads
and feed-forward width for Transformers; cell type, hidden width and
bidirectionality for recurrent models; and state/convolution/expansion controls
for Mamba. On supported systems, the optional `mamba-ssm` package supplies the
optimized native blocks. Other systems use an explicitly recorded pure-PyTorch
selective-state-space backend, so the dashboard stays usable without claiming
source-code identity with the optimized implementation.

The **Ablation Lab** discovers compatible modules from a trained checkpoint and
can zero, mean-replace, or sample-permute selected neurons, channels, recurrent
features, attention features, or Mamba features. It saves baseline and ablated
reconstruction error, relative error change, latent shift and per-sample
effects. Captum Layer Feature Ablation is installed for unit ranking; direct
PyTorch hooks remain available for causal interventions on every supported
architecture.

The **Custom analysis registry** adds activation, parameter, or representation
statistics to the ordinary analysis plan. Definitions are declarative JSON—not
Python source—and their results are saved in `custom_analysis_results.json`
and displayed beside built-in scientific metrics.

Data can be generated with the general linear/nonlinear factor renderer,
uploaded as NPZ/CSV, or imported from dSprites, MNIST, Fashion-MNIST, and
CelebA. Dataset downloads are explicit and cached under
`dashboard/experiments/geometry_lab/_datasets/`; nothing large is fetched when
the dashboard merely opens. dSprites is retained in its publisher's native NPZ
format. MNIST and Fashion-MNIST are fetched as their published IDX gzip files,
and CelebA as its image/annotation archives, before the selected seeded subset
is converted to the lab's standardized `dataset.npz`. Every derived NPZ gets a
SHA-256 digest, while the experiment JSON/YAML records the source URL, source
format, split, conversion policy, and checksum-verification method.
Split counts use largest-remainder allocation so requested train, validation,
and test partitions are deterministic and their exact counts are persisted.
Older named-paper experiments remain readable through compatibility routes,
but new studies are created through the single paper-agnostic workflow.

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
