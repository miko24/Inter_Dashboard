# Reproduction plan: *Variational Autoencoders Pursue PCA Directions (by Accident)*

## Objective

Reproduce the empirical claims, tables, and figures in Rolínek, Zietlow, and Martius, CVPR 2019, from arXiv:1812.06775v2. The primary claim is that a diagonal Gaussian VAE posterior plus sampling noise encourages locally orthogonal decoder directions, which aligns latent axes with high-variance data directions and improves disentanglement. The full-covariance VAE, deterministic autoencoder, and random decoder are the negative controls.

This is a **paper reproduction**, not merely a dashboard demonstration. No final experiment should start until the dashboard passes the parity gate below.

## Reproduction boundary

The paper specifies datasets, major architectures, optimizers, learning rates, epochs, latent dimensions, beta values, test metrics, and several target results. It does **not** specify batch size, weight initialization, exact random seeds, the number of repetitions behind most means and standard deviations, every beta-sweep coordinate, all image preprocessing, or the exact random nonlinear generator parameters. No public author code was located in a primary-source search.

Therefore:

- Treat values explicitly printed in the paper as locked.
- Store every otherwise unspecified choice as an explicit reproduction assumption.
- Use 20 paired seeds as the primary replication policy, matching the apparent count of points per epoch group in Figure 4; label this as an inference, not a reported paper setting.
- Use batch size 256 as the primary declared assumption, then repeat a smaller sensitivity panel at 64, 128, and 512. Do not present 256 as a paper fact.
- A result is a faithful reproduction when the paper-defined metrics, data, and model families are used and the paper's mean/order/trend is recovered within sampling uncertainty. Bit-for-bit or pixel-identical output is not possible from the paper alone.

## Mandatory dashboard parity gate

The current lab dashboard must be corrected before running the study. At present, several controls with paper names compute materially different experiments.

1. **Loss scaling**
   - Reconstruction must be the squared error summed over observation dimensions and averaged over the batch.
   - KL must be summed over latent dimensions and averaged over the batch.
   - Preserve the paper's beta against those reductions. The current all-element means change the effective beta.

2. **Optimizer support**
   - Add AdaGrad to the dashboard and backend.
   - Required setting: learning rate `1e-2` for dSprites, MNIST, and Fashion-MNIST.
   - Adam at `1e-3` is required for both synthetic datasets; Adam at `1e-4` is required for CelebA.

3. **Three-way data split**
   - Add seeded 80% train / 10% validation / 10% test splits.
   - Use validation only for development decisions.
   - Compute every reported table and figure from the untouched test split.
   - Reuse the identical split across model variants and paired seeds.

4. **Exact paper DtO**
   - For every test sample, compute the decoder Jacobian at the posterior mean: `J_i = d Dec(mu(x_i)) / d mu(x_i)`.
   - Compute `J_i = U_i Sigma_i V_i^T`.
   - Find the signed permutation `P(V_i)` minimizing the paper's L1 objective.
   - Report `mean_i ||V_i - P(V_i)||_F`.
   - Keep the dashboard's current normalized off-diagonal `J^T J` score only as an auxiliary metric and rename it; it is not Equation 29 DtO.

5. **Exact paper Disentanglement Score**
   - Work one ground-truth factor and one latent coordinate at a time, never all latent coordinates jointly.
   - Use SciPy/scikit-learn 5-nearest-neighbor regressor or classifier with default settings.
   - Within the held-out test set, use a seeded 80/20 probe split.
   - Continuous factor sensitivity: `A_ij = sqrt(var(w_i)) - sqrt(MSE(z_j -> w_i))`; normalize by `sqrt(var(w_i))`.
   - Discrete factor sensitivity: classification accuracy; normalize by the best constant-classifier accuracy.
   - For each factor, subtract the second-best latent-coordinate score from the best, normalize, and average over factors as Equations 65-70 specify.
   - The existing dashboard factor-recoverability/MLP score is auxiliary and must not be reported as the paper's Disentanglement Score.

6. **Polarized-regime instrumentation**
   - Log the implemented `L_KL` and the simplified `L_approximately_KL` every 500 minibatches.
   - Define active coordinates exactly as `var(mu_j(x)) > 0.5`.
   - Compute `Delta_KL = |L_KL - L_approximately_KL| / L_KL`.
   - Report the percentage of training time for which `Delta_KL < 3%` continuously until the end.

7. **Dataset fidelity**
   - dSprites: use the official native dataset, its five nonconstant factors (shape, scale, orientation, x, y), all 64x64 binary pixels, and the full split. Remove the dashboard's 50,000-sample cap and 16x16 area downsampling for the paper run.
   - Synthetic linear: generate exactly 50,000 points from `[0,1]^2`; stretch one source axis by 2, embed in three dimensions, then rotate 45 degrees around the axis parallel to `(1,-1,1)`. Save the sampled factors, final 3D observations, rotation matrix, and seed in the NPZ.
   - Synthetic nonlinear: generate exactly 50,000 points from `[0,1]^2` through a fixed randomly initialized `2 -> 10 -> 6` MLP with biases and tanh nonlinearities. Save every generator weight and bias. The current dashboard generator has the wrong hidden width.
   - MNIST and Fashion-MNIST: lock the source version and checksum; do not silently limit to a dashboard subset.
   - CelebA: lock the source version, crop/resize/normalization, split, and image tensor format. Preserve the images at the resolution required by the paper's convolutional architecture.

8. **Architecture fidelity**
   - Support different encoder and decoder activations; the paper uses ReLU in several encoders and Tanh in the corresponding decoders.
   - Support the paper's CelebA convolution/deconvolution model.
   - Add an explicit untrained/random-decoder evaluation path.
   - Retain the full-covariance Gaussian posterior using a stable Cholesky factor and the exact multivariate-normal KL.

9. **Complete test evaluation**
   - Compute final metrics over the complete test split in batches.
   - Do not use the current 1,024-sample scientific-analysis cap or the 1,500-point latent export for paper numbers.
   - Sampling caps may remain for interactive plots, clearly labeled as visualization-only.

The parity gate passes only after unit tests verify the loss reductions, exact DtO on known matrices, the KNN score on toy factors, the 80/10/10 split, full-covariance KL, and full-test aggregation.

## Locked paper protocols

| Dataset | Model architecture | Latent dim. | Optimizer | LR | Epochs | beta |
|---|---|---:|---|---:|---:|---:|
| dSprites | Encoder `1200-1200`, ReLU; decoder `1200-1200-1200`, Tanh; biases | 5 | AdaGrad | 1e-2 | 50 | 4 for beta-VAE; 1 for VAE |
| Synthetic linear | No hidden layers; linear encoder and decoder; biases | 2 | Adam | 1e-3 | 600 | 1e-4 |
| Synthetic nonlinear | Encoder `60-40-20`, Tanh; decoder `60-40-20`, Tanh; biases | 2 | Adam | 1e-3 | 600 | 1e-3 |
| MNIST | Encoder `400`, ReLU; decoder `500-500`, Tanh; biases | 6 | AdaGrad | 1e-2 | 400 | 1 |
| Fashion-MNIST | Encoder `400`, ReLU; decoder `500-500`, Tanh; biases | 6 | AdaGrad | 1e-2 | 500 | 1 |
| CelebA | Encoder conv blocks `[32,4,2],[32,4,2],[64,4,2],[64,4,2]`, ReLU; decoder connecting MLP `[64]`, then `[64,4,2],[32,4,2],[32,4,2],[3,4,2]`, ReLU | 32 | Adam | 1e-4 | 50 | 4 |

For each applicable dataset, train the diagonal-posterior beta-VAE, standard VAE, deterministic AE, and full-covariance beta-VAE control with matched architecture, initialization policy, split, and seed. Evaluate a random decoder with the same decoder architecture and initialization distribution. Follow the omissions in Table 1: synthetic standard-VAEs are expected to overprune, MNIST/Fashion-MNIST have no disentanglement score, and the paper does not report beta-VAE rows for those two datasets.

## Dashboard experiment workflow

### 1. Create a registered dataset experiment

In **Experiments & Data**:

- Experiment name: `Rolinek2019_<dataset>_<model>_seed<seed>`.
- Run label: `table1`, `table2`, `figure4`, `figure5`, `table4`, or `figure7`.
- Tags: `paper-reproduction, cvpr2019, arxiv-1812.06775, <dataset>, <model>`.
- Paper title: `Variational Autoencoders Pursue PCA Directions (by Accident)`.
- Citation: `M. Rolinek, D. Zietlow, G. Martius, CVPR 2019, DOI 10.1109/CVPR.2019.01269`.
- Paper URL: `https://arxiv.org/abs/1812.06775`.
- Hypothesis: `Diagonal posterior VAEs have lower paper-defined decoder DtO than deterministic and full-covariance controls; lower DtO is associated with higher paper-defined disentanglement.`
- Import the exact prepared NPZ or locked official benchmark artifact.
- Confirm the displayed SHA-256 digest, sample count, observation dimension, factor count, and split counts before continuing.

Never regenerate data independently for different model variants. Select or clone the same registered dataset version.

### 2. Apply a model protocol

In **Model Protocol**, paste a protocol JSON, apply it, and visually verify every training field. Example for the synthetic-linear beta-VAE:

```json
{
  "model_type": "beta_vae",
  "latent_dim": 2,
  "encoder_depth": 0,
  "decoder_depth": 0,
  "activation": "tanh",
  "beta": 0.0001,
  "epochs": 600,
  "learning_rate": 0.001,
  "batch_size": 256,
  "optimizer": "adam",
  "seed": 1
}
```

For the synthetic nonlinear beta-VAE, use `encoder_layers: [60,40,20]`, `decoder_layers: [60,40,20]`, Tanh, beta `0.001`, 600 epochs, and Adam `0.001`. Switch `model_type` to `beta_vae_full_cov` for the covariance control. Use the deterministic AE variant with the same encoder/decoder widths and no KL term. Use `beta: 1` for a standard VAE.

After the parity patch, dSprites/MNIST/Fashion-MNIST protocols must include distinct `encoder_activation` and `decoder_activation` fields; the current single `activation` field is insufficient.

### 3. Select the analysis plan

Select:

- Representation overview
- Decoder geometry
- Precision and polarization
- Factor probes
- Latent explorer
- Factor traversal for qualitative checks

The paper-defined DtO, Disentanglement Score, and `Delta_KL` report must be present in the planned-output list. Tangent, topology, folding, and causal-interference metrics may be retained as dashboard extensions but must be separated from the paper reproduction tables.

### 4. Train and analyze

- Use seeds 1-20 for the primary replication.
- Use identical paired seeds across architectures within each dataset.
- Train once per model/seed; do not select the best seed.
- Save the checkpoint, protocol JSON, environment manifest, full learning curves, dataset hash, test indices, final latent means/variances, local decoder Jacobians or reproducible batched summaries, and paper-defined metrics.
- Fail the run if the loss is nonfinite, a requested test sample is omitted, or the saved protocol differs from the effective backend configuration.

### 5. Aggregate without cherry-picking

For each table cell, report `n`, arithmetic mean, sample standard deviation, 95% bootstrap confidence interval, and all seed-level points. Preserve failed runs in an audit log and rerun only under a written, model-independent failure rule.

## Required experiment matrix

### A. Table 1: orthogonality and disentanglement

Run the paper's model columns against dSprites, synthetic linear, synthetic nonlinear, MNIST, and Fashion-MNIST. Primary paper targets:

| Dataset / metric | beta-VAE | VAE | AE | full-cov beta-VAE | Random decoder |
|---|---:|---:|---:|---:|---:|
| dSprites Disent. | 0.33 +/- 0.15 | 0.21 +/- 0.10 | 0.09 +/- 0.04 | 0.12 +/- 0.06 | not reported |
| dSprites DtO | 0.76 +/- 0.08 | 1.08 +/- 0.15 | 1.62 +/- 0.03 | 1.73 +/- 0.14 | 1.86 +/- 0.11 |
| Synth. linear Disent. | 0.99 +/- 0.01 | omitted | 0.71 +/- 0.19 | 0.71 +/- 0.31 | not reported |
| Synth. linear DtO | 0.00 +/- 0.00 | omitted | 0.33 +/- 0.18 | 0.34 +/- 0.35 | 0.79 +/- 0.21 |
| Synth. nonlinear Disent. | 0.73 +/- 0.16 | omitted | 0.59 +/- 0.30 | 0.42 +/- 0.24 | not reported |
| Synth. nonlinear DtO | 0.18 +/- 0.02 | omitted | 0.54 +/- 0.13 | 0.55 +/- 0.02 | 0.89 +/- 0.16 |
| MNIST DtO | not reported | 1.59 +/- 0.08 | 1.83 +/- 0.05 | 1.93 +/- 0.08 | 2.11 +/- 0.11 |
| Fashion-MNIST DtO | not reported | 1.36 +/- 0.05 | 1.87 +/- 0.03 | 2.02 +/- 0.08 | 2.11 +/- 0.11 |

### B. Table 2: polarized regime

For each of dSprites, Fashion-MNIST, MNIST, synthetic linear, and synthetic nonlinear, run the dataset-dependent latent dimension and latent dimension 10. Reproduce the percentage of training time for which `Delta_KL < 3%` continuously to the end. Targets are respectively:

- dSprites: 97.8% and 90.6%
- Fashion-MNIST: 99.8% and 97.7%
- MNIST: 99.8% and 99.5%
- Synthetic linear: 99.8% and 96.7%
- Synthetic nonlinear: 99.9% and 98.5%

### C. Figure 4: DtO versus disentanglement

- Dataset/model: dSprites beta-VAE, beta 4.
- Train independent paired groups for 10, 30, and 50 epochs.
- Use 20 seeds per epoch group as the declared inference from the plotted point count.
- Plot paper-defined Disentanglement Score on x and paper-defined DtO on y, with different markers for the three epoch counts.
- Report Pearson and Spearman association in addition to reproducing the scatter. The expected direction is negative: lower DtO accompanies higher disentanglement.

### D. Figure 5: beta sensitivity

- dSprites: run a log-spaced/explicit grid covering beta 0.1 to 10 and include beta 4 exactly.
- Synthetic linear and nonlinear: run a log-spaced grid covering beta `1e-6` to `1`, including `1e-4` and `1e-3` exactly.
- Because the exact grid is not printed, record the selected grid as a reproduction assumption; use at least five values per decade for the synthetic curves and paired seeds at every beta.
- Plot mean +/- sample standard deviation for Disentanglement Score and DtO.
- Confirm the qualitative regimes: too-small beta gives weak orthogonalization; too-large beta overprunes; the selected paper values lie near the useful region.
- Do not use the dashboard's current beta preset unchanged: it uses a different dataset, a sparse-factor configuration, a short 20-epoch protocol, and a non-paper metric.

### E. Table 4 and Figure 6: degenerate singular values

- Use the linear synthetic dataset with the two factors at relative scale ratios 1.0, 1.2, and 1.5.
- Train the linear beta-VAE for multiple paired seeds.
- Reproduce the targets:
  - Ratio 1.0: Disent. `0.51 +/- 0.28`, DtO `0.49 +/- 0.32`.
  - Ratio 1.2: Disent. `0.76 +/- 0.25`, DtO `0.20 +/- 0.24`.
  - Ratio 1.5: Disent. `0.98 +/- 0.06`, DtO `0.01 +/- 0.06`.
- Plot at least four representative latent embeddings at ratio 1.0, selected by a fixed seed rule rather than visual appeal.

### F. Figure 7: nonlinear eigenfaces

- Train the paper's CelebA beta-VAE for 50 epochs, latent dimension 32, beta 4.
- Select 300 test datapoints by a saved random seed and compute the mean input face and its mean latent representation `z_mean`.
- Sort latent coordinates by mean posterior sigma and use the first five coordinates.
- Decode `z_mean - 2.5 e_i`, `z_mean`, and `z_mean + 2.5 e_i` for each coordinate.
- Export the full image grid and record the sorted coordinate indices and mean sigma values. Confirm that traversals capture semantic as well as photometric changes, without using this qualitative result as a substitute for Table 1.

## Acceptance criteria

A reproduction release is accepted only if all of the following hold:

1. Every locked paper setting matches the effective saved backend configuration.
2. Every final metric uses the untouched test split and the exact paper definition.
3. Table 1 reproduces the ordering `diagonal VAE DtO < AE/full-covariance/random` on each applicable dataset, with synthetic linear beta-VAE near zero DtO and near-one disentanglement.
4. Paper target means fall inside the reproduction's 95% confidence interval, or the discrepancy is reported with a sensitivity analysis rather than hidden.
5. Table 2 shows the polarized approximation dominating the great majority of training after the early phase.
6. Figure 4 has the expected negative association.
7. Figure 5 shows both under-regularized and overpruned regimes and includes the paper's selected beta values.
8. Table 4 recovers the collapse of ambiguity as the factor-scale ratio departs from 1.
9. Every figure can be regenerated from saved seed-level result files by one aggregation command.
10. A final deviation table lists every paper detail that was unspecified, every inferred value, every dashboard extension, and every failed acceptance item.

## Execution order

1. Implement and unit-test the parity gate.
2. Run one-seed smoke tests on synthetic linear for all four trained model families plus random decoder.
3. Run the 20-seed synthetic linear and nonlinear Table 1 panel.
4. Run the degeneracy panel and synthetic beta sweep.
5. Run dSprites Table 1, Figure 4, beta sweep, and polarized-regime panel.
6. Run MNIST and Fashion-MNIST controls.
7. Run CelebA eigenfaces last because it is the highest-cost and least quantitatively central experiment.
8. Freeze seed-level results, aggregate tables/figures, and write the deviation report.

## Final artifact checklist

For each run retain `config.json`, `config.yaml`, exact dataset digest and source metadata, split indices, `training_config.json`, environment manifest, `model.pt`, per-batch/per-epoch metrics, test latent statistics, exact DtO, exact Disentanglement Score, polarization trace, and run status. At study level retain the seed list, aggregation script/version, Table 1/2/4 CSVs, Figure 4/5/6/7 source data and images, confidence intervals, sensitivity results, and deviations from the paper.
