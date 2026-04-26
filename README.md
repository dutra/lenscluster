# Lenscluster Workflow

This repository now uses one public fitting workflow:

1. Fit the large-scale cluster model with SVI.
2. Fit the joint large+small model with SVI, initialized from the large-scale SVI solution.
3. Optionally run NUTS on the joint model with `--fit-method svi+nuts`.

Use `--fit-method svi` for a fast variational result, or `--fit-method svi+nuts` for SVI initialization followed by joint posterior sampling.

## Run

Fast NUTS smoke test:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_nuts_smoke \
  --fit-method svi+nuts \
  --svi-steps 1000 \
  --warmup 100 \
  --samples 50 \
  --chains 1 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-galaxies 32 \
  --target-accept 0.75 \
  --max-tree-depth 5 \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --z-bin-tol 0.25 \
  --profile-variant original
```

Longer joint run:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_workflow \
  --fit-method svi+nuts \
  --warmup 1000 \
  --samples 250 \
  --chains 4 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --validation-approx adaptive \
  --z-bin-tol 0.25 \
  --profile-variant original \
  --plot-caustics
```

The same command with `--fit-method svi` skips NUTS and writes the AutoNormal guide posterior.

## Mock-Cluster Validation

The validation runner builds a synthetic single-BCG cluster, runs the normal
parser/build/inference workflow, and writes PDF-only recovery figures. By
default it uses a mildly realistic mock: source families cycle through
`z=1.5,2.0,3.0`, the BCG is slightly offset from the cluster halo, the image
position uncertainty is `0.15"` and the reported source-scatter truth is
`0.05"`. It uses SVI initialization followed by NumPyro NUTS:

```bash
python -m lenscluster.validation \
  --n-subhalos 50
```

Useful explicit configuration for the current subhalo validation setup:

```bash
python -m lenscluster.validation \
  --n-subhalos 50 \
  --source-redshifts 1.5,2.0,3.0 \
  --pos-sigma-arcsec 0.15 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-selection adaptive \
  --active-scaling-cumulative-fraction 0.995 \
  --active-scaling-min 4
```

The adaptive subhalo selection ranks potfile galaxies by brightness and
proximity to the observed multiple images, then chooses the active exact
subhalo cutoff from the cumulative-importance curve. The default keeps enough
ranked subhalos to capture 99.5% of the ranking importance, with at least four
active subhalos per potfile. The remaining subhalos are retained through the
refreshing surrogate rather than removed from the model.

For a faster variational-only validation run:

```bash
python -m lenscluster.validation \
  --n-subhalos 50 \
  --fit-method svi \
  --svi-steps 1000 \
  --samples 500
```

For a smaller NUTS test:

```bash
python -m lenscluster.validation \
  --n-subhalos 50 \
  --fit-method svi+nuts \
  --svi-steps 500 \
  --warmup 100 \
  --samples 200
```

Validation outputs are written to:

```text
validation_runs/single_bcg/<run-name>/seed_<seed>/
```

The default run name and seed produce:

```text
validation_runs/single_bcg/single_bcg_recovery/seed_12345/
```

Main validation PDFs:

- `parameter_recovery.pdf`
- `mass_profile_recovery.pdf`
- `magnification_recovery.pdf`
- `image_recovery.pdf`
- `source_recovery.pdf`
- `subhalo_population.pdf`
- `validation_summary.pdf`
- `corner.pdf`
- `potfile_corner.pdf` when potfile scaling parameters are present

The mass-profile validation figure decomposes the recovered deflection profile
into total, halo, BCG, subhalos, and BCG+subhalos. This is important because
strong-lensing image positions mostly constrain the total deflection field; the
halo, BCG, and subhalo components can trade mass unless the priors and image
configuration break that degeneracy.

The posterior artifacts used to make these PDFs are saved under:

```text
validation_runs/single_bcg/<run-name>/seed_<seed>/solver/fit/stage2_joint/artifacts/plot_bundle.h5
```

All validation figures are saved as PDFs. The validation runner does not write
CSV tables.

## Model

The sampled model is assembled from the priors in the Lenstool `.par` file and
any potfile scaling priors. A quantity is free only if it has a decoded prior;
otherwise it remains fixed at the input value.

Free large-scale parameters come from supported potential profiles with priors
in the `.par` file. Supported profiles are dPIE (`81`) and shear (`14`), with
fields such as `x_centre`, `y_centre`, `ellipticite`, `angle_pos`,
`core_radius_kpc`, `cut_radius_kpc`, `v_disp`, and `gamma`.

Member-galaxy subhalos are dPIE components generated from potfiles. If the
potfile provides priors, the free scaling hyperparameters are `sigma`, `cutkpc`,
`corekpc`, `vdslope`, and `slope`. Their nominal scaling relation is:

```text
sigma_i = sigma_ref * L_i^(1 / vdslope)
core_i  = core_ref  * L_i^0.5
cut_i   = cut_ref   * L_i^(2 / slope)
```

where `L_i` is the catalog luminosity ratio relative to `mag0`.

With `--scaling-scatter`, the model also samples positive scatter
hyperparameters per potfile for the requested fields: `sigma_log_scatter`,
`core_log_scatter`, and/or `cut_log_scatter`. These are marginalized with a
linearized source-plane covariance approximation rather than sampled as
per-member latent offsets. The approximation perturbs each enabled scaling
field around the current model, estimates the first-order source-plane
sensitivity, and adds that variance to the source-plane likelihood.

```text
Cov_extra ~= J_scatter diag(scatter^2) J_scatter^T
```

Positive quantities such as scaling `sigma` and `cutkpc` are sampled in latent
log space and converted back to physical units when building the lens model and
writing outputs.

The source-plane likelihood also includes a global positive intrinsic scatter
parameter, `source.sigma_int`, sampled in log space. It is combined in
quadrature with the configured positional uncertainty:

```text
sigma_eff_i^2 = sigma_arcsec_i^2 + source_sigma_int^2
```

## Likelihood

Inference uses a source-plane Gaussian likelihood. For each observed image, the
current lens model ray-shoots the image position to the source plane. Within
each multiply imaged family, the source position is estimated as the weighted
centroid of those ray-shot source positions. The log likelihood penalizes the
scatter around that family centroid:

```text
log L = -0.5 * sum_i [
  ((beta_x_i - beta_bar_x)^2 + (beta_y_i - beta_bar_y)^2) / sigma_eff_i^2
  + 2 log(2 pi sigma_eff_i^2)
]
```

`sigma_eff_i` is the positional uncertainty plus inferred intrinsic scatter.
When scaling scatter is enabled, the likelihood uses separate diagonal
`x`/`y` variances with the linearized scaling-scatter contribution added on top
of `sigma_eff_i^2`. Single-image families are excluded from this source-plane
scatter likelihood because their source-plane residual is zero by construction.
Exact image-plane solving is used for validation, diagnostics, and plots; it is
not currently the sampled posterior likelihood.

## Outputs

For `--run-name joint_workflow`, outputs are written to:

- `plots/m0416_original/joint_workflow/stage1_large_only/`
- `plots/m0416_original/joint_workflow/stage2_joint/`
- `plots/m0416_original/joint_workflow/sequential_summary.json`

Each stage writes:

- `artifacts/plot_bundle.h5`
- `tables/run_summary.json`
- `tables/potential_summary.csv`
- `tables/family_diagnostics.csv`
- diagnostic PNGs in the stage directory

`run.xsh` is a thin wrapper around the same command.

The implementation uses a standard `src/lenscluster/` package layout. The
repository-root `cluster_solver.py` remains as a compatibility shim, so existing
`python -m cluster_solver` commands continue to work.
