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
`core_log_scatter`, and/or `cut_log_scatter`. Each member then gets a latent
offset for each enabled field:

```text
delta_i ~ Normal(0, scatter)
scaled_value_i *= exp(delta_i)
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
Single-image families are excluded from this source-plane scatter likelihood
because their source-plane residual is zero by construction. Exact image-plane
solving is used for validation, diagnostics, and plots; it is not currently the
sampled posterior likelihood.

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
