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
  --z-bin-efficiency-tol 0.01 \
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
  --z-bin-efficiency-tol 0.01 \
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

When subhalos are enabled, the mock injects intrinsic log-normal scatter around
the member-galaxy scaling relations by default:

- `--subhalo-sigma-scatter-dex 0.07`
- `--subhalo-cut-scatter-dex 0.20`

Member-galaxy core radii remain fixed at the tiny configured value. The
validation runner also fits matching scaling-scatter hyperparameters by default
for the injected fields; disable that with `--no-fit-scaling-scatter`.

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

The validation runner first runs the normal `lenscluster.cluster_solver`
pipeline on the mock `.par` file, so the standard real-data stage outputs are
also present under:

```text
validation_runs/single_bcg/<run-name>/seed_<seed>/solver/fit/stage1_large_only/
validation_runs/single_bcg/<run-name>/seed_<seed>/solver/fit/stage2_joint/
```

Mock-truth recovery PDFs are additionally written at the seed directory level:

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

All figures are saved as PDFs. The standard solver stage still writes its usual
diagnostic tables under each stage's `tables/` directory. Use `--skip-plots` on
the validation command only when you want to suppress the standard solver plot
suite; the mock-truth recovery PDFs are still generated.

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

The sampled likelihood uses a cached magnification-weighted source-plane
metric. It periodically computes the local magnification at each observed image
and uses the equal-area circularized source-plane variance
`sigma_img^2 / |mu| + sigma_int^2`. It preserves the local area scaling of the
full Jacobian covariance while keeping likelihood evaluations scalar and fast.
The covariance floor for this metric is controlled with
`--source-plane-covariance-floor`.

Nearby source redshifts are grouped by fractional lensing-efficiency
`D_ls / D_s` tolerance instead of raw redshift. The default
`--z-bin-efficiency-tol 0.01` keeps each effective source plane within about
1% in lensing strength, which automatically makes bins finer near the cluster
and coarser at high source redshift.

When `--svi-steps` is larger than `--refresh-every`, SVI is run in blocks.
Between blocks the inactive-subhalo surrogate, scaling-scatter cache, and
magnification weights are refreshed at the current guide median, then the final
fixed cache is used for NUTS. This incorporates updated Jacobian information
without changing the target density inside a NUTS trajectory.

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
- diagnostic PDFs in the stage directory

`run.xsh` is a thin wrapper around the same command.

The implementation uses a standard `src/lenscluster/` package layout. The
repository-root `cluster_solver.py` remains as a compatibility shim, so existing
`python -m cluster_solver` commands continue to work.

## HFF Pagul21 Catalogs

The Pagul21 HFF Zenodo record can be read and split into cluster-specific
tables with:

```bash
python -m lenscluster.hff_pagul21 --summary-only
python -m lenscluster.hff_pagul21
```

This writes prepared directories under `data/HFF_Pagul21/prepared/<cluster>/`
with photometric catalogs, magnification summaries, an `obs_arcs.cat`, and a
bootstrap `.par` file. By default the bootstrap `.par` includes a Pagul21
photometry-selected BCG candidate plus a `cluster_members_potfile.cat` for
likely cluster members; use `--no-pagul21-members` to omit these. For example:

```bash
python -m lenscluster.cluster_solver \
  --par-path data/HFF_Pagul21/prepared/a2744/a2744_bootstrap.par \
  --fit-mode joint \
  --fit-method svi
```

The upstream `ZSPEC` values in the Pagul21 files are unset. To enrich the
prepared catalogs with coordinate-matched SIMBAD spectroscopic redshifts, run:

```bash
python -m lenscluster.hff_pagul21 --query-simbad-specz --simbad-match-arcsec 1.0
```

This leaves the raw Zenodo files unchanged, writes `simbad_sources.csv` and
match tables into each prepared cluster directory, and uses a matched SIMBAD
redshift in `obs_arcs.cat` only when it is behind the cluster lens and
consistent with the catalog redshift. Add `--no-use-simbad-specz` to write the
match diagnostics without changing the generated arc redshifts.

To also assign heuristic candidate family labels from close SIMBAD spec-z, HST
color agreement, and a strong-lensing-scale angular span, add:

```bash
python -m lenscluster.hff_pagul21 \
  --query-simbad-specz \
  --assign-families
```

This writes `candidate_families.csv` and `candidate_family_members.csv`, then
uses those labels in `obs_arcs.cat`. These are candidate labels for inspection,
not a replacement for a vetted multiple-image catalog. The default maximum
within-family span is 120 arcsec and can be changed with
`--family-max-separation-arcsec`. Candidate-family rows also get a fixed
`family_reliability` value from their redshift offset, color offset, and
angular distance from the family centroid. The source-plane likelihood uses
that value in a reliability-weighted per-image mixture, with a broad outlier
term controlled by `--source-plane-outlier-sigma-arcsec`.

These generated `.par` files are parser/integration bootstrap inputs only. The
Pagul21 catalogs do not include multiple-image family memberships, so each
generated image is a single-image pseudo-family and does not by itself provide
a science-ready strong-lensing constraint. Replace `obs_arcs.cat` with real HFF
multiple-image family labels before using these files for an actual mass fit.
