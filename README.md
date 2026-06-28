# Lenscluster Workflow

This repository now has two public fitting workflows: the sequential optimizer/sampler path and the one-shot `evidence-ns` path. The sequential path runs:

1. Fit the large-scale cluster model with SVI.
2. Fit the joint large+small model with SVI, initialized from the large-scale SVI solution.
3. Optionally run NUTS on the joint model with `--fit-method svi+nuts`, or use a direct sampler with `--fit-method mchmc` / `--fit-method mclmc`.
4. Optionally run an image-plane refinement stage with `--image-plane-mode local-jacobian`.
5. Optionally run a final image-plane stage using `--image-plane-mode linearized-forward-beta-image-plane`, `--image-plane-mode forward-metric-image-plane`, `--image-plane-mode critical-arc-mixture-image-plane`, or `--image-plane-mode fold-regularized-forward-beta-image-plane`.

Use `--fit-method svi` for a fast variational result, `--fit-method svi+nuts` for SVI initialization followed by posterior sampling, or `--fit-method mchmc` / `--fit-method mclmc` for direct BlackJAX microcanonical sampling in the model's latent parameter space. In the current sequential workflow, `--fit-method`, `--warmup`, `--samples`, and `--max-tree-depth` configure the sampled production stages, while `--svi-steps` and `--refresh-every` configure the initializer, backprojected-centroid fit, and optional free-source forward fit. Nested sampling is reserved for the one-shot evidence workflow: use `--fit-mode evidence-ns` with an explicit `--evidence-source-prior-sigma-arcsec`; `--fit-method`, `--svi-steps`, `--warmup`, and `--samples` are ignored in that mode.

For `mchmc` and `mclmc`, `--warmup` controls BlackJAX tuning/adaptation and `--samples` controls production transitions per chain. Prefer `mchmc` when asymptotic Metropolis correction is required. Unadjusted `mclmc` can be faster, but each saved draw is one integration transition and the run should be checked by increasing samples or reducing the tuned step size in a sensitivity run.

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
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies 32 \
  --target-accept 0.75 \
  --max-tree-depth 5 \
  --z-bin-efficiency-tol 0.01 \
  --profile-variant original
```

Longer joint run:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_workflow \
  --fit-method svi+nuts svi+nuts svi \
  --image-plane-mode forward-metric-image-plane \
  --image-plane-newton-steps 0 \
  --warmup 1000 1000 0 \
  --svi-steps 2000 2000 500 \
  --samples 250 250 100 \
  --chains 4 \
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 8 6 \
  --z-bin-efficiency-tol 0.01 \
  --profile-variant original \
  --plot-caustics
```

The same command with scalar `--fit-method svi` skips NUTS in every sampled stage and writes the AutoNormal guide posterior. A mixed command such as `--fit-method svi+nuts svi --svi-steps 2000 500 --warmup 1000 0 --samples 250 100 --max-tree-depth 8 6` runs NUTS in stage 2 and an SVI-only image-plane stage 3; when stage 3 is skipped before stage 4, the second value controls stage 4, and when both stage 3 and stage 4 run, a third value controls the final stage-4 directory. A scalar `--fit-method mchmc` or `--fit-method mclmc` runs the selected microcanonical sampler in every sampled stage; staged values can mix them with `svi` or `svi+nuts`.

One-shot evidence nested sampling example:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_workflow_evidence_ns \
  --fit-mode evidence-ns \
  --evidence-source-prior-sigma-arcsec 20.0 \
  --ns-num-live-points 2000 \
  --ns-max-samples 200000 \
  --ns-dlogz 1e-3 \
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies -1
```

For mock runs with known truth values, pass a truth JSON directly to the solver:

```bash
python -m cluster_solver \
  --par-path data/clustersim/input.par \
  --truth data/clustersim/truth.json \
  --output-dir plots/clustersim \
  --run-name joint_workflow
```

When `--truth` is provided, recovery validation PDFs are written under the run's
`validation/` directory after solver artifacts are saved.

## Mock-Cluster Validation

Mock validation is configured from Python dataclasses, matching the main
`lenscluster.config` API. There is no mock-validation CLI. The runner generates
the mock inputs, converts them directly to a `LensModelConfig`, compiles a
`LensClusterSolverConfig` with `compile_run_plan`, runs `LensClusterRunner`, and
writes recovery diagnostics from the final compiled solver stage.

A minimal single-BCG validation run looks like:

```python
from lenscluster.config import (
    ImageDiagnosticsConfig,
    LensClusterSolverConfig,
    RuntimeConfig,
    StageScheduleConfig,
    TruthRecoveryConfig,
)
from lenscluster.mock_validation import (
    MockValidationConfig,
    MockValidationPathsConfig,
    MockValidationRuntimeConfig,
    MockValidationSolverConfig,
    SingleBCGMockConfig,
    run_single_bcg_validation,
)

config = MockValidationConfig(
    mock=SingleBCGMockConfig(
        n_primary_families=20,
        n_subhalo_families=0,
        n_subhalos=50,
        min_images_per_family=3,
        primary_source_redshifts=(1.5, 2.0, 3.0),
        subhalo_source_redshifts=(1.5, 2.0, 3.0),
        pos_sigma_arcsec=0.15,
    ),
    paths=MockValidationPathsConfig(
        output_dir="validation_runs",
        campaign_name="covariance_test",
        run_name="single_bcg_recovery",
        variant_name="anisotropic",
    ),
    runtime=MockValidationRuntimeConfig(realizations=1, seed=12345),
    solver=MockValidationSolverConfig(
        template=LensClusterSolverConfig(
            runtime=RuntimeConfig(skip_plots=True),
            image_diagnostics=ImageDiagnosticsConfig(
                posterior_image_diagnostic_draws=8,
                posterior_image_diagnostic_mode="exact",
            ),
            truth=TruthRecoveryConfig(
                posterior_truth_recovery_draws=128,
                caustic_plot_grid_scale_arcsec=0.2,
            ),
            schedule=StageScheduleConfig(
                fit_method=("svi+nuts",),
                svi_steps=(2000, 2000),
                refresh_every=(250, 250),
                warmup=(300,),
                samples=(500,),
                sampling_refresh_runs=(1,),
                max_tree_depth=(10,),
            ),
        ),
        run_name="fit",
    ),
)

outputs = run_single_bcg_validation(config)
```

The default mock samples primary source families inside the largest tangential
caustic, accepts sources only when they satisfy the configured image-count
bounds, offsets the BCG slightly from the cluster halo, and records mock truth
for the halo, BCG, sources, and optional subhalo population. `max_images_per_family=None`
keeps the upper multiplicity unbounded.

When subhalos are enabled, the mock draws a Natarajan-style count-matched
Schechter cluster-member luminosity function. It samples a parent population,
maps luminosity to the dPIE mass proxy through the configured scaling relation,
applies the faint member limit, and selects the requested number of members.
The generated member catalog is used only when subhalos are requested.

Validation outputs are written to:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/
```

The default run name and seed produce:

```text
validation_runs/covariance_test/single_bcg_recovery/seed_00012345/anisotropic/
```

Solver outputs are written under the realization directory using the current
compiled stage names:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage0_fast_initializer/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage1_backprojected_centroid_fit/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage2_free_source_forward_fit/  # when stage2_forward_mode != "none"
```

Mock inputs and pre-fit diagnostics are shared by seed:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/mock/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/prefit/
```

Mock-truth recovery PDFs are additionally written at the variant directory level:

- `parameter_recovery.pdf`
- `mass_profile_recovery.pdf`
- `surface_density_recovery.pdf`
- `critical_caustic_recovery.pdf`
- `magnification_recovery.pdf`
- `absolute_magnification_recovery.pdf`
- `image_recovery.pdf`
- `image_residual_histogram.pdf`
- `source_recovery.pdf`
- `subhalo_shmf.pdf`
- `subhalo_recovery_shmf.pdf`
- `subhalo_recovery_radial.pdf`
- `validation_summary.pdf`
- `corner.pdf`
- `potfile_corner.pdf` when potfile scaling parameters are present

The mass-profile validation figures decompose the recovered deflection profile
and annular projected surface density into total, halo, BCG, subhalos, and
BCG+subhalos. Strong-lensing image positions mostly constrain the total
deflection field, so these component-level plots help expose degeneracies.

The posterior artifacts used to make these PDFs are saved under the final
solver stage, for example:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage2_free_source_forward_fit/artifacts/plot_bundle.h5
```

All figures are saved as PDFs. Standard solver diagnostic tables remain under
each stage's `tables/` directory. Set `RuntimeConfig(skip_plots=True)` on the
solver template to suppress standard solver plots; mock-truth recovery PDFs are
still generated.

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
core_i  = core_ref  * L_i^(2 / slope)
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
Exact image-plane solving is used for fit-quality diagnostics and plots; it is
not currently the sampled posterior likelihood.

The sampled likelihood uses a cached magnification-weighted source-plane
metric. It periodically computes the local magnification at each observed image
and uses the equal-area circularized source-plane variance
`sigma_img^2 / |mu| + sigma_int^2`. It preserves the local area scaling of the
full Jacobian covariance while keeping likelihood evaluations scalar and fast.
The covariance floor for this metric is controlled with
`--source-plane-covariance-floor`.

When `--image-plane-mode local-jacobian` is selected, the sequential workflow
adds a third `stage3_image_plane` fit initialized from the source-plane joint
stage. This stage uses the same SVI/NUTS method as stage 2, but replaces the
scalar magnification weighting with the full local 2x2 lensing Jacobian
covariance at each observed image. It is still a differentiable local
approximation, not a full image-finding likelihood.

Stage 3 can also be selected explicitly with `--stage3-image-plane-mode`.
The default `auto` preserves the historical behavior: local-Jacobian stage 3
for `--image-plane-mode local-jacobian`, and local-Jacobian stage 3 before a
final stage 4 unless `--skip-stage3-image-plane-local-jacobian` is passed.
Use `--image-plane-mode none --stage3-image-plane-mode critical-arc-mixture-image-plane`
to run `stage3_image_plane` with a critical-arc point/arc mixture likelihood
but without sampled source coordinates. In that stage-3 centroid mode, each
family source position is recomputed at every likelihood evaluation as the
current weighted centroid of the ray-shot image positions, with weights
proportional to reliability divided by image-plane variance.

When `--image-plane-mode linearized-forward-beta-image-plane` is selected, the
workflow adds `stage4_linearized_image_plane`. By default it also runs
`stage3_image_plane` first; pass `--skip-stage3-image-plane-local-jacobian` or
`--stage3-image-plane-mode none` to initialize stage 4 directly from
`stage2_free_source_forward_fit`. This final stage samples the lens
parameters plus explicit 2D source positions for each multiply imaged family,
initialized from the previous sampled stage's source centroids. The sampled likelihood computes one
local image-plane correction at each observed image even when
`--image-plane-newton-steps 0`; positive values add that many further Newton
updates before scoring the image-plane displacement. It uses its own
`image_sigma_int` scatter parameter in image-plane units.

For sequential resumes, `--resume` is equivalent to `--resume all`: completed
stages are reused/finalized and their outputs may be refreshed. Use
`--resume fast` to load existing previous-stage artifacts and run only the final
enabled stage, such as a missing stage 4 initialized from an existing
`stage3_image_plane` run.

When `--image-plane-mode forward-metric-image-plane` is selected, the workflow
adds `stage4_forward_metric_image_plane`. It samples the same explicit 2D source
positions as the linearized forward-beta stage, but leaves the residual in source
coordinates and scores it with the proposal-current forward image covariance
`A Sigma_img A^T`, where `A = d beta / d theta` at the observed image. This
avoids the inverse-Jacobian image displacement while still using image-plane
positional uncertainties. This mode requires `--image-plane-newton-steps 0` and
does not support `source-position-parameterization=conditional-whitened`.

When `--image-plane-mode fold-regularized-forward-beta-image-plane` is selected,
the workflow adds `stage4_fold_regularized_image_plane`. It uses the same
explicit source-position target as `forward-metric-image-plane`, but near a
critical Jacobian it scores the residual by solving the local signed fold
equation in the singular-vector frame of `A = d beta / d theta`. The constrained
component uses the linear image-plane displacement, while the critical component
uses the minimum real root distance of
`0.5 kappa_eff theta_crit^2 + s_min theta_crit + r_crit = 0`. `kappa_eff` is
estimated by finite-differencing the lensing Jacobian along the observed
critical image-plane direction; with `--sampling-engine refreshing_surrogate_flat`,
the curvature and near-critical row mask are refreshed with the surrogate cache
and reused between refreshes. `--fold-curvature-arcsec-inv` remains only as a
fallback scale for direct helper use. Away from criticality, or when no local
real root exists, the row falls back to the ordinary forward-metric inlier. It
requires `--image-plane-newton-steps 0` and does not support
`source-position-parameterization=conditional-whitened`.

When `--image-plane-mode critical-arc-mixture-image-plane` is selected, the
stage-4 likelihood uses explicit sampled source coordinates, an observed-anchor
LM image-plane correction, and a point/arc residual mixture whose broad
critical-direction branch turns on near singular local Jacobians. The
critical-arc likelihood always evaluates the full exact point/arc mixture;
there are no speed-mode approximations or masked near-critical row splits. Use
`--stage3-image-plane-mode critical-arc-mixture-image-plane` when the same
critical-arc residual model should be used in stage 3 with centroid source
positions instead of sampled explicit source coordinates.

Optional CAB arc-morphology constraints are supplied by `image.arcfile` as an
independent arc catalog, not as image annotations. Rows have the form
`arc_id coord_1 coord_2 z_arc tangent_angle_rad curvature_arcsec_inv
sigma_tangent_angle_rad sigma_curvature_arcsec_inv [reliability]`; `arc_id`
must be unique, and `z_arc` is retained only for provenance diagnostics
(`z_arc >= 0` or `-1`). CAB uses the arc anchor, tangent, and curvature only,
so it never links an arc to an image family. CAB rows use a smoothed local
tangent frame, so constraints automatically fade to an axial-uniform outlier
branch when the local stretch direction or branch choice is ill-defined.

If `--fit-cosmology-flat-wcdm` is enabled, every executed sequential fitting
stage samples the flat wCDM cosmology parameters.

The explicit-beta stage-4 image-plane modes also include a smooth
observed-image presence penalty by default. For each observed image, the local
image-plane residual is converted to a soft presence probability with
`--image-presence-match-radius-arcsec` and
`--image-presence-temperature-arcsec`; each family then receives a smooth
penalty when the reliability-weighted number of present observed images falls
below the catalog count. The effective default
`--image-presence-penalty-weight` is `2.0` for sequential stage 4 and `0.0`
for evidence, non-image-plane likelihoods, and critical-arc mixture likelihoods;
pass a positive value to enable it. In `critical-arc-mixture-image-plane`, the
presence probability reuses the critical-arc branch probability and counts an
image as present when it is close in the noncritical direction, even if displaced
along the arc. This is a differentiable local surrogate for missing
observed-image anchors, not an exact predicted-image multiplicity count from the
full image solver.

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

For `--fit-mode evidence-ns`, the solver skips the sequential stages and builds
one joint evidence target with analytically marginalized source positions under
the configured Gaussian source prior by default. For correctness checks, pass
`--evidence-likelihood-mode linearized-forward-beta-image-plane` to sample one
source position per family from the same evidence source prior and evaluate the
linearized image-plane residual likelihood. After the full NS run, the saved posterior
table is always 4096 posterior draws produced by `NestedSampler.get_samples(...)`.
Those resampled draws are used for log-probability postprocessing, artifact
writing, plotting, and best-fit selection. This resampling step does not change
the evidence run or the `--ns-max-samples` termination limit. `run_summary.json`
records `ns_posterior_samples=4096` and `ns_posterior_resampling` alongside the
JAXNS evidence diagnostics `ns_log_z_mean` and `ns_log_z_uncert`.

## Outputs

For `--run-name joint_workflow`, outputs are written to:

- `plots/m0416_original/joint_workflow/stage0_fast_initializer/`
- `plots/m0416_original/joint_workflow/stage1_backprojected_centroid_fit/`
- `plots/m0416_original/joint_workflow/stage2_free_source_forward_fit/` when `--stage2-forward-mode` is enabled
- `plots/m0416_original/joint_workflow/stage3_image_plane/` when `--image-plane-mode local-jacobian`, when `--stage3-image-plane-mode critical-arc-mixture-image-plane`, or before stage 4 unless skipped
- `plots/m0416_original/joint_workflow/stage4_linearized_image_plane/` when `--image-plane-mode linearized-forward-beta-image-plane`
- `plots/m0416_original/joint_workflow/stage4_forward_metric_image_plane/` when `--image-plane-mode forward-metric-image-plane`
- `plots/m0416_original/joint_workflow/stage4_fold_regularized_image_plane/` when `--image-plane-mode fold-regularized-forward-beta-image-plane`
- `plots/m0416_original/joint_workflow/sequential_summary.json`

Each stage writes:

- `artifacts/plot_bundle.h5`
- `tables/run_summary.json`
- `tables/potential_summary.csv`
- `tables/family_diagnostics.csv`
- `tables/subhalo_properties.csv`
- diagnostic PDFs in the stage directory, including `image_residual_histogram.pdf`;
  potfile subhalo runs also include `subhalo_mass_function.pdf` and
  `subhalo_radial_distribution.pdf`;
  with `--plot-caustics`, this also includes `absolute_magnification.pdf`

`run.xsh` is a thin wrapper around the same command.

The implementation uses a standard `src/lenscluster/` package layout. The
repository-root `cluster_solver.py` remains as a compatibility shim, so existing
`python -m cluster_solver` commands continue to work.

## HFF Pagul21 Catalogs

Generated HFF master/member/image-family catalogs now default to
`results/hff_master_catalogs`, while raw catalog inputs remain under `data/`:

```bash
python scripts/build_hff_master_catalog.py all
python scripts/build_hff_master_catalog.py plots
```

The `all` command builds the catalogs and plot products. Use `plots` when you
only need to regenerate figures from existing CSV outputs.

Rendered HFF `lenscluster.cluster_solver` parameter folders likewise default to
`results/hff_lenscluster_pars`:

```bash
python scripts/build_hff_lenscluster_pars.py render
```

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
