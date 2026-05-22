$XONSH_SHOW_TRACEBACK = True

import time

$JAX_NUM_CPU_DEVICES = "30"

PYTHON = "/home/dutra/.conda/envs/lenstronomy/bin/python"
OUTPUT_DIR = f"plots/clustersim"
PAR_PATH = "data/clustersim/input.par"
TRUTH_PATH = "data/clustersim/truth.json"
active_scaling_galaxies = [-1]

# The sequential solver runs:
# 1. large-scale SVI
# 2. joint source-plane large+small fit
# 3. optional local-Jacobian image-plane fit
# 4. optional explicit-beta image-plane fit
OUTPUT_DIR = "runs_noscatter_may18_cosmology"

workflow = "nuts_sequential"
quick_diagnostics = False

if workflow == "evidence_ns":
    solver_fit_mode = "evidence-ns"
    image_plane_mode = "none"
    run_name = "single_bcg_recovery_evidence_ns_marginal_beta"
    evidence_source_prior_sigma_arcsec = 20.0
    evidence_source_prior_mean_x_arcsec = 0.0
    evidence_source_prior_mean_y_arcsec = 0.0
    evidence_likelihood_mode = "linearized-forward-beta-image-plane"
    image_plane_newton_steps = 0 # use 1 or 2 for accuracy, but slower
    source_position_parameterization = "prior-whitened" # or "direct"
    ns_num_live_points = 2000
    ns_max_samples = "none"
    ns_dlogz = 0.01
    workflow_args = [
        "--solver-fit-mode", solver_fit_mode,
        "--image-plane-mode", image_plane_mode,
        "--evidence-likelihood-mode", evidence_likelihood_mode,
        "--image-plane-newton-steps", image_plane_newton_steps,
        "--source-position-parameterization", source_position_parameterization,
        "--evidence-source-prior-sigma-arcsec", evidence_source_prior_sigma_arcsec,
        "--evidence-source-prior-mean-x-arcsec", evidence_source_prior_mean_x_arcsec,
        "--evidence-source-prior-mean-y-arcsec", evidence_source_prior_mean_y_arcsec,
        "--ns-num-live-points", ns_num_live_points,
        "--ns-max-samples", ns_max_samples,
        "--ns-dlogz", ns_dlogz,
        *(["--quick-diagnostics"] if quick_diagnostics else []),
    ]

elif workflow == "nuts_sequential":
    solver_fit_mode = "sequential"
    image_plane_mode = "linearized-forward-beta-image-plane"
    run_name = "single_bcg_recovery_nuts_sequential_conditionalwhitened_refreshsurrogate_approximate"
    fit_method = ["svi+nuts", "svi+nuts", "svi+nuts"]
    warmup = [500, 1000, 2000]
    samples = [250, 250, 250]
    skip_stage3_image_plane_local_jacobian = False
    image_plane_newton_steps = 0 # use 1 or 2 for accuracy, but slower
    linearized_beta_prior_sigma_arcsec = 0.3
    source_position_parameterization = "conditional-whitened" # options: "prior-whitened" (safer), "conditional-whitened" (experimental, faster)
    target_accept = 0.9
    max_tree_depth = 8
    workflow_args = [
        "--solver-fit-mode", solver_fit_mode,
        "--image-plane-mode", image_plane_mode,
        *(["--skip-stage3-image-plane-local-jacobian"] if skip_stage3_image_plane_local_jacobian else []),
        *(["--quick-diagnostics"] if quick_diagnostics else []),
        "--image-plane-newton-steps", image_plane_newton_steps,
        "--linearized-beta-prior-sigma-arcsec", linearized_beta_prior_sigma_arcsec,
        "--source-position-parameterization", source_position_parameterization,
        "--fit-method", *fit_method,
        "--warmup", *warmup,
        "--samples", *samples,
        "--svi-steps", 250,
        "--target-accept", target_accept,
        "--max-tree-depth", max_tree_depth,
    ]
else:
    raise ValueError(f"Unknown workflow={workflow!r}")

active_scaling_galaxies = [-1]
subhalo_sigma_scatter_dex = 0.0
subhalo_cut_scatter_dex = 0.0
source_sigma_int_arcsec = 0.0

n_primary_families = 20
n_subhalo_families = 5
nfamilies_total = n_primary_families + n_subhalo_families
min_images_per_family = 3
caustic_compute_window_arcsec = 160
caustic_grid_scale_arcsec = 0.2
caustic_min_area_arcsec2 = 1e-5
caustic_boundary_margin_arcsec = 0.5
if nfamilies_total <= 1:
    source_redshifts = "2.0"
else:
    source_redshifts = ",".join(str(1.5 + idx * (7.0 - 1.5) / (nfamilies_total - 1)) for idx in range(nfamilies_total))

_run_start_monotonic = time.monotonic()
try:
    @(PYTHON) -m lenscluster.validation --resume \
      --output-dir @(OUTPUT_DIR) \
      --run-name @(run_name) \
      --realizations 1 \
      --n-primary-families @(n_primary_families) \
      --n-subhalo-families @(n_subhalo_families) \
      --min-images-per-family @(min_images_per_family) \
      --caustic-compute-window-arcsec @(caustic_compute_window_arcsec) \
      --caustic-grid-scale-arcsec @(caustic_grid_scale_arcsec) \
      --caustic-min-area-arcsec2 @(caustic_min_area_arcsec2) \
      --caustic-boundary-margin-arcsec @(caustic_boundary_margin_arcsec) \
      --n-subhalos 50 \
      --subhalo-sigma-scatter-dex @(subhalo_sigma_scatter_dex) \
      --subhalo-cut-scatter-dex @(subhalo_cut_scatter_dex) \
      --source-redshifts @(source_redshifts) \
      --pos-sigma-arcsec 0 \
      --source-sigma-int-arcsec @(source_sigma_int_arcsec) \
      --no-fit-scaling-scatter \
      @(workflow_args) \
      --chains 4 \
      --sampling-engine refreshing_surrogate \
      --active-scaling-galaxies @(active_scaling_galaxies) \
      --active-scaling-selection adaptive \
      --active-scaling-cumulative-fraction 0.995 \
      --active-scaling-min 4 \
      --posterior-diagnostic-workers 30 \
      --posterior-diagnostic-mode approximate \
      --z-bin-efficiency-tol 0.01 \
      --image-plane-scatter-upper-arcsec 0.2
    #   --posterior-diagnostic-mode approximate \
    #   --fit-cosmology-flat-wcdm
finally:
    print(f"[timing] run.xsh elapsed={time.monotonic() - _run_start_monotonic:.2f}s")
