$XONSH_SHOW_TRACEBACK = True

import time
import sys
from pathlib import Path

cores = 4

$JAX_NUM_CPU_DEVICES = cores

PYTHON = "/home/dutra/.conda/envs/lenstronomy/bin/python"

output_dir = f"jun20i_stage3_blend_topksurrogate_freetopk_refresh_faster_tokpsurrogateflat_20topk_hyperscaling_bergamini"

HFF_RGB_BANDS = ["F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W"]
HFF_RGB_DISPLAY = {"q": 6.4, "stretch": 0.0145, "minimum": -5.5e-4, "red_gain": 0.47, "green_gain": 0.91, "blue_gain": 3.95}
# h10 neutral warm from the HERA FF-SIMS ACS RGB tuning pass.
FF_RGB_DISPLAY = {"q": 6.8, "stretch": 0.0158, "minimum": 0.00105, "red_gain": 0.62, "green_gain": 0.78, "blue_gain": 3.65}

VALID_CLUSTERS = "A2744, M0416, M1206, AS1063, A307, AS1063_CAMINHA, ARES, HERA"
BERGAMINI_CLUSTERS = {
    "A2744": {
        "cluster_key": "a2744",
        "par_path": "data/Bergamini/A2744_Bergamini23/Bergamini23_A2744_Normal.par",
        "output_dir": f"results/{output_dir}/a2744_bergamini23",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "A2744_CATS": {
        "cluster_key": "a2744_cats",
        "par_path": "data/a2744_cats_v31/input.par",
        "output_dir": f"results/{output_dir}/a2744_cats_v31",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "M0416": {
        "cluster_key": "m0416",
        "par_path": "data/Bergamini/M0416_Bergamini22/Bergamini22_MACS0416_Normal.par",
        "output_dir": f"results/{output_dir}/m0416_bergamini22",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "M1206": {
        "cluster_key": "m1206",
        "par_path": "data/Bergamini/M1206_Bergamini19/Bergamini19_MACSJ1206_lenstool_Normal.par",
        "output_dir": f"results/{output_dir}/m1206_bergamini19",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "AS1063": {
        "cluster_key": "as1063",
        "par_path": "data/Bergamini/RXJ2248_Bergamini19/Bergamini_RXCJ2248_lenstool_Normal.par",
        "output_dir": f"results/{output_dir}/as1063_bergamini19",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "AS1063_CAMINHA": {
        "cluster_key": "as1063_caminha",
        "par_path": "data/Bergamini/A1063_Caminha16/A1063_lenstool_Caminha2016.par",
        "output_dir": f"results/{output_dir}/as1063_caminha16",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "A307": {
        "cluster_key": "a307",
        "par_path": "data/a370_niemiec/a_sl_normal.par",
        "output_dir": f"results/{output_dir}/a307_niemiec",
        "image_catalog_family_cutout_rgb": dict(HFF_RGB_DISPLAY),
    },
    "ARES": {
        "cluster_key": "ares",
        "par_path": "data/ff_sims/ares/ares_lenscluster.par",
        "output_dir": f"results/{output_dir}/ares",
        "image_catalog_family_cutout_image_dir": "data/ff_sims",
        "image_catalog_family_cutout_image_scale": "auto",
        "image_catalog_family_cutout_bands": ["F435W", "F606W", "F814W"],
        "image_catalog_family_cutout_rgb": dict(FF_RGB_DISPLAY),
        "kappa_true_fits": "data/ff_sims/ares/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/ares/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/ares/gammay_z9_0.fits",
    },
    "HERA": {
        "cluster_key": "hera",
        "par_path": "data/ff_sims/hera/hera_lenscluster.par",
        "output_dir": f"results/{output_dir}/hera",
        "image_catalog_family_cutout_image_dir": "data/ff_sims",
        "image_catalog_family_cutout_image_scale": "auto",
        "image_catalog_family_cutout_bands": ["F435W", "F606W", "F814W"],
        "image_catalog_family_cutout_rgb": dict(FF_RGB_DISPLAY),
        "kappa_true_fits": "data/ff_sims/hera/kappa_z9_0.fits",
        "gammax_true_fits": "data/ff_sims/hera/gammax_z9_0.fits",
        "gammay_true_fits": "data/ff_sims/hera/gammay_z9_0.fits",
    },
}
CLUSTER_ALIASES = {}


def _usage() -> str:
    return f"Usage: xonsh run.xsh <cluster>\nValid clusters: {VALID_CLUSTERS}"


if len(sys.argv) != 2:
    print(_usage())
    raise SystemExit(2)

requested_cluster = sys.argv[1].strip().upper()
canonical_cluster = CLUSTER_ALIASES.get(requested_cluster, requested_cluster)
if canonical_cluster not in BERGAMINI_CLUSTERS:
    print(f"Unknown cluster: {sys.argv[1]!r}\n{_usage()}")
    raise SystemExit(2)

cluster_config = BERGAMINI_CLUSTERS[canonical_cluster]
PAR_PATH = cluster_config["par_path"]
OUTPUT_DIR = cluster_config["output_dir"]
if not Path(PAR_PATH).is_file():
    raise SystemExit(f"Bergamini par file does not exist: {PAR_PATH}")
BAYES_PATH = Path(PAR_PATH).with_name("bayes.dat")
bayes_overlay_args = ["--corner-overlay-bayes-dat", str(BAYES_PATH)] if BAYES_PATH.is_file() else []
BEST_PAR_PATH = Path(PAR_PATH).with_name("best.par")
best_par_overlay_args = ["--corner-overlay-best-par", str(BEST_PAR_PATH)] if BEST_PAR_PATH.is_file() else []

# Mirrors the active solver-control settings in run_validation.xsh, but runs the
# real-data cluster solver directly instead of the mock validation wrapper.
mode = "none"  # "linear", "metric", "anchored", "critical_arc", or "fold_regularized"
run_name = f"{cluster_config['cluster_key']}_{mode}_nuts_nozeff"
fit_mode = "sequential"
image_plane_modes = {
    "none": "none",
    "linear": "linearized-forward-beta-image-plane",
    "metric": "forward-metric-image-plane",
    "anchored": "anchored-solved-forward-beta-image-plane",
    "critical_arc": "critical-arc-mixture-image-plane",
    "fold_regularized": "fold-regularized-forward-beta-image-plane",
    "catastrophe": "catastrophe-normal-form-image-plane",
}
image_plane_mode = image_plane_modes[mode]
stage3_mode = "local-jacobian"#"local-jacobian" #"critical-arc-mixture-image-plane"

fit_method = ["svi+nuts", "svi+nuts"]#, "svi+nuts"]
refresh_every = 1000
svi_steps = [3000, 4000]#, 4000]
warmup = [500, 1000]#, 2000]
samples = [150, 250]#, 500]
sampling_refresh_runs = [1, 1]#, 1]
max_tree_depth = [8, 8]#, 8]
quick_diagnostics = False
target_accept = 0.8
chains = cores
active_scaling_galaxies = [25]
z_bin_efficiency_tol = 0.0

# Short stage4 conditioning check before a full production run: confirm the magnification fold
# un-sticks the chains (adapted step size recovers, tree saturation drops, chains move) cheaply.
pilot = False
if pilot:
    refresh_every = 500
    target_accept = 0.8
    svi_steps = [500, 500, 500]
    fit_method = ["svi+nuts", "mchmc", "mchmc"]
    fit_method = ["svi+nuts", "svi+nuts", "svi+nuts"]
    warmup = [200, 5000, 500]
    samples = [100, 500, 250]
    max_tree_depth = [6, 8, 8]
    #quick_diagnostics = True
    chains = cores
    active_scaling_galaxies = [128]
    z_bin_efficiency_tol = 0.0000001


OUTPUT_DIR = f"{OUTPUT_DIR}_ASG{active_scaling_galaxies[-1]}_T{max_tree_depth[-1]}W{warmup[-1]}S{samples[-1]}"
#OUTPUT_DIR = f"{OUTPUT_DIR}_ASG{active_scaling_galaxies[-1]}_T{max_tree_depth[-2]}W{warmup[-2]}S{samples[-2]}_T{max_tree_depth[-1]}W{warmup[-1]}S{samples[-1]}"

start_at_stage2 = False
start_at_stage3 = True
skip_stage3_image_plane_local_jacobian = False
exact_image_diagnostics_stage2 = True
image_catalog_family_cutout_image_dir = cluster_config.get("image_catalog_family_cutout_image_dir", "data/BUFFALO_Images")
image_catalog_family_cutout_image_scale = cluster_config.get("image_catalog_family_cutout_image_scale", "30mas")
image_catalog_family_cutout_bands = cluster_config.get(
    "image_catalog_family_cutout_bands",
    HFF_RGB_BANDS if image_catalog_family_cutout_image_dir == "data/BUFFALO_Images" else None,
)
image_catalog_family_cutout_rgb = cluster_config.get("image_catalog_family_cutout_rgb", {})
image_catalog_family_cutout_rgb_args = [
    "--image-catalog-family-cutout-rgb-q", image_catalog_family_cutout_rgb["q"],
    "--image-catalog-family-cutout-rgb-stretch", image_catalog_family_cutout_rgb["stretch"],
    "--image-catalog-family-cutout-rgb-minimum", image_catalog_family_cutout_rgb["minimum"],
    "--image-catalog-family-cutout-rgb-red-gain", image_catalog_family_cutout_rgb["red_gain"],
    "--image-catalog-family-cutout-rgb-green-gain", image_catalog_family_cutout_rgb["green_gain"],
    "--image-catalog-family-cutout-rgb-blue-gain", image_catalog_family_cutout_rgb["blue_gain"],
] if image_catalog_family_cutout_rgb else []
image_catalog_family_cutout_args = [
    "--image-catalog-family-cutout-image-dir", image_catalog_family_cutout_image_dir,
    "--image-catalog-family-cutout-image-scale", image_catalog_family_cutout_image_scale,
    *(["--image-catalog-family-cutout-bands", *image_catalog_family_cutout_bands] if image_catalog_family_cutout_bands else []),
    *(image_catalog_family_cutout_rgb_args),
] if image_catalog_family_cutout_image_dir else []
kappa_true_fits = cluster_config.get("kappa_true_fits")
kappa_true_args = ["--kappa-true-fits", kappa_true_fits] if kappa_true_fits else []
gammax_true_fits = cluster_config.get("gammax_true_fits")
gammay_true_fits = cluster_config.get("gammay_true_fits")
gamma_true_args = [
    *(["--gammax-true-fits", gammax_true_fits] if gammax_true_fits else []),
    *(["--gammay-true-fits", gammay_true_fits] if gammay_true_fits else []),
]
image_plane_newton_steps = 0
linearized_beta_prior_sigma_arcsec = 3.0
source_position_parameterization = "prior-whitened" #conditional-whitened" #"prior-whitened"
source_plane_covariance_mode = "magnification"
sampling_engine = "topk_discovery_flat" # "refreshing_surrogate_flat" # "full" #"refreshing_surrogate"
stage4_sampling_engine = "topk_discovery_flat" # "refreshing_surrogate_flat" # full_flat
scaling_relation_mode = "bergamini-ml"
topk_discovery_scaling_galaxies = 10
topk_discovery_score = "alpha_jacobian"
topk_discovery_jacobian_weight = 1.0
topk_discovery_final_engine = "refreshing_surrogate_flat"
topk_discovery_final_svi_polish_steps = 2000
independent_scaling_free_log_vdisp_tau_prior_median = 0.20
independent_scaling_free_log_core_tau_prior_median = 0.30
independent_scaling_free_log_cut_tau_prior_median = 0.30
independent_scaling_free_log_tau_prior_sigma = 0.40
pos_sigma_arcsec = 0.01
anchored_args = [
    "--anchored-image-plane-solve-steps", 0,
    "--anchored-image-plane-trust-radius-arcsec", 0.3,
    "--anchored-image-plane-lm-damping-relative", 1.0e-3,
    "--anchored-image-plane-lm-damping-absolute", 1.0e-6,
] if mode == "anchored" else []
critical_arc_args = [
    "--critical-arc-critical-direction-sigma-arcsec", 10.0,
    "--critical-arc-base-prob", 0.10,
    "--critical-arc-max-prob", 0.85,
    # Arc-mixture gate only -- the magnification fold uses its own baked-in threshold/softness.
    "--sample-critical-arc-singular-threshold",
    "--critical-arc-singular-threshold", 0.4,
    "--critical-arc-singular-threshold-prior-median", 0.15,
    "--critical-arc-singular-threshold-prior-log-sigma", 0.5,
    "--critical-arc-singular-threshold-lower", 0.03,
    "--critical-arc-singular-threshold-upper", 0.40,
    "--sample-critical-arc-singular-softness",
    "--critical-arc-singular-softness", 0.05,
    "--critical-arc-singular-softness-prior-median", 0.05,
    "--critical-arc-singular-softness-prior-log-sigma", 0.5,
    "--critical-arc-singular-softness-lower", 0.005,
    "--critical-arc-singular-softness-upper", 0.20,
    "--critical-arc-lm-damping-relative", 1.0e-3,
    "--critical-arc-lm-damping-absolute", 1.0e-6,
    "--critical-arc-lm-trust-radius-arcsec", 10.0,
    "--arc-recovery-p-arc-threshold", 0.5,
    "--arc-aware-max-arclength-arcsec", 10.0,
    "--arc-aware-curve-step-arcsec", 0.01,
] if mode == "critical_arc" else []
fold_regularized_args = [
    "--fold-curvature-arcsec-inv", 1.0,
] if mode == "fold_regularized" else []
smc_args = [
    "--jax-default-device", "cpu",
    "--smc-device", "cpu",
    "--smc-particles", 4096,
    "--smc-mcmc-kernel", "mala",
    "--smc-mcmc-steps", 8,
    "--smc-target-ess-frac", 0.85,
    "--smc-max-temperature-steps", 256,
    "--smc-mala-step-size", 0.03,
] if mode == "critical_arc" else []


workflow_args = [
    "--potfile-member-mag-max", 22.0,
    "--sampling-refresh-runs", *sampling_refresh_runs,
    *(["--start-at-stage2"] if start_at_stage2 else []),
    *(["--start-at-stage3"] if start_at_stage3 else []),
    "--fit-mode", fit_mode,
    "--stage3-image-plane-mode", stage3_mode,
    "--image-plane-mode", image_plane_mode,
    *(["--skip-stage3-image-plane-local-jacobian"] if skip_stage3_image_plane_local_jacobian else []),
    *(["--exact-image-diagnostics-stage2"] if exact_image_diagnostics_stage2 else []),
    "--image-plane-newton-steps", image_plane_newton_steps,
    "--linearized-beta-prior-sigma-arcsec", linearized_beta_prior_sigma_arcsec,
    "--source-position-parameterization", source_position_parameterization,
    "--source-plane-covariance-mode", source_plane_covariance_mode,
    "--stage4-fresh-process",
    "--stage4-sampling-engine", stage4_sampling_engine,
    "--fit-method", *fit_method,
    "--warmup", *warmup,
    "--samples", *samples,
    "--refresh-every", refresh_every,
    "--svi-steps", *svi_steps,
    "--chains", chains,
    "--target-accept", target_accept,
    "--max-tree-depth", *max_tree_depth,
    "--z-bin-efficiency-tol", z_bin_efficiency_tol,

    *(anchored_args),
    *(critical_arc_args),
    *(fold_regularized_args),
    *(smc_args),
]

active_scaling_args = [
    "--sampling-engine", sampling_engine,
    "--scaling-relation-mode", scaling_relation_mode,
    "--topk-discovery-scaling-galaxies", topk_discovery_scaling_galaxies,
    "--topk-discovery-score", topk_discovery_score,
    "--topk-discovery-jacobian-weight", topk_discovery_jacobian_weight,
    "--topk-discovery-final-engine", topk_discovery_final_engine,
    "--topk-discovery-final-svi-polish-steps", topk_discovery_final_svi_polish_steps,
    "--independent-scaling-free-log-vdisp-tau-prior-median", independent_scaling_free_log_vdisp_tau_prior_median,
    "--independent-scaling-free-log-core-tau-prior-median", independent_scaling_free_log_core_tau_prior_median,
    "--independent-scaling-free-log-cut-tau-prior-median", independent_scaling_free_log_cut_tau_prior_median,
    "--independent-scaling-free-log-tau-prior-sigma", independent_scaling_free_log_tau_prior_sigma,
    "--active-scaling-galaxies", *active_scaling_galaxies,
    "--active-scaling-selection", "fixed",
    "--active-scaling-cumulative-fraction", 0.995,
    "--active-scaling-min", 4,
    #"--infer-active-scaling",
    #"--active-scaling-prior-prob", 0.5,
    #"--active-scaling-freeze-threshold", 0.5,
    #"--active-scaling-inference-likelihood", "population",
    #"--active-scaling-local-logit-prior-sigma", 0.5,
]

scatter_and_stabilizer_args = [
    "--pos-sigma-arcsec", pos_sigma_arcsec,
    "--image-presence-penalty-weight", 2.0,
    "--image-presence-match-radius-arcsec", 1.0,
    "--image-presence-temperature-arcsec", 0.5,
    "--image-plane-scatter-prior", "log-uniform",
    "--image-plane-scatter-floor-arcsec", 0.01,
    "--image-plane-scatter-upper-arcsec", 1.0,
    "--scaling-scatter",
    "--scaling-scatter-fields", "sigma,cut",
    #"--likelihood-stabilizer-max-gain", "50",
    # "--likelihood-stabilizer-max-residual-arcsec", "5",
    # "--likelihood-stabilizer-residual-loss", "student-t",
    # "--likelihood-stabilizer-student-t-nu", "4",
    

]

validation_args = [
    *(["--quick-diagnostics"] if quick_diagnostics else []),
    "--exact-image-min-distance-arcsec", 0.5,
    "--exact-image-precision-limit", 1.0e-2,
    "--exact-image-num-iter-max", 50,
    "--match-tolerance-arcsec", 2.0,
    "--caustic-source-redshift", 9.0,
    *(image_catalog_family_cutout_args),
    *(kappa_true_args),
    *(gamma_true_args),
]

real_data_args = [
    #"--fov-limit-radius", 200,
    #"--fit-cosmology-flat-wcdm"
]

debug_args = [
    "--no-jax-clear-caches-after-svi-refresh",
    #"--quick-diagnostics",
    "--image-catalog-family-cutout-mode", "fast",
    "--numpyro-print-summary",
    "--nuts-chain-method", "parallel",
    #"--skip-validation",
    "--dense-mass", "structured",
    #"--plots-only",
    "--debug-sampler-diagnostics",
    #"--fix-image-sigma-int-arcsec", "0.5",
]

_run_start_monotonic = time.monotonic()
try:
    @(PYTHON) -m lenscluster.cluster_solver --resume all \
      --par-path @(PAR_PATH) \
      --output-dir @(OUTPUT_DIR) \
      --run-name @(run_name) \
      @(debug_args) \
      @(workflow_args) \
      @(active_scaling_args) \
      @(scatter_and_stabilizer_args) \
      @(validation_args) \
      @(real_data_args) \
      @(bayes_overlay_args) \
      @(best_par_overlay_args)

 #   --jax-default-device cpu --smc-device gpu \

# --quick-diagnostics \

# --smc-particles 4096 \
# --smc-mcmc-kernel rmh \
# --smc-rmh-scale 0.05 \
# --smc-mcmc-steps 32 \
# --smc-target-ess-frac 0.95 \
# --smc-max-temperature-steps 1024 \
#rmh or mala
    #   --quick-diagnostics
    #   --likelihood-stabilizer-max-gain 50 \
    #   --likelihood-stabilizer-max-residual-arcsec 3 \
    #   --likelihood-stabilizer-residual-loss student-t \
    #   --likelihood-stabilizer-student-t-nu 4 \

finally:
    print(f"[timing] elapsed={time.monotonic() - _run_start_monotonic:.2f}s")
    printf '\033[?1000l\033[?1002l\033[?1003l\033[?1005l\033[?1006l\033[?1015l'
