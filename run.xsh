$XONSH_SHOW_TRACEBACK = True

import time
import sys
from pathlib import Path

$JAX_NUM_CPU_DEVICES = "30"

PYTHON = "/home/dutra/.conda/envs/lenstronomy/bin/python"
output_dir = f"bergamini_may30a"

VALID_CLUSTERS = "A2744, M0416, M1206, AS1063, A307"
BERGAMINI_CLUSTERS = {
    "A2744": {
        "cluster_key": "a2744",
        "par_path": "data/Bergamini/A2744_Bergamini23/Bergamini23_A2744_Normal.par",
        "output_dir": f"results/{output_dir}/a2744_bergamini23",
    },
    "A2744_CATS": {
        "cluster_key": "a2744_cats",
        "par_path": "data/a2744_cats_v31/input.par",
        "output_dir": f"results/{output_dir}/a2744_cats_v31",
    },
    "M0416": {
        "cluster_key": "m0416",
        "par_path": "data/Bergamini/M0416_Bergamini22/Bergamini22_MACS0416_Normal.par",
        "output_dir": f"results/{output_dir}/m0416_bergamini22",
    },
    "M1206": {
        "cluster_key": "m1206",
        "par_path": "data/Bergamini/M1206_Bergamini19/Bergamini19_MACSJ1206_lenstool_Normal.par",
        "output_dir": f"results/{output_dir}/m1206_bergamini19",
    },
    "AS1063": {
        "cluster_key": "as1063",
        "par_path": "data/Bergamini/RXJ2248_Bergamini19/Bergamini_RXCJ2248_lenstool_Normal.par",
        "output_dir": f"results/{output_dir}/as1063_bergamini19",
    },
    "A307": {
        "cluster_key": "a307",
        "par_path": "data/a370_niemiec/a_sl_normal.par",
        "output_dir": f"results/{output_dir}/a307_niemiec",
    },
}
CLUSTER_ALIASES = {}


def _usage() -> str:
    return f"Usage: xonsh run_as1063.xsh <cluster>\nValid clusters: {VALID_CLUSTERS}"


if len(sys.argv) != 2:
    print(_usage())
    raise SystemExit(2)

requested_cluster = sys.argv[1].strip().upper()
canonical_cluster = CLUSTER_ALIASES.get(requested_cluster, requested_cluster)
if canonical_cluster not in BERGAMINI_CLUSTERS:
    print(f"Unknown Bergamini cluster: {sys.argv[1]!r}\n{_usage()}")
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

# Mirrors the active nuts_sequential fitting settings in run.xsh, but runs the
# real-data cluster solver directly instead of the mock validation wrapper.
run_name = f"{cluster_config['cluster_key']}_linearizedforwardbeta_refreshing_surrogate"
fit_mode = "sequential"
image_plane_mode = "linearized-forward-beta-image-plane"
fit_method = ["svi+nuts", "svi+nuts", "svi+nuts"]
warmup = [500, 1000, 1000]
samples = [150, 250, 250]
skip_stage3_image_plane_local_jacobian = False
quick_diagnostics = False
image_plane_newton_steps = 0
linearized_beta_prior_sigma_arcsec = 0.5
source_position_parameterization = "conditional-whitened"
target_accept = 0.85
max_tree_depth = [8, 8, 8]
active_scaling_galaxies = [50]
cosmology_init_om0 = 0.4
cosmology_init_w0 = -0.9

cosmology_init_args = [
    *(["--cosmology-init-om0", cosmology_init_om0] if cosmology_init_om0 is not None else []),
    *(["--cosmology-init-w0", cosmology_init_w0] if cosmology_init_w0 is not None else []),
]

workflow_args = [
    "--fit-mode", fit_mode,
    "--image-plane-mode", image_plane_mode,
    *(["--skip-stage3-image-plane-local-jacobian"] if skip_stage3_image_plane_local_jacobian else []),
    *(["--quick-diagnostics"] if quick_diagnostics else []),
    "--image-plane-newton-steps", image_plane_newton_steps,
    "--linearized-beta-prior-sigma-arcsec", linearized_beta_prior_sigma_arcsec,
    "--source-position-parameterization", source_position_parameterization,
    "--fit-method", *fit_method,
    "--warmup", *warmup,
    "--samples", *samples,
    "--svi-steps", 1000,
    "--target-accept", target_accept,
    "--max-tree-depth", *max_tree_depth,
    *cosmology_init_args,
]

_run_start_monotonic = time.monotonic()
try:
    @(PYTHON) -m lenscluster.cluster_solver --resume \
      --par-path @(PAR_PATH) \
      --output-dir @(OUTPUT_DIR) \
      --run-name @(run_name) \
      @(workflow_args) \
      @(bayes_overlay_args) \
      @(best_par_overlay_args) \
      --corner-suppress-fit-markers \
      --chains 4 \
      --sampling-engine refreshing_surrogate \
      --active-scaling-galaxies @(active_scaling_galaxies) \
      --active-scaling-selection adaptive \
      --active-scaling-cumulative-fraction 0.995 \
      --active-scaling-min 4 \
      --scaling-scatter \
      --scaling-scatter-fields sigma,cut \
      --scaling-scatter-max 0.5 \
      --z-bin-efficiency-tol 0.02 \
      --plot-caustics \
      --fit-quality-workers 32 \
      --fov-limit-radius 200 \
      --likelihood-stabilizer-residual-loss student-t \
      --likelihood-stabilizer-max-gain 100 \
      --image-plane-scatter-upper-arcsec 2.0 \
      --image-plane-scatter-floor-arcsec 0.20 \
      --image-plane-scatter-prior lognormal \
      --image-plane-scatter-prior-median-arcsec 0.5 \
      --image-plane-scatter-prior-log-sigma 0.5 \
    --image-presence-penalty-weight 1.0 \
    --image-presence-match-radius-arcsec 0.7 \
    --image-presence-temperature-arcsec 0.35
#     --fit-cosmology-flat-wcdm

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
