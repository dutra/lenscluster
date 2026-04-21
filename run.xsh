$JAX_NUM_CPU_DEVICES = "30"

profile_variant = "original"

OUTPUT_DIR = f"plots/m0416_{profile_variant}"
PAR_PATH = "data/M0416_Bergamini22/Bergamini22_MACS0416.par"
active_scaling_galaxies = [-1]

# The solver now always runs:
# 1. large-scale SVI
# 2. joint large+small SVI
# 3. optional joint NUTS when --fit-method svi+nuts is selected
fit_method = "svi+nuts"

python -m cluster_solver \
  --par-path @(PAR_PATH) \
  --output-dir @(OUTPUT_DIR) \
  --run-name joint_workflow \
  --fit-method @(fit_method) \
  --warmup 1000 \
  --samples 250 \
  --chains 4 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-galaxies @(active_scaling_galaxies) \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --validation-approx adaptive \
  --z-bin-tol 0.25 \
  --profile-variant @(profile_variant) \
  --plot-caustics
