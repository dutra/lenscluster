$JAX_NUM_CPU_DEVICES = "30"
#$JAX_PLATFORMS = "cpu"
#$XLA_FLAGS= "--xla_force_host_platform_device_count=30"

profile_variant = "original"

# Set these once.
OUTPUT_DIR = f"plots/m0416_{profile_variant}"

PAR_PATH = f"data/M0416_Bergamini22/Bergamini22_MACS0416.par"
STAGE1_RUN_DIR = f"{OUTPUT_DIR}/large"
SMC_RUN_DIR = f"{OUTPUT_DIR}/exp_blackjax_smc_gpu"

stages = ['large', 'ranked']#, 'smc', 'smc_refine']
# stages = ['ranked']

active_scaling_galaxies = [-1]

if 'large' in stages:
  print("-" * 80, end="\n\n")
  print("Running large-only reference stage...", flush=True)
  
  # Large-only reference
  python -m cluster_solver \
    --par-path @(PAR_PATH) \
    --output-dir @(OUTPUT_DIR) \
    --run-name large \
    --fit-mode large-only \
    --map-broad-seeds 3 \
    --map-local-refine-seeds 3 \
    --map-local-jitter-scale 0.06 \
    --continuation-sigma-scale 2.5 \
    --continuation-validation-top-k 0 \
    --warmup 1000 \
    --samples 250 \
    --likelihood-mode source \
    --nuts-init-strategy prior_center \
    --validate-top-k-families 0 \
    --validation-approx adaptive \
    --z-bin-tol 0.25 \
    --chains 6 \
    --profile-variant @(profile_variant) \
    --plot-caustics

if 'smc' in stages:
  print("-" * 80, end="\n\n")
  print("Running SMC stage...", flush=True)

  # Experimental small-only: blackjax_smc population sampler for RTX 5070 (stage 1 / GPU)
  /home/dutra/.conda/envs/lenstronomygpu/bin/python -m cluster_solver \
    --par-path @(PAR_PATH) \
    --output-dir @(OUTPUT_DIR) \
    --run-name exp_blackjax_smc_gpu \
    --fit-mode small-only \
    --stage1-run-dir @(STAGE1_RUN_DIR) \
    --sampler blackjax_smc \
    --smc-particles @(1024) \
    --smc-ess-threshold 0.5 \
    --smc-move-steps 12 \
    --smc-move-scale 0.02 \
    --smc-seed-mode prior \
    --map-broad-seeds 1 \
    --map-local-refine-seeds 1 \
    --map-local-jitter-scale 0.05 \
    --continuation-sigma-scale 2.0 \
    --continuation-validation-top-k 0 \
    --warmup 1000 \
    --samples 250 \
    --chains 4 \
    --sampling-engine refreshing_surrogate \
    --active-scaling-galaxies @(active_scaling_galaxies) \
    --refresh-param-drift-frac 0.08 \
    --target-accept 0.9 \
    --max-tree-depth 8 \
    --nuts-init-strategy prior_center \
    --likelihood-mode source \
    --validate-top-k-families 0 \
    --validation-approx adaptive \
    --z-bin-tol 0.25 \
    --profile-variant @(profile_variant) \
    --plot-caustics


$JAX_NUM_CPU_DEVICES = "30"
#$JAX_PLATFORMS = "cpu"
$XLA_FLAGS= "--xla_force_host_platform_device_count=30"

if 'smc_refine' in stages:
  print("-" * 80, end="\n\n")
  print("Running SMC GPU refinement stage...", flush=True)
  # Experimental small-only: refine saved SMC particles with CPU-only NUTS (stage 2 / CPU)
python -m cluster_solver \
    --refine-from-run-dir @(SMC_RUN_DIR) \
    --output-dir @(OUTPUT_DIR) \
    --run-name exp_blackjax_smc_gpu_cpu_refine \
    --warmup 500 \
    --samples 250 \
    --chains 4 \
    --sampling-engine refreshing_surrogate \
    --active-scaling-galaxies @(active_scaling_galaxies) \
    --refresh-param-drift-frac 0.08 \
    --target-accept 0.9 \
    --max-tree-depth 8 \
    --nuts-init-strategy prior_center \
    --likelihood-mode source \
    --validate-top-k-families 0 \
    --validation-approx adaptive \
    --profile-variant @(profile_variant) \
    --plot-caustics

if 'ranked' in stages:
  print("-" * 80, end="\n\n")
  print("Running ranked-MAP NUTS initialization stage...", flush=True)
  # Experimental small-only: ranked-MAP NUTS initialization
  python -m playground.cluster_solver \
    --par-path @(PAR_PATH) \
    --output-dir @(OUTPUT_DIR) \
    --run-name exp_ranked \
    --fit-mode small-only \
    --stage1-run-dir @(STAGE1_RUN_DIR) \
    --map-broad-seeds 1 \
    --map-local-refine-seeds 1 \
    --map-local-jitter-scale 0.05 \
    --continuation-sigma-scale 2.0 \
    --continuation-validation-top-k 4 \
    --warmup 1000 \
    --samples 250 \
    --chains 4 \
    --sampling-engine refreshing_surrogate \
    --active-scaling-galaxies @(active_scaling_galaxies) \
    --refresh-param-drift-frac 0.08 \
    --target-accept 0.9 \
    --max-tree-depth 8 \
    --nuts-init-strategy prior_center \
    --likelihood-mode source \
    --validate-top-k-families 0 \
    --validation-approx adaptive \
    --z-bin-tol 0.25 \
    --profile-variant @(profile_variant) \
    --plot-caustics 
