# Playground Cluster-Solver Workflow

`playground` is the local sandbox for runnable `lenstronomy` experiments, prototypes, and one-off utilities. The current main workflow is the MACS0416 cluster-solver pipeline driven by [`playground/run.xsh`](/home/dutra/dev/lenstronomy/playground/run.xsh), which orchestrates staged runs of [`playground.cluster_solver`](/home/dutra/dev/lenstronomy/playground/cluster_solver.py).

## Prerequisites

- Use the required interpreter for standard project runs:
  `/home/dutra/.conda/envs/lenstronomy/bin/python`
- Run commands from the repository root: `/home/dutra/dev/lenstronomy`
- The current runner assumes JAX CPU execution with:
  `JAX_NUM_CPU_DEVICES=30`
- Some stages also set:
  `XLA_FLAGS=--xla_force_host_platform_device_count=30`

## Current Experiment Configuration

The committed [`run.xsh`](/home/dutra/dev/lenstronomy/playground/run.xsh) is currently configured as:

```text
profile_variant = "original"
PAR_PATH = playground/M0416_Bergamini22/Bergamini22_MACS0416.par
OUTPUT_DIR = plots/m0416_original
STAGE1_RUN_DIR = plots/m0416_original/large
SMC_RUN_DIR = plots/m0416_original/exp_blackjax_smc_gpu
stages = ['large', 'ranked']
active_scaling_galaxies = [-1]
```

This means the default run executes:

- `large`: a large-scale-halo reference fit
- `ranked`: a small-only fit seeded from the saved `large` run

The same file also contains two optional experimental stages:

- `smc`: GPU BlackJAX SMC population sampling
- `smc_refine`: CPU NUTS refinement from saved SMC particles

## Running from `run.xsh`

[`run.xsh`](/home/dutra/dev/lenstronomy/playground/run.xsh) is the authoritative source for the active workflow. If you change `stages`, `PAR_PATH`, `OUTPUT_DIR`, or `profile_variant` there, you are changing the documented workflow.

The file is an Xonsh script that expands the staged commands below. In practice, the important part is the exact `python -m playground.cluster_solver` invocations embedded in it.

## Stage Commands

### `large`

Large-only reference stage. This writes the stage-1 results used by later small-only runs.

```bash
JAX_NUM_CPU_DEVICES=30 \
/home/dutra/.conda/envs/lenstronomy/bin/python -m playground.cluster_solver \
  --par-path playground/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
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
  --profile-variant original \
  --plot-caustics
```

### `ranked`

Ranked-MAP initialization for the small-only stage, using the saved `large` run as `--stage1-run-dir`.

```bash
JAX_NUM_CPU_DEVICES=30 \
/home/dutra/.conda/envs/lenstronomy/bin/python -m playground.cluster_solver \
  --par-path playground/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name exp_ranked \
  --fit-mode small-only \
  --stage1-run-dir plots/m0416_original/large \
  --map-broad-seeds 1 \
  --map-local-refine-seeds 1 \
  --map-local-jitter-scale 0.05 \
  --continuation-sigma-scale 2.0 \
  --continuation-validation-top-k 4 \
  --warmup 1000 \
  --samples 250 \
  --chains 4 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 \
  --nuts-init-strategy prior_center \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --validation-approx adaptive \
  --z-bin-tol 0.25 \
  --profile-variant original \
  --plot-caustics
```

### `smc` (optional, experimental)

Experimental small-only SMC stage. This is the one stage in `run.xsh` that switches to the GPU environment.

```bash
JAX_NUM_CPU_DEVICES=30 \
/home/dutra/.conda/envs/lenstronomygpu/bin/python -m playground.cluster_solver \
  --par-path playground/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name exp_blackjax_smc_gpu \
  --fit-mode small-only \
  --stage1-run-dir plots/m0416_original/large \
  --sampler blackjax_smc \
  --smc-particles 1024 \
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
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 \
  --nuts-init-strategy prior_center \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --validation-approx adaptive \
  --z-bin-tol 0.25 \
  --profile-variant original \
  --plot-caustics
```

### `smc_refine` (optional, experimental)

CPU refinement stage that consumes the saved SMC run directory.

```bash
JAX_NUM_CPU_DEVICES=30 \
XLA_FLAGS=--xla_force_host_platform_device_count=30 \
/home/dutra/.conda/envs/lenstronomy/bin/python -m playground.cluster_solver \
  --refine-from-run-dir plots/m0416_original/exp_blackjax_smc_gpu \
  --output-dir plots/m0416_original \
  --run-name exp_blackjax_smc_gpu_cpu_refine \
  --warmup 500 \
  --samples 250 \
  --chains 4 \
  --sampling-engine refreshing_surrogate \
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 \
  --nuts-init-strategy prior_center \
  --likelihood-mode source \
  --validate-top-k-families 0 \
  --validation-approx adaptive \
  --profile-variant original \
  --plot-caustics
```

## Outputs

Runs land under `plots/m0416_original/<run-name>/`. Each run directory typically contains:

- `artifacts/`
- generated plots in the run directory itself
- generated tables in `tables/` under the run directory

Common saved artifacts and summaries include:

- `artifacts/posterior_arrays.npz`
- `artifacts/cli_args.json`
- `artifacts/init_diagnostics.json`
- `artifacts/plot_bundle.h5`
- `artifacts/stage1_prior_summary.json` for stage-1 compatible runs
- `tables/run_summary.json`
- `tables/potential_summary.csv`
- `tables/family_diagnostics.csv`
- `tables/potfile_summary.txt`
- `tables/potfile_constraint_diagnostics.csv`
- `tables/potfile_constraint_summary.txt`
- `tables/scaling_rank_diagnostics.csv`

Common plots include:

- `corner.png`
- `potfile_corner.png`
- `potfile_histograms.png`
- `trace_plot.png`
- `run_diagnostics.png`
- `residuals_by_family.png`
- `image_plane_fit.png`
- `source_plane_scatter.png`
- `per_potential_summary.png`
- `refresh_diagnostics.png`
- `timing_profile.png`
- `caustic_overlay.png` when caustic plotting is enabled

## Notes

- `run.xsh` currently enables `large` and `ranked` by default.
- `smc` and `smc_refine` are present in the script but commented out of the active `stages` list.
- Standard documented runs use `/home/dutra/.conda/envs/lenstronomy/bin/python`.
- The `smc` path is intentionally documented separately because it uses `/home/dutra/.conda/envs/lenstronomygpu/bin/python`.
