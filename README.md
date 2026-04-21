# Lenscluster Workflow

This repository currently centers on the staged MACS0416 cluster-solver workflow driven by [`run.xsh`](/home/dutra/dev/lenscluster/run.xsh). The script orchestrates runs of [`cluster_solver.py`](/home/dutra/dev/lenscluster/cluster_solver.py) with a small set of named stages.

## Repository Layout

- [`run.xsh`](/home/dutra/dev/lenscluster/run.xsh): authoritative staged runner
- [`cluster_solver.py`](/home/dutra/dev/lenscluster/cluster_solver.py): main CLI entrypoint
- [`data/M0416_Bergamini22/Bergamini22_MACS0416.par`](/home/dutra/dev/lenscluster/data/M0416_Bergamini22/Bergamini22_MACS0416.par): active input model
- `plots/`: run outputs

## Current `run.xsh` Configuration

The committed runner is currently configured as:

```python
profile_variant = "original"
OUTPUT_DIR = "plots/m0416_original"
PAR_PATH = "data/M0416_Bergamini22/Bergamini22_MACS0416.par"
STAGE1_RUN_DIR = "plots/m0416_original/large"
SMC_RUN_DIR = "plots/m0416_original/exp_blackjax_smc_gpu"
stages = ["large", "ranked"]
active_scaling_galaxies = [-1]
```

That means the default run executes:

- `large`: large-only reference fit
- `ranked`: small-only run initialized from the saved `large` stage

Two additional stages are present in the script but disabled by default:

- `smc`: GPU BlackJAX SMC population sampling
- `smc_refine`: CPU refinement from a saved SMC run

## Runtime Assumptions

`run.xsh` currently sets:

- `JAX_NUM_CPU_DEVICES=30`
- `XLA_FLAGS=--xla_force_host_platform_device_count=30` for the refinement section

Most stages call:

```bash
python -m cluster_solver ...
```

The `smc` stage is the exception and explicitly uses:

```bash
/home/dutra/.conda/envs/lenstronomygpu/bin/python -m cluster_solver ...
```

## Stage Commands

### `large`

Large-only reference stage that seeds later small-only runs.

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
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

Small-only ranked-MAP initialization using the saved `large` run as `--stage1-run-dir`.

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
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

### `smc` (optional)

Experimental GPU population-sampling stage.

```bash
/home/dutra/.conda/envs/lenstronomygpu/bin/python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
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

### `smc_refine` (optional)

CPU refinement stage that resumes from the saved SMC particles.

```bash
python -m cluster_solver \
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

Runs land under `plots/m0416_original/<run-name>/`.

Common contents include:

- `artifacts/`
- `tables/`
- diagnostic plots written into the run directory

Typical saved files include:

- `artifacts/posterior_arrays.npz`
- `artifacts/cli_args.json`
- `artifacts/init_diagnostics.json`
- `artifacts/plot_bundle.h5`
- `tables/run_summary.json`
- `tables/potential_summary.csv`
- `tables/family_diagnostics.csv`

## Notes

- [`run.xsh`](/home/dutra/dev/lenscluster/run.xsh) is the source of truth for the active workflow.
- Changing `stages`, `PAR_PATH`, `OUTPUT_DIR`, or `profile_variant` in that script changes the documented run configuration.
- The current default workflow enables `large` and `ranked`.
