# SWASHES smooth-bump benchmark

This directory contains an executable driver for the four standard fully wet
steady-flow tests over the SWASHES smooth bump,

`B(x) = 0.2 - 0.05 (x-10)^2` for `|x-10| <= 2`, and zero elsewhere,
on `[0,25]` m.

The four parameter sets are:

- subcritical: discharge `q=4.42 m^2/s`, downstream depth `h_R=2 m`;
- supercritical: discharge `q=10 m^2/s`, upstream depth `h_L=1 m`;
- transcritical without shock: `q=1.53 m^2/s`;
- transcritical with hydraulic jump: `q=0.18 m^2/s`, downstream depth `h_R=0.33 m`.

## Run

From the repository root:

```bash
python benchmarks/smooth_bump/run_swashes_smooth_bump.py
```

The default refinement sequence is `N=40,80,160,320`. The driver uses the
current `src/hp_pccu_1d_core.py` and initializes each calculation from the
analytical steady solution, then advances it for a 0.2 s drift test. This is
the short steady-reference protocol used during development of the manuscript
smooth-bump refinement study.

For a fast smoke test:

```bash
python benchmarks/smooth_bump/run_swashes_smooth_bump.py --quick
```

Outputs are written to `benchmarks/smooth_bump/results/` and include a CSV error
table plus NPZ profiles for each case/grid. The `results/` directory is ignored
by git.

## Analytical reference

The driver solves the steady Bernoulli relation on the appropriate hydraulic
branch. For the hydraulic-jump case it locates the non-trivial stationary shock
by matching the momentum function across the supercritical and subcritical
branches.

## Reference data

`reference_results/smooth_bump_L1_errors.csv` contains the compact L1 thickness-error table reported in the manuscript. `reference_results/smooth_bump_detailed_errors.csv` contains the complete current-core error diagnostics for the same four cases and four grids. Running the driver reproduces these values in `benchmarks/smooth_bump/results/`.
