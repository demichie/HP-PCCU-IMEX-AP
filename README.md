# HP-PCCU / IMEX-AP reference implementation and reproducibility benchmarks

This repository package accompanies the manuscript on the hydrostatic-path PCCU method with local implicit rheology and relaxation-aware asymptotic stabilization.

The package is intentionally code-focused. It contains Python reference cores, verification tests, benchmark drivers, and machine-readable reference results. LaTeX sources, compiled manuscript files, and manuscript figures are not included here.

## Layout

- `src/`
  - `hp_pccu_1d_core.py`: 1-D hydrostatic-path PCCU reference core using solid mass fraction as the reconstructed composition coordinate.
  - `hp_pccu_1d_blended_ap.py`: 1-D Newtonian IMEX/AP layer.
  - `hp_pccu_1d_blended_ap_bingham.py`: 1-D Bingham extension with the yield gate used in the manuscript benchmarks.
  - `hp_pccu_2d_core.py`: pure-liquid 2-D reference core used for the manuscript 2-D reproducibility calculations.
  - `hp_pccu_2d_imex_ap.py`: 2-D Newtonian IMEX/AP extension used for the 2-D benchmarks.
- `tests/`: direct verification scripts for dry-safe reconstruction, composition reconstruction, 2-D well balancing, split equivalence, and wet/dry robustness.
- `benchmarks/`: benchmark drivers grouped by manuscript test.
- `reference_results/`: machine-readable outputs underlying the numerical values reported in the manuscript. No figure files are stored here.

## Python requirements

Python 3.10+ is recommended. The reference scripts require only:

```bash
python -m pip install numpy pandas
```

`pandas` is used only by some benchmark/result scripts; the numerical cores themselves require NumPy only.

## Quick verification

From the repository root:

```bash
python run_tests.py
```

The verification scripts are intentionally small and are not a replacement for the full benchmark suite.

## Benchmark drivers

The main benchmark entry points are:

```bash
python benchmarks/hydrodynamic_1d/run_hydrodynamic.py
python benchmarks/hydrodynamic_1d/run_excavation_zero_rheology.py
python benchmarks/smooth_bump/run_swashes_smooth_bump.py
python benchmarks/damped_pulses/run_damped_pulses.py
python benchmarks/damped_pulses/run_apoff_dt_sensitivity.py
python benchmarks/relaxation_lake/run_relaxation_lake.py
python benchmarks/rough_newtonian/run_rough_newtonian.py
python benchmarks/bingham_arrest/run_bingham_arrest.py
python benchmarks/excavation_1d/run_excavation_rheology.py
python benchmarks/excavation_2d/run_excavation_2d.py
python benchmarks/circular_2d/run_circular_expansion.py
```

Each driver writes numerical output into a local `results/` directory. These result directories are excluded by `.gitignore` so that reruns do not pollute the repository.

The smooth-bump cases use the standard SWASHES analytical steady-flow benchmark. An executable driver based on the released 1-D core is provided in `benchmarks/smooth_bump/run_swashes_smooth_bump.py`. The L1 values reported in the manuscript are stored in `reference_results/smooth_bump_L1_errors.csv`, while `reference_results/smooth_bump_detailed_errors.csv` contains the complete error diagnostics produced by the same current-core protocol.

## Damped-pulse comparison

The damped Gaussian and square-pulse benchmark is the cleanest test of the relaxation-aware modification. The AP-off calculation and the two AP calculations use the same HP-PCCU/IMEX discretization; the only difference is whether the relaxation-aware replacement is active and whether the AP calculation uses the common small step or its natural accepted step. The reference solution is the exact linear damped shallow-water solution in Fourier space.

## Reference results

`reference_results/` contains the numerical tables and histories used for manuscript cross-checks, including:

- hydrodynamic well-balancing / Ritter / excavation diagnostics;
- damped-pulse errors, refinement, and AP-off time-step sensitivity;
- strongly viscous rough-topography metrics and time histories;
- Bingham profiles, histories, and grid self-differences for N=40,80,160,320;
- 1-D and 2-D excavation rheology diagnostics;
- circular-expansion isotropy and time-step values;
- the smooth-bump L1 error table.

The reference CSV/JSON files are kept separate from generated `results/` so benchmark reruns can be compared without overwriting the manuscript reference values.

## Scope

These Python files are compact scientific reference implementations and reproducibility drivers. They are not intended to replace the production Fortran code. The numerical choices match the manuscript benchmarks: CFL 0.24 unless stated otherwise, generalized minmod reconstruction with theta=1.3, dry tolerance 1e-10 m, and the wet/smooth AP threshold 0.10.
