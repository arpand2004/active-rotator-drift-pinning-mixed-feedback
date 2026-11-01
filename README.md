# Collective drift and pinning in active rotator networks with Kuramoto coupling and mixed-sign feedback disorder

Numerical code, simulation data, figures, and supplementary tables accompanying the V3 revision of *Collective drift and pinning in active rotator networks with Kuramoto coupling and mixed-sign feedback disorder*.

## Repository structure

- `scripts/` — simulations, stationary mean-field calculations, figure generation, and numerical validation.
- `data/simulations/` — simulation results, summaries, and completion manifests.
- `data/theory/` — stationary solutions, stability calculations, and fold data.
- `data/comparisons/` — matched hysteresis and independent-scan comparisons.
- `data/checks/` and `data/validations/` — numerical robustness and disorder-sampling controls.
- `figures/` — eight main and six supplementary figures in PDF and PNG formats.
- `tables/` — two supplementary tables in LaTeX and CSV formats.
- `requirements.txt` — Python dependencies.

The full map and coupling-cut trial datasets, together with production checkpoints, are included in the GitHub V3 release assets.

## Version 3 updates

This revision includes refined stationary mean-field branch and stability calculations, finite-size comparisons, collective-phase rotation and spectral diagnostics, and reentrant drift analysis. Additional controls examine upward/downward coupling sweeps, numerical integration and measurement-window sensitivity, backward phase excursions, and antithetic versus independent Gaussian disorder sampling.