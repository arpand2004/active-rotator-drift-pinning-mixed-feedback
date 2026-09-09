# Active Rotator Drift Pinning with Mixed Feedback

This repository contains simulation code, processed numerical data, and figures associated with the preprint:

**Collective drift and pinning in active rotator networks with Kuramoto coupling and mixed-sign feedback disorder**

arXiv: https://arxiv.org/abs/2606.10032

## Repository structure

```text
active-rotator-drift-pinning-mixed-feedback/
├── data/
│   ├── n100/
│   │   ├── drift_map.csv
│   │   ├── drift_fractions.csv
│   │   ├── order_parameter_cuts.csv
│   │   └── stationary_mean_field_boundary.csv
│   ├── n200/
│   │   ├── representative_dynamics.npz
│   │   └── representative_dynamics_summary.csv
│   └── finite_size/
│       ├── drift_vs_kappa.csv
│       └── coupling_windows.csv
├── figures/
│   ├── figure_1.png
│   ├── figure_1.pdf
│   ├── ...
│   ├── figure_7.png
│   └── figure_7.pdf
├── scripts/
│   ├── figures_1_2.py
│   ├── simulate_main_results.py
│   ├── simulate_finite_size.py
│   ├── plot_figures_3_6.py
│   ├── plot_figure_7.py
│   ├── simulators/
│   └── plotters/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data

The repository includes the processed numerical datasets used to generate the manuscript figures.

The standard parameter sweeps for Figures 3–5 use `N=100`. Representative dynamical realizations and ensemble diagnostics for Figure 6 use `N=200`. Finite-size calculations for Figure 7 use `N=20,40,80,120,200`.

## Reproducing the figures

Figures 1 and 2:

```bash
python scripts/figures_1_2.py
```

Figures 3–6 from the saved datasets:

```bash
python scripts/plot_figures_3_6.py
```

Figure 7 from the saved finite-size datasets:

```bash
python scripts/plot_figure_7.py
```

## Regenerating the numerical data

The main simulation datasets used by Figures 3–6 can be regenerated with:

```bash
python scripts/simulate_main_results.py
```

Individual groups can also be run separately:

```bash
python scripts/simulate_main_results.py --only grid
python scripts/simulate_main_results.py --only order
python scripts/simulate_main_results.py --only representative
```

The finite-size datasets used by Figure 7 can be regenerated with:

```bash
python scripts/simulate_finite_size.py
```

These simulations are substantially more computationally expensive than plotting from the supplied processed data.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The numerical and plotting code uses NumPy, pandas, and Matplotlib.

## License

See `LICENSE`.