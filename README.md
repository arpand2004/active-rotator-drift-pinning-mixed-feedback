# Active Rotator Drift Pinning with Mixed Feedback

This repository contains simulation scripts, processed data, and figures for the study of drift, pinning, and finite-size effects in fully connected active-rotator systems with fixed intrinsic drive, Kuramoto coupling and mixed-sign local feedback.

The model studied is a coupled active-rotator system of the form

\[
\dot{\theta}_i = \omega -
A_i \sin\theta_i
+
\frac{K}{N-1}
\sum_{j\neq i}
\sin(\theta_j-\theta_i),
\]

where the local feedback amplitudes \(A_i\) are drawn from a zero-mean distribution. The scripts reproduce the figures associated with local fixed-point structure and wrapped time series, drift/pinning regime maps, analytical benchmarks, finite-size effects, and the zero-mean intrinsic-frequency extension.

## Repository structure

```text
active-rotator-drift-pinning-mixed-feedback/
├── data/
│   ├── figure_3.csv
│   ├── figure_4.csv
│   ├── figure_5.csv
│   ├── figure_6.csv
│   ├── figure_7.csv
│   ├── figure_8.csv
│   └── figure_9.csv
├── figures/
│   ├── figure_1.png
│   ├── figure_1.pdf
│   ├── ...
│   ├── figure_9.png
│   └── figure_9.pdf
├── scripts/
│   ├── figures_1_2.py
│   ├── figures_3_4.py
│   ├── figure_5.py
│   ├── figure_6.py
│   ├── figures_7_8.py
│   └── figure_9.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
