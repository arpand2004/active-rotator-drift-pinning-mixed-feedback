#!/usr/bin/env python3

from pathlib import Path
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_SIMULATION = True

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_3_DATA = DATA_DIR / "figure_3.csv"
FIGURE_4_DATA = DATA_DIR / "figure_4.csv"

FIGURE_3_PNG = FIGURES_DIR / "figure_3.png"
FIGURE_3_PDF = FIGURES_DIR / "figure_3.pdf"
FIGURE_4_PNG = FIGURES_DIR / "figure_4.png"
FIGURE_4_PDF = FIGURES_DIR / "figure_4.pdf"


@dataclass(frozen=True)
class Config:
    N: int = 100
    omega: float = 0.5

    T: float = 1000.0
    dt: float = 0.01
    late_fraction: float = 0.30

    drift_tol_fraction: float = 1e-3

    sigma_min: float = 0.0
    sigma_max: float = 6.0
    K_min: float = 0.0
    K_max: float = 6.0

    n_sigma: int = 100
    n_k: int = 100
    trials: int = 50

    seed: int = 55500


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figures 3 and 4.")

    parser.add_argument("--plot-only", action="store_true")

    parser.add_argument("--N", type=int, default=Config.N)
    parser.add_argument("--omega", type=float, default=Config.omega)

    parser.add_argument("--T", type=float, default=Config.T)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--late-fraction", type=float, default=Config.late_fraction)

    parser.add_argument("--drift-tol-fraction", type=float, default=Config.drift_tol_fraction)

    parser.add_argument("--sigma-min", type=float, default=Config.sigma_min)
    parser.add_argument("--sigma-max", type=float, default=Config.sigma_max)
    parser.add_argument("--K-min", type=float, default=Config.K_min)
    parser.add_argument("--K-max", type=float, default=Config.K_max)

    parser.add_argument("--n-sigma", type=int, default=Config.n_sigma)
    parser.add_argument("--n-k", type=int, default=Config.n_k)
    parser.add_argument("--trials", type=int, default=Config.trials)

    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )

    args = parser.parse_args()

    cfg = Config(
        N=args.N,
        omega=args.omega,
        T=args.T,
        dt=args.dt,
        late_fraction=args.late_fraction,
        drift_tol_fraction=args.drift_tol_fraction,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        K_min=args.K_min,
        K_max=args.K_max,
        n_sigma=args.n_sigma,
        n_k=args.n_k,
        trials=args.trials,
        seed=args.seed,
    )

    return cfg, args.jobs, args.plot_only


def simulate_sigma_trial(job):
    trial, isg, sigma_ratio, K_ratios, cfg_dict = job
    cfg = Config(**cfg_dict)

    N = cfg.N
    omega = cfg.omega

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt

    drift_tol = cfg.drift_tol_fraction * abs(omega)

    K_values = K_ratios * omega
    n_k = len(K_values)

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = sigma_ratio * omega * xi_A

    rng_theta = np.random.default_rng(cfg.seed + 9176 * trial + 37 * isg + 12345)
    theta0 = rng_theta.uniform(0.0, 2.0 * np.pi, N)

    theta = np.repeat(theta0[None, :], n_k, axis=0)
    theta_late_start = None

    fc_factor = N / (N - 1) if N > 1 else 1.0

    for step in range(steps):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        mean_sin = np.mean(sin_theta, axis=1)
        mean_cos = np.mean(cos_theta, axis=1)

        if step == late_start:
            theta_late_start = theta.copy()

        coupling = fc_factor * (
            mean_sin[:, None] * cos_theta
            -
            mean_cos[:, None] * sin_theta
        )

        dtheta = omega - A_i[None, :] * sin_theta + K_values[:, None] * coupling
        theta += cfg.dt * dtheta

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T and late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time

    return {
        "trial": trial,
        "isg": isg,
        "net_drift": np.mean(Omega_i, axis=1),
        "mean_abs_drift": np.mean(np.abs(Omega_i), axis=1),
        "positive_fraction": np.mean(Omega_i > drift_tol, axis=1),
        "negative_fraction": np.mean(Omega_i < -drift_tol, axis=1),
    }


def run_sweep(cfg, jobs):
    sigma_ratios = np.linspace(cfg.sigma_min, cfg.sigma_max, cfg.n_sigma)
    K_ratios = np.linspace(cfg.K_min, cfg.K_max, cfg.n_k)

    shape = (cfg.n_k, cfg.n_sigma)

    sums = {
        "net_drift": np.zeros(shape),
        "mean_abs_drift": np.zeros(shape),
        "positive_fraction": np.zeros(shape),
        "negative_fraction": np.zeros(shape),
    }

    cfg_dict = asdict(cfg)

    job_list = [
        (trial, isg, float(sigma_ratio), K_ratios, cfg_dict)
        for trial in range(cfg.trials)
        for isg, sigma_ratio in enumerate(sigma_ratios)
    ]

    with mp.get_context("spawn").Pool(processes=jobs) as pool:
        for result in pool.imap_unordered(simulate_sigma_trial, job_list, chunksize=1):
            isg = result["isg"]

            for key in sums:
                sums[key][:, isg] += result[key]

    maps = {
        "net_drift_over_omega": sums["net_drift"] / cfg.trials / cfg.omega,
        "mean_abs_drift_over_omega": sums["mean_abs_drift"] / cfg.trials / cfg.omega,
        "positive_fraction": sums["positive_fraction"] / cfg.trials,
        "negative_fraction": sums["negative_fraction"] / cfg.trials,
    }

    return sigma_ratios, K_ratios, maps


def save_figure_3_data(sigma_ratios, K_ratios, maps):
    rows = []
    D = maps["mean_abs_drift_over_omega"]

    for ik, K_ratio in enumerate(K_ratios):
        for isg, sigma_ratio in enumerate(sigma_ratios):
            rows.append([sigma_ratio, K_ratio, D[ik, isg]])

    np.savetxt(
        FIGURE_3_DATA,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="sigma_A_over_omega,K_over_omega,D",
        comments="",
        fmt="%.10g",
    )


def save_figure_4_data(sigma_ratios, K_ratios, maps):
    rows = []

    f_plus = maps["positive_fraction"]
    f_minus = maps["negative_fraction"]

    for ik, K_ratio in enumerate(K_ratios):
        for isg, sigma_ratio in enumerate(sigma_ratios):
            rows.append([sigma_ratio, K_ratio, f_plus[ik, isg], f_minus[ik, isg]])

    np.savetxt(
        FIGURE_4_DATA,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="sigma_A_over_omega,K_over_omega,positive_fraction,negative_fraction",
        comments="",
        fmt="%.10g",
    )


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_grid_from_csv(filename, value_columns):
    df = pd.read_csv(filename)

    x = np.sort(df["sigma_A_over_omega"].unique())
    y = np.sort(df["K_over_omega"].unique())

    out = {}
    for col in value_columns:
        pivot = df.pivot(index="K_over_omega", columns="sigma_A_over_omega", values=col)
        pivot = pivot.reindex(index=y, columns=x)
        out[col] = pivot.to_numpy()

    return x, y, out


def style_main_axes(ax):
    ax.grid(False)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        top=False,
        right=False,
        colors="black",
        width=1.1,
        length=4.5,
    )


def style_inset_axes_white(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        colors="white",
        width=0.9,
        length=3.0,
        labelsize=8,
    )


def make_figure_3():
    sigma_vals, kappa_vals, data = load_grid_from_csv(FIGURE_3_DATA, ["D"])
    D = data["D"]

    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)

    im = ax.pcolormesh(
        sigma_vals,
        kappa_vals,
        D,
        shading="auto",
        cmap="bone",
        vmin=0.0,
        vmax=1.05,
        rasterized=True,
    )

    ax.set_xlabel(r"$r$", fontsize=20)
    ax.set_ylabel(r"$\kappa$", fontsize=20)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    style_main_axes(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.018)
    cbar.set_label(r"$D$", fontsize=20)
    cbar.set_ticks(np.arange(0.0, 1.01, 0.2))

    fig.savefig(FIGURE_3_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_3_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_figure_4():
    sigma_vals, kappa_vals, data = load_grid_from_csv(
        FIGURE_4_DATA,
        ["positive_fraction", "negative_fraction"]
    )

    f_plus = data["positive_fraction"]
    f_minus = data["negative_fraction"]

    fig, ax = plt.subplots(figsize=(9.2, 6.8), constrained_layout=True)

    im = ax.pcolormesh(
        sigma_vals,
        kappa_vals,
        f_plus,
        shading="auto",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    ax.set_xlabel(r"$r$", fontsize=20)
    ax.set_ylabel(r"$\kappa$", fontsize=20)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    style_main_axes(ax)

    cbar = fig.colorbar(im, ax=ax, pad=0.018)
    cbar.set_label(r"$f +$", fontsize=20)
    cbar.set_ticks(np.arange(0.0, 1.01, 0.2))

    axins = ax.inset_axes([0.07, 0.62, 0.29, 0.29])
    cax_in = ax.inset_axes([0.385, 0.62, 0.022, 0.29])

    neg_vmax = float(np.nanmax(f_minus))
    neg_vmax = max(0.4, np.ceil(neg_vmax / 0.2) * 0.2)

    im_in = axins.pcolormesh(
        sigma_vals,
        kappa_vals,
        f_minus,
        shading="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=neg_vmax,
        rasterized=True,
    )

    axins.set_xlim(0, 6)
    axins.set_ylim(0, 6)
    axins.set_xticks([0, 2, 4, 6])
    axins.set_yticks([0, 2, 4, 6])

    style_inset_axes_white(axins)

    axins.set_title(r"$f -$", fontsize=12, color="white", pad=2)

    cbar_in = fig.colorbar(im_in, cax=cax_in)
    cbar_in.set_ticks(np.arange(0.0, neg_vmax + 0.001, 0.2))
    cbar_in.ax.tick_params(labelsize=8, colors="white")
    cbar_in.outline.set_edgecolor("white")
    cbar_in.outline.set_linewidth(1.0)

    for spine in cax_in.spines.values():
        spine.set_color("white")
        spine.set_linewidth(1.0)

    fig.savefig(FIGURE_4_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_4_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, jobs, plot_only = parse_args()

    if RUN_SIMULATION and not plot_only:
        sigma_ratios, K_ratios, maps = run_sweep(cfg, jobs)
        save_figure_3_data(sigma_ratios, K_ratios, maps)
        save_figure_4_data(sigma_ratios, K_ratios, maps)

    set_plot_style()
    make_figure_3()
    make_figure_4()


if __name__ == "__main__":
    mp.freeze_support()
    main()