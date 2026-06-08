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

DATA_PATH = DATA_DIR / "figure_6.csv"
FIGURE_PNG = FIGURES_DIR / "figure_6.png"
FIGURE_PDF = FIGURES_DIR / "figure_6.pdf"


@dataclass(frozen=True)
class Config:
    N: int = 100
    omega: float = 0.5

    T: float = 1000.0
    dt: float = 0.01
    late_fraction: float = 0.30

    sigma_min: float = 0.0
    sigma_max: float = 6.0
    K_min: float = 0.0
    K_max: float = 6.0

    n_sigma: int = 100
    n_k: int = 100
    trials: int = 50

    theta_min_deg: float = 130.0
    theta_max_deg: float = 140.0

    seed: int = 55500


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 6.")

    parser.add_argument("--plot-only", action="store_true")

    parser.add_argument("--N", type=int, default=Config.N)
    parser.add_argument("--omega", type=float, default=Config.omega)

    parser.add_argument("--T", type=float, default=Config.T)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--late-fraction", type=float, default=Config.late_fraction)

    parser.add_argument("--sigma-min", type=float, default=Config.sigma_min)
    parser.add_argument("--sigma-max", type=float, default=Config.sigma_max)
    parser.add_argument("--K-min", type=float, default=Config.K_min)
    parser.add_argument("--K-max", type=float, default=Config.K_max)

    parser.add_argument("--n-sigma", type=int, default=Config.n_sigma)
    parser.add_argument("--n-k", type=int, default=Config.n_k)
    parser.add_argument("--trials", type=int, default=Config.trials)

    parser.add_argument("--theta-min-deg", type=float, default=Config.theta_min_deg)
    parser.add_argument("--theta-max-deg", type=float, default=Config.theta_max_deg)

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
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        K_min=args.K_min,
        K_max=args.K_max,
        n_sigma=args.n_sigma,
        n_k=args.n_k,
        trials=args.trials,
        theta_min_deg=args.theta_min_deg,
        theta_max_deg=args.theta_max_deg,
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

    K_values = K_ratios * omega
    n_k = len(K_values)

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = sigma_ratio * omega * xi_A

    theta_min = np.deg2rad(cfg.theta_min_deg)
    theta_max = np.deg2rad(cfg.theta_max_deg)

    rng_theta = np.random.default_rng(cfg.seed + 9176 * trial + 37 * isg + 12345)
    theta0 = rng_theta.uniform(theta_min, theta_max, N)

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
            - mean_cos[:, None] * sin_theta
        )

        dtheta = omega - A_i[None, :] * sin_theta + K_values[:, None] * coupling
        theta += cfg.dt * dtheta

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T or late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time
    mean_abs_drift = np.mean(np.abs(Omega_i), axis=1)

    return {
        "isg": isg,
        "mean_abs_drift": mean_abs_drift,
    }


def run_sweep(cfg, jobs):
    sigma_ratios = np.linspace(cfg.sigma_min, cfg.sigma_max, cfg.n_sigma)
    K_ratios = np.linspace(cfg.K_min, cfg.K_max, cfg.n_k)

    mean_abs_drift_sum = np.zeros((cfg.n_k, cfg.n_sigma))
    cfg_dict = asdict(cfg)

    job_list = [
        (trial, isg, float(sigma_ratio), K_ratios, cfg_dict)
        for trial in range(cfg.trials)
        for isg, sigma_ratio in enumerate(sigma_ratios)
    ]

    if jobs == 1:
        for result in map(simulate_sigma_trial, job_list):
            mean_abs_drift_sum[:, result["isg"]] += result["mean_abs_drift"]
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(simulate_sigma_trial, job_list, chunksize=1):
                mean_abs_drift_sum[:, result["isg"]] += result["mean_abs_drift"]

    D_map = mean_abs_drift_sum / cfg.trials / cfg.omega

    return sigma_ratios, K_ratios, D_map


def save_data(r_values, kappa_values, D_map):
    rows = []

    for ik, kappa in enumerate(kappa_values):
        for ir, r in enumerate(r_values):
            rows.append({
                "r": float(r),
                "kappa": float(kappa),
                "D": float(D_map[ik, ir]),
            })

    pd.DataFrame(rows).to_csv(DATA_PATH, index=False)


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 20,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)

    required_cols = {"r", "kappa", "D"}
    missing = required_cols.difference(df.columns)

    if missing:
        raise ValueError(f"{DATA_PATH} is missing required columns: {missing}")

    r_values = np.sort(df["r"].unique())
    kappa_values = np.sort(df["kappa"].unique())

    pivot = df.pivot(index="kappa", columns="r", values="D")
    pivot = pivot.reindex(index=kappa_values, columns=r_values)

    return r_values, kappa_values, pivot.to_numpy()


def add_theory_lines(ax, r_values, kappa_values):
    r_min = np.min(r_values)
    r_max = np.max(r_values)
    k_min = np.min(kappa_values)
    k_max = np.max(kappa_values)

    r_line = np.linspace(r_min, r_max, 600)
    kappa_parabola = 0.5 * r_line**2

    mask = (kappa_parabola >= k_min) & (kappa_parabola <= k_max)

    ax.plot(
        r_line[mask],
        kappa_parabola[mask],
        color="black",
        lw=2.2,
        linestyle="solid",
        zorder=10,
    )

    r_crit = np.sqrt(np.pi / 2.0)

    if r_min <= r_crit <= r_max:
        ax.axvline(
            r_crit,
            color="black",
            lw=2.2,
            linestyle="--",
            zorder=10,
        )


def add_custom_legend_box(ax):
    legend_text = (
        r"$\theta_i(0) \in [130°,140°]$"
        "\n"
        r"$\,$"
        "\n"
        "━━━  " + r"$\kappa = r^{2}/2$"
        "\n"
        r"$\,$"
        "\n"
        "━ ━  " + r"$r = \sqrt{\pi/2}$"
    )

    ax.text(
        0.69,
        0.07,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        linespacing=0.9,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
            linewidth=1.0,
        ),
        zorder=20,
    )


def make_plot(r_values, kappa_values, D_map):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True)

    im = ax.pcolormesh(
        r_values,
        kappa_values,
        D_map,
        shading="auto",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    ax.set_xlabel(r"$r$", fontsize=20)
    ax.set_ylabel(r"$\kappa$", fontsize=20)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))

    add_theory_lines(ax, r_values, kappa_values)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$D$", fontsize=20)
    cbar.set_ticks(np.arange(0.0, 1.01, 0.2))
    cbar.ax.tick_params(labelsize=12)

    add_custom_legend_box(ax)

    fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, jobs, plot_only = parse_args()

    if RUN_SIMULATION and not plot_only:
        r_values, kappa_values, D_map = run_sweep(cfg, jobs)
        save_data(r_values, kappa_values, D_map)

    r_values, kappa_values, D_map = load_data()
    make_plot(r_values, kappa_values, D_map)


if __name__ == "__main__":
    mp.freeze_support()
    main()