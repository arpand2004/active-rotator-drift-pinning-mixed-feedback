#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import os
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "figure_5.csv"
FIGURE_PDF = FIGURES_DIR / "figure_5.pdf"
FIGURE_PNG = FIGURES_DIR / "figure_5.png"


@dataclass(frozen=True)
class Config:
    N: int = 100
    omega: float = 0.5

    T: float = 1000.0
    dt: float = 0.01
    late_fraction: float = 0.30

    r_numerics: tuple = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    r_min: float = 0.0
    r_max: float = 6.0
    n_r_analytics: int = 600

    trials: int = 50

    seed: int = 55500


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 5.")

    parser.add_argument("--N", type=int, default=Config.N)
    parser.add_argument("--omega", type=float, default=Config.omega)

    parser.add_argument("--T", type=float, default=Config.T)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--late-fraction", type=float, default=Config.late_fraction)

    parser.add_argument(
        "--r-numerics",
        type=float,
        nargs="+",
        default=list(Config.r_numerics),
    )

    parser.add_argument("--r-min", type=float, default=Config.r_min)
    parser.add_argument("--r-max", type=float, default=Config.r_max)
    parser.add_argument("--n-r-analytics", type=int, default=Config.n_r_analytics)

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
        r_numerics=tuple(args.r_numerics),
        r_min=args.r_min,
        r_max=args.r_max,
        n_r_analytics=args.n_r_analytics,
        trials=args.trials,
        seed=args.seed,
    )

    return cfg, args.jobs


def trapz_manual(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.shape != x.shape:
        raise ValueError("trapz_manual requires y and x to have the same shape.")

    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def analytical_D0(r_values, n_quad=20001):
    r_values = np.asarray(r_values, dtype=float)

    a = np.linspace(-1.0, 1.0, n_quad)
    base = np.sqrt(np.maximum(1.0 - a**2, 0.0))

    D0 = np.zeros_like(r_values, dtype=float)

    for idx, r in enumerate(r_values):
        if np.isclose(r, 0.0):
            D0[idx] = 1.0
        else:
            g = (1.0 / (np.sqrt(2.0 * np.pi) * r)) * np.exp(-a**2 / (2.0 * r**2))
            integrand = base * g
            D0[idx] = trapz_manual(integrand, a)

    return D0


def simulate_kappa0_r_trial(job):
    trial, ir, r_value, cfg_dict = job
    cfg = Config(**cfg_dict)

    N = cfg.N
    omega = cfg.omega

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = r_value * omega * xi_A

    rng_theta = np.random.default_rng(cfg.seed + 9176 * trial + 37 * ir + 12345)
    theta = rng_theta.uniform(0.0, 2.0 * np.pi, N)

    theta_late_start = None

    for step in range(steps):
        sin_theta = np.sin(theta)

        if step == late_start:
            theta_late_start = theta.copy()

        dtheta = omega - A_i * sin_theta
        theta += cfg.dt * dtheta

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T and late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time
    D_trial = np.mean(np.abs(Omega_i)) / omega

    return {
        "trial": trial,
        "ir": ir,
        "r_value": r_value,
        "D_trial": D_trial,
    }


def run_kappa0_sweep(cfg, jobs):
    r_numerics = np.asarray(cfg.r_numerics, dtype=float)
    n_r = len(r_numerics)

    D_trials = np.zeros((cfg.trials, n_r), dtype=float)

    cfg_dict = asdict(cfg)

    job_list = [
        (trial, ir, float(r_value), cfg_dict)
        for trial in range(cfg.trials)
        for ir, r_value in enumerate(r_numerics)
    ]

    def store_result(result):
        trial = result["trial"]
        ir = result["ir"]
        D_trials[trial, ir] = result["D_trial"]

    if jobs == 1:
        for result in map(simulate_kappa0_r_trial, job_list):
            store_result(result)
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(simulate_kappa0_r_trial, job_list, chunksize=1):
                store_result(result)

    D_numerics = np.mean(D_trials, axis=0)
    D_numerics_std = np.std(D_trials, axis=0, ddof=1) if cfg.trials > 1 else np.zeros(n_r)
    D_numerics_sem = D_numerics_std / np.sqrt(cfg.trials)

    r_analytics = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r_analytics)
    D_analytics = analytical_D0(r_analytics)

    return r_numerics, D_numerics, D_numerics_std, D_numerics_sem, r_analytics, D_analytics


def save_data(
    r_numerics,
    D_numerics,
    D_numerics_std,
    D_numerics_sem,
    r_analytics,
    D_analytics,
):
    with open(DATA_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "r", "D", "D_std", "D_sem"])

        for r, D in zip(r_analytics, D_analytics):
            writer.writerow(["analytics", f"{r:.10g}", f"{D:.10g}", "", ""])

        for r, D, std, sem in zip(r_numerics, D_numerics, D_numerics_std, D_numerics_sem):
            writer.writerow([
                "numerics",
                f"{r:.10g}",
                f"{D:.10g}",
                f"{std:.10g}",
                f"{sem:.10g}",
            ])


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax):
    ax.grid(False)

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


def make_figure(r_numerics, D_numerics, r_analytics, D_analytics):
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8.4, 6.1), constrained_layout=True)

    ax.plot(
        r_analytics,
        D_analytics,
        color="black",
        lw=3.2,
        label="Analytics",
        zorder=3,
    )

    ax.plot(
        r_numerics,
        D_numerics,
        linestyle="none",
        marker="s",
        markersize=12.0,
        markerfacecolor="#B22222",
        markeredgecolor="black",
        markeredgewidth=0.7,
        label="Numerics",
        zorder=4,
    )

    ax.set_xlabel(r"$r$", fontsize=20)
    ax.set_ylabel(r"$D_0(r)$", fontsize=20)

    x_pad = 0.03 * (np.max(r_analytics) - np.min(r_analytics))
    ax.set_xlim(np.min(r_analytics) - x_pad, np.max(r_analytics) + x_pad)

    y_min = min(np.nanmin(D_numerics), np.nanmin(D_analytics))
    y_max = max(np.nanmax(D_numerics), np.nanmax(D_analytics))
    y_pad = 0.06 * (y_max - y_min)

    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    style_axes(ax)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.82",
        fontsize=15,
        handlelength=2.5,
        borderpad=0.95,
        labelspacing=0.72,
        handletextpad=0.85,
    )

    fig.savefig(FIGURE_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, jobs = parse_args()

    (
        r_numerics,
        D_numerics,
        D_numerics_std,
        D_numerics_sem,
        r_analytics,
        D_analytics,
    ) = run_kappa0_sweep(cfg, jobs)

    save_data(
        r_numerics,
        D_numerics,
        D_numerics_std,
        D_numerics_sem,
        r_analytics,
        D_analytics,
    )

    make_figure(r_numerics, D_numerics, r_analytics, D_analytics)


if __name__ == "__main__":
    mp.freeze_support()
    main()