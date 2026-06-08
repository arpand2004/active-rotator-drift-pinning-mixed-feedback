#!/usr/bin/env python3

from pathlib import Path
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt


RUN_SIMULATION = True

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "figure_9.csv"
FIGURE_PDF = FIGURES_DIR / "figure_9.pdf"
FIGURE_PNG = FIGURES_DIR / "figure_9.png"


@dataclass(frozen=True)
class Config:
    N: int = 100

    omega_mean: float = 0.0
    omega_variance: float = 0.5

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
    parser = argparse.ArgumentParser(description="Generate Figure 9.")

    parser.add_argument("--plot-only", action="store_true")

    parser.add_argument("--N", type=int, default=Config.N)

    parser.add_argument("--omega-mean", type=float, default=Config.omega_mean)
    parser.add_argument("--omega-variance", type=float, default=Config.omega_variance)

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

    if args.omega_variance <= 0:
        raise ValueError("omega_variance must be positive.")

    cfg = Config(
        N=args.N,
        omega_mean=args.omega_mean,
        omega_variance=args.omega_variance,
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


def omega_ref(cfg):
    return np.sqrt(cfg.omega_variance)


def simulate_sigma_trial(job):
    trial, isg, sigma_ratio, K_ratios, cfg_dict = job
    cfg = Config(**cfg_dict)

    N = cfg.N
    omega_scale = omega_ref(cfg)
    omega_std = omega_scale

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt

    drift_tol = cfg.drift_tol_fraction * omega_scale

    K_values = K_ratios * omega_scale
    n_k = len(K_values)

    rng_omega = np.random.default_rng(cfg.seed + 424242 * trial + 73)
    omega_i = rng_omega.normal(cfg.omega_mean, omega_std, N)

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = sigma_ratio * omega_scale * xi_A

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

        dtheta = omega_i[None, :] - A_i[None, :] * sin_theta + K_values[:, None] * coupling
        theta += cfg.dt * dtheta

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T and late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time

    return {
        "isg": isg,
        "mean_abs_drift": np.mean(np.abs(Omega_i), axis=1),
        "positive_fraction": np.mean(Omega_i > drift_tol, axis=1),
        "negative_fraction": np.mean(Omega_i < -drift_tol, axis=1),
    }


def run_sweep(cfg, jobs):
    omega_scale = omega_ref(cfg)

    sigma_ratios = np.linspace(cfg.sigma_min, cfg.sigma_max, cfg.n_sigma)
    K_ratios = np.linspace(cfg.K_min, cfg.K_max, cfg.n_k)

    shape = (cfg.n_k, cfg.n_sigma)

    sums = {
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

    if jobs == 1:
        for result in map(simulate_sigma_trial, job_list):
            isg = result["isg"]

            for key in sums:
                sums[key][:, isg] += result[key]
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(simulate_sigma_trial, job_list, chunksize=1):
                isg = result["isg"]

                for key in sums:
                    sums[key][:, isg] += result[key]

    maps = {
        "mean_abs_drift_over_sigma_omega": sums["mean_abs_drift"] / cfg.trials / omega_scale,
        "positive_fraction": sums["positive_fraction"] / cfg.trials,
        "negative_fraction": sums["negative_fraction"] / cfg.trials,
    }

    return sigma_ratios, K_ratios, maps


def save_data(sigma_ratios, K_ratios, maps):
    rows = []

    for ik, K_ratio in enumerate(K_ratios):
        for isg, sigma_ratio in enumerate(sigma_ratios):
            rows.append([
                sigma_ratio,
                K_ratio,
                maps["mean_abs_drift_over_sigma_omega"][ik, isg],
                maps["positive_fraction"][ik, isg],
                maps["negative_fraction"][ik, isg],
            ])

    header = (
        "sigma_A_over_sigma_omega,"
        "K_over_sigma_omega,"
        "mean_abs_drift_over_sigma_omega,"
        "positive_fraction,"
        "negative_fraction"
    )

    np.savetxt(
        DATA_PATH,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10g",
    )


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.2,
        "mathtext.fontset": "dejavusans",
    })


def load_saved_maps():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    data = np.genfromtxt(DATA_PATH, delimiter=",", names=True)

    sigma_vals = np.sort(np.unique(data["sigma_A_over_sigma_omega"]))
    K_vals = np.sort(np.unique(data["K_over_sigma_omega"]))

    n_sigma = len(sigma_vals)
    n_k = len(K_vals)

    D_map = np.full((n_k, n_sigma), np.nan)
    fplus_map = np.full((n_k, n_sigma), np.nan)
    fminus_map = np.full((n_k, n_sigma), np.nan)

    sigma_to_idx = {val: i for i, val in enumerate(sigma_vals)}
    K_to_idx = {val: i for i, val in enumerate(K_vals)}

    for row in data:
        isg = sigma_to_idx[row["sigma_A_over_sigma_omega"]]
        ik = K_to_idx[row["K_over_sigma_omega"]]

        D_map[ik, isg] = row["mean_abs_drift_over_sigma_omega"]
        fplus_map[ik, isg] = row["positive_fraction"]
        fminus_map[ik, isg] = row["negative_fraction"]

    return sigma_vals, K_vals, D_map, fplus_map, fminus_map


def style_main_axes(ax):
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


def style_inset_axes_black(axins):
    axins.set_facecolor("white")

    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    axins.tick_params(
        axis="both",
        which="both",
        direction="out",
        colors="black",
        width=0.9,
        length=3.5,
        labelsize=9,
    )

    axins.title.set_color("black")
    axins.xaxis.label.set_color("black")
    axins.yaxis.label.set_color("black")

    axins.set_xticks([0, 2, 4, 6])
    axins.set_yticks([0, 2, 4, 6])


def style_inset_colorbar_black(cbar_in):
    cbar_in.ax.tick_params(labelsize=9, colors="black")
    cbar_in.outline.set_edgecolor("black")
    cbar_in.outline.set_linewidth(1.0)
    cbar_in.ax.yaxis.label.set_color("black")


def plot_combined_figure(sigma_vals, K_vals, D_map, fplus_map, fminus_map):
    set_plot_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.4, 6.8),
        constrained_layout=False,
    )

    fig.subplots_adjust(wspace=0.12)
    axD, axF = axes

    vmax_use = max(1.05, float(np.nanmax(D_map)))

    imD = axD.pcolormesh(
        sigma_vals,
        K_vals,
        D_map,
        shading="auto",
        cmap="bone",
        vmin=0.0,
        vmax=vmax_use,
        rasterized=True,
    )

    axD.set_xlabel(r"$\sigma_A/\sigma_\omega$")
    axD.set_ylabel(r"$K/\sigma_\omega$")

    cbarD = fig.colorbar(imD, ax=axD, pad=0.018)
    cbarD.set_label(r"$D$")

    style_main_axes(axD)

    imF = axF.pcolormesh(
        sigma_vals,
        K_vals,
        fplus_map,
        shading="auto",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    axF.set_xlabel(r"$\sigma_A/\sigma_\omega$")
    axF.set_ylabel(r"$K/\sigma_\omega$")

    cbarF = fig.colorbar(imF, ax=axF, pad=0.018)
    cbarF.set_label(r"$f +$", fontstyle="italic")

    inset_bounds = [0.19, 0.60, 0.33, 0.33]
    cbar_bounds = [0.553, 0.60, 0.020, 0.33]

    axins = axF.inset_axes(inset_bounds)
    cax_in = axF.inset_axes(cbar_bounds)

    neg_vmax = max(0.15, float(np.nanmax(fminus_map)))

    im_in = axins.pcolormesh(
        sigma_vals,
        K_vals,
        fminus_map,
        shading="auto",
        cmap="Greys",
        vmin=0.0,
        vmax=neg_vmax,
        rasterized=True,
    )

    axins.set_title(r"$f -$", fontsize=14, pad=5, color="black")

    axins.set_xlim(np.min(sigma_vals), np.max(sigma_vals))
    axins.set_ylim(np.min(K_vals), np.max(K_vals))

    style_inset_axes_black(axins)

    cbar_in = fig.colorbar(
        im_in,
        cax=cax_in,
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    style_inset_colorbar_black(cbar_in)

    style_main_axes(axF)

    fig.savefig(FIGURE_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, jobs, plot_only = parse_args()

    if RUN_SIMULATION and not plot_only:
        sigma_vals, K_vals, maps = run_sweep(cfg, jobs)
        save_data(sigma_vals, K_vals, maps)

    sigma_vals, K_vals, D_map, fplus_map, fminus_map = load_saved_maps()
    plot_combined_figure(sigma_vals, K_vals, D_map, fplus_map, fminus_map)


if __name__ == "__main__":
    mp.freeze_support()
    main()