#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
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

DATA_7_PATH = DATA_DIR / "figure_7.csv"
DATA_8_PATH = DATA_DIR / "figure_8.csv"

FIGURE_7_PNG = FIGURES_DIR / "figure_7.png"
FIGURE_7_PDF = FIGURES_DIR / "figure_7.pdf"
FIGURE_8_PNG = FIGURES_DIR / "figure_8.png"
FIGURE_8_PDF = FIGURES_DIR / "figure_8.pdf"


@dataclass(frozen=True)
class Config:
    omega: float = 0.5

    T: float = 1000.0
    dt: float = 0.01
    late_fraction: float = 0.30

    K_min: float = 0.0
    K_max: float = 6.0
    n_k: int = 100

    low_kappa_min: float = 0.0
    low_kappa_max: float = 0.5
    high_kappa_min: float = 5.5
    high_kappa_max: float = 6.0

    trials: int = 50

    N_values: tuple = (20, 40, 80, 120, 200)

    r_fixed_for_curve: float = 2.0

    r_values: tuple = (
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5, 5.0, 5.5, 6.0
    )

    fig8_r_values: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    seed: int = 55500


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figures 7 and 8.")

    parser.add_argument("--plot-only", action="store_true")

    parser.add_argument("--omega", type=float, default=Config.omega)

    parser.add_argument("--T", type=float, default=Config.T)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--late-fraction", type=float, default=Config.late_fraction)

    parser.add_argument("--K-min", type=float, default=Config.K_min)
    parser.add_argument("--K-max", type=float, default=Config.K_max)
    parser.add_argument("--n-k", type=int, default=Config.n_k)

    parser.add_argument("--low-kappa-min", type=float, default=Config.low_kappa_min)
    parser.add_argument("--low-kappa-max", type=float, default=Config.low_kappa_max)
    parser.add_argument("--high-kappa-min", type=float, default=Config.high_kappa_min)
    parser.add_argument("--high-kappa-max", type=float, default=Config.high_kappa_max)

    parser.add_argument("--trials", type=int, default=Config.trials)

    parser.add_argument(
        "--N-values",
        type=int,
        nargs="+",
        default=list(Config.N_values),
    )

    parser.add_argument(
        "--r-values",
        type=float,
        nargs="+",
        default=list(Config.r_values),
    )

    parser.add_argument(
        "--fig8-r-values",
        type=float,
        nargs="+",
        default=list(Config.fig8_r_values),
    )

    parser.add_argument("--r-fixed-for-curve", type=float, default=Config.r_fixed_for_curve)
    parser.add_argument("--seed", type=int, default=Config.seed)

    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )

    args = parser.parse_args()

    cfg = Config(
        omega=args.omega,
        T=args.T,
        dt=args.dt,
        late_fraction=args.late_fraction,
        K_min=args.K_min,
        K_max=args.K_max,
        n_k=args.n_k,
        low_kappa_min=args.low_kappa_min,
        low_kappa_max=args.low_kappa_max,
        high_kappa_min=args.high_kappa_min,
        high_kappa_max=args.high_kappa_max,
        trials=args.trials,
        N_values=tuple(args.N_values),
        r_fixed_for_curve=args.r_fixed_for_curve,
        r_values=tuple(args.r_values),
        fig8_r_values=tuple(args.fig8_r_values),
        seed=args.seed,
    )

    return cfg, args.jobs, args.plot_only


FINITE_SIZE_COLORS = {
    20: "#66A6D9",
    40: "#6CC08B",
    80: "#E9B44C",
    120: "#9D8AC7",
    200: "#D95F5F",
}

FALLBACK_N_COLORS = [
    "#66A6D9",
    "#6CC08B",
    "#E9B44C",
    "#9D8AC7",
    "#D95F5F",
    "#4C78A8",
    "#54A24B",
    "#F58518",
    "#B279A2",
    "#B22222",
]

BROWN_R_COLORS = {
    1.0: "#D7B98E",
    2.0: "#BE9362",
    3.0: "#A2764A",
    4.0: "#7F5735",
    5.0: "#604026",
    6.0: "#3F2818",
}

R_MARKERS = {
    1.0: "D",
    2.0: "^",
    3.0: "v",
    4.0: "X",
    5.0: "s",
    6.0: "o",
}

BASE_MARKER_SIZE = 7.6
MARKER_EDGE_WIDTH = 1.55

R_MARKER_SIZES = {
    1.0: BASE_MARKER_SIZE,
    2.0: BASE_MARKER_SIZE,
    3.0: BASE_MARKER_SIZE,
    4.0: 8.6,
    5.0: 6.8,
    6.0: 10.2,
}


def get_color_for_N(N, index):
    if int(N) in FINITE_SIZE_COLORS:
        return FINITE_SIZE_COLORS[int(N)]
    return FALLBACK_N_COLORS[index % len(FALLBACK_N_COLORS)]


def simulate_one_N_r_trial(job):
    N, r_value, trial, K_ratios, low_mask, high_mask, cfg_dict = job
    cfg = Config(**cfg_dict)

    omega = cfg.omega

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt

    K_values = K_ratios * omega
    n_k = len(K_values)

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial + 7919 * N)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = r_value * omega * xi_A

    rng_theta = np.random.default_rng(cfg.seed + 9176 * trial + 104729 * N + 12345)
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

    D_curve = np.mean(np.abs(Omega_i), axis=1) / omega

    D_low = float(np.mean(D_curve[low_mask]))
    D_high = float(np.mean(D_curve[high_mask]))
    Delta_D = D_high - D_low

    return {
        "N": N,
        "r_value": float(r_value),
        "trial": trial,
        "D_curve": D_curve,
        "D_low": D_low,
        "D_high": D_high,
        "Delta_D": Delta_D,
    }


def run_combined_sweep(cfg, jobs):
    K_ratios = np.linspace(cfg.K_min, cfg.K_max, cfg.n_k)

    low_mask = (
        (K_ratios >= cfg.low_kappa_min)
        &
        (K_ratios <= cfg.low_kappa_max)
    )

    high_mask = (
        (K_ratios >= cfg.high_kappa_min)
        &
        (K_ratios <= cfg.high_kappa_max)
    )

    if not np.any(low_mask):
        raise ValueError("Low-kappa window contains no K grid points.")

    if not np.any(high_mask):
        raise ValueError("High-kappa window contains no K grid points.")

    N_values = list(cfg.N_values)
    r_values = np.array(cfg.r_values, dtype=float)
    n_r = len(r_values)

    if not np.any(np.isclose(r_values, cfg.r_fixed_for_curve)):
        raise ValueError(
            f"r_fixed_for_curve={cfg.r_fixed_for_curve} is not in r_values."
        )

    r_to_index = {float(r): i for i, r in enumerate(r_values)}
    cfg_dict = asdict(cfg)

    D_low_trials = {
        N: np.zeros((cfg.trials, n_r))
        for N in N_values
    }

    D_high_trials = {
        N: np.zeros((cfg.trials, n_r))
        for N in N_values
    }

    Delta_trials = {
        N: np.zeros((cfg.trials, n_r))
        for N in N_values
    }

    D_curve_trials_fixed_r = {
        N: np.zeros((cfg.trials, cfg.n_k))
        for N in N_values
    }

    job_list = [
        (N, float(r_value), trial, K_ratios, low_mask, high_mask, cfg_dict)
        for N in N_values
        for r_value in r_values
        for trial in range(cfg.trials)
    ]

    def store_result(result):
        N = result["N"]
        r_value = result["r_value"]
        trial = result["trial"]
        ir = r_to_index[float(r_value)]

        D_low_trials[N][trial, ir] = result["D_low"]
        D_high_trials[N][trial, ir] = result["D_high"]
        Delta_trials[N][trial, ir] = result["Delta_D"]

        if np.isclose(r_value, cfg.r_fixed_for_curve):
            D_curve_trials_fixed_r[N][trial, :] = result["D_curve"]

    if jobs == 1:
        for result in map(simulate_one_N_r_trial, job_list):
            store_result(result)
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(simulate_one_N_r_trial, job_list, chunksize=1):
                store_result(result)

    D_curve_mean_fixed_r = {}
    D_curve_std_fixed_r = {}

    for N in N_values:
        D_curve_mean_fixed_r[N] = np.mean(D_curve_trials_fixed_r[N], axis=0)
        D_curve_std_fixed_r[N] = (
            np.std(D_curve_trials_fixed_r[N], axis=0, ddof=1)
            if cfg.trials > 1 else np.zeros(cfg.n_k)
        )

    D_low_mean = {}
    D_low_std = {}
    D_high_mean = {}
    D_high_std = {}
    Delta_mean = {}
    Delta_std = {}

    for N in N_values:
        D_low_mean[N] = np.mean(D_low_trials[N], axis=0)
        D_low_std[N] = (
            np.std(D_low_trials[N], axis=0, ddof=1)
            if cfg.trials > 1 else np.zeros(n_r)
        )

        D_high_mean[N] = np.mean(D_high_trials[N], axis=0)
        D_high_std[N] = (
            np.std(D_high_trials[N], axis=0, ddof=1)
            if cfg.trials > 1 else np.zeros(n_r)
        )

        Delta_mean[N] = np.mean(Delta_trials[N], axis=0)
        Delta_std[N] = (
            np.std(Delta_trials[N], axis=0, ddof=1)
            if cfg.trials > 1 else np.zeros(n_r)
        )

    return {
        "K_ratios": K_ratios,
        "r_values": r_values,
        "N_values": N_values,
        "D_curve_mean_fixed_r": D_curve_mean_fixed_r,
        "D_curve_std_fixed_r": D_curve_std_fixed_r,
        "D_low_mean": D_low_mean,
        "D_low_std": D_low_std,
        "D_high_mean": D_high_mean,
        "D_high_std": D_high_std,
        "Delta_mean": Delta_mean,
        "Delta_std": Delta_std,
    }


def save_figure_7_data(cfg, K_ratios, N_values, D_curve_mean_fixed_r, D_curve_std_fixed_r):
    fieldnames = [
        "N",
        "kappa",
        "r_fixed",
        "D_mean",
        "D_std",
        "D_sem",
        "trials",
        "omega",
        "T",
        "dt",
        "late_fraction",
    ]

    with open(DATA_7_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_values:
            for ik, kappa in enumerate(K_ratios):
                D_std = D_curve_std_fixed_r[N][ik]
                D_sem = D_std / np.sqrt(cfg.trials)

                writer.writerow({
                    "N": N,
                    "kappa": float(kappa),
                    "r_fixed": cfg.r_fixed_for_curve,
                    "D_mean": D_curve_mean_fixed_r[N][ik],
                    "D_std": D_std,
                    "D_sem": D_sem,
                    "trials": cfg.trials,
                    "omega": cfg.omega,
                    "T": cfg.T,
                    "dt": cfg.dt,
                    "late_fraction": cfg.late_fraction,
                })


def save_figure_8_data(
    cfg,
    r_values,
    N_values,
    D_low_mean,
    D_low_std,
    D_high_mean,
    D_high_std,
    Delta_mean,
    Delta_std,
):
    fieldnames = [
        "N",
        "r",
        "D_low_mean",
        "D_low_std",
        "D_low_sem",
        "D_high_mean",
        "D_high_std",
        "D_high_sem",
        "Delta_mean",
        "Delta_std",
        "Delta_sem",
        "trials",
        "omega",
        "T",
        "dt",
        "late_fraction",
        "K_min",
        "K_max",
        "n_k",
        "low_kappa_min",
        "low_kappa_max",
        "high_kappa_min",
        "high_kappa_max",
    ]

    with open(DATA_8_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for N in N_values:
            for ir, r_value in enumerate(r_values):
                D_low_std_val = D_low_std[N][ir]
                D_high_std_val = D_high_std[N][ir]
                Delta_std_val = Delta_std[N][ir]

                writer.writerow({
                    "N": N,
                    "r": float(r_value),
                    "D_low_mean": D_low_mean[N][ir],
                    "D_low_std": D_low_std_val,
                    "D_low_sem": D_low_std_val / np.sqrt(cfg.trials),
                    "D_high_mean": D_high_mean[N][ir],
                    "D_high_std": D_high_std_val,
                    "D_high_sem": D_high_std_val / np.sqrt(cfg.trials),
                    "Delta_mean": Delta_mean[N][ir],
                    "Delta_std": Delta_std_val,
                    "Delta_sem": Delta_std_val / np.sqrt(cfg.trials),
                    "trials": cfg.trials,
                    "omega": cfg.omega,
                    "T": cfg.T,
                    "dt": cfg.dt,
                    "late_fraction": cfg.late_fraction,
                    "K_min": cfg.K_min,
                    "K_max": cfg.K_max,
                    "n_k": cfg.n_k,
                    "low_kappa_min": cfg.low_kappa_min,
                    "low_kappa_max": cfg.low_kappa_max,
                    "high_kappa_min": cfg.high_kappa_min,
                    "high_kappa_max": cfg.high_kappa_max,
                })


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
        "legend.fontsize": 11,
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


def require_columns(df, required, filename):
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(
            f"{filename} is missing required columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )


def padded_limits(values, frac=0.08, min_pad=0.02):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -0.1, 1.0

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin

    if span <= 0:
        pad = max(min_pad, 0.05 * abs(ymax) if ymax != 0 else min_pad)
    else:
        pad = max(frac * span, min_pad)

    return ymin - pad, ymax + pad


def load_figure_7_data():
    if not DATA_7_PATH.exists():
        raise FileNotFoundError(f"{DATA_7_PATH} not found.")

    df = pd.read_csv(DATA_7_PATH)

    required = [
        "N",
        "kappa",
        "r_fixed",
        "D_mean",
        "D_sem",
    ]

    require_columns(df, required, DATA_7_PATH)

    return df.sort_values(["N", "kappa"]).reset_index(drop=True)


def load_figure_8_data():
    if not DATA_8_PATH.exists():
        raise FileNotFoundError(f"{DATA_8_PATH} not found.")

    df = pd.read_csv(DATA_8_PATH)

    required = [
        "N",
        "r",
        "D_low_mean",
        "D_low_sem",
        "D_high_mean",
        "D_high_sem",
        "Delta_mean",
        "Delta_sem",
    ]

    require_columns(df, required, DATA_8_PATH)

    return df.sort_values(["N", "r"]).reset_index(drop=True)


def make_figure_7(df7, df8):
    set_plot_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13.2, 5.8),
        constrained_layout=False,
    )
    fig.subplots_adjust(wspace=0.25)

    ax_left, ax_right = axes

    all_y_left = []

    for idx, N in enumerate(sorted(df7["N"].unique())):
        N_int = int(N)
        color = get_color_for_N(N_int, idx)
        sub = df7[df7["N"] == N].sort_values("kappa")

        x = sub["kappa"].to_numpy()
        y = sub["D_mean"].to_numpy()
        err = sub["D_sem"].to_numpy()

        all_y_left.extend((y - err).tolist())
        all_y_left.extend((y + err).tolist())

        ax_left.plot(
            x,
            y,
            color=color,
            lw=2.8,
            marker="o",
            markersize=4.2,
            markevery=max(1, len(x) // 12),
            label=rf"$N={N_int}$",
            zorder=3,
        )

        ax_left.fill_between(
            x,
            y - err,
            y + err,
            color=color,
            alpha=0.17,
            linewidth=0.0,
            zorder=2,
        )

    ax_left.set_xlabel(r"$\kappa$")
    ax_left.set_ylabel(r"$D_N(r=2,\kappa)$")

    x_min = df7["kappa"].min()
    x_max = df7["kappa"].max()
    x_pad = 0.015 * (x_max - x_min)
    ax_left.set_xlim(x_min - x_pad, x_max + x_pad)

    y_low, y_high = padded_limits(all_y_left, frac=0.06, min_pad=0.015)
    ax_left.set_ylim(max(0.0, y_low), y_high)

    style_axes(ax_left)

    ax_left.legend(
        loc="upper left",
        fontsize=13,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.1,
        borderpad=0.75,
        labelspacing=0.60,
        handletextpad=0.75,
    )

    all_y_right = []

    for idx, N in enumerate(sorted(df8["N"].unique())):
        N_int = int(N)
        color = get_color_for_N(N_int, idx)
        sub = df8[df8["N"] == N].sort_values("r")

        x = sub["r"].to_numpy()
        y = sub["Delta_mean"].to_numpy()
        err = sub["Delta_sem"].to_numpy()

        all_y_right.extend((y - err).tolist())
        all_y_right.extend((y + err).tolist())

        ax_right.plot(
            x,
            y,
            color=color,
            lw=2.8,
            marker="o",
            markersize=5.0,
            label=rf"$N={N_int}$",
            zorder=3,
        )

        ax_right.fill_between(
            x,
            y - err,
            y + err,
            color=color,
            alpha=0.17,
            linewidth=0.0,
            zorder=2,
        )

    ax_right.set_xlabel(r"$r$")
    ax_right.set_ylabel(r"$\Delta D_N(r)$")

    r_min = df8["r"].min()
    r_max = df8["r"].max()
    r_pad = 0.03 * (r_max - r_min)
    ax_right.set_xlim(r_min - r_pad, r_max + r_pad)

    y_low, y_high = padded_limits(all_y_right, frac=0.10, min_pad=0.035)
    ax_right.set_ylim(y_low, y_high)

    style_axes(ax_right)

    ax_right.legend(
        loc="upper right",
        fontsize=13,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.1,
        borderpad=0.75,
        labelspacing=0.60,
        handletextpad=0.75,
    )

    fig.savefig(FIGURE_7_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_7_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_figure_8_curve(ax, x, y, err, r_value, panel="low", zorder_override=None):
    color = BROWN_R_COLORS[r_value]
    marker = R_MARKERS[r_value]
    msize = R_MARKER_SIZES[r_value]

    alpha_fill = 0.12 if panel == "low" else 0.08

    if zorder_override is None:
        z_line = 5 + int(r_value)
    else:
        z_line = zorder_override

    ax.fill_between(
        x,
        y - err,
        y + err,
        color=color,
        alpha=alpha_fill,
        linewidth=0.0,
        zorder=max(1, z_line - 2),
    )

    ax.plot(
        x,
        y,
        color=color,
        lw=2.7,
        linestyle="solid",
        marker=marker,
        markersize=msize,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=MARKER_EDGE_WIDTH,
        zorder=z_line,
        label=rf"$r={int(r_value)}$",
    )


def make_figure_8(df8):
    set_plot_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.6, 5.8),
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(wspace=0.15)

    ax_low, ax_high = axes

    selected_r = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    N_values = sorted(df8["N"].unique())
    N_array = np.array(N_values, dtype=float)
    N_pad = 0.03 * (np.max(N_array) - np.min(N_array))

    all_y_low = []

    for r_value in selected_r:
        sub = df8[np.isclose(df8["r"], r_value)].sort_values("N")

        if len(sub) == 0:
            continue

        x = sub["N"].to_numpy()
        y = sub["D_low_mean"].to_numpy()
        err = sub["D_low_sem"].to_numpy()

        all_y_low.extend((y - err).tolist())
        all_y_low.extend((y + err).tolist())

        plot_figure_8_curve(ax_low, x, y, err, r_value, panel="low")

    ax_low.set_xlabel(r"$N$")
    ax_low.set_ylabel(r"$D_N$")
    ax_low.set_title(r"Weak coupling: $\kappa\in[0,0.5]$")

    ax_low.set_xlim(np.min(N_array) - N_pad, np.max(N_array) + N_pad)
    ax_low.set_xticks(N_array)

    y_low_min, y_low_max = padded_limits(all_y_low, frac=0.08, min_pad=0.03)
    ax_low.set_ylim(y_low_min, y_low_max)

    style_axes(ax_low)

    ax_low.legend(
        loc="upper left",
        bbox_to_anchor=(0.04, 0.84),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.2,
        borderpad=0.65,
        labelspacing=0.50,
        handletextpad=0.70,
    )

    all_y_high = []
    plot_order_high = [1.0, 2.0, 3.0, 4.0, 6.0, 5.0]

    for r_value in plot_order_high:
        sub = df8[np.isclose(df8["r"], r_value)].sort_values("N")

        if len(sub) == 0:
            continue

        x = sub["N"].to_numpy()
        y = sub["D_high_mean"].to_numpy()
        err = sub["D_high_sem"].to_numpy()

        all_y_high.extend((y - err).tolist())
        all_y_high.extend((y + err).tolist())

        if r_value == 6.0:
            z_override = 8
        elif r_value == 5.0:
            z_override = 12
        else:
            z_override = 9 + int(r_value)

        plot_figure_8_curve(
            ax_high,
            x,
            y,
            err,
            r_value,
            panel="high",
            zorder_override=z_override,
        )

    ax_high.set_xlabel(r"$N$")
    ax_high.set_title(r"Strong coupling: $\kappa\in[5.5,6]$")

    ax_high.set_xlim(np.min(N_array) - N_pad, np.max(N_array) + N_pad)
    ax_high.set_xticks(N_array)

    y_high_min, y_high_max = padded_limits(all_y_high, frac=0.08, min_pad=0.04)
    ax_high.set_ylim(y_high_min, y_high_max)

    style_axes(ax_high)

    fig.savefig(FIGURE_8_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_8_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg, jobs, plot_only = parse_args()

    if RUN_SIMULATION and not plot_only:
        results = run_combined_sweep(cfg, jobs)

        save_figure_7_data(
            cfg,
            results["K_ratios"],
            results["N_values"],
            results["D_curve_mean_fixed_r"],
            results["D_curve_std_fixed_r"],
        )

        save_figure_8_data(
            cfg,
            results["r_values"],
            results["N_values"],
            results["D_low_mean"],
            results["D_low_std"],
            results["D_high_mean"],
            results["D_high_std"],
            results["Delta_mean"],
            results["Delta_std"],
        )

    df7 = load_figure_7_data()
    df8 = load_figure_8_data()

    make_figure_7(df7, df8)
    make_figure_8(df8)


if __name__ == "__main__":
    mp.freeze_support()
    main()