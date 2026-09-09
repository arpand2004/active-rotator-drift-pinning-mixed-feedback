#!/usr/bin/env python3

from pathlib import Path
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_ARCHIVE_PATH = DATA_DIR / "n200" / "representative_dynamics.npz"
SUMMARY_PATH = DATA_DIR / "n200" / "representative_dynamics_summary.csv"

FIGURE_PNG = FIGURES_DIR / "figure_6.png"
FIGURE_PDF = FIGURES_DIR / "figure_6.pdf"


PANEL_POINTS = (
    (0.5, 6.0, "Coherent drift"),
    (5.0, 0.0, "Uncoupled pinning"),
    (4.0, 2.7, "Backward slips"),
    (5.0, 6.0, "Coherent pinning"),
)


@dataclass(frozen=True)
class Config:
    N: int = 200
    omega: float = 0.5

    T: float = 1000.0
    dt: float = 0.01
    late_fraction: float = 0.30

    T_plot: float = 200.0
    traj_stride: int = 20
    trace_stride: int = 20
    hist_stride: int = 100

    n_traj_plot: int = 60
    n_bins: int = 36

    drift_tol_fraction: float = 1e-3



    trials: int = 250

    seed: int = 55500


TRAJ_CMAP = plt.cm.viridis
R1_COLOR = "#B22222"
R2_COLOR = "#4C78A8"
HIST_COLOR = "#7F5735"
SHIFT_COLOR = "#B22222"


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
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


def plot_wrapped_trajectory(ax, tvals, yvals, color, lw=0.9, alpha=0.85):
    y = yvals.astype(float).copy()
    for idx in range(1, len(y)):
        if abs(y[idx] - y[idx - 1]) > np.pi:
            y[idx] = np.nan
    ax.plot(tvals, y, color=color, lw=lw, alpha=alpha)


def make_figure(cfg):
    if not DATA_ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_ARCHIVE_PATH} not found. Generate the representative-dynamics data first."
        )
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Generate the representative-dynamics data first."
        )

    set_plot_style()

    payload = np.load(DATA_ARCHIVE_PATH, allow_pickle=True)
    summary = pd.read_csv(SUMMARY_PATH)

    n_points = len(PANEL_POINTS)

    fig, axes = plt.subplots(3, n_points, figsize=(17.4, 10.4), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.075, top=0.905,
                        wspace=0.22, hspace=0.42)

    pi_ticks = [0.0, np.pi / 2, np.pi, 3.0 * np.pi / 2, 2.0 * np.pi]
    pi_labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    for ip, (r_value, kappa_value, label) in enumerate(PANEL_POINTS):
        traj = payload[f"traj_{ip}"]
        trace = payload[f"trace_{ip}"]
        row = summary.iloc[ip]

        ax_traj = axes[0, ip]
        ax_order = axes[1, ip]
        ax_hist = axes[2, ip]

        t_traj = np.arange(traj.shape[0]) * cfg.traj_stride * cfg.dt
        n_show = min(cfg.n_traj_plot, traj.shape[1])
        indices = np.linspace(0, traj.shape[1] - 1, n_show).astype(int)
        colors = TRAJ_CMAP(np.linspace(0.12, 0.92, n_show))

        for k, i in enumerate(indices):
            plot_wrapped_trajectory(ax_traj, t_traj, traj[:, i], colors[k], lw=0.9, alpha=0.85)

        ax_traj.set_xlim(0.0, cfg.T_plot)
        ax_traj.set_ylim(0.0, 2.0 * np.pi)
        ax_traj.set_yticks(pi_ticks)
        ax_traj.set_xlabel(r"$t$")
        ax_traj.set_title(rf"$r={r_value:g},\ \kappa={kappa_value:g}$" + "\n" + label, pad=8)
        if ip == 0:
            ax_traj.set_yticklabels(pi_labels)
            ax_traj.set_ylabel(r"$\theta_i\ (\mathrm{mod}\ 2\pi)$")
        else:
            ax_traj.set_yticklabels([])
        style_axes(ax_traj)

        ax_order.plot(trace[:, 0], trace[:, 1], color=R1_COLOR, lw=1.6,
                      label=r"$R_1$", zorder=4)
        ax_order.plot(trace[:, 0], trace[:, 2], color=R2_COLOR, lw=1.6,
                      label=r"$R_2$", zorder=3)

        ax_order.set_xlim(0.0, cfg.T_plot)
        ax_order.set_ylim(0.0, 1.42)
        ax_order.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax_order.set_xlabel(r"$t$")
        if ip == 0:
            ax_order.set_ylabel(r"$R_1,\ R_2$")
        else:
            ax_order.set_yticklabels([])
        style_axes(ax_order)


        ax_order.text(
            0.5, 0.94,
            rf"$\delta R_1 = {row['R1_std_mean']:.3f}$"
            rf"$,\quad f_- = {row['f_minus_mean']:.2f}$",
            transform=ax_order.transAxes,
            ha="center", va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="0.82",
                      boxstyle="round,pad=0.38", alpha=0.94),
            zorder=10,
        )

        order_legend_kwargs = dict(
            loc="lower left",
            ncol=2,
            frameon=True,
            framealpha=0.94,
            facecolor="white",
            edgecolor="0.82",
            fontsize=11,
            handlelength=1.6,
            borderpad=0.5,
            columnspacing=1.0,
            handletextpad=0.55,
        )

        if ip == 1:
            order_legend_kwargs["bbox_to_anchor"] = (0.03, 0.08)
        elif ip == 2:


            order_legend_kwargs["bbox_to_anchor"] = (0.08, 0.00)
        elif ip == 3:


            order_legend_kwargs["bbox_to_anchor"] = (0.13, 0.00)

        ax_order.legend(**order_legend_kwargs)

        edges = payload["hist_edges"]
        counts = payload["hist_counts"][ip]
        density = counts / (counts.sum() * (edges[1] - edges[0]))
        centers = 0.5 * (edges[1:] + edges[:-1])
        shifted = np.roll(density, cfg.n_bins // 2)

        ax_hist.bar(
            centers, density, width=edges[1] - edges[0],
            color=HIST_COLOR, edgecolor="white", linewidth=0.6, zorder=3
        )
        ax_hist.step(
            centers, shifted, where="mid", color=SHIFT_COLOR,
            lw=2.0, linestyle="--", zorder=5,
            label=r"shifted by $\pi$"
        )

        ax_hist.set_xlim(0.0, 2.0 * np.pi)
        ax_hist.set_xticks(pi_ticks)
        ax_hist.set_xticklabels(pi_labels)
        ax_hist.set_xlabel(r"$\theta_i\ (\mathrm{mod}\ 2\pi)$")
        ymax = 1.42 * max(float(np.max(density)), float(np.max(shifted)))
        ax_hist.set_ylim(0.0, ymax if ymax > 0 else 1.0)
        if ip == 0:
            ax_hist.set_ylabel(r"$p(\theta)$")
        style_axes(ax_hist)

        ax_hist.legend(
            loc="upper left", frameon=True, framealpha=0.94,
            facecolor="white", edgecolor="0.82", fontsize=11,
            handlelength=1.8, borderpad=0.5, handletextpad=0.55,
        )




    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tmp_png = FIGURES_DIR / "figure_6.__tmp__.png"
    tmp_pdf = FIGURES_DIR / "figure_6.__tmp__.pdf"


    for tmp in (tmp_png, tmp_pdf):
        if tmp.exists():
            tmp.unlink()

    fig.savefig(
        tmp_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        tmp_pdf,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)



    for tmp in (tmp_png, tmp_pdf):
        if not tmp.exists():
            raise RuntimeError(f"Figure save failed: {tmp} was not created.")
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"Figure save failed: {tmp} is empty.")


    tmp_png.replace(FIGURE_PNG)
    tmp_pdf.replace(FIGURE_PDF)


    if not FIGURE_PNG.exists() or FIGURE_PNG.stat().st_size == 0:
        raise RuntimeError(f"PNG save failed: {FIGURE_PNG}")
    if not FIGURE_PDF.exists() or FIGURE_PDF.stat().st_size == 0:
        raise RuntimeError(f"PDF save failed: {FIGURE_PDF}")

    print("\nFIGURE 6 SAVED SUCCESSFULLY:")
    print(f"  PNG: {FIGURE_PNG.resolve()}  ({FIGURE_PNG.stat().st_size / 1024:.1f} kB)")
    print(f"  PDF: {FIGURE_PDF.resolve()}  ({FIGURE_PDF.stat().st_size / 1024:.1f} kB)")


def main():
    cfg = Config()
    make_figure(cfg)


if __name__ == "__main__":
    mp.freeze_support()
    main()
