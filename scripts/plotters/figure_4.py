#!/usr/bin/env python3

from pathlib import Path
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "n100" / "order_parameter_cuts.csv"
FIGURE_PNG = FIGURES_DIR / "figure_4.png"
FIGURE_PDF = FIGURES_DIR / "figure_4.pdf"


N_DEFAULT = 100
OMEGA_DEFAULT = 0.5

T_DEFAULT = 1000.0
DT_DEFAULT = 0.01
LATE_FRACTION_DEFAULT = 0.30
N_LATE_SAMPLES_DEFAULT = 60

KAPPA_MIN_DEFAULT = 0.0
KAPPA_MAX_DEFAULT = 6.0
N_KAPPA_DEFAULT = 61

R_CUTS_DEFAULT = (1.0, 2.0, 4.0, 6.0)
R_KAPPA0_DEFAULT = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

TRIALS_DEFAULT = 50
SEED_DEFAULT = 55500


R_COLORS = {
    1.0: "#D7B98E",
    2.0: "#BE9362",
    4.0: "#7F5735",
    6.0: "#3F2818",
}

R_MARKERS = {
    1.0: "D",
    2.0: "^",
    4.0: "X",
    6.0: "o",
}

R_MARKER_SIZES = {
    1.0: 7.4,
    2.0: 7.4,
    4.0: 8.3,
    6.0: 9.6,
}

MARKER_EDGE_WIDTH = 1.45


def trapz_manual(y, x):
    return float(
        np.sum(
            0.5 * (y[1:] + y[:-1]) * np.diff(x)
        )
    )


def gaussian_density(a, r):
    return (
        1.0
        / (np.sqrt(2.0 * np.pi) * r)
        * np.exp(-a**2 / (2.0 * r**2))
    )


def analytical_D0(r_values, n_quad=20001):
    r_values = np.asarray(r_values, dtype=float)

    a = np.linspace(-1.0, 1.0, n_quad)

    base = np.sqrt(
        np.maximum(
            1.0 - a**2,
            0.0,
        )
    )

    out = np.zeros_like(
        r_values,
        dtype=float,
    )

    for i, r in enumerate(r_values):
        if np.isclose(r, 0.0):
            out[i] = 1.0
        else:
            out[i] = trapz_manual(
                base * gaussian_density(a, r),
                a,
            )

    return out


def drifting_cos2_kernel(a):
    a = np.asarray(a, dtype=float)

    small = np.abs(a) < 1.0e-4
    safe = np.where(
        small,
        1.0,
        a,
    )

    full = (
        1.0
        - (2.0 / safe**2)
        * (
            1.0
            - np.sqrt(
                np.maximum(
                    1.0 - safe**2,
                    0.0,
                )
            )
        )
    )

    series = (
        -a**2 / 4.0
        - a**4 / 8.0
    )

    return np.where(
        small,
        series,
        full,
    )


def analytical_R2_kappa0(
    r_values,
    n_quad=20001,
    tail_factor=40.0,
):
    r_values = np.asarray(
        r_values,
        dtype=float,
    )

    a_drift = np.linspace(
        -1.0,
        1.0,
        n_quad,
    )

    cos_drift = drifting_cos2_kernel(
        a_drift
    )

    out = np.zeros_like(
        r_values,
        dtype=float,
    )

    for i, r in enumerate(r_values):
        if np.isclose(r, 0.0):
            out[i] = 0.0
            continue

        g_drift = gaussian_density(
            a_drift,
            r,
        )

        cos_part = trapz_manual(
            cos_drift * g_drift,
            a_drift,
        )

        a_max = 1.0 + tail_factor * r

        a_pin = np.linspace(
            1.0,
            a_max,
            n_quad,
        )

        g_pin = gaussian_density(
            a_pin,
            r,
        )

        cos_pin = trapz_manual(
            (
                1.0
                - 2.0 / a_pin**2
            )
            * g_pin,
            a_pin,
        )

        sin_pin = trapz_manual(
            (
                2.0
                * np.sqrt(
                    np.maximum(
                        1.0
                        - 1.0 / a_pin**2,
                        0.0,
                    )
                )
                / a_pin
            )
            * g_pin,
            a_pin,
        )

        real_part = (
            cos_part
            + 2.0 * cos_pin
        )

        imag_part = (
            2.0 * sin_pin
        )

        out[i] = float(
            np.hypot(
                real_part,
                imag_part,
            )
        )

    return out


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 12.5,
            "axes.labelsize": 16.5,
            "axes.titlesize": 15.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax):
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.15)

    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        top=False,
        right=False,
        colors="black",
        width=1.05,
        length=4.5,
    )


def legend_style(**overrides):
    kwargs = dict(
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.82",
        fancybox=False,
        handlelength=1.8,
        borderpad=0.45,
        labelspacing=0.36,
        handletextpad=0.50,
        columnspacing=0.75,
    )

    kwargs.update(
        overrides
    )

    return kwargs


def validate_data(df):
    required = {
        "dataset",
        "r",
        "kappa",
        "R1",
        "R1_sem",
        "R2",
        "R2_sem",
        "D",
        "D_sem",
    }

    if not required.issubset(
        set(df.columns)
    ):
        return False

    datasets = set(
        df["dataset"]
        .astype(str)
        .unique()
    )

    return {
        "cuts_numerics",
        "kappa0_numerics",
        "kappa0_analytics",
    }.issubset(datasets)


def make_figure_4():
    set_plot_style()

    if not DATA_PATH.exists():
        sys.exit(
            "\nNo data/n100/order_parameter_cuts.csv exists yet.\n"
            "Generate the order-parameter dataset first.\n"
        )

    data = pd.read_csv(
        DATA_PATH
    )

    if not validate_data(
        data
    ):
        sys.exit(
            "\ndata/n100/order_parameter_cuts.csv is still the dataset.\n"
            "Generate the order-parameter dataset first.\n"
        )

    cuts = (
        data[
            data["dataset"]
            == "cuts_numerics"
        ]
        .copy()
    )

    num = (
        data[
            data["dataset"]
            == "kappa0_numerics"
        ]
        .sort_values("r")
        .copy()
    )

    ana = (
        data[
            data["dataset"]
            == "kappa0_analytics"
        ]
        .sort_values("r")
        .copy()
    )

    r_values = sorted(
        cuts["r"].unique()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 10.5),
        constrained_layout=False,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.965,
        wspace=0.235,
        hspace=0.285,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]





    h_R2a, = ax_a.plot(
        ana["r"],
        ana["R2"],
        color="black",
        lw=3.0,
        label=r"$R_2^{0}(r)$ analytics",
        zorder=3,
    )

    h_D0a, = ax_a.plot(
        ana["r"],
        ana["D"],
        color="0.48",
        lw=2.45,
        linestyle="--",
        label=r"$D_0(r)$ analytics",
        zorder=2,
    )

    h_R2n, = ax_a.plot(
        num["r"],
        num["R2"],
        linestyle="none",
        marker="s",
        markersize=12.4,
        markerfacecolor="0.22",
        markeredgecolor="0.22",
        markeredgewidth=0.85,
        label=r"$R_2^{0}$ numerics",
        zorder=6,
    )

    h_R1n, = ax_a.plot(
        num["r"],
        num["R1"],
        linestyle="none",
        marker="o",
        markersize=10.9,
        markerfacecolor="0.45",
        markeredgecolor="0.45",
        markeredgewidth=1.20,
        label=r"$R_1^{0}$ numerics",
        zorder=7,
    )

    h_D0n, = ax_a.plot(
        num["r"],
        num["D"],
        linestyle="none",
        marker="^",
        markersize=10.9,
        markerfacecolor="0.48",
        markeredgecolor="0.48",
        markeredgewidth=1.20,
        label=r"$D_0$ numerics",
        zorder=8,
    )

    ax_a.set_xlabel(
        r"$r$"
    )

    ax_a.set_ylabel(
        r"$R_1^{0},\ R_2^{0},\ D_0$"
    )

    ax_a.set_title(
        r"(a) Uncoupled limit, $\kappa=0$",
        loc="center",
        pad=10,
    )

    ax_a.set_xlim(
        -0.16,
        6.16,
    )

    ax_a.set_ylim(
        -0.05,
        1.27,
    )

    ax_a.set_yticks(
        np.arange(
            0.0,
            1.01,
            0.2,
        )
    )

    style_axes(
        ax_a
    )










    blank = Line2D(
        [],
        [],
        linestyle="none",
        marker=None,
        alpha=0.0,
        label="",
    )

    ax_a.legend(
        handles=[
            h_R2a,
            h_D0a,
            blank,
            h_R2n,
            h_R1n,
            h_D0n,
        ],
        labels=[
            r"$R_2^{0}(r)$ analytics",
            r"$D_0(r)$ analytics",
            "",
            r"$R_2^{0}$ numerics",
            r"$R_1^{0}$ numerics",
            r"$D_0$ numerics",
        ],
        loc="upper center",
        ncol=2,
        fontsize=10.9,
        bbox_to_anchor=(
            0.52,
            0.985,
        ),
        **legend_style(
            handlelength=2.00,
            columnspacing=1.15,
            handletextpad=0.58,
            borderpad=0.54,
            labelspacing=0.42,
        ),
    )





    for idx, r_value in enumerate(
        r_values
    ):
        color = R_COLORS.get(
            float(r_value),
            "#666666",
        )

        marker = R_MARKERS.get(
            float(r_value),
            "o",
        )

        msize = R_MARKER_SIZES.get(
            float(r_value),
            7.5,
        )

        sub = (
            cuts[
                np.isclose(
                    cuts["r"],
                    r_value,
                )
            ]
            .sort_values("kappa")
        )

        x = sub[
            "kappa"
        ].to_numpy()

        markevery = max(
            1,
            len(x) // 8,
        )

        for ax, key in (
            (ax_b, "R1"),
            (ax_c, "R2"),
            (ax_d, "D"),
        ):
            y = sub[
                key
            ].to_numpy()

            err = sub[
                f"{key}_sem"
            ].to_numpy()

            ax.fill_between(
                x,
                y - err,
                y + err,
                color=color,
                alpha=0.14,
                linewidth=0.0,
                zorder=2,
            )

            ax.plot(
                x,
                y,
                color=color,
                lw=2.55,
                marker=marker,
                markersize=msize,
                markevery=markevery,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=rf"$r={r_value:g}$",
                zorder=5 + idx,
            )

    panel_info = (
        (
            ax_b,
            r"$R_1$",
            r"(b) Polar (Kuramoto) order",
        ),
        (
            ax_c,
            r"$R_2$",
            r"(c) Nematic order",
        ),
        (
            ax_d,
            r"$D$",
            r"(d) Mean absolute late-time drift",
        ),
    )

    for ax, ylabel, title in panel_info:
        ax.set_xlabel(
            r"$\kappa$"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            title,
            loc="center",
            pad=10,
        )

        ax.set_xlim(
            -0.14,
            6.14,
        )


        ax.set_ylim(
            -0.04,
            1.28,
        )

        ax.set_yticks(
            np.arange(
                0.0,
                1.01,
                0.2,
            )
        )

        style_axes(
            ax
        )

        ax.legend(
            loc="upper center",
            ncol=4,
            fontsize=10.8,
            bbox_to_anchor=(
                0.50,
                0.985,
            ),
            **legend_style(
                columnspacing=1.00,
                handlelength=1.58,
                handletextpad=0.46,
                borderpad=0.42,
                labelspacing=0.38,
            ),
        )

    fig.savefig(
        FIGURE_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        FIGURE_PDF,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved:"
    )
    print(
        " ",
        FIGURE_PNG.relative_to(
            REPO_ROOT
        )
    )
    print(
        " ",
        FIGURE_PDF.relative_to(
            REPO_ROOT
        )
    )


def main():
    make_figure_4()


if __name__ == "__main__":
    mp.freeze_support()
    main()
