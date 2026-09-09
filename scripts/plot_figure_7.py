#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "finite_size"
FIGURES_DIR = REPO_ROOT / "figures"

DRIFT_VS_KAPPA_PATH = DATA_DIR / "drift_vs_kappa.csv"
COUPLING_WINDOWS_PATH = DATA_DIR / "coupling_windows.csv"

FIGURE_7_PNG = FIGURES_DIR / "figure_7.png"
FIGURE_7_PDF = FIGURES_DIR / "figure_7.pdf"


FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
]


KAPPA_BLUE_SHADES = [
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#084594",
]


STRONG_COLOR = "#B45F2A"
WEAK_COLOR = "#2C7FB8"
REFERENCE_GREY = "#4A4A4A"


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
        "axes.titleweight": "normal",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
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


def padded_limits(values, frac=0.08, min_pad=0.02):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return -0.1, 1.0

    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin

    if span <= 0:
        pad = max(min_pad, 0.05 * abs(ymax) if ymax != 0 else min_pad)
    else:
        pad = max(frac * span, min_pad)

    return ymin - pad, ymax + pad


def require_columns(df, required, path):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def load_drift_vs_kappa_data():
    if not DRIFT_VS_KAPPA_PATH.exists():
        raise FileNotFoundError(f"Missing required saved data file: {DRIFT_VS_KAPPA_PATH}")

    df = pd.read_csv(DRIFT_VS_KAPPA_PATH)

    required = [
        "N",
        "kappa",
        "r_fixed",
        "D_mean",
        "D_sem",
    ]
    require_columns(df, required, DRIFT_VS_KAPPA_PATH)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df.sort_values(["N", "kappa"]).reset_index(drop=True)


def load_coupling_windows_data():
    if not COUPLING_WINDOWS_PATH.exists():
        raise FileNotFoundError(f"Missing required saved data file: {COUPLING_WINDOWS_PATH}")

    df = pd.read_csv(COUPLING_WINDOWS_PATH)

    required = [
        "N",
        "r",
        "D_low_mean",
        "D_low_sem",
        "D_high_mean",
        "D_high_sem",
        "Delta_mean",
        "Delta_sem",
        "low_kappa_min",
        "low_kappa_max",
        "high_kappa_min",
        "high_kappa_max",
    ]
    require_columns(df, required, COUPLING_WINDOWS_PATH)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df.sort_values(["N", "r"]).reset_index(drop=True)


def get_color_for_N(N, index):
    N_int = int(round(float(N)))
    return FINITE_SIZE_COLORS.get(
        N_int,
        FALLBACK_N_COLORS[index % len(FALLBACK_N_COLORS)]
    )


def linear_fit_extrapolation(invN, values):
    x = np.asarray(invN, dtype=float)
    y = np.asarray(values, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("Need at least two finite points for 1/N extrapolation.")

    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def build_Dinf_vs_kappa(df_drift):
    kappas = np.sort(df_drift["kappa"].unique())
    Dinf = []

    for kappa in kappas:
        sub = df_drift[np.isclose(df_drift["kappa"], kappa)].sort_values("N")
        N = sub["N"].to_numpy(dtype=float)
        D = sub["D_mean"].to_numpy(dtype=float)

        _, intercept = linear_fit_extrapolation(1.0 / N, D)
        Dinf.append(intercept)

    return np.asarray(kappas, dtype=float), np.asarray(Dinf, dtype=float)


def estimate_max_slope_location(kappa, Dvals, n_dense=4000):
    kappa = np.asarray(kappa, dtype=float)
    Dvals = np.asarray(Dvals, dtype=float)

    order = np.argsort(kappa)
    kappa = kappa[order]
    Dvals = Dvals[order]

    k_dense = np.linspace(kappa.min(), kappa.max(), n_dense)
    D_dense = np.interp(k_dense, kappa, Dvals)
    grad = np.gradient(D_dense, k_dense)

    return float(k_dense[np.argmax(grad)])


def nearest_saved_kappa_map(targets, available):
    available = np.sort(np.asarray(available, dtype=float))
    mapping = {}

    for target in targets:
        idx = int(np.argmin(np.abs(available - target)))
        mapping[float(target)] = float(available[idx])

    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError(
            "Two requested kappa cuts mapped to the same saved grid point."
        )

    return mapping


def add_top_N_axis(ax):
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())

    positions = np.array([1/200, 1/80, 1/40, 1/20], dtype=float)
    labels = ["200", "80", "40", "20"]

    xmin, xmax = ax.get_xlim()
    mask = (positions >= xmin) & (positions <= xmax)

    top.set_xticks(positions[mask])
    top.set_xticklabels(np.asarray(labels)[mask])
    top.set_xlabel(r"$N$", labelpad=4, fontsize=16)

    top.tick_params(
        axis="x",
        which="both",
        direction="out",
        top=True,
        bottom=False,
        colors="black",
        width=1.0,
        length=4.5,
        labelsize=12,
    )


    for spine in top.spines.values():
        spine.set_visible(False)

    return top


def plot_panel_a(ax, df_drift):
    r_fixed_values = np.sort(df_drift["r_fixed"].unique())
    if len(r_fixed_values) != 1:
        raise ValueError(
            f"Expected one fixed r in data/finite_size/drift_vs_kappa.csv; found {r_fixed_values}."
        )

    r_fixed = float(r_fixed_values[0])
    all_y = []


    N_handles = []

    for idx, N in enumerate(sorted(df_drift["N"].unique())):
        N_int = int(round(float(N)))
        color = get_color_for_N(N, idx)
        sub = df_drift[np.isclose(df_drift["N"], N)].sort_values("kappa")

        kappa = sub["kappa"].to_numpy(dtype=float)
        D = sub["D_mean"].to_numpy(dtype=float)
        Dsem = sub["D_sem"].to_numpy(dtype=float)

        all_y.extend((D - Dsem).tolist())
        all_y.extend((D + Dsem).tolist())

        line, = ax.plot(
            kappa,
            D,
            color=color,
            lw=2.8,
            marker="o",
            markersize=4.2,
            markevery=max(1, len(kappa) // 12),
            zorder=3,
            label=rf"$N={N_int}$",
        )

        ax.fill_between(
            kappa,
            D - Dsem,
            D + Dsem,
            color=color,
            alpha=0.17,
            linewidth=0.0,
            zorder=2,
        )

        N_handles.append(line)


    kappas, Dinf = build_Dinf_vs_kappa(df_drift)
    all_y.extend(Dinf.tolist())

    inf_line, = ax.plot(
        kappas,
        Dinf,
        color="black",
        lw=2.7,
        linestyle="-",
        zorder=5,
        label=r"$N\to\infty$",
    )


    kappa_ref = 0.5 * r_fixed**2
    kappa_max_slope = estimate_max_slope_location(kappas, Dinf)

    ref_line = ax.axvline(
        kappa_ref,
        color=REFERENCE_GREY,
        lw=1.8,
        linestyle="-",
        zorder=1,
        label=rf"$\kappa=r^2/2={kappa_ref:.2f}$",
    )

    slope_line = ax.axvline(
        kappa_max_slope,
        color=REFERENCE_GREY,
        lw=1.8,
        linestyle="--",
        zorder=1,
        label=rf"max slope, $\kappa={kappa_max_slope:.2f}$",
    )

    ax.set_title(
        r"(a) Mean absolute late-time drift",
        loc="center",
        pad=10,
    )
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$D_N(r=2,\kappa)$")

    x_min = float(df_drift["kappa"].min())
    x_max = float(df_drift["kappa"].max())
    x_pad = 0.015 * (x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    y_low, y_high = padded_limits(all_y, frac=0.055, min_pad=0.015)
    ax.set_ylim(max(0.0, y_low), y_high)

    style_axes(ax)


    leg_N = ax.legend(
        handles=N_handles,
        loc="upper left",
        fontsize=12.2,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.0,
        borderpad=0.65,
        labelspacing=0.50,
        handletextpad=0.70,
    )
    ax.add_artist(leg_N)



    ax.legend(
        handles=[inf_line, ref_line, slope_line],
        loc="lower right",
        fontsize=10.8,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.2,
        borderpad=0.60,
        labelspacing=0.45,
        handletextpad=0.65,
    )


def plot_panel_b(ax, df_windows):
    all_y = []

    for idx, N in enumerate(sorted(df_windows["N"].unique())):
        N_int = int(round(float(N)))
        color = get_color_for_N(N, idx)
        sub = df_windows[np.isclose(df_windows["N"], N)].sort_values("r")

        r = sub["r"].to_numpy(dtype=float)
        delta = sub["Delta_mean"].to_numpy(dtype=float)
        delta_sem = sub["Delta_sem"].to_numpy(dtype=float)

        all_y.extend((delta - delta_sem).tolist())
        all_y.extend((delta + delta_sem).tolist())

        ax.plot(
            r,
            delta,
            color=color,
            lw=2.8,
            marker="o",
            markersize=5.0,
            label=rf"$N={N_int}$",
            zorder=3,
        )

        ax.fill_between(
            r,
            delta - delta_sem,
            delta + delta_sem,
            color=color,
            alpha=0.17,
            linewidth=0.0,
            zorder=2,
        )

    ax.set_title(
        r"(b) Drift recovery",
        loc="center",
        pad=10,
    )
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\Delta D_N(r)$")

    r_min = float(df_windows["r"].min())
    r_max = float(df_windows["r"].max())
    r_pad = 0.03 * (r_max - r_min)
    ax.set_xlim(r_min - r_pad, r_max + r_pad)

    y_low, y_high = padded_limits(all_y, frac=0.10, min_pad=0.035)
    ax.set_ylim(y_low, y_high)

    style_axes(ax)

    ax.legend(
        loc="upper right",
        fontsize=12.2,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        handlelength=2.0,
        borderpad=0.65,
        labelspacing=0.50,
        handletextpad=0.70,
    )


def plot_panel_c(ax, df_drift):
    r_fixed_values = np.sort(df_drift["r_fixed"].unique())
    if len(r_fixed_values) != 1:
        raise ValueError(
            f"Expected one fixed r in data/finite_size/drift_vs_kappa.csv; found {r_fixed_values}."
        )

    r_fixed = float(r_fixed_values[0])

    target_kappas = [0.0, 1.2, 2.4, 3.6, 6.0]
    kappa_map = nearest_saved_kappa_map(
        target_kappas,
        df_drift["kappa"].unique(),
    )

    legend_handles = []
    all_x = []
    all_y = []

    for color, nominal_kappa in zip(KAPPA_BLUE_SHADES, target_kappas):
        saved_kappa = kappa_map[nominal_kappa]

        sub = df_drift[np.isclose(df_drift["kappa"], saved_kappa)].sort_values("N")

        N = sub["N"].to_numpy(dtype=float)
        invN = 1.0 / N
        D = sub["D_mean"].to_numpy(dtype=float)
        Dsem = sub["D_sem"].to_numpy(dtype=float)

        slope, intercept = linear_fit_extrapolation(invN, D)

        xfit = np.linspace(0.0, max(invN) * 1.03, 200)
        yfit = slope * xfit + intercept

        ax.plot(
            xfit,
            yfit,
            color=color,
            lw=2.4,
            zorder=2,
        )

        ax.errorbar(
            invN,
            D,
            yerr=Dsem,
            fmt="o",
            ms=8.3,
            mfc="white",
            mec=color,
            mew=1.9,
            ecolor=color,
            elinewidth=1.4,
            capsize=3.2,
            linestyle="none",
            zorder=3,
        )

        ax.plot(
            0.0,
            intercept,
            marker="*",
            markersize=18,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=4,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.9,
                markersize=7.7,
                label=rf"$\kappa={nominal_kappa:.1f}$",
            )
        )

        all_x.extend(invN.tolist())
        all_y.extend((D - Dsem).tolist())
        all_y.extend((D + Dsem).tolist())
        all_y.append(intercept)

    ax.set_title(
        r"(c) Finite-size extrapolation",
        loc="center",
        y=1.20,
        pad=0,
    )
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$D_N(r=2,\kappa)$")

    xmax_data = max(all_x)



    xmax = xmax_data + 0.0115
    ax.set_xlim(-0.0025, xmax)
    ax.set_xticks(np.arange(0.0, 0.051, 0.01))

    y_low, y_high = padded_limits(all_y, frac=0.06, min_pad=0.025)
    ax.set_ylim(max(0.20, y_low), y_high)

    style_axes(ax)
    add_top_N_axis(ax)




    label_y_x = xmax_data + 0.0012

    for color, nominal_kappa in zip(KAPPA_BLUE_SHADES, target_kappas):
        saved_kappa = kappa_map[nominal_kappa]
        sub = df_drift[np.isclose(df_drift["kappa"], saved_kappa)].sort_values("N")
        N = sub["N"].to_numpy(dtype=float)
        D = sub["D_mean"].to_numpy(dtype=float)
        slope, intercept = linear_fit_extrapolation(1.0 / N, D)
        label_y = slope * label_y_x + intercept

        ax.text(
            0.965,
            label_y,
            rf"$\kappa={nominal_kappa:.1f}$",
            transform=ax.get_yaxis_transform(),
            color=color,
            fontsize=13.0,
            ha="right",
            va="center",
            clip_on=True,
        )


def plot_panel_d(ax, df_windows):
    r_vals = np.sort(df_windows["r"].unique())

    Dinf_low = []
    Dinf_high = []

    for r in r_vals:
        sub = df_windows[np.isclose(df_windows["r"], r)].sort_values("N")

        N = sub["N"].to_numpy(dtype=float)
        invN = 1.0 / N

        Dlow = sub["D_low_mean"].to_numpy(dtype=float)
        Dhigh = sub["D_high_mean"].to_numpy(dtype=float)

        _, low_intercept = linear_fit_extrapolation(invN, Dlow)
        _, high_intercept = linear_fit_extrapolation(invN, Dhigh)

        Dinf_low.append(low_intercept)
        Dinf_high.append(high_intercept)

    Dinf_low = np.asarray(Dinf_low, dtype=float)
    Dinf_high = np.asarray(Dinf_high, dtype=float)

    low_range = df_windows[
        ["low_kappa_min", "low_kappa_max"]
    ].iloc[0].to_numpy(dtype=float)

    high_range = df_windows[
        ["high_kappa_min", "high_kappa_max"]
    ].iloc[0].to_numpy(dtype=float)

    ax.plot(
        r_vals,
        Dinf_high,
        color=STRONG_COLOR,
        lw=2.7,
        marker="o",
        markersize=8.0,
        markerfacecolor="white",
        markeredgecolor=STRONG_COLOR,
        markeredgewidth=1.8,
        label=rf"$\kappa\in[{high_range[0]:g},{high_range[1]:g}]$",
        zorder=3,
    )

    ax.plot(
        r_vals,
        Dinf_low,
        color=WEAK_COLOR,
        lw=2.7,
        marker="s",
        markersize=7.5,
        markerfacecolor="white",
        markeredgecolor=WEAK_COLOR,
        markeredgewidth=1.8,
        label=rf"$\kappa\in[{low_range[0]:g},{low_range[1]:g}]$",
        zorder=3,
    )

    ax.axvline(
        np.sqrt(np.pi / 2.0),
        color="black",
        lw=1.8,
        linestyle="--",
        label=r"$r=\sqrt{\pi/2}$",
        zorder=1,
    )

    ax.axhline(
        0.0,
        color="0.65",
        lw=1.0,
        zorder=1,
    )

    ax.set_title(
        r"(d) Thermodynamic-limit drift",
        loc="center",
        pad=10,
    )
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$D_{\infty}$", labelpad=7)

    ax.set_xlim(r_vals.min() - 0.15, r_vals.max() + 0.15)

    all_y = np.concatenate([Dinf_low, Dinf_high, [0.0]])
    y_low, y_high = padded_limits(all_y, frac=0.08, min_pad=0.05)
    ax.set_ylim(min(-0.05, y_low), y_high)

    style_axes(ax)

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.82",
        fontsize=11.5,
        borderpad=0.65,
        labelspacing=0.50,
        handlelength=2.2,
        handletextpad=0.65,
    )


def make_figure(df_drift, df_windows):
    set_plot_style()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.8, 10.7),
        constrained_layout=False,
    )



    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.965,
        wspace=0.22,
        hspace=0.46,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    plot_panel_a(ax_a, df_drift)
    plot_panel_b(ax_b, df_windows)
    plot_panel_c(ax_c, df_drift)
    plot_panel_d(ax_d, df_windows)





    tmp_png = FIGURES_DIR / "figure_7.__tmp__.png"
    tmp_pdf = FIGURES_DIR / "figure_7.__tmp__.pdf"

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
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Figure save failed: {tmp}")


    if FIGURE_7_PNG.exists():
        FIGURE_7_PNG.unlink()
    if FIGURE_7_PDF.exists():
        FIGURE_7_PDF.unlink()

    tmp_png.replace(FIGURE_7_PNG)
    tmp_pdf.replace(FIGURE_7_PDF)

    if not FIGURE_7_PNG.exists() or FIGURE_7_PNG.stat().st_size == 0:
        raise RuntimeError(f"PNG overwrite failed: {FIGURE_7_PNG}")

    if not FIGURE_7_PDF.exists() or FIGURE_7_PDF.stat().st_size == 0:
        raise RuntimeError(f"PDF overwrite failed: {FIGURE_7_PDF}")



    print("\nFINAL FIGURE 7 SAVED AND VERIFIED:")
    print(
        f"  PNG: {FIGURE_7_PNG.resolve()} "
        f"({FIGURE_7_PNG.stat().st_size / 1024:.1f} kB)"
    )
    print(
        f"  PDF: {FIGURE_7_PDF.resolve()} "
        f"({FIGURE_7_PDF.stat().st_size / 1024:.1f} kB)"
    )
    print("\nInputs used:")
    print(" ", DRIFT_VS_KAPPA_PATH.resolve())
    print(" ", COUPLING_WINDOWS_PATH.resolve())
    print("\n")


def main():
    print("Repository root:", REPO_ROOT)
    print("Reading saved finite-size data:")
    print(" ", DRIFT_VS_KAPPA_PATH)
    print(" ", COUPLING_WINDOWS_PATH)

    df_drift = load_drift_vs_kappa_data()
    df_windows = load_coupling_windows_data()

    make_figure(df_drift, df_windows)


if __name__ == "__main__":
    main()
