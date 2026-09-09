#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_PNG = FIGURES_DIR / "figure_5.png"
FIGURE_PDF = FIGURES_DIR / "figure_5.pdf"


def set_plot_style():
    plt.rcParams.update({
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
        "axes.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax):
    ax.grid(False)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 7, 1))

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


def truncated_cmap(name, minval, maxval, n=256):
    base = plt.get_cmap(name)
    vals = base(np.linspace(minval, maxval, n))
    return colors.LinearSegmentedColormap.from_list(
        f"{name}_soft",
        vals,
    )


REQUIRED_COLUMNS = {
    "sigma_A_over_omega",
    "K_over_omega",
    "positive_fraction",
    "negative_fraction",
}


def find_fraction_csv():

    preferred = [
        DATA_DIR / "n100" / "drift_fractions.csv",
    ]

    candidates = preferred + [
        p for p in sorted(DATA_DIR.rglob("*.csv"))
        if p not in preferred
    ]

    checked = []

    for path in candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        checked.append((path, list(df.columns)))

        if REQUIRED_COLUMNS.issubset(df.columns):
            print("\nUsing drift-fraction data:")
            print(" ", path.relative_to(REPO_ROOT))
            return path, df

    print("\nNo saved CSV containing the positive/negative drift fractions was found.")
    print("\nCSV files checked:")
    for path, cols in checked:
        print(f"  {path.relative_to(REPO_ROOT)}")
        print(f"    columns = {cols}")

    sys.exit(
        "\nThe required columns are:\n"
        "  sigma_A_over_omega, K_over_omega, "
        "positive_fraction, negative_fraction\n"
    )


def grid_from_dataframe(df, column):
    r_vals = np.sort(
        pd.to_numeric(
            df["sigma_A_over_omega"],
            errors="raise",
        ).unique()
    )

    kappa_vals = np.sort(
        pd.to_numeric(
            df["K_over_omega"],
            errors="raise",
        ).unique()
    )

    pivot = (
        df.pivot(
            index="K_over_omega",
            columns="sigma_A_over_omega",
            values=column,
        )
        .reindex(
            index=kappa_vals,
            columns=r_vals,
        )
    )

    return r_vals, kappa_vals, pivot.to_numpy(dtype=float)


def make_figure():
    set_plot_style()

    _, df = find_fraction_csv()

    r_vals, kappa_vals, f_plus = grid_from_dataframe(
        df,
        "positive_fraction",
    )

    r2, k2, f_minus = grid_from_dataframe(
        df,
        "negative_fraction",
    )

    if not (
        np.array_equal(r_vals, r2)
        and np.array_equal(kappa_vals, k2)
    ):
        raise RuntimeError(
            "Positive- and negative-fraction grids do not match."
        )




    cmap_plus = truncated_cmap(
        "RdBu_r",
        0.12,
        0.88,
    )

    cmap_minus = truncated_cmap(
        "YlGn",
        0.08,
        0.88,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.9),
        constrained_layout=False,
    )



    fig.subplots_adjust(
        left=0.060,
        right=0.975,
        bottom=0.12,
        top=0.91,
        wspace=0.2,
    )

    ax_a, ax_b = axes

    im_a = ax_a.pcolormesh(
        r_vals,
        kappa_vals,
        f_plus,
        shading="auto",
        cmap=cmap_plus,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    neg_vmax = float(np.nanmax(f_minus))
    neg_vmax = max(
        0.4,
        np.ceil(neg_vmax / 0.2) * 0.2,
    )

    im_b = ax_b.pcolormesh(
        r_vals,
        kappa_vals,
        f_minus,
        shading="auto",
        cmap=cmap_minus,
        vmin=0.0,
        vmax=neg_vmax,
        rasterized=True,
    )

    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\kappa$")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        style_axes(ax)

    ax_a.set_title(
        r"(a) Positive late-time drift",
        loc="center",
        pad=10,
    )

    ax_b.set_title(
        r"(b) Negative late-time drift",
        loc="center",
        pad=10,
    )




    cbar_a = fig.colorbar(
        im_a,
        ax=ax_a,
        pad=0.020,
        fraction=0.046,
    )

    cbar_a.set_ticks(
        np.arange(0.0, 1.01, 0.2)
    )

    cbar_a.ax.tick_params(
        labelsize=12.5,
        width=1.0,
        length=4.0,
    )



    cbar_a.ax.set_title(
        r"$f+$",
        fontsize=18,
        pad=8,
    )




    cbar_b = fig.colorbar(
        im_b,
        ax=ax_b,
        pad=0.020,
        fraction=0.046,
    )

    cbar_b.set_ticks(
        np.arange(
            0.0,
            neg_vmax + 0.001,
            0.2,
        )
    )

    cbar_b.ax.tick_params(
        labelsize=12.5,
        width=1.0,
        length=4.0,
    )


    cbar_b.ax.set_title(
        r"$f-$",
        fontsize=18,
        pad=8,
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

    plt.close(fig)

    print("\nSaved:")
    print(" ", FIGURE_PNG.relative_to(REPO_ROOT))
    print(" ", FIGURE_PDF.relative_to(REPO_ROOT))


if __name__ == "__main__":
    make_figure()
