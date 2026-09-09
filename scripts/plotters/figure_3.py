#!/usr/bin/env python3

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root


SHOW_NECESSARY_LINE = False


R_VALUES = np.arange(6.0, 3.99, -0.10)
KAPPA_TOP = 6.0
KAPPA_BOTTOM = 2.5
KAPPA_STEP = 0.02


N_GH = 140


FIGSIZE = (12.2, 8.4)
PNG_DPI = 600
CMAP = "bone"


ROOT = Path.cwd()

if not (ROOT / "data").exists() or not (ROOT / "scripts").exists():
    sys.exit(
        "\nERROR: Run this script from the repository root, i.e. the folder "
        "containing data/, figures/, and scripts/.\n\n"
        "Example:\n"
        "  Run this script from the repository root.\n"
        "  python scripts/plotters/figure_3.py\n"
    )

DATA_DIR = ROOT / "data" / "n100"
FIG_DIR = ROOT / "figures"
PROCESSED_DIR = DATA_DIR

FIG_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / "figure_3.png"
OUT_PDF = FIG_DIR / "figure_3.pdf"
OUT_MF = DATA_DIR / "stationary_mean_field_boundary.csv"


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_column(columns, exact=(), contains=()):
    nmap = {norm(c): c for c in columns}

    for candidate in exact:
        nc = norm(candidate)
        if nc in nmap:
            return nmap[nc]

    for c in columns:
        nc = norm(c)
        for token in contains:
            if norm(token) in nc:
                return c

    return None


def try_long_form_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if len(df) < 100:
        return None

    cols = list(df.columns)

    rcol = pick_column(
        cols,
        exact=("r", "r_value", "feedback_disorder", "sigma_ratio"),
        contains=("sigmaaoveromega", "feedbackdisorder"),
    )

    kcol = pick_column(
        cols,
        exact=("kappa", "K", "k", "coupling", "kappa_value", "K_over_omega"),
        contains=("kappa", "koveromega", "coupling"),
    )

    dcol = pick_column(
        cols,
        exact=("D", "D_mean", "mean_D", "drift_mean", "mean_drift", "D_avg"),
        contains=("dmean", "meandrift", "driftmean"),
    )

    if rcol is None or kcol is None or dcol is None:
        return None

    work = df[[rcol, kcol, dcol]].copy()
    work.columns = ["r", "kappa", "D"]

    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna()

    nr = work["r"].nunique()
    nk = work["kappa"].nunique()

    if nr < 15 or nk < 15:
        return None

    name_bonus = 1000 if (
        "figure_3" in path.name.lower()
        or "figure3" in path.name.lower()
    ) else 0

    score = name_bonus + nr * nk

    pivot = work.pivot_table(
        index="kappa",
        columns="r",
        values="D",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    r = pivot.columns.to_numpy(float)
    kappa = pivot.index.to_numpy(float)
    D = pivot.to_numpy(float)

    return score, r, kappa, D


def try_matrix_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.shape[0] < 15 or df.shape[1] < 16:
        return None

    try:
        axis0 = pd.to_numeric(
            df.iloc[:, 0],
            errors="raise"
        ).to_numpy(float)

        axis1 = np.array(
            [float(c) for c in df.columns[1:]],
            dtype=float
        )

        values = (
            df.iloc[:, 1:]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
    except Exception:
        return None

    if not (
        np.nanmin(axis0) >= -0.1
        and np.nanmax(axis0) <= 6.5
        and np.nanmin(axis1) >= -0.1
        and np.nanmax(axis1) <= 6.5
    ):
        return None

    r = np.sort(axis0)
    kappa = np.sort(axis1)

    ir = np.argsort(axis0)
    ik = np.argsort(axis1)

    D = values[ir][:, ik].T

    name_bonus = 1000 if (
        "figure_3" in path.name.lower()
        or "figure3" in path.name.lower()
    ) else 0

    score = name_bonus + len(r) * len(kappa)

    return score, r, kappa, D


def locate_heatmap():
    csvs = [
        p for p in DATA_DIR.rglob("*.csv")
        if "stationary_mean_field_boundary" not in p.name.lower()
    ]

    if not csvs:
        sys.exit(
            "\nERROR: No CSV files found anywhere inside ./data.\n"
        )

    candidates = []

    for p in csvs:
        out = try_long_form_csv(p)
        if out is not None:
            candidates.append((out[0], p, out[1], out[2], out[3]))
            continue

        out = try_matrix_csv(p)
        if out is not None:
            candidates.append((out[0], p, out[1], out[2], out[3]))

    if not candidates:
        print(
            "\nCSV files were found, but none looks like "
            "the dense r-kappa-D heat-map table."
        )
        print("\nFiles present under ./data:")
        for p in csvs:
            print("  ", p.relative_to(ROOT))
        sys.exit(1)

    candidates.sort(key=lambda x: x[0], reverse=True)

    _, path, r, kappa, D = candidates[0]

    print(f"\nUsing heat-map data:\n  {path.relative_to(ROOT)}")
    print(
        f"Detected grid: {len(r)} r-values x "
        f"{len(kappa)} kappa-values"
    )
    print(
        f"D range: {np.nanmin(D):.4f} to "
        f"{np.nanmax(D):.4f}"
    )

    return path, r, kappa, D


GH_X, GH_W_RAW = hermgauss(N_GH)
GH_W = GH_W_RAW / np.sqrt(np.pi)


def f1(B):

    B = np.asarray(B, dtype=float)
    out = np.zeros(B.shape, dtype=np.complex128)

    drifting = B < 1.0
    locked = ~drifting

    if np.any(drifting):
        b = B[drifting]
        s = np.sqrt(
            np.clip(1.0 - b * b, 0.0, None)
        )



        out[drifting] = 1j * b / (1.0 + s)

    if np.any(locked):
        b = B[locked]

        out[locked] = (
            np.sqrt(
                np.clip(
                    1.0 - 1.0 / (b * b),
                    0.0,
                    None
                )
            )
            + 1j / b
        )

    return out


def I_of_Z(Z, r):

    a = np.sqrt(2.0) * r * GH_X

    w = a + Z
    B = np.abs(w)

    unit = np.zeros_like(
        w,
        dtype=np.complex128
    )

    mask = B > 1e-13
    unit[mask] = w[mask] / B[mask]

    return np.sum(
        GH_W * unit * f1(B)
    )


def residual(xy, r, kappa):
    Z = complex(xy[0], xy[1])



    F = Z - kappa * I_of_Z(Z, r)

    return np.array(
        [F.real, F.imag]
    )


def solve_nonzero(r, kappa, seed):

    seeds = [
        seed,
        0.97 * seed,
        1.03 * seed,
        seed * np.exp(1j * 0.03),
        seed * np.exp(-1j * 0.03),
        0.90 * seed,
        1.08 * seed,
    ]

    good = []

    for z0 in seeds:
        sol = root(
            residual,
            x0=[z0.real, z0.imag],
            args=(r, kappa),
            method="hybr",
            tol=1e-10,
        )

        Z = complex(
            sol.x[0],
            sol.x[1]
        )

        res = np.linalg.norm(
            residual(
                sol.x,
                r,
                kappa
            )
        )

        R1 = abs(Z) / kappa

        if (
            res < 2e-8
            and abs(Z) > 2e-3
            and 0.0 < R1 <= 1.02
        ):

            if Z.imag < 0:
                Z = -Z

            good.append(
                (res, Z)
            )

    if not good:
        return None

    good.sort(
        key=lambda item: item[0]
    )

    return good[0][1]


def boundary_at_r(r, seed_at_k6):

    Z = solve_nonzero(
        r,
        KAPPA_TOP,
        seed_at_k6
    )

    if Z is None:
        for Rguess, angle_deg in [
            (0.80, 130),
            (0.70, 135),
            (0.60, 140),
            (0.90, 125),
        ]:
            guess = (
                KAPPA_TOP
                * Rguess
                * np.exp(
                    1j * np.deg2rad(angle_deg)
                )
            )

            Z = solve_nonzero(
                r,
                KAPPA_TOP,
                guess
            )

            if Z is not None:
                break

    if Z is None:
        return None

    high_k_solution = Z

    last_good_k = KAPPA_TOP
    last_good_Z = Z

    k = (
        KAPPA_TOP
        - KAPPA_STEP
    )

    first_bad_k = None

    while (
        k
        >= KAPPA_BOTTOM - 1e-12
    ):
        scaled_seed = (
            last_good_Z
            * (k / last_good_k)
        )

        test = solve_nonzero(
            r,
            k,
            scaled_seed
        )

        if test is None:
            first_bad_k = k
            break

        last_good_k = k
        last_good_Z = test
        k -= KAPPA_STEP

    if first_bad_k is None:
        return (
            last_good_k,
            last_good_Z,
            high_k_solution
        )

    bad = first_bad_k
    good = last_good_k
    good_Z = last_good_Z


    for _ in range(10):
        mid = 0.5 * (bad + good)

        test = solve_nonzero(
            r,
            mid,
            good_Z * (mid / good)
        )

        if test is None:
            bad = mid
        else:
            good = mid
            good_Z = test

    return (
        good,
        good_Z,
        high_k_solution
    )


def trace_boundary():
    rows = []

    k6_seed = (
        KAPPA_TOP
        * 0.78
        * np.exp(
            1j * np.deg2rad(130.0)
        )
    )

    print(
        "\nComputing stationary "
        "mean-field boundary ..."
    )

    for r in R_VALUES:
        result = boundary_at_r(
            r,
            k6_seed
        )

        if result is None:
            print(
                f"  r={r:.2f}: "
                "no nonzero stationary solution found "
                "for kappa <= 6"
            )
            continue

        kc, Zedge, Zk6 = result
        k6_seed = Zk6

        R1edge = abs(Zedge) / kc
        psi = (
            np.degrees(
                np.angle(Zedge)
            )
            % 360.0
        )

        rows.append(
            {
                "r": r,
                "kappa_c_MF": kc,
                "R1_at_edge": R1edge,
                "psi1_deg_at_edge": psi,
                "ReZ_at_edge": Zedge.real,
                "ImZ_at_edge": Zedge.imag,
            }
        )

        print(
            f"  r={r:.2f}  "
            f"kappa_edge≈{kc:.3f}  "
            f"R1≈{R1edge:.3f}  "
            f"psi1≈{psi:.1f}°"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("r")
    )

    df.to_csv(
        OUT_MF,
        index=False
    )

    return df


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "axes.labelsize": 26,
        "axes.labelweight": "normal",
        "axes.linewidth": 1.15,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_region_label(
    ax,
    x,
    y,
    text,
):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=19.0,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="0.45",
            linewidth=0.55,
            alpha=0.78,
        ),
        zorder=20,
    )


def main():
    _, r_grid, k_grid, D = (
        locate_heatmap()
    )

    mf = trace_boundary()

    if mf.empty:
        sys.exit(
            "\nERROR: Mean-field continuation "
            "produced no boundary points.\n"
        )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    mesh = ax.pcolormesh(
        r_grid,
        k_grid,
        D,
        shading="auto",
        cmap=CMAP,
        vmin=0.0,
        vmax=max(
            1.0,
            float(np.nanmax(D))
        ),
        rasterized=True,
    )


    cbar = fig.colorbar(
        mesh,
        ax=ax,
        pad=0.022,
        fraction=0.046,
    )

    cbar.set_label(
        r"$D$",
        fontsize=26,
        rotation=0,
        labelpad=20,
    )

    cbar.ax.tick_params(
        labelsize=15
    )





    mf_line, = ax.plot(
        mf["r"],
        mf["kappa_c_MF"],
        lw=1.25,
        color="#e5a62f",
        label="Stationary mean-field boundary",
        zorder=12,
    )

    mf_line.set_path_effects(
        [
            pe.Stroke(
                linewidth=2.0,
                foreground="black",
                alpha=0.62,
            ),
            pe.Normal(),
        ]
    )





    if SHOW_NECESSARY_LINE:
        rnec = np.sqrt(
            np.pi / 2.0
        )

        ax.axvline(
            rnec,
            color="0.20",
            lw=1.2,
            ls=(0, (4, 3)),
            zorder=11,
        )




    add_region_label(
        ax,
        0.78,
        0.62,
        "Independent\ndrift",
    )

    add_region_label(
        ax,
        1.00,
        4.80,
        "Coherent\ndrift",
    )

    add_region_label(
        ax,
        5.02,
        0.85,
        "Antipodal\npinning",
    )

    add_region_label(
        ax,
        5.15,
        5.20,
        "Coherent\npinning",
    )


    ax.set_xlabel(
        r"$r$",
        labelpad=6,
    )

    ax.set_ylabel(
        r"$\kappa$",
        labelpad=8,
    )

    ax.set_xlim(
        float(np.nanmin(r_grid)),
        float(np.nanmax(r_grid)),
    )

    ax.set_ylim(
        float(np.nanmin(k_grid)),
        float(np.nanmax(k_grid)),
    )




    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.018, 0.985),
        frameon=True,
        fancybox=False,
        framealpha=0.90,
        facecolor="white",
        edgecolor="0.45",
        borderpad=0.42,
        handlelength=2.2,
    )

    fig.tight_layout()

    fig.savefig(
        OUT_PNG,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("\nDONE. Saved:")
    print(
        " ",
        OUT_PNG.relative_to(ROOT)
    )
    print(
        " ",
        OUT_PDF.relative_to(ROOT)
    )
    print(
        " ",
        OUT_MF.relative_to(ROOT)
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            RuntimeWarning
        )
        main()
