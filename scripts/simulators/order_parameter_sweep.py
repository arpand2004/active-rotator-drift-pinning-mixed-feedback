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


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate N=100 order-parameter and drift-cut dataset."
    )


    p.add_argument("--N", type=int, default=N_DEFAULT)
    p.add_argument("--omega", type=float, default=OMEGA_DEFAULT)
    p.add_argument("--T", type=float, default=T_DEFAULT)
    p.add_argument("--dt", type=float, default=DT_DEFAULT)
    p.add_argument(
        "--late-fraction",
        type=float,
        default=LATE_FRACTION_DEFAULT,
    )
    p.add_argument(
        "--n-late-samples",
        type=int,
        default=N_LATE_SAMPLES_DEFAULT,
    )
    p.add_argument(
        "--trials",
        type=int,
        default=TRIALS_DEFAULT,
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )

    return p.parse_args()


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


def simulate_one_job(job):
    (
        trial,
        r_index,
        r_value,
        kappas,
        N,
        omega,
        T,
        dt,
        late_fraction,
        n_late_samples,
        seed,
    ) = job

    steps = int(round(T / dt))

    late_start = int(
        round(
            (1.0 - late_fraction)
            * steps
        )
    )

    late_time = (
        steps - late_start
    ) * dt

    stride = max(
        1,
        (steps - late_start)
        // n_late_samples,
    )

    K_values = (
        kappas * omega
    )

    n_k = len(
        kappas
    )

    rng_A = np.random.default_rng(
        seed
        + 1000003 * trial
        + 1009 * r_index
    )

    A_i = (
        r_value
        * omega
        * rng_A.normal(
            0.0,
            1.0,
            N,
        )
    )

    rng_theta = np.random.default_rng(
        seed
        + 9176 * trial
        + 37 * r_index
        + 12345
    )

    theta0 = rng_theta.uniform(
        0.0,
        2.0 * np.pi,
        N,
    )

    theta = np.repeat(
        theta0[None, :],
        n_k,
        axis=0,
    )

    theta_late_start = None

    fc_factor = (
        N / (N - 1)
        if N > 1
        else 1.0
    )

    R1_sum = np.zeros(
        n_k
    )

    R2_sum = np.zeros(
        n_k
    )

    n_samples = 0

    for step in range(steps):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        mean_sin = np.mean(
            sin_theta,
            axis=1,
        )

        mean_cos = np.mean(
            cos_theta,
            axis=1,
        )

        if step == late_start:
            theta_late_start = (
                theta.copy()
            )

        if (
            step >= late_start
            and (step - late_start) % stride == 0
        ):
            mean_sin2 = np.mean(
                2.0
                * sin_theta
                * cos_theta,
                axis=1,
            )

            mean_cos2 = np.mean(
                1.0
                - 2.0 * sin_theta**2,
                axis=1,
            )

            R1_sum += np.hypot(
                mean_sin,
                mean_cos,
            )

            R2_sum += np.hypot(
                mean_sin2,
                mean_cos2,
            )

            n_samples += 1

        coupling = (
            fc_factor
            * (
                mean_sin[:, None]
                * cos_theta
                - mean_cos[:, None]
                * sin_theta
            )
        )

        dtheta = (
            omega
            - A_i[None, :]
            * sin_theta
            + K_values[:, None]
            * coupling
        )

        theta += (
            dt * dtheta
        )

    if theta_late_start is None:
        raise RuntimeError(
            "late_start was not reached."
        )

    Omega_i = (
        theta
        - theta_late_start
    ) / late_time

    return {
        "trial": trial,
        "r_index": r_index,
        "R1": (
            R1_sum / n_samples
        ),
        "R2": (
            R2_sum / n_samples
        ),
        "D": (
            np.mean(
                np.abs(Omega_i),
                axis=1,
            )
            / omega
        ),
    }


def run_sweep(
    r_values,
    kappas,
    *,
    N,
    omega,
    T,
    dt,
    late_fraction,
    n_late_samples,
    trials,
    seed,
    jobs,
):
    n_r = len(
        r_values
    )

    n_k = len(
        kappas
    )

    R1_trials = np.zeros(
        (trials, n_r, n_k)
    )

    R2_trials = np.zeros(
        (trials, n_r, n_k)
    )

    D_trials = np.zeros(
        (trials, n_r, n_k)
    )

    job_list = [
        (
            trial,
            ir,
            float(r_value),
            kappas,
            N,
            omega,
            T,
            dt,
            late_fraction,
            n_late_samples,
            seed,
        )
        for trial in range(trials)
        for ir, r_value in enumerate(
            r_values
        )
    ]

    def store(result):
        t = result["trial"]
        ir = result["r_index"]

        R1_trials[t, ir, :] = (
            result["R1"]
        )

        R2_trials[t, ir, :] = (
            result["R2"]
        )

        D_trials[t, ir, :] = (
            result["D"]
        )

    if jobs == 1:
        for result in map(
            simulate_one_job,
            job_list,
        ):
            store(result)
    else:
        with mp.get_context(
            "spawn"
        ).Pool(
            processes=jobs
        ) as pool:
            for result in pool.imap_unordered(
                simulate_one_job,
                job_list,
                chunksize=1,
            ):
                store(result)

    def summarise(arr):
        mean = np.mean(
            arr,
            axis=0,
        )

        std = (
            np.std(
                arr,
                axis=0,
                ddof=1,
            )
            if trials > 1
            else np.zeros_like(mean)
        )

        sem = (
            std / np.sqrt(trials)
        )

        return (
            mean,
            std,
            sem,
        )

    return {
        "R1": summarise(
            R1_trials
        ),
        "R2": summarise(
            R2_trials
        ),
        "D": summarise(
            D_trials
        ),
    }


def build_and_save_figure4_data(args):

    kappas = np.linspace(
        KAPPA_MIN_DEFAULT,
        KAPPA_MAX_DEFAULT,
        N_KAPPA_DEFAULT,
    )

    r_cuts = np.asarray(
        R_CUTS_DEFAULT,
        dtype=float,
    )

    r_kappa0 = np.asarray(
        R_KAPPA0_DEFAULT,
        dtype=float,
    )

    print(
        "\nRunning Figure-4 fixed-r cuts ..."
    )

    cuts = run_sweep(
        r_cuts,
        kappas,
        N=args.N,
        omega=args.omega,
        T=args.T,
        dt=args.dt,
        late_fraction=args.late_fraction,
        n_late_samples=args.n_late_samples,
        trials=args.trials,
        seed=args.seed,
        jobs=args.jobs,
    )

    print(
        "\nRunning Figure-4 uncoupled points ..."
    )

    kappa_zero = np.array(
        [0.0],
        dtype=float,
    )

    k0 = run_sweep(
        r_kappa0,
        kappa_zero,
        N=args.N,
        omega=args.omega,
        T=args.T,
        dt=args.dt,
        late_fraction=args.late_fraction,
        n_late_samples=args.n_late_samples,
        trials=args.trials,
        seed=args.seed + 700000,
        jobs=args.jobs,
    )

    rows = []


    for ir, r_value in enumerate(
        r_cuts
    ):
        for ik, kappa in enumerate(
            kappas
        ):
            rows.append(
                {
                    "dataset": "cuts_numerics",
                    "r": float(r_value),
                    "kappa": float(kappa),
                    "R1": float(
                        cuts["R1"][0][ir, ik]
                    ),
                    "R1_std": float(
                        cuts["R1"][1][ir, ik]
                    ),
                    "R1_sem": float(
                        cuts["R1"][2][ir, ik]
                    ),
                    "R2": float(
                        cuts["R2"][0][ir, ik]
                    ),
                    "R2_std": float(
                        cuts["R2"][1][ir, ik]
                    ),
                    "R2_sem": float(
                        cuts["R2"][2][ir, ik]
                    ),
                    "D": float(
                        cuts["D"][0][ir, ik]
                    ),
                    "D_std": float(
                        cuts["D"][1][ir, ik]
                    ),
                    "D_sem": float(
                        cuts["D"][2][ir, ik]
                    ),
                    "N": args.N,
                    "trials": args.trials,
                    "omega": args.omega,
                    "T": args.T,
                    "dt": args.dt,
                    "late_fraction": args.late_fraction,
                }
            )


    for ir, r_value in enumerate(
        r_kappa0
    ):
        rows.append(
            {
                "dataset": "kappa0_numerics",
                "r": float(r_value),
                "kappa": 0.0,
                "R1": float(
                    k0["R1"][0][ir, 0]
                ),
                "R1_std": float(
                    k0["R1"][1][ir, 0]
                ),
                "R1_sem": float(
                    k0["R1"][2][ir, 0]
                ),
                "R2": float(
                    k0["R2"][0][ir, 0]
                ),
                "R2_std": float(
                    k0["R2"][1][ir, 0]
                ),
                "R2_sem": float(
                    k0["R2"][2][ir, 0]
                ),
                "D": float(
                    k0["D"][0][ir, 0]
                ),
                "D_std": float(
                    k0["D"][1][ir, 0]
                ),
                "D_sem": float(
                    k0["D"][2][ir, 0]
                ),
                "N": args.N,
                "trials": args.trials,
                "omega": args.omega,
                "T": args.T,
                "dt": args.dt,
                "late_fraction": args.late_fraction,
            }
        )


    r_analytics = np.linspace(
        0.0,
        6.0,
        601,
    )

    R2_analytics = (
        analytical_R2_kappa0(
            r_analytics
        )
    )

    D0_analytics = (
        analytical_D0(
            r_analytics
        )
    )

    for r_value, R2_value, D0_value in zip(
        r_analytics,
        R2_analytics,
        D0_analytics,
    ):
        rows.append(
            {
                "dataset": "kappa0_analytics",
                "r": float(r_value),
                "kappa": 0.0,
                "R1": 0.0,
                "R1_std": np.nan,
                "R1_sem": np.nan,
                "R2": float(R2_value),
                "R2_std": np.nan,
                "R2_sem": np.nan,
                "D": float(D0_value),
                "D_std": np.nan,
                "D_sem": np.nan,
                "N": np.nan,
                "trials": np.nan,
                "omega": args.omega,
                "T": np.nan,
                "dt": np.nan,
                "late_fraction": np.nan,
            }
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        DATA_PATH,
        index=False,
    )

    print(
        "\nSaved Figure-4 data:"
        f"\n  {DATA_PATH.relative_to(REPO_ROOT)}"
    )


def main():
    args = parse_args()
    build_and_save_figure4_data(args)

    print("\nOrder-parameter dataset saved:")
    print(f"  {DATA_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
