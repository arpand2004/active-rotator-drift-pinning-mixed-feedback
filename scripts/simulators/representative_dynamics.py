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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Figure 6.")


    parser.add_argument("--N", type=int, default=Config.N)
    parser.add_argument("--omega", type=float, default=Config.omega)

    parser.add_argument("--T", type=float, default=Config.T)
    parser.add_argument("--dt", type=float, default=Config.dt)
    parser.add_argument("--late-fraction", type=float, default=Config.late_fraction)

    parser.add_argument("--T-plot", type=float, default=Config.T_plot)
    parser.add_argument("--traj-stride", type=int, default=Config.traj_stride)
    parser.add_argument("--trace-stride", type=int, default=Config.trace_stride)
    parser.add_argument("--hist-stride", type=int, default=Config.hist_stride)

    parser.add_argument("--n-traj-plot", type=int, default=Config.n_traj_plot)
    parser.add_argument("--n-bins", type=int, default=Config.n_bins)

    parser.add_argument("--drift-tol-fraction", type=float,
                        default=Config.drift_tol_fraction)
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
        T_plot=args.T_plot,
        traj_stride=args.traj_stride,
        trace_stride=args.trace_stride,
        hist_stride=args.hist_stride,
        n_traj_plot=args.n_traj_plot,
        n_bins=args.n_bins,
        drift_tol_fraction=args.drift_tol_fraction,
        trials=args.trials,
        seed=args.seed,
    )

    return cfg, args.jobs


def simulate_point_trial(job):
    ip, trial, r_value, kappa_value, cfg_dict = job
    cfg = Config(**cfg_dict)

    N = cfg.N
    omega = cfg.omega

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt
    plot_steps = int(round(cfg.T_plot / cfg.dt))

    drift_tol = cfg.drift_tol_fraction * abs(omega)
    K_value = kappa_value * omega

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial + 7919 * ip)
    A_i = r_value * omega * rng_A.normal(0.0, 1.0, N)

    rng_theta = np.random.default_rng(cfg.seed + 9176 * trial + 37 * ip + 12345)
    theta = rng_theta.uniform(0.0, 2.0 * np.pi, N)

    fc_factor = N / (N - 1) if N > 1 else 1.0

    keep_traj = (trial == 0)

    traj = []
    trace = []
    R1_late = []

    hist_counts = np.zeros(cfg.n_bins, dtype=float)
    hist_edges = np.linspace(0.0, 2.0 * np.pi, cfg.n_bins + 1)
    theta_late_start = None

    for step in range(steps):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        mean_sin = float(np.mean(sin_theta))
        mean_cos = float(np.mean(cos_theta))

        if step == late_start:
            theta_late_start = theta.copy()

        R1 = np.hypot(mean_sin, mean_cos)

        if step >= late_start and step % cfg.trace_stride == 0:
            R1_late.append(R1)

        if step >= late_start and step % cfg.hist_stride == 0:
            hist_counts += np.histogram(
                np.mod(theta, 2.0 * np.pi), bins=hist_edges
            )[0]

        if keep_traj and step <= plot_steps and step % cfg.traj_stride == 0:
            traj.append(np.mod(theta, 2.0 * np.pi).copy())
            mean_sin2 = float(np.mean(2.0 * sin_theta * cos_theta))
            mean_cos2 = float(np.mean(1.0 - 2.0 * sin_theta**2))
            R2 = np.hypot(mean_sin2, mean_cos2)
            trace.append((step * cfg.dt, R1, R2))

        coupling = fc_factor * (mean_sin * cos_theta - mean_cos * sin_theta)
        theta = theta + cfg.dt * (omega - A_i * sin_theta + K_value * coupling)

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T and late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time
    R1_late = np.asarray(R1_late)

    sin_final = np.sin(theta)
    cos_final = np.cos(theta)
    mean_sin2 = float(np.mean(2.0 * sin_final * cos_final))
    mean_cos2 = float(np.mean(1.0 - 2.0 * sin_final**2))

    return {
        "ip": ip,
        "trial": trial,
        "hist_counts": hist_counts,
        "Theta": float(np.angle(np.mean(np.exp(1j * theta)))),
        "Omega_over_omega": Omega_i / omega,
        "f_plus": float(np.mean(Omega_i > drift_tol)),
        "f_minus": float(np.mean(Omega_i < -drift_tol)),
        "D": float(np.mean(np.abs(Omega_i)) / omega),
        "R1_mean": float(np.mean(R1_late)),
        "R1_std": float(np.std(R1_late)),
        "R2_final": float(np.hypot(mean_sin2, mean_cos2)),
        "traj": np.asarray(traj) if keep_traj else None,
        "trace": np.asarray(trace) if keep_traj else None,
    }


def run_points(cfg, jobs):
    n_points = len(PANEL_POINTS)

    hist_totals = np.zeros((n_points, cfg.n_bins), dtype=float)
    Theta = np.zeros((n_points, cfg.trials), dtype=float)
    scalars = {
        key: np.zeros((n_points, cfg.trials), dtype=float)
        for key in ("f_plus", "f_minus", "D", "R1_mean", "R1_std", "R2_final")
    }

    trajectories = [None] * n_points
    traces = [None] * n_points

    cfg_dict = asdict(cfg)
    job_list = [
        (ip, trial, float(r_value), float(kappa_value), cfg_dict)
        for ip, (r_value, kappa_value, _) in enumerate(PANEL_POINTS)
        for trial in range(cfg.trials)
    ]

    def store_result(result):
        ip = result["ip"]
        trial = result["trial"]

        hist_totals[ip] += result["hist_counts"]
        Theta[ip, trial] = result["Theta"]

        for key in scalars:
            scalars[key][ip, trial] = result[key]

        if result["traj"] is not None:
            trajectories[ip] = result["traj"]
            traces[ip] = result["trace"]

    if jobs == 1:
        for result in map(simulate_point_trial, job_list):
            store_result(result)
    else:
        with mp.get_context("spawn").Pool(processes=jobs) as pool:
            for result in pool.imap_unordered(simulate_point_trial, job_list, chunksize=1):
                store_result(result)

    return hist_totals, Theta, scalars, trajectories, traces


def save_data(cfg, hist_totals, Theta, scalars, trajectories, traces):
    payload = {}

    for ip, (r_value, kappa_value, label) in enumerate(PANEL_POINTS):
        payload[f"traj_{ip}"] = trajectories[ip]
        payload[f"trace_{ip}"] = traces[ip]

    payload["hist_counts"] = hist_totals
    payload["hist_edges"] = np.linspace(0.0, 2.0 * np.pi, cfg.n_bins + 1)
    payload["Theta"] = Theta
    payload["points"] = np.array([(r, k, lab) for r, k, lab in PANEL_POINTS], dtype=object)

    np.savez_compressed(DATA_ARCHIVE_PATH, **payload)

    rows = []
    for ip, (r_value, kappa_value, label) in enumerate(PANEL_POINTS):
        row = {
            "r": r_value,
            "kappa": kappa_value,
            "label": label,
            "N": cfg.N,
            "trials": cfg.trials,
            "omega": cfg.omega,
            "T": cfg.T,
            "dt": cfg.dt,
            "late_fraction": cfg.late_fraction,
        }
        for key in scalars:
            row[f"{key}_mean"] = float(np.mean(scalars[key][ip]))
            row[f"{key}_sem"] = (
                float(np.std(scalars[key][ip], ddof=1) / np.sqrt(cfg.trials))
                if cfg.trials > 1 else 0.0
            )
        rows.append(row)

    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False)


def main():
    cfg, jobs = parse_args()

    hist_totals, Theta, scalars, trajectories, traces = run_points(cfg, jobs)
    save_data(cfg, hist_totals, Theta, scalars, trajectories, traces)

    print("\nRepresentative-dynamics datasets saved:")
    print(f"  {DATA_ARCHIVE_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
