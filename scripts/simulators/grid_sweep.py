#!/usr/bin/env python3

from pathlib import Path
import argparse
import os
import multiprocessing as mp
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


DATA_DIR = REPO_ROOT / "data" / "n100"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_MAP_PATH = DATA_DIR / "drift_map.csv"
DRIFT_FRACTIONS_PATH = DATA_DIR / "drift_fractions.csv"


@dataclass(frozen=True)
class Config:
    N: int = 100
    omega: float = 0.5

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
    parser = argparse.ArgumentParser(description="Generate N=100 drift-map and drift-fraction datasets.")


    parser.add_argument("--N", type=int, default=Config.N)
    parser.add_argument("--omega", type=float, default=Config.omega)

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

    cfg = Config(
        N=args.N,
        omega=args.omega,
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

    return cfg, args.jobs


def simulate_sigma_trial(job):
    trial, isg, sigma_ratio, K_ratios, cfg_dict = job
    cfg = Config(**cfg_dict)

    N = cfg.N
    omega = cfg.omega

    steps = int(round(cfg.T / cfg.dt))
    late_start = int(round((1.0 - cfg.late_fraction) * steps))
    late_time = (steps - late_start) * cfg.dt

    drift_tol = cfg.drift_tol_fraction * abs(omega)

    K_values = K_ratios * omega
    n_k = len(K_values)

    rng_A = np.random.default_rng(cfg.seed + 1000003 * trial)
    xi_A = rng_A.normal(0.0, 1.0, N)
    A_i = sigma_ratio * omega * xi_A

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

        dtheta = omega - A_i[None, :] * sin_theta + K_values[:, None] * coupling
        theta += cfg.dt * dtheta

    if theta_late_start is None:
        raise RuntimeError("late_start was not reached. Check T and late_fraction.")

    Omega_i = (theta - theta_late_start) / late_time

    return {
        "trial": trial,
        "isg": isg,
        "net_drift": np.mean(Omega_i, axis=1),
        "mean_abs_drift": np.mean(np.abs(Omega_i), axis=1),
        "positive_fraction": np.mean(Omega_i > drift_tol, axis=1),
        "negative_fraction": np.mean(Omega_i < -drift_tol, axis=1),
    }


def run_sweep(cfg, jobs):
    sigma_ratios = np.linspace(cfg.sigma_min, cfg.sigma_max, cfg.n_sigma)
    K_ratios = np.linspace(cfg.K_min, cfg.K_max, cfg.n_k)

    shape = (cfg.n_k, cfg.n_sigma)

    sums = {
        "net_drift": np.zeros(shape),
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

    with mp.get_context("spawn").Pool(processes=jobs) as pool:
        for result in pool.imap_unordered(simulate_sigma_trial, job_list, chunksize=1):
            isg = result["isg"]

            for key in sums:
                sums[key][:, isg] += result[key]

    maps = {
        "net_drift_over_omega": sums["net_drift"] / cfg.trials / cfg.omega,
        "mean_abs_drift_over_omega": sums["mean_abs_drift"] / cfg.trials / cfg.omega,
        "positive_fraction": sums["positive_fraction"] / cfg.trials,
        "negative_fraction": sums["negative_fraction"] / cfg.trials,
    }

    return sigma_ratios, K_ratios, maps


def save_drift_map_data(sigma_ratios, K_ratios, maps):
    rows = []
    D = maps["mean_abs_drift_over_omega"]

    for ik, K_ratio in enumerate(K_ratios):
        for isg, sigma_ratio in enumerate(sigma_ratios):
            rows.append([sigma_ratio, K_ratio, D[ik, isg]])

    np.savetxt(
        DRIFT_MAP_PATH,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="sigma_A_over_omega,K_over_omega,D",
        comments="",
        fmt="%.10g",
    )


def save_drift_fraction_data(sigma_ratios, K_ratios, maps):
    rows = []

    f_plus = maps["positive_fraction"]
    f_minus = maps["negative_fraction"]

    for ik, K_ratio in enumerate(K_ratios):
        for isg, sigma_ratio in enumerate(sigma_ratios):
            rows.append([sigma_ratio, K_ratio, f_plus[ik, isg], f_minus[ik, isg]])

    np.savetxt(
        DRIFT_FRACTIONS_PATH,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="sigma_A_over_omega,K_over_omega,positive_fraction,negative_fraction",
        comments="",
        fmt="%.10g",
    )


def main():
    cfg, jobs = parse_args()

    sigma_ratios, K_ratios, maps = run_sweep(cfg, jobs)

    save_drift_map_data(
        sigma_ratios,
        K_ratios,
        maps,
    )

    save_drift_fraction_data(
        sigma_ratios,
        K_ratios,
        maps,
    )

    print("\nGrid-sweep datasets saved:")
    print(f"  {DRIFT_MAP_PATH}")
    print(f"  {DRIFT_FRACTIONS_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
