from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
import numpy as np
import pandas as pd
from protocol import ROOT, SIM_OUTPUT, CHECK_OUTPUT, DT, SAMPLE_DT, T_TRANSIENT, T_MEASURE, GRID_STEP, SEED, EPS, TRIALS_BY_N, MAP_R, MAP_K, MAP_N, CUT_SPECS, HYST_R, HYST_K, HYST_N, OSC_POINTS, OSC_SCAN_R, OSC_SCAN_K, OSC_SCAN_N, TRACE_POINTS, TRACE_N, CONV_POINTS, CONV_N, CONV_TRIALS, PRECISION_POINTS, PRECISION_N, PRECISION_TRIALS, DISORDER_CHECK_POINTS, DISORDER_CHECK_N, DISORDER_CHECK_TRIALS, production_validate

REPO_ROOT = ROOT
OUT_DIR = SIM_OUTPUT
MAP_TRIALS = TRIALS_BY_N[MAP_N]
CUT_DK = GRID_STEP
CUT_TRIALS = TRIALS_BY_N
HYST_REALIZATIONS = TRIALS_BY_N[HYST_N]
HYST_SETTLE = T_TRANSIENT
HYST_MEASURE = T_MEASURE
HYST_INITIAL = 0.0
OSC_TRIALS = TRIALS_BY_N
OSC_SCAN_NTRIALS = {n: TRIALS_BY_N[n] for n in OSC_SCAN_N}
TRACE_TRIALS = {n: TRIALS_BY_N[n] for n in TRACE_N}

OBS = [
    "D", "D_endpoint", "f_plus", "f_minus", "back_rev", "fwd_rev",
    "persistent_minus", "persistent_plus", "backward_turn_episode",
    "max_backward_excursion_turns", "min_omega", "max_omega",
    "min_winding", "max_winding", "mean_omega", "omega_abs_half_difference",
    "R1", "R2", "dR1", "dR2", "R1_first", "R1_second",
    "dR1_first", "dR1_second", "R1_trend", "R1_min", "R1_final",
    "R2_final", "Theta_final", "peak_freq", "peak_frac",
    "peak_freq_hann", "peak_frac_hann", "peak_power", "peak_freq_R2",
    "peak_frac_R2", "psi_rate", "psi_weak_fraction", "disorder_mean",
    "disorder_std",
] + [f"f_plus_{e:.0e}" for e in EPS] + [f"f_minus_{e:.0e}" for e in EPS]
def rhs(th: np.ndarray, a: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    s, c = np.sin(th), np.cos(th)
    ms = s.mean(axis=-1, keepdims=True)
    mc = c.mean(axis=-1, keepdims=True)
    return 1.0 - a * s + kappa * (ms * c - mc * s)
def rk4(th: np.ndarray, a: np.ndarray, kappa: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(th, a, kappa)
    k2 = rhs(th + 0.5 * dt * k1, a, kappa)
    k3 = rhs(th + 0.5 * dt * k2, a, kappa)
    k4 = rhs(th + dt * k3, a, kappa)
    return th + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
def disorder(rng: np.random.Generator, N: int, r: float, scheme: str) -> np.ndarray:
    if scheme == "iid":
        return r * rng.normal(0.0, 1.0, N)
    if scheme == "antithetic":
        if N % 2:
            raise ValueError("Antithetic disorder requires even N")
        half = rng.normal(0.0, 1.0, N // 2)
        return r * np.concatenate([half, -half])
    raise ValueError(f"Unknown disorder scheme: {scheme}")
def initial_conditions(seed: int, r: float, N: int, trial: int, scheme: str):
    stem = [seed, int(round(10000 * r)), N, trial]
    rng_a = np.random.default_rng(stem + [101])
    rng_th = np.random.default_rng(stem + [202])
    return disorder(rng_a, N, r, scheme), rng_th.uniform(0.0, 2.0 * np.pi, N)
def integrate(th: np.ndarray, a: np.ndarray, kappa: np.ndarray,
              T: float, dt: float) -> np.ndarray:
    steps = int(round(T / dt))
    if not np.isclose(steps * dt, T, rtol=0, atol=1e-9):
        raise ValueError(f"T={T} must be divisible by dt={dt}")
    for _ in range(steps):
        th = rk4(th, a, kappa, dt)
    return th
def _slopes(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    centered = t - t.mean()
    denom = np.dot(centered, centered)
    if denom <= 0:
        raise ValueError("Measurement window too short for least-squares slope")
    return np.tensordot(centered, x, axes=(0, 0)) / denom
def _spectrum(x: np.ndarray, sample_dt: float, hann: bool = False):
    n = x.shape[0]
    if n < 8:
        raise ValueError("Need at least eight samples for spectral diagnostics")
    x = x - x.mean(axis=0, keepdims=True)
    if hann:
        x = x * np.hanning(n).reshape((n,) + (1,) * (x.ndim - 1))
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    P[0] = 0.0
    freqs = np.fft.rfftfreq(n, sample_dt)
    ip = np.argmax(P, axis=0)
    flat_p = P.reshape((len(freqs), -1))
    flat_i = np.asarray(ip).reshape(-1)
    total = flat_p.sum(axis=0)
    fractions = np.zeros_like(total)
    for j, i in enumerate(flat_i):
        if total[j] > 1e-25:
            fractions[j] = flat_p[max(int(i) - 2, 1):int(i) + 3, j].sum() / total[j]
    shape = np.shape(ip)
    return (freqs[ip], fractions.reshape(shape), P.max(axis=0), freqs, P)
def measure(th: np.ndarray, a: np.ndarray, kappa: np.ndarray,
            T: float, dt: float, *, capture: bool = False,
            tracer_indices: np.ndarray | None = None) -> dict:
    n = int(round(T / dt))
    if not np.isclose(n * dt, T, rtol=0, atol=1e-9):
        raise ValueError("Measurement T must be divisible by dt")
    every = max(1, int(round(SAMPLE_DT / dt)))
    if not np.isclose(every * dt, SAMPLE_DT, rtol=0, atol=1e-10):
        raise ValueError("SAMPLE_DT must be an integer multiple of dt")
    th0 = th.copy()
    running_max = th.copy()
    max_backward_excursion = np.zeros_like(th)
    S0 = St = Stt = 0.0
    Sx = np.zeros_like(th)
    Stx = np.zeros_like(th)
    h_n = [0, 0]
    h_t = [0.0, 0.0]
    h_tt = [0.0, 0.0]
    h_x = [np.zeros_like(th), np.zeros_like(th)]
    h_tx = [np.zeros_like(th), np.zeros_like(th)]
    Z1, Z2, ts = [], [], []
    selected = []
    for k in range(n + 1):
        if k % every == 0:
            t = k * dt
            x = th - th0
            S0 += 1.0
            St += t
            Stt += t * t
            Sx += x
            Stx += t * x
            half = 0 if t < T * 0.5 else 1
            h_n[half] += 1
            h_t[half] += t
            h_tt[half] += t * t
            h_x[half] += x
            h_tx[half] += t * x
            e = np.exp(1j * th)
            Z1.append(e.mean(axis=-1))
            Z2.append((e * e).mean(axis=-1))
            ts.append(t)
            if capture:
                if tracer_indices is None:
                    raise ValueError("capture=True requires tracer_indices")
                selected.append(th[:, tracer_indices].copy())
        if k < n:
            th = rk4(th, a, kappa, dt)
            running_max = np.maximum(running_max, th)
            max_backward_excursion = np.maximum(max_backward_excursion, running_max - th)
    if S0 < 8 or min(h_n) < 3:
        raise ValueError("Too few samples to estimate angular velocities")
    Om = (S0 * Stx - St * Sx) / (S0 * Stt - St ** 2)
    half_omega = [(h_n[v] * h_tx[v] - h_t[v] * h_x[v]) /
                  (h_n[v] * h_tt[v] - h_t[v] ** 2) for v in (0, 1)]
    W = (th - th0) / (2.0 * np.pi)
    endpoint_omega = (th - th0) / T
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    ts = np.asarray(ts)
    R1 = np.abs(Z1)
    R2 = np.abs(Z2)
    frq, frac, power, _, _ = _spectrum(R1, every * dt)
    frq_h, frac_h, _, _, _ = _spectrum(R1, every * dt, hann=True)
    frq2, frac2, _, _, _ = _spectrum(R2, every * dt)
    psi = np.unwrap(np.angle(Z1), axis=0)
    tc = ts - ts.mean()
    psi_rate = np.sum(tc[:, None] * (psi - psi.mean(axis=0)), axis=0) / np.sum(tc ** 2)
    weak_threshold = max(0.02, 1.0 / math.sqrt(th.shape[-1]))
    weak_fraction = (R1 < weak_threshold).mean(axis=0)
    mid = len(R1) // 2
    trend = np.sum(tc[:, None] * (R1 - R1.mean(axis=0)), axis=0) / np.sum(tc ** 2)
    out = dict(
        D=np.abs(Om).mean(axis=-1),
        D_endpoint=np.abs(endpoint_omega).mean(axis=-1),
        f_plus=(Om > 1e-3).mean(axis=-1),
        f_minus=(Om < -1e-3).mean(axis=-1),
        back_rev=(W <= -1.0).mean(axis=-1),
        fwd_rev=(W >= 1.0).mean(axis=-1),
        persistent_minus=((half_omega[0] < -1e-3) & (half_omega[1] < -1e-3)).mean(axis=-1),
        persistent_plus=((half_omega[0] > 1e-3) & (half_omega[1] > 1e-3)).mean(axis=-1),
        backward_turn_episode=(max_backward_excursion >= 2.0 * np.pi).mean(axis=-1),
        max_backward_excursion_turns=max_backward_excursion.max(axis=-1) / (2.0 * np.pi),
        min_omega=Om.min(axis=-1), max_omega=Om.max(axis=-1),
        min_winding=W.min(axis=-1), max_winding=W.max(axis=-1),
        mean_omega=Om.mean(axis=-1),
        omega_abs_half_difference=np.abs(half_omega[0] - half_omega[1]).mean(axis=-1),
        R1=R1.mean(axis=0), R2=R2.mean(axis=0),
        dR1=R1.std(axis=0), dR2=R2.std(axis=0),
        R1_first=R1[:mid].mean(axis=0), R1_second=R1[mid:].mean(axis=0),
        dR1_first=R1[:mid].std(axis=0), dR1_second=R1[mid:].std(axis=0),
        R1_trend=trend, R1_min=R1.min(axis=0),
        R1_final=R1[-1], R2_final=R2[-1], Theta_final=np.angle(Z1[-1]),
        peak_freq=frq, peak_frac=frac, peak_freq_hann=frq_h,
        peak_frac_hann=frac_h, peak_power=power,
        peak_freq_R2=frq2, peak_frac_R2=frac2,
        psi_rate=psi_rate, psi_weak_fraction=weak_fraction,
        disorder_mean=np.full(th.shape[0], a.mean()),
        disorder_std=np.full(th.shape[0], a.std()),
    )
    for eps in EPS:
        out[f"f_plus_{eps:.0e}"] = (Om > eps).mean(axis=-1)
        out[f"f_minus_{eps:.0e}"] = (Om < -eps).mean(axis=-1)
    if not all(np.isfinite(out[c]).all() for c in OBS):
        raise FloatingPointError("Nonfinite observable detected")
    stationary_r1 = out["dR1"] < 1e-9
    stationary_r2 = out["dR2"] < 1e-9
    for col in ("peak_freq", "peak_frac", "peak_freq_hann", "peak_frac_hann", "peak_power"):
        out[col] = np.where(stationary_r1, 0.0, out[col])
    for col in ("peak_freq_R2", "peak_frac_R2"):
        out[col] = np.where(stationary_r2, 0.0, out[col])
    out["th_end"] = th
    if capture:
        out["capture"] = dict(time=ts, z1=Z1, z2=Z2,
                              theta_selected=np.asarray(selected),
                              tracer_indices=np.asarray(tracer_indices),
                              omega_lsq=Om, omega_first=half_omega[0],
                              omega_second=half_omega[1],
                              omega_endpoint=endpoint_omega, winding=W,
                              theta0=th0, theta_end=th)
    return out
def _base_row(r, kk, N, trial, dt, T_tr, T_ms, seed, scheme):
    return dict(r=float(r), kappa=float(kk), N=int(N), trial=int(trial),
                dt=float(dt), T_transient=float(T_tr), T_measure=float(T_ms),
                seed=int(seed), disorder_scheme=scheme)
def job_batch(job):
    r, kappas, N, trial, T_tr, T_ms, dt, seed, tag, scheme = job
    a, theta0 = initial_conditions(seed, r, N, trial, scheme)
    kappas = np.asarray(kappas, dtype=float)
    th = np.repeat(theta0[None, :], len(kappas), axis=0)
    K = kappas[:, None]
    th = integrate(th, a[None, :], K, T_tr, dt)
    obs = measure(th, a[None, :], K, T_ms, dt)
    rows = []
    for j, kk in enumerate(kappas):
        row = _base_row(r, kk, N, trial, dt, T_tr, T_ms, seed, scheme)
        row["job_tag"] = int(tag)
        for key in OBS:
            row[key] = float(obs[key][j])
        rows.append(row)
    return rows
def job_hysteresis(job):
    r, direction, real, N, ks, T_init, T_settle, T_ms, dt, seed, scheme = job
    a, theta0 = initial_conditions(seed, r, N, real, scheme)
    th = theta0[None, :]
    seq = np.asarray(ks) if direction == "up" else np.asarray(ks)[::-1]
    th = integrate(th, a[None, :], np.array([[seq[0]]]), T_init, dt)
    rows = []
    for kk in seq:
        K = np.array([[kk]])
        th = integrate(th, a[None, :], K, T_settle, dt)
        obs = measure(th, a[None, :], K, T_ms, dt)
        th = obs["th_end"]
        row = _base_row(r, kk, N, real, dt, T_settle, T_ms, seed, scheme)
        row.update(direction=direction, realization=int(real), T_initial=T_init,
                   T_settle=T_settle, sweep_step=float(ks[1] - ks[0]))
        for key in OBS:
            row[key] = float(obs[key][0])
        rows.append(row)
    return rows
def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, dropna=False, sort=True)
    mean = g[OBS].mean().add_suffix("_mean")
    sem = g[OBS].sem().add_suffix("_sem")
    std = g[OBS].std().add_suffix("_std")
    out = pd.concat([mean, sem, std], axis=1)
    out["trials"] = g.size()
    return out.reset_index()
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _serializable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_serializable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    return value
def _atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="_partial_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
def _atomic_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="_partial_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            df.to_csv(f, index=False, float_format="%.16g")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
def _run_dir(mode, args):
    base = CHECK_OUTPUT if args.smoke else SIM_OUTPUT
    suffix = f"_{args.tag}" if args.tag else ""
    return base / f"{mode}{suffix}"
def _prepare_dir(mode, args, scientific_config):
    directory = _run_dir(mode, args)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(mode=mode, smoke=bool(args.smoke), seed=args.seed,
                       disorder_scheme=args.disorder, script_sha256=_sha(Path(__file__)), protocol_sha256=_sha(Path(__file__).with_name("protocol.py")),
                       scientific_config=_serializable(scientific_config))
    token = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    manifest = directory / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("fingerprint") != token:
            raise RuntimeError(
                f"Existing output at {directory} was produced with different script/settings. "
                "Nothing was overwritten. Use --tag to create a new independent dataset."
            )
    else:
        if any(directory.iterdir()):
            raise RuntimeError(f"Nonempty output directory without manifest: {directory}")
        information = dict(fingerprint=token, config=fingerprint,
                           python=platform.python_version(), numpy=np.__version__,
                           pandas=pd.__version__, model="dtheta_i/dtau = 1 - a_i sin(theta_i) "
                           "+ kappa*Im(Z1*exp(-i theta_i))", coupling_normalization="1/N",
                           omega_definition="least-squares slope of unwrapped theta_i(tau)",
                           spectrum_units="cycles per dimensionless tau")
        _atomic_text(manifest, json.dumps(information, indent=2, sort_keys=True) + "\n")
    return directory
def run_pool(func, jobs: list, args, mode: str, group_columns: list[str],
             scientific_config: dict):
    directory = _prepare_dir(mode, args, scientific_config)
    parts = directory / "_chunks"
    parts.mkdir(exist_ok=True)
    pending = [(i, job) for i, job in enumerate(jobs) if not (parts / f"job_{i:06d}.csv").exists()]
    print(f"{mode}: {len(jobs) - len(pending)}/{len(jobs)} checkpointed, "
          f"{len(pending)} remaining; folder={directory}", flush=True)
    if pending:
        def store_job(i, result):
            if not result:
                raise RuntimeError(f"Job {i} returned no data")
            df = pd.DataFrame(result)
            if not set(OBS).issubset(df.columns):
                raise RuntimeError(f"Job {i} is missing observables")
            if not np.isfinite(df[OBS].to_numpy(dtype=float)).all():
                raise RuntimeError(f"Job {i} produced nonfinite values")
            df.insert(0, "job_id", i)
            _atomic_csv(parts / f"job_{i:06d}.csv", df)
        if args.jobs == 1:
            for finished, (i, job) in enumerate(pending, 1):
                store_job(i, func(job))
                if finished % max(1, len(pending) // 20) == 0 or finished == len(pending):
                    print(f"  {finished}/{len(pending)} new jobs", flush=True)
        else:
            with mp.get_context("spawn").Pool(processes=args.jobs) as pool:
                for finished, result in enumerate(pool.imap(func, [j for _, j in pending], chunksize=1), 1):
                    i = pending[finished - 1][0]
                    store_job(i, result)
                    if finished % max(1, len(pending) // 20) == 0 or finished == len(pending):
                        print(f"  {finished}/{len(pending)} new jobs", flush=True)
    paths = [parts / f"job_{i:06d}.csv" for i in range(len(jobs))]
    if not all(p.exists() for p in paths):
        raise RuntimeError("Missing one or more checkpoint files")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    if df.empty or not set(OBS).issubset(df.columns):
        raise RuntimeError("Empty or malformed aggregate dataset")
    sorting = [c for c in ["r", "kappa", "N", "trial", "disorder_scheme",
                            "direction", "realization", "dt", "T_transient", "T_measure"] if c in df.columns]
    if sorting:
        df.sort_values(sorting, kind="stable", inplace=True, ignore_index=True)
    if len(df) != sum(len(pd.read_csv(p)) for p in paths):
        raise RuntimeError("Trial aggregation row count mismatch")
    summary = summarise(df, group_columns)
    _atomic_csv(directory / "trials.csv", df)
    _atomic_csv(directory / "summary.csv", summary)
    _atomic_text(directory / "COMPLETE.json", json.dumps(dict(jobs=len(jobs),
                 rows=len(df), summary_rows=len(summary), script_sha256=_sha(Path(__file__))), indent=2) + "\n")
    print(f"{mode}: saved {len(df)} individual rows, {len(summary)} summary rows", flush=True)
    return df, summary
def mode_map(args):
    r_vals = MAP_R if not args.smoke else np.array([0.0, 5.0])
    k_vals = MAP_K if not args.smoke else np.array([0.0, 6.0])
    N = MAP_N if not args.smoke else 40
    trials = MAP_TRIALS if not args.smoke else 2
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = [(float(r), k_vals, N, t, T_tr, T_ms, DT, args.seed, 1, args.disorder)
            for r in r_vals for t in range(trials)]
    run_pool(job_batch, jobs, args, "map", ["r", "kappa", "N", "disorder_scheme"],
             dict(r=r_vals, k=k_vals, N=N, trials=trials, transient=T_tr,
                  measure=T_ms, dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
def mode_cuts(args):
    specs = CUT_SPECS if not args.smoke else {4.5: (4.0, 6.0)}
    sizes = CUT_TRIALS if not args.smoke else {40: 2, 80: 2}
    dk = CUT_DK if not args.smoke else 2.0
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = []
    for r, (k0, k1) in specs.items():
        ks = np.round(np.arange(k0, k1 + 1e-9, dk), 6)
        for N, B in sizes.items():
            for t in range(B):
                jobs.append((float(r), ks, N, t, T_tr, T_ms, DT, args.seed, 2, args.disorder))
    jobs.sort(key=lambda j: -j[2] * len(j[1]))
    run_pool(job_batch, jobs, args, "cuts", ["r", "kappa", "N", "disorder_scheme"],
             dict(specs=specs, sizes=sizes, dk=dk, transient=T_tr, measure=T_ms,
                  dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
def mode_hysteresis(args):
    r_vals = HYST_R if not args.smoke else (6.0,)
    ks = HYST_K if not args.smoke else np.array([6.0, 5.0, 4.0])
    N = HYST_N if not args.smoke else 40
    realizations = HYST_REALIZATIONS if not args.smoke else 1
    T_init, T_set, T_ms = (HYST_INITIAL, HYST_SETTLE, HYST_MEASURE) if not args.smoke else (5.0, 5.0, 5.0)
    jobs = [(float(r), direction, q, N, ks, T_init, T_set, T_ms, DT, args.seed, args.disorder)
            for r in r_vals for direction in ("up", "down") for q in range(realizations)]
    run_pool(job_hysteresis, jobs, args, "hysteresis",
             ["r", "direction", "kappa", "N", "disorder_scheme"],
             dict(r=r_vals, k=ks, N=N, realizations=realizations, initial=T_init,
                  settle=T_set, measure=T_ms, dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
def mode_convergence(args):
    pts = CONV_POINTS if not args.smoke else CONV_POINTS[:1]
    N = CONV_N if not args.smoke else 40
    B = CONV_TRIALS if not args.smoke else 2
    settings = ((DT, T_TRANSIENT, T_MEASURE), (DT / 2.0, T_TRANSIENT, T_MEASURE),
                (DT, 2 * T_TRANSIENT, T_MEASURE),
                (DT, T_TRANSIENT, 2 * T_MEASURE),
                (DT, 2 * T_TRANSIENT, 2 * T_MEASURE))
    if args.smoke:
        settings = ((DT, 10.0, 10.0), (DT / 2, 10.0, 10.0), (DT, 10.0, 20.0))
    jobs = [(float(r), np.array([k]), N, t, T_tr, T_ms, dt, args.seed, 3, args.disorder)
            for dt, T_tr, T_ms in settings for r, k in pts for t in range(B)]
    _, summary = run_pool(job_batch, jobs, args, "convergence",
                          ["r", "kappa", "N", "dt", "T_transient", "T_measure", "disorder_scheme"],
                          dict(points=pts, N=N, trials=B, settings=settings,
                               sample_dt=SAMPLE_DT, eps=EPS))
    print("Convergence diagnostics: check drift, negative fractions and dR1 across settings")
    for (r, k), sub in summary.groupby(["r", "kappa"]):
        print(f"(r,k)=({r},{k}):")
        for _, row in sub.iterrows():
            print(f"  dt={row['dt']:g} tr={row['T_transient']:g} meas={row['T_measure']:g} "
                  f"D={row['D_mean']:.5g} f-={row['f_minus_mean']:.5g} "
                  f"dR1={row['dR1_mean']:.5g} peak={row['peak_freq_mean']:.5g}")
def mode_precision(args):
    pts = PRECISION_POINTS if not args.smoke else PRECISION_POINTS[:1]
    N = PRECISION_N if not args.smoke else 40
    B = PRECISION_TRIALS if not args.smoke else 2
    settings = ((DT, T_TRANSIENT, T_MEASURE),
                (DT / 2.0, T_TRANSIENT, T_MEASURE),
                (DT / 10.0, T_TRANSIENT, T_MEASURE))
    if args.smoke:
        settings = ((DT, 10.0, 10.0), (DT / 2.0, 10.0, 10.0),
                    (DT / 10.0, 10.0, 10.0))
    jobs = [(float(r), np.array([k]), N, t, tr, ms, dt,
             args.seed, 7, args.disorder)
            for dt, tr, ms in settings for r, k in pts for t in range(B)]
    _, summary = run_pool(job_batch, jobs, args, "precision",
                          ["r", "kappa", "N", "dt", "disorder_scheme"],
                          dict(points=pts, N=N, trials=B, settings=settings,
                               sample_dt=SAMPLE_DT, eps=EPS))
    for (r, k), sub in summary.groupby(["r", "kappa"]):
        for _, row in sub.iterrows():
            print(f"precision (r,k)=({r},{k}) dt={row['dt']:g} "
                  f"D={row['D_mean']:.6g} dR1={row['dR1_mean']:.6g}")
def mode_oscillation(args):
    pts = OSC_POINTS if not args.smoke else OSC_POINTS[:2]
    sizes = OSC_TRIALS if not args.smoke else {40: 2, 80: 2}
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = [(float(r), np.array([k]), N, t, T_tr, T_ms, DT, args.seed, 4, args.disorder)
            for r, k in pts for N, B in sizes.items() for t in range(B)]
    jobs.sort(key=lambda j: -j[2])
    run_pool(job_batch, jobs, args, "oscillation", ["r", "kappa", "N", "disorder_scheme"],
             dict(points=pts, sizes=sizes, transient=T_tr, measure=T_ms,
                  dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
def mode_oscillation_scan(args):
    r_vals = OSC_SCAN_R if not args.smoke else (4.24,)
    k_vals = OSC_SCAN_K if not args.smoke else np.array([2.5, 2.7, 2.9])
    sizes = OSC_SCAN_NTRIALS if not args.smoke else {40: 2, 80: 2}
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = [(float(r), k_vals, N, t, T_tr, T_ms, DT, args.seed, 6, args.disorder)
            for r in r_vals for N, B in sizes.items() for t in range(B)]
    jobs.sort(key=lambda j: -j[2] * len(j[1]))
    run_pool(job_batch, jobs, args, "oscillation_scan",
             ["r", "kappa", "N", "disorder_scheme"],
             dict(r=r_vals, k=k_vals, sizes=sizes, transient=T_tr, measure=T_ms,
                  dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
def mode_disorder_check(args):
    pts = DISORDER_CHECK_POINTS if not args.smoke else DISORDER_CHECK_POINTS[:1]
    sizes = DISORDER_CHECK_N if not args.smoke else (40, 80)
    B = DISORDER_CHECK_TRIALS if not args.smoke else 2
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = [(float(r), np.array([k]), N, t, T_tr, T_ms, DT, args.seed, 5, scheme)
            for scheme in ("iid", "antithetic") for r, k in pts for N in sizes for t in range(B)]
    run_pool(job_batch, jobs, args, "disorder_check",
             ["r", "kappa", "N", "disorder_scheme"],
             dict(points=pts, sizes=sizes, trials=B, schemes=("iid", "antithetic"),
                  transient=T_tr, measure=T_ms, dt=DT,
                  sample_dt=SAMPLE_DT, eps=EPS))
def _trace_indices(a: np.ndarray, count: int = 20):
    N = len(a)
    indices = np.round(np.linspace(0, N - 1, min(count // 2, N))).astype(int).tolist()
    ordered = np.argsort(a)
    indices += ordered[np.linspace(0, N - 1, min(count, N)).astype(int)].tolist()
    return np.array(sorted(set(indices)), dtype=int)
def job_trace(job):
    r, k, N, trial, T_tr, T_ms, dt, seed, scheme = job
    a, theta0 = initial_conditions(seed, r, N, trial, scheme)
    th = integrate(theta0[None, :], a[None, :], np.array([[k]]), T_tr, dt)
    trace_idx = _trace_indices(a)
    res = measure(th, a[None, :], np.array([[k]]), T_ms, dt,
                  capture=True, tracer_indices=trace_idx)
    cap = res["capture"]
    result = _base_row(r, k, N, trial, dt, T_tr, T_ms, seed, scheme)
    for ob in OBS:
        result[ob] = float(res[ob][0])
    return result, a, cap
def _save_trace(path: Path, row: dict, a: np.ndarray, cap: dict):
    path.mkdir(parents=True, exist_ok=True)
    t = cap["time"]
    z1 = cap["z1"][:, 0]
    z2 = cap["z2"][:, 0]
    df = pd.DataFrame(dict(tau=t, Z1_Re=z1.real, Z1_Im=z1.imag,
                           R1=np.abs(z1), psi1_unwrapped=np.unwrap(np.angle(z1)),
                           Z2_Re=z2.real, Z2_Im=z2.imag, R2=np.abs(z2),
                           psi2_unwrapped=np.unwrap(np.angle(z2))))
    _atomic_csv(path / "order_parameters.csv", df)
    xx1 = np.abs(z1)
    xx2 = np.abs(z2)
    _, _, _, f, P1 = _spectrum(xx1[:, None], SAMPLE_DT, hann=True)
    _, _, _, _, P2 = _spectrum(xx2[:, None], SAMPLE_DT, hann=True)
    _atomic_csv(path / "spectra.csv", pd.DataFrame(dict(frequency_cycles_per_tau=f,
                                                          R1_hann_power=P1[:, 0],
                                                          R2_hann_power=P2[:, 0])))
    tidx = cap["tracer_indices"]
    traces = pd.DataFrame(cap["theta_selected"][:, 0, :], columns=[f"theta_{i}" for i in tidx])
    traces.insert(0, "tau", t)
    _atomic_csv(path / "sampled_unwrapped_phases.csv", traces)
    _atomic_csv(path / "oscillator_diagnostics.csv",
                pd.DataFrame(dict(index=np.arange(len(a)), a=a,
                                  omega_lsq=cap["omega_lsq"][0],
                                  omega_first_half=cap["omega_first"][0],
                                  omega_second_half=cap["omega_second"][0],
                                  omega_endpoint=cap["omega_endpoint"][0],
                                  net_revolutions=cap["winding"][0],
                                  theta_initial=cap["theta0"][0],
                                  theta_final=cap["theta_end"][0])))
    fd, temp = tempfile.mkstemp(prefix="_partial_", suffix=".npz", dir=path)
    os.close(fd)
    try:
        np.savez_compressed(temp, time=t, Z1=z1, Z2=z2, disorder=a,
                            tracer_indices=tidx,
                            theta_selected=cap["theta_selected"][:, 0, :],
                            omega_lsq=cap["omega_lsq"][0],
                            omega_first_half=cap["omega_first"][0],
                            omega_second_half=cap["omega_second"][0],
                            winding=cap["winding"][0],
                            theta_initial=cap["theta0"][0],
                            theta_final=cap["theta_end"][0])
        os.replace(temp, path / "trajectory.npz")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    _atomic_text(path / "observables.json", json.dumps(_serializable(row), indent=2) + "\n")
def mode_traces(args):
    pts = TRACE_POINTS if not args.smoke else TRACE_POINTS[:1]
    sizes = TRACE_N if not args.smoke else (40,)
    trial_counts = TRACE_TRIALS if not args.smoke else {n: 1 for n in sizes}
    T_tr, T_ms = (T_TRANSIENT, T_MEASURE) if not args.smoke else (10.0, 10.0)
    jobs = [(float(r), float(k), N, t, T_tr, T_ms, DT, args.seed, args.disorder)
            for r, k in pts for N in sizes for t in range(trial_counts[N])]
    directory = _prepare_dir("traces", args, dict(points=pts, sizes=sizes,
                 trials_per_N=trial_counts, transient=T_tr, measure=T_ms,
                 dt=DT, sample_dt=SAMPLE_DT, eps=EPS))
    for i, job in enumerate(jobs):
        target = directory / f"trace_{i:04d}"
        if (target / "trajectory.npz").exists() and (target / "observables.json").exists():
            continue
        row, a, cap = job_trace(job)
        _save_trace(target, row, a, cap)
        print(f"traces: saved {i+1}/{len(jobs)} (r={row['r']}, k={row['kappa']}, N={row['N']})", flush=True)
    records = [json.loads((directory / f"trace_{i:04d}" / "observables.json").read_text())
               for i in range(len(jobs))]
    df = pd.DataFrame(records)
    _atomic_csv(directory / "trials.csv", df)
    _atomic_csv(directory / "summary.csv", summarise(df, ["r", "kappa", "N", "disorder_scheme"]))
    _atomic_text(directory / "COMPLETE.json", json.dumps(dict(jobs=len(jobs), rows=len(df)), indent=2) + "\n")
def check_production_protocol():
    production_validate()
    if not np.isclose(HYST_SETTLE, T_TRANSIENT) or not np.isclose(HYST_MEASURE, T_MEASURE) or HYST_INITIAL != 0:
        raise RuntimeError("Hysteresis observation windows are not standardized")
    if HYST_REALIZATIONS != TRIALS_BY_N[HYST_N]:
        raise RuntimeError("Hysteresis trial count mismatch")
    if MAP_TRIALS != TRIALS_BY_N[MAP_N]:
        raise RuntimeError("Map trial count mismatch")
    if CUT_TRIALS != TRIALS_BY_N or OSC_TRIALS != TRIALS_BY_N:
        raise RuntimeError("Finite-size trial count mismatch")
    if any(OSC_SCAN_NTRIALS[n] != TRIALS_BY_N[n] for n in OSC_SCAN_N):
        raise RuntimeError("Scan trial count mismatch")
    if any(TRACE_TRIALS[n] != TRIALS_BY_N[n] for n in TRACE_N):
        raise RuntimeError("Trace trial count mismatch")
    print(f"Shared production protocol valid: RK4, 300+300, grid {GRID_STEP:g}, matched trial counts")

def self_test():
    check_production_protocol()
    for scheme in ("iid", "antithetic"):
        a, th = initial_conditions(SEED, 4.5, 40, 0, scheme)
        a2, th2 = initial_conditions(SEED, 4.5, 40, 0, scheme)
        assert np.array_equal(a, a2) and np.array_equal(th, th2)
        assert a.shape == (40,) and th.shape == (40,)
        if scheme == "antithetic":
            assert abs(a.mean()) < 1e-12
    th = np.arange(12) * 0.37
    k = np.array([[2.0]])
    v = rhs(th[None, :], np.zeros((1, 12)), k)
    v_rot = rhs((th + 0.81)[None, :], np.zeros((1, 12)), k)
    assert np.allclose(v, v_rot)
    a = np.zeros((1, 40))
    v = np.zeros((1, 40))
    z = measure(v, a, np.array([[0.0]]), 10.0, DT)
    assert abs(z["D"][0] - 1.0) < 1e-10
    assert z["f_plus"][0] == 1 and z["back_rev"][0] == 0
    theta = np.linspace(-3, 3, 40)
    K = 4.0
    direct = K * np.sin(theta[None, :] - theta[:, None]).sum(axis=1) / len(theta)
    assert np.allclose(rhs(theta[None, :], np.zeros((1, 40)), np.array([[K]]))[0] - 1, direct)
    print("self-test passed: deterministic seeds, iid/antithetic schemes, RK4 model, "
          "drift estimator, Kuramoto normalization")
def main():
    parser = argparse.ArgumentParser(description="Production active-rotator simulations, checkpointed")
    parser.add_argument("--mode", required=True,
                        choices=["map", "cuts", "hysteresis", "convergence", "oscillation",
                                 "oscillation_scan", "precision", "disorder_check", "traces", "self_test"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--disorder", choices=["iid", "antithetic"], default="antithetic",
                        help="antithetic enforces finite-N sign symmetry (referee m12); iid is a control")
    parser.add_argument("--tag", default="", help="Independent output directory suffix")
    args = parser.parse_args()
    if args.mode not in ("self_test", "convergence", "precision", "disorder_check") and (args.seed != SEED or args.disorder != "antithetic"):
        parser.error("Production mode requires shared seed and antithetic disorder")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.tag and (not args.tag.replace("-", "").replace("_", "").isalnum()):
        parser.error("--tag may contain only letters, numbers, hyphen, underscore")
    if args.mode == "self_test":
        self_test()
        return
    modes = {"map": mode_map, "cuts": mode_cuts, "hysteresis": mode_hysteresis,
             "convergence": mode_convergence, "oscillation": mode_oscillation,
             "oscillation_scan": mode_oscillation_scan,
             "precision": mode_precision, "disorder_check": mode_disorder_check, "traces": mode_traces}
    check_production_protocol()
    modes[args.mode](args)
    print(f"Mode {args.mode}: completed. Output: {_run_dir(args.mode, args)}", flush=True)
if __name__ == "__main__":
    mp.freeze_support()
    main()
