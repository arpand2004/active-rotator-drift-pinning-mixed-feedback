from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from protocol import MAP_K, SEED, DT, T_TRANSIENT, T_MEASURE, CHECK_OUTPUT, GRID_STEP, production_validate
from simulate import OBS, initial_conditions, integrate, job_hysteresis, job_batch, check_production_protocol
from mf_theory import h1

def origin_linearization(r, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    def piece(lo, hi):
        aa = (hi + lo) / 2.0 + (hi - lo) / 2.0 * nodes
        ww = (hi - lo) / 2.0 * weights
        gg = np.exp(-0.5 * (aa / r) ** 2) / (np.sqrt(2.0 * np.pi) * r)
        hh = h1(aa)
        return (np.dot(ww, aa * gg * hh), np.dot(ww, gg * hh / aa))
    p, q = piece(0.0, 1.0)
    p2, q2 = piece(1.0, 12.0 * r)
    dx = 2.0 * (p + p2) / (r * r)
    dy = 2.0j * (q + q2)
    M = np.array([[dx.real, dy.real], [dx.imag, dy.imag]])
    delta = (M[0, 0] - M[1, 1]) ** 2 + 4.0 * M[0, 1] * M[1, 0]
    return float(delta), M

def batch_equality():
    a, th0 = initial_conditions(SEED, 4.24, 200, 0, 'antithetic')
    j = int(np.argmin(abs(MAP_K - 2.75)))
    multi = integrate(np.repeat(th0[None, :], len(MAP_K), axis=0), a[None, :], MAP_K[:, None], 150.0, DT)
    one = integrate(th0[None, :].copy(), a[None, :], np.array([[MAP_K[j]]]), 150.0, DT)
    error = float(np.max(np.abs(multi[j] - one[0])))
    if error != 0.0:
        raise RuntimeError(f'Batched integration differs from single integration: {error:.17g}')
    return error

def hysteresis_initial_match():
    r, N, trial = 4.5, 40, 0
    ks = np.array([3.0, 3.0 + GRID_STEP])
    hj = (r, 'up', trial, N, ks, 0.0, T_TRANSIENT, T_MEASURE, DT, SEED, 'antithetic')
    h = job_hysteresis(hj)[0]
    bj = (r, np.array([3.0]), N, trial, T_TRANSIENT, T_MEASURE, DT, SEED, 0, 'antithetic')
    b = job_batch(bj)[0]
    err = float(max(abs(h[c] - b[c]) for c in OBS))
    if err != 0.0:
        raise RuntimeError(f'First hysteresis point differs from independent initialization: {err:.17g}')
    return err

def main():
    production_validate()
    check_production_protocol()
    d = []
    for r in np.round(np.arange(3.8, 6.0000001, GRID_STEP), 6):
        delta, M = origin_linearization(float(r), 256)
        finer, _ = origin_linearization(float(r), 512)
        if delta >= 0.0 or finer >= 0.0 or abs(finer - delta) > 2e-4:
            raise RuntimeError(f'Zero-branch linearization failed at r={r}: {delta}, {finer}')
        d.append(dict(r=float(r), discriminant=finer, M11=float(M[0, 0]), M12=float(M[0, 1]), M21=float(M[1, 0]), M22=float(M[1, 1])))
    CHECK_OUTPUT.mkdir(parents=True, exist_ok=True)
    p = CHECK_OUTPUT / 'zero_linearization.csv'
    report_path = CHECK_OUTPUT / 'preflight_report.json'
    error = batch_equality()
    herr = hysteresis_initial_match()
    pd.DataFrame(d).to_csv(p, index=False, float_format='%.16g')
    hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ('simulate.py', 'mf_theory.py', 'protocol.py', 'compare_hysteresis.py', 'verify_all.py', 'preflight.py')}
    report = dict(source_sha256=hashes, batch_max_absolute_error=error, initial_hysteresis_max_absolute_error=herr,
                  max_zero_branch_discriminant=max(x['discriminant'] for x in d),
                  min_zero_branch_discriminant=min(x['discriminant'] for x in d),
                  r_values=len(d), time_protocol=[DT,T_TRANSIENT,T_MEASURE],
                  status='passed')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
