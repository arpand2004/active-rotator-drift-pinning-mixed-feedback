import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import ROOT, DT, T_TRANSIENT, T_MEASURE, SEED
from simulate import job_trace, _save_trace

target = ROOT / "data/simulations/traces/figure_support/trace_r4p5_k3p3_N3200_trial0"
previous = ROOT / "data/figure_support/fig5_r4p5_k3p3_N3200_trial0"
required = ("observables.json", "order_parameters.csv", "oscillator_diagnostics.csv",
            "sampled_unwrapped_phases.csv", "spectra.csv", "trajectory.npz")

if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    if previous.is_dir() and all((previous / name).is_file() for name in required):
        shutil.copytree(previous, target)
        print("Copied previously generated trace into the permanent traces folder")
    else:
        row, disorder, capture = job_trace((4.5, 3.3, 3200, 0, T_TRANSIENT,
                                            T_MEASURE, DT, SEED, "antithetic"))
        _save_trace(target, row, disorder, capture)
        print("Generated and saved one deterministic support trace")

if not all((target / name).is_file() for name in required):
    raise RuntimeError("Support trace folder is incomplete")

meta = json.loads((target / "observables.json").read_text())

for key, expected in (("r", 4.5), ("kappa", 3.3), ("N", 3200), ("trial", 0),
                      ("seed", SEED), ("dt", DT), ("T_transient", T_TRANSIENT),
                      ("T_measure", T_MEASURE)):
    if not np.isclose(float(meta[key]), expected, atol=1e-10, rtol=0):
        raise RuntimeError(f"Saved support trace has incorrect {key}")

if meta["disorder_scheme"] != "antithetic":
    raise RuntimeError("Incorrect disorder scheme")

fields = ["r", "kappa", "N", "trial", "D", "R1", "R2", "dR1", "peak_freq_hann", "psi_rate"]
trials = pd.read_csv(ROOT / "data/simulations/cuts/trials.csv", usecols=fields)
match = trials[np.isclose(trials.r, 4.5, atol=1e-9)
               & np.isclose(trials.kappa, 3.3, atol=1e-9)
               & (trials.N == 3200) & (trials.trial == 0)]

if len(match) != 1:
    raise RuntimeError("Expected exactly one matching audited cuts row")

for key in fields[4:]:
    if not np.isclose(float(meta[key]), float(match.iloc[0][key]),
                      atol=1e-8, rtol=1e-7):
        raise RuntimeError(f"Saved trace disagrees with audited cuts data: {key}")

print("Verified the complete saved trace against the audited cuts data")
print(target)
