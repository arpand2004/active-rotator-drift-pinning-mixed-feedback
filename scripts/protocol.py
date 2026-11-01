from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_OUTPUT = ROOT / "data" / "simulations"
THEORY_OUTPUT = ROOT / "data" / "theory"
COMPARE_OUTPUT = ROOT / "data" / "comparisons"
CHECK_OUTPUT = ROOT / "data" / "checks"
DT = 0.02
SAMPLE_DT = 0.2
T_TRANSIENT = 300.0
T_MEASURE = 300.0
GRID_STEP = 0.05
SEED = 55500
EPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
TRIALS_BY_N = {100: 64, 200: 24, 400: 24, 800: 16, 1600: 12, 3200: 8, 6400: 6}
MAP_R = np.round(np.arange(0.0, 6.0 + GRID_STEP / 2, GRID_STEP), 6)
MAP_K = np.round(np.arange(0.0, 8.0 + GRID_STEP / 2, GRID_STEP), 6)
MAP_N = 200
CUT_SPECS = {2.0: (0.0, 6.0), 4.0: (0.0, 8.0), 4.5: (0.0, 8.0), 5.0: (0.0, 8.0), 6.0: (2.0, 8.0)}
HYST_R = (4.25, 4.5, 5.0, 6.0)
HYST_K = np.round(np.arange(3.0, 8.0 + GRID_STEP / 2, GRID_STEP), 6)
HYST_N = 3200
OSC_POINTS = ((4.24, 2.73), (4.5, 4.6), (6.0, 6.0), (2.0, 2.0))
OSC_SCAN_R = (2.0, 4.0, 4.24, 4.25, 4.5, 5.0, 6.0)
OSC_SCAN_K = MAP_K
OSC_SCAN_N = (200, 800, 3200)
TRACE_POINTS = OSC_POINTS
TRACE_N = (800, 3200)
CONV_POINTS = OSC_POINTS
CONV_N = 800
CONV_TRIALS = 16
PRECISION_POINTS = OSC_POINTS
PRECISION_N = 400
PRECISION_TRIALS = 4
DISORDER_CHECK_POINTS = ((4.24, 2.73), (4.5, 4.6), (6.0, 6.0))
DISORDER_CHECK_N = (200, 800, 3200)
DISORDER_CHECK_TRIALS = 12
MF_R_MIN = 3.8
MF_R_MAX = 6.0
MF_EXTRA_R = (4.25,)
MF_K_MIN = 0.0
MF_K_MAX = 8.0

def production_validate():
    if (not np.isclose(DT, 0.02) or not np.isclose(T_TRANSIENT, 300) or not np.isclose(T_MEASURE, 300)
        or not np.isclose(SAMPLE_DT, 0.2)):
        raise RuntimeError("Production time protocol changed")
    if not np.isclose(SAMPLE_DT / DT, round(SAMPLE_DT / DT)):
        raise RuntimeError("Sampling step is not a multiple of RK4 step")
    if any(n % 2 for n in TRIALS_BY_N):
        raise RuntimeError("Antithetic disorder requires even N")
    for x in (MAP_R, MAP_K, HYST_K, OSC_SCAN_K):
        if not np.allclose(np.diff(x), GRID_STEP, rtol=0, atol=1e-9):
            raise RuntimeError("Nonuniform production grid")
    if not np.allclose(np.round(np.asarray(list(CUT_SPECS)), 6) / GRID_STEP, np.round(np.asarray(list(CUT_SPECS)) / GRID_STEP), atol=1e-8):
        raise RuntimeError("Cut values not aligned with the production grid")
    if not set(HYST_R).issubset(set(OSC_SCAN_R)):
        raise RuntimeError("Missing hysteresis disorder in scan")
    if not set(HYST_K).issubset(set(OSC_SCAN_K)):
        raise RuntimeError("Missing hysteresis coupling in scan")
    if HYST_N not in OSC_SCAN_N:
        raise RuntimeError("Missing hysteresis system size in scan")
    if MAP_N not in TRIALS_BY_N:
        raise RuntimeError("Missing map sample count")
    return True
