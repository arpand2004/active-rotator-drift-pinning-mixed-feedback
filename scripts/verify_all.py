import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import ROOT, SIM_OUTPUT, THEORY_OUTPUT, GRID_STEP, TRIALS_BY_N, DT, SAMPLE_DT, T_TRANSIENT, T_MEASURE, SEED, MAP_R, MAP_K, MAP_N, CUT_SPECS, OSC_SCAN_R, OSC_SCAN_K, OSC_SCAN_N, HYST_R, HYST_K, HYST_N, TRACE_N, TRACE_POINTS, OSC_POINTS, production_validate

MODES = ('hysteresis', 'oscillation_scan', 'cuts', 'oscillation', 'map', 'traces')

def verify_mode(mode):
    directory = SIM_OUTPUT / mode
    if not (directory / 'COMPLETE.json').exists() or not (directory / 'manifest.json').exists():
        raise RuntimeError(f'{mode}: missing completion metadata')
    manifest = json.loads((directory / 'manifest.json').read_text())
    meta = manifest['config']
    if meta['seed'] != SEED or meta['disorder_scheme'] != 'antithetic':
        raise RuntimeError(f'{mode}: incorrect seed/disorder scheme')
    script_digest = hashlib.sha256((ROOT / 'scripts' / 'simulate.py').read_bytes()).hexdigest()
    protocol_digest = hashlib.sha256((ROOT / 'scripts' / 'protocol.py').read_bytes()).hexdigest()
    if meta['script_sha256'] != script_digest or meta['protocol_sha256'] != protocol_digest:
        raise RuntimeError(f'{mode}: manifest does not match current scripts')
    d = pd.read_csv(directory / 'trials.csv')
    c = json.loads((directory / 'COMPLETE.json').read_text())
    if len(d) != c['rows'] or len(d) == 0:
        raise RuntimeError(f'{mode}: incomplete trial data')
    for col, val in [('dt', DT), ('T_transient', T_TRANSIENT), ('T_measure', T_MEASURE), ('seed', SEED)]:
        if col not in d or not np.allclose(d[col], val, rtol=0, atol=1e-8):
            raise RuntimeError(f'{mode}: incompatible {col}')
    if not (d['disorder_scheme'] == 'antithetic').all():
        raise RuntimeError(f'{mode}: scheme mismatch')
    k = ['r', 'kappa', 'N', 'trial', 'disorder_scheme'] + (['direction'] if mode == 'hysteresis' else [])
    if d.duplicated(k).any():
        raise RuntimeError(f'{mode}: duplicate realization')
    if not np.isfinite(d.select_dtypes(include='number').to_numpy()).all():
        raise RuntimeError(f'{mode}: nonfinite numeric data')
    for n, group in d.groupby('N'):
        intended = TRIALS_BY_N[int(n)]
        sizes = group.groupby(['r','kappa'] + (['direction'] if mode == 'hysteresis' else [])).size()
        if not (sizes == intended).all():
            raise RuntimeError(f'{mode}: N={n} trial count mismatch')
    for e in (1e-4,3e-4,1e-3,3e-3,1e-2):
        if f'f_minus_{e:.0e}' not in d.columns:
            raise RuntimeError(f'{mode}: missing epsilon diagnostic {e}')
    if (np.diff(d[[f'f_minus_{e:.0e}' for e in (1e-4,3e-4,1e-3,3e-3,1e-2)]].to_numpy(),axis=1)>1e-10).any():
        raise RuntimeError(f'{mode}: epsilon fractions not monotonic')
    if (d['disorder_mean'].abs()>1e-9).any():
        raise RuntimeError(f'{mode}: antithetic symmetry broken')
    if mode == 'hysteresis':
        if set(d.direction) != {'up','down'} or not np.allclose(d.T_settle,300) or not np.allclose(d.T_initial,0):
            raise RuntimeError('Hysteresis protocol mismatch')
    if mode == 'map':
        if not (np.array_equal(np.sort(d.r.unique()), MAP_R) and np.array_equal(np.sort(d.kappa.unique()), MAP_K) and set(d.N)=={MAP_N}):
            raise RuntimeError('Map grid mismatch')
    if mode == 'oscillation_scan':
        if set(d.r) != set(OSC_SCAN_R) or not np.array_equal(np.sort(d.kappa.unique()),OSC_SCAN_K) or set(d.N)!=set(OSC_SCAN_N):
            raise RuntimeError('Oscillation scan grid mismatch')
    if mode == 'hysteresis':
        if set(d.r)!=set(HYST_R) or not np.array_equal(np.sort(d.kappa.unique()),HYST_K) or set(d.N)!={HYST_N}:
            raise RuntimeError('Hysteresis grid mismatch')
    return d

def overlap_equal(a,b,name):
    k=['r','kappa','N','trial','seed','disorder_scheme']
    m=a.merge(b,on=k,suffixes=('_a','_b'),how='inner',validate='one_to_one')
    if m.empty:
        raise RuntimeError(f'{name}: no overlapping trials to verify')
    for col in ('D','R1','R2','dR1','dR2','f_plus','f_minus','peak_freq_hann'):
        if not np.allclose(m[col+'_a'],m[col+'_b'],rtol=0,atol=1e-12):
            raise RuntimeError(f'{name}: mismatch in {col}')
    return int(len(m))

def main():
    production_validate()
    data={mode: verify_mode(mode) for mode in MODES}
    matches={}
    for a,b in [('cuts','oscillation_scan'),('cuts','map'),('oscillation_scan','map'),('oscillation','oscillation_scan'),('oscillation','traces')]:
        matches[a+'_vs_'+b]=overlap_equal(data[a],data[b],a+' vs '+b)
    if not (THEORY_OUTPUT/'COMPLETE.json').exists():
        raise RuntimeError('Mean-field theory incomplete')
    theory_root = pd.read_csv(THEORY_OUTPUT/'mf_roots_on_grid.csv')
    theory_branch = pd.read_csv(THEORY_OUTPUT/'mf_branches.csv')
    if (theory_root.n_unstable < 0).any() or (theory_branch.n_unstable < 0).any():
        raise RuntimeError('Mean-field stability is not complete across the exported stationary branches')
    output={'status':'passed', 'datasets':{mode:int(len(df)) for mode,df in data.items()}, 'identical_records_across_modes':matches, 'time_protocol':[DT,SAMPLE_DT,T_TRANSIENT,T_MEASURE], 'grid_step':GRID_STEP, 'trial_counts':TRIALS_BY_N}
    path=ROOT/'data'/'audit.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__ == '__main__':
    main()
