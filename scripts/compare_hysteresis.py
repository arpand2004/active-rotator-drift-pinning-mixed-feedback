import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import DT, T_TRANSIENT, T_MEASURE, SEED, COMPARE_OUTPUT, production_validate
KEY = ['r', 'kappa', 'N', 'trial', 'seed', 'disorder_scheme']
MEASURES = ('D', 'R1', 'R2', 'dR1', 'dR2', 'peak_freq_hann')
def read_checked(path, label):
    data = pd.read_csv(path)
    required = set(KEY) | set(MEASURES) | {'dt', 'T_transient', 'T_measure'}
    if label == 'hyst':
        required |= {'direction', 'T_settle', 'T_initial'}
    missing = required - set(data)
    if missing:
        raise ValueError(f'{label}: missing columns: {sorted(missing)}')
    for field, expected in [('dt', DT), ('T_transient', T_TRANSIENT), ('T_measure', T_MEASURE)]:
        if not np.all(np.isclose(data[field].to_numpy(),expected,rtol=0,atol=1e-9)):
            raise ValueError(f'{label}: {field} != {expected} in one or more rows; refusing mixed protocol')
    if label == 'hyst':
        if not np.all(np.isclose(data.T_settle,T_TRANSIENT)) or not np.all(np.isclose(data.T_initial,0)):
            raise ValueError('hyst: settling or initial preparation differs from audited standard')
    if set(data.disorder_scheme) != {'antithetic'}:
        raise ValueError(f'{label}: expected only antithetic disorder, got {set(data.disorder_scheme)}')
    if set(data.seed) != {SEED}:
        raise ValueError(f'{label}: expected seed=55500; got {set(data.seed)}')
    for col in ['r','kappa']:
        data[col] = data[col].round(6)
    uniqueness = KEY + (['direction'] if label == 'hyst' else [])
    dup = data.duplicated(uniqueness)
    if dup.any():
        raise ValueError(f'{label}: {dup.sum()} duplicate exact-realization keys')
    return data
def paired_comparison(h, s):
    if set(h.direction) != {'up', 'down'}:
        raise ValueError('Hysteresis must include both up and down directions')
    if not set(h.N).issubset(set(s.N)):
        raise ValueError('Scan missing hysteresis N')
    pairs = h.merge(s, on=KEY, how='left', validate='many_to_one', suffixes=('_hyst', '_scan'), indicator=True)
    unmatched = pairs._merge.ne('both')
    if unmatched.any():
        raise ValueError(f'{unmatched.sum()} hysteresis records lack identical scan realization IDs; refusing unequal ensembles')
    pairs = pairs.drop(columns=['_merge']).copy()
    measures = ('D', 'R1', 'R2', 'dR1', 'dR2', 'peak_freq_hann')
    for measure in measures:
        pairs[measure + '_difference'] = pairs[measure + '_hyst'] - pairs[measure + '_scan']
    keys = ['r', 'kappa', 'N', 'direction']
    agg = {'matched_trials': ('trial', 'size')}
    for x in ('D', 'R1', 'R2', 'dR1'):
        agg[x + '_gap_mean'] = (x + '_difference', 'mean')
        agg[x + '_gap_sem'] = (x + '_difference', 'sem')
        agg[x + '_hyst_mean'] = (x + '_hyst', 'mean')
        agg[x + '_scan_mean'] = (x + '_scan', 'mean')
    stats = pairs.groupby(keys, sort=True).agg(**agg).reset_index()
    expected = h.groupby(keys).size().rename('original_trials').reset_index()
    check = stats.merge(expected, on=keys, validate='one_to_one')
    if not (check.matched_trials == check.original_trials).all():
        raise ValueError('Comparison lost directional trials')
    directional = h.pivot(index=KEY, columns='direction', values=['D', 'R1', 'R2', 'dR1'])
    if directional.isna().any().any():
        raise ValueError('Hysteresis up/down have unmatched disorder realizations')
    directional.columns = [str(a) + '_' + str(b) for a, b in directional.columns]
    directional = directional.reset_index()
    for x in ('D', 'R1', 'R2', 'dR1'):
        directional[x + '_up_minus_down'] = directional[x + '_up'] - directional[x + '_down']
    return pairs, stats, directional

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hysteresis',type=Path,required=True)
    parser.add_argument('--scan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    h=read_checked(args.hysteresis,'hyst')
    s=read_checked(args.scan,'scan')
    production_validate()
    paired,summary,directional=paired_comparison(h,s)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Comparison folder is not empty; refusing overwrite: {args.output}")
    args.output.mkdir(parents=True,exist_ok=True)
    paired.to_csv(args.output/'paired_trials.csv',index=False)
    summary.to_csv(args.output/'paired_summary.csv',index=False)
    directional.to_csv(args.output/'up_down_paired.csv',index=False)
    print(f'EXACT PAIRING: {len(paired)} directional trial rows, '
          f'{len(summary)} directional parameter groups; '\
          f'{paired.trial.nunique()} distinct realization IDs.')
    print('Comparison uses identical disorder, seed, (r,kappa,N,trial) and 300+300 windows; '
          'differences reflect independent initialization versus history-dependent sweeps.')
    print('Saved:',args.output)
if __name__=='__main__':
    main()
