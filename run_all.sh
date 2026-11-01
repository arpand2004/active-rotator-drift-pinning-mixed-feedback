set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 - <<'PY'
from pathlib import Path
import json
root = Path('data/checks')
report = root / 'preflight_report.json'
if not report.exists():
    raise RuntimeError('First run preflight.sh and inspect its completed diagnostics')
info = json.loads(report.read_text())
if info.get('status') != 'passed':
    raise RuntimeError('Preflight did not pass')
import hashlib
for name, checksum in info['source_sha256'].items():
    current = hashlib.sha256((Path('scripts') / name).read_bytes()).hexdigest()
    if current != checksum:
        raise RuntimeError(f'Source changed after preflight: {name}')
for mode in ('convergence', 'precision', 'disorder_check'):
    directory = Path('data/simulations') / mode
    if not (directory / 'COMPLETE.json').exists():
        raise RuntimeError(f'Missing completed {mode} control results; run preflight.sh first')
    manifest = json.loads((directory / 'manifest.json').read_text())['config']
    if manifest['script_sha256'] != info['source_sha256']['simulate.py'] or manifest['protocol_sha256'] != info['source_sha256']['protocol.py']:
        raise RuntimeError(f'Control result for {mode} belongs to different source')
PY
python3 -u scripts/mf_theory.py --jobs 2
python3 -u scripts/simulate.py --mode hysteresis --jobs 2
python3 -u scripts/simulate.py --mode oscillation_scan --jobs 2
python3 -u scripts/compare_hysteresis.py --hysteresis data/simulations/hysteresis/trials.csv --scan data/simulations/oscillation_scan/trials.csv --output data/comparisons/hysteresis_vs_scan
python3 -u scripts/simulate.py --mode cuts --jobs 2
python3 -u scripts/simulate.py --mode oscillation --jobs 2
python3 -u scripts/simulate.py --mode map --jobs 2
python3 -u scripts/simulate.py --mode traces --jobs 2
python3 -u scripts/verify_all.py
