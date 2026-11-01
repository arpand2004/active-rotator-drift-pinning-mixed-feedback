set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python3 -u scripts/simulate.py --mode self_test
python3 -u scripts/preflight.py
for mode in map cuts hysteresis oscillation oscillation_scan traces convergence precision disorder_check; do
    python3 -u scripts/simulate.py --mode "$mode" --smoke --jobs 2
done
python3 -u scripts/mf_theory.py --check --jobs 2
python3 -u scripts/simulate.py --mode convergence --jobs 2
python3 -u scripts/simulate.py --mode precision --jobs 2
python3 -u scripts/simulate.py --mode disorder_check --jobs 2
printf 'Preflight computations completed. Review the convergence, precision and disorder-check summaries before running run_all.sh.\n'
