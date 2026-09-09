#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SIMULATORS = ROOT / "scripts" / "simulators"

SCRIPTS = {
    "grid": "grid_sweep.py",
    "order": "order_parameter_sweep.py",
    "representative": "representative_dynamics.py",
}


def run_simulation(name, jobs=None):
    cmd = [sys.executable, str(SIMULATORS / SCRIPTS[name])]

    if jobs is not None:
        cmd += ["--jobs", str(jobs)]

    print(f"\nRunning {name} simulation...")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation datasets used by Figures 3–6."
    )
    parser.add_argument(
        "--only",
        choices=["all", "grid", "order", "representative"],
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=None)

    args = parser.parse_args()

    targets = (
        ["grid", "order", "representative"]
        if args.only == "all"
        else [args.only]
    )

    for target in targets:
        run_simulation(target, args.jobs)


if __name__ == "__main__":
    main()
