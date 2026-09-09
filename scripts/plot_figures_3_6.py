#!/usr/bin/env python3


from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTTERS = REPO_ROOT / "scripts" / "plotters"


def run(script, *args):
    path = PLOTTERS / script
    print(f"\n{'=' * 68}")
    print(f"Running {script}")
    print("=" * 68)

    subprocess.run(
        [sys.executable, str(path), *args],
        cwd=REPO_ROOT,
        check=True,
    )


def main():


    run("figure_3.py")


    run("figure_4.py")


    run("figure_5.py")

    run("figure_6.py")

    print("\nAll manuscript Figures 3–6 regenerated successfully.")


if __name__ == "__main__":
    main()
