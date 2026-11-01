from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import ROOT, DT, T_TRANSIENT, T_MEASURE, SEED, TRIALS_BY_N
from simulate import job_batch

POINTS = {
    4.0: (0.0, 2.95),
    4.5: (3.0, 3.3, 6.0, 8.0),
    5.0: (3.5, 6.0),
    6.0: (3.85, 6.0),
}

OUT = ROOT / "data" / "validations" / "iid_targeted_v3"
FIELDS_TO_CHECK = (
    "D", "R1", "R2", "dR1", "peak_freq_hann", "psi_rate"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planned_jobs(smoke=False):
    sizes = {40: 1} if smoke else TRIALS_BY_N
    points = {4.5: (3.3,)} if smoke else POINTS
    items = []

    for r, kappas in points.items():
        for n, count in sizes.items():
            for trial in range(count):
                job = (
                    float(r),
                    np.array(kappas, dtype=float),
                    int(n),
                    int(trial),
                    4.0 if smoke else T_TRANSIENT,
                    4.0 if smoke else T_MEASURE,
                    DT,
                    SEED,
                    9,
                    "iid",
                )
                items.append(job)

    items.sort(key=lambda job: (-job[2] * len(job[1]), job[0], job[3]))
    return items


def _work(item):
    i, job = item
    return i, job_batch(job)


def validate_baseline():
    source = ROOT / "data" / "simulations" / "cuts" / "trials.csv"

    if not source.exists():
        raise RuntimeError(f"Missing V3 cuts data: {source}")

    cols = ["r", "kappa", "N", "trial", *FIELDS_TO_CHECK]
    df = pd.read_csv(source, usecols=cols)

    rows = df[
        np.isclose(df.r, 4.5)
        & np.isclose(df.kappa, 3.3)
        & (df.N == 100)
        & (df.trial == 0)
    ]

    if len(rows) != 1:
        raise RuntimeError("Cannot locate the unique baseline trial")

    job = (
        4.5, np.array([3.3]), 100, 0,
        T_TRANSIENT, T_MEASURE, DT, SEED, 9, "antithetic"
    )

    calculated = job_batch(job)[0]

    for field in FIELDS_TO_CHECK:
        original = float(rows.iloc[0][field])
        reproduced = float(calculated[field])

        if not np.isclose(
            original, reproduced, rtol=1e-10, atol=1e-10
        ):
            raise RuntimeError(
                f"Baseline mismatch in {field}: "
                f"{original} versus {reproduced}"
            )

    print(
        "PASS: current solver reproduces six original "
        "production observables",
        flush=True,
    )


def run(args):
    jobs = planned_jobs(args.smoke)

    dest = (
        OUT.with_name("iid_targeted_v3_smoke")
        if args.smoke else OUT
    )

    chunks = dest / "_chunks"
    manifest_path = dest / "manifest.json"

    points = {4.5: (3.3,)} if args.smoke else POINTS

    manifest = {
        "validation": "Targeted independent Gaussian V3 comparison",
        "source_sha256": digest(ROOT / "scripts" / "simulate.py"),
        "protocol_sha256": digest(ROOT / "scripts" / "protocol.py"),
        "driver_sha256": digest(Path(__file__)),
        "points": {str(k): list(v) for k, v in points.items()},
        "trials_by_N": (
            {"40": 1}
            if args.smoke
            else {str(k): v for k, v in TRIALS_BY_N.items()}
        ),
        "seed": SEED,
        "dt": DT,
        "T_transient": 4.0 if args.smoke else T_TRANSIENT,
        "T_measure": 4.0 if args.smoke else T_MEASURE,
        "disorder_scheme": "iid",
        "jobs": len(jobs),
    }

    total = sum(len(job[1]) for job in jobs)

    print(f"Output directory: {dest}", flush=True)
    print(f"Jobs: {len(jobs)}", flush=True)
    print(f"Point-realizations: {total}", flush=True)

    if not args.run:
        print(
            "PLAN ONLY: no simulations performed.",
            flush=True,
        )
        return

    if not args.smoke and not args.skip_baseline:
        validate_baseline()

    chunks.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())

        if previous != manifest:
            raise RuntimeError(
                "Manifest differs from current source or settings. "
                "Refusing to mix datasets."
            )
    else:
        if list(chunks.glob("job_*.csv")):
            raise RuntimeError(
                "Checkpoint files exist without a manifest."
            )

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    missing = [
        (i, job)
        for i, job in enumerate(jobs)
        if not (chunks / f"job_{i:06d}.csv").exists()
    ]

    print(
        f"{len(jobs) - len(missing)} jobs already checkpointed; "
        f"{len(missing)} remaining",
        flush=True,
    )

    if missing:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(_work, item)
                for item in missing
            ]

            for finished, future in enumerate(
                as_completed(futures), 1
            ):
                i, records = future.result()

                part = chunks / f"job_{i:06d}.csv"
                temporary = chunks / f"job_{i:06d}.partial"

                pd.DataFrame.from_records(records).to_csv(
                    temporary, index=False
                )
                temporary.replace(part)

                if (
                    finished == 1
                    or finished % 20 == 0
                    or finished == len(missing)
                ):
                    print(
                        f"Completed {finished}/{len(missing)} new jobs",
                        flush=True,
                    )

    pieces = []

    for i in range(len(jobs)):
        path = chunks / f"job_{i:06d}.csv"

        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint: {path}")

        part = pd.read_csv(path)
        part.insert(0, "validation_job_id", i)
        pieces.append(part)

    trials = pd.concat(pieces, ignore_index=True)

    if len(trials) != total:
        raise RuntimeError(
            f"Incorrect row count: {len(trials)} instead of {total}"
        )

    if set(trials.disorder_scheme) != {"iid"}:
        raise RuntimeError("Unexpected disorder scheme")

    keys = ["r", "kappa", "N", "trial"]

    if trials.duplicated(keys).any():
        raise RuntimeError("Duplicate realization keys")

    expected_counts = (
        {40: 1} if args.smoke else TRIALS_BY_N
    )

    for (r, k, n), group in trials.groupby(
        ["r", "kappa", "N"]
    ):
        if len(group) != expected_counts[int(n)]:
            raise RuntimeError(
                f"Incorrect realization count at {(r, k, n)}"
            )

    tmp = dest / "trials.partial"
    trials.to_csv(tmp, index=False)
    tmp.replace(dest / "trials.csv")

    grouped = trials.groupby(
        ["r", "kappa", "N"], sort=True
    )[list(FIELDS_TO_CHECK)]

    summary = grouped.agg(["mean", "sem", "size"])
    summary.columns = [
        "_".join(x) for x in summary.columns
    ]

    summary.reset_index().to_csv(
        dest / "summary.csv", index=False
    )

    (dest / "COMPLETE.json").write_text(
        json.dumps(
            {
                "rows": len(trials),
                "jobs": len(jobs),
                "source_sha256": manifest["source_sha256"],
                "driver_sha256": manifest["driver_sha256"],
            },
            indent=2,
        ) + "\n"
    )

    print(
        f"PASS: saved {len(trials)} independent-Gaussian "
        f"records in {dest}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resumable targeted i.i.d. V3 validation"
    )

    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--verify-baseline", action="store_true"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true"
    )

    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    if args.verify_baseline:
        validate_baseline()
    else:
        run(args)


if __name__ == "__main__":
    main()
