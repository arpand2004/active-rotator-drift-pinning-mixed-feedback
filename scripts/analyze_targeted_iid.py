from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import ROOT, TRIALS_BY_N

OUT = ROOT / "data" / "validations" / "iid_targeted_v3"
IID = OUT / "trials.csv"
PAIRED = ROOT / "data" / "simulations" / "cuts" / "trials.csv"

KEY = ["r", "kappa", "N", "trial"]
OBS = ["D", "R1", "R2", "dR1", "peak_freq_hann", "psi_rate"]

RIDGE = {
    (4.0, 2.95),
    (4.5, 3.3),
    (5.0, 3.5),
    (6.0, 3.85),
}

CONTROL = (4.0, 0.0)

MF_CHECK = {
    (4.5, 6.0),
    (5.0, 6.0),
    (6.0, 6.0),
}

REENTRANCE = {
    (4.5, 3.0),
    (4.5, 6.0),
    (4.5, 8.0),
}

ALL = RIDGE | {CONTROL} | MF_CHECK | REENTRANCE
SIZE_GRID = sorted(TRIALS_BY_N)


def save_csv(frame, filename):
    path = OUT / filename
    temporary = OUT / (filename + ".partial")

    frame.to_csv(
        temporary,
        index=False,
        float_format="%.16g"
    )

    temporary.replace(path)


def failed_fit(status, sizes):
    return dict(
        intercept=np.nan,
        intercept_se=np.nan,
        slope=np.nan,
        slope_se=np.nan,
        sizes=int(sizes),
        reduced_chi2=np.nan,
        zero_sem_sizes=0,
        fit_status=status,
    )


def weighted_fit(x, y, sem):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sem)
        & (sem >= 0)
    )

    x, y, sem = x[valid], y[valid], sem[valid]

    if len(x) < 3 or len(np.unique(x)) < 3:
        return failed_fit("insufficient_sizes", len(x))

    zero_count = int(np.count_nonzero(sem == 0))
    positive = sem[sem > 0]

    if not len(positive):
        if np.allclose(y, y[0], rtol=0, atol=1e-14):
            return dict(
                intercept=float(y[0]),
                intercept_se=0.0,
                slope=0.0,
                slope_se=0.0,
                sizes=int(len(x)),
                reduced_chi2=np.nan,
                zero_sem_sizes=zero_count,
                fit_status=(
                    "constant_zero_sem_no_statistical_error_estimate"
                ),
            )

        return failed_fit(
            "nonconstant_with_all_zero_sem",
            len(x),
        )

    sem = np.where(
        sem > 0,
        sem,
        positive.min(),
    )

    design = np.column_stack(
        (np.ones(len(x)), x)
    )

    w = 1.0 / (sem * sem)

    normal = design.T @ (
        design * w[:, None]
    )

    try:
        covariance = np.linalg.inv(normal)

        beta = np.linalg.solve(
            normal,
            design.T @ (w * y),
        )

    except np.linalg.LinAlgError:
        return failed_fit(
            "singular_design",
            len(x),
        )

    residuals = (y - design @ beta) / sem

    return dict(
        intercept=float(beta[0]),
        intercept_se=float(
            np.sqrt(max(0, covariance[0, 0]))
        ),
        slope=float(beta[1]),
        slope_se=float(
            np.sqrt(max(0, covariance[1, 1]))
        ),
        sizes=int(len(x)),
        reduced_chi2=float(
            np.dot(residuals, residuals) / (len(x) - 2)
        ),
        zero_sem_sizes=zero_count,
        fit_status=(
            "sem_floored" if zero_count else "weighted"
        ),
    )


def loglog_slope(n, mean, sem):
    n = np.asarray(n, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sem = np.asarray(sem, dtype=float)

    good = (
        np.isfinite(n)
        & (n > 0)
        & np.isfinite(mean)
        & (mean > 0)
        & np.isfinite(sem)
        & (sem >= 0)
    )

    n, mean, sem = n[good], mean[good], sem[good]

    if len(n) < 3:
        return dict(
            loglog_slope=np.nan,
            loglog_slope_se=np.nan,
            loglog_sizes=len(n),
            loglog_status="insufficient_sizes",
        )

    fit = weighted_fit(
        np.log(n),
        np.log(mean),
        sem / mean,
    )

    return dict(
        loglog_slope=fit["slope"],
        loglog_slope_se=fit["slope_se"],
        loglog_sizes=fit["sizes"],
        loglog_status=fit["fit_status"],
    )


def load_data():
    if (
        not IID.is_file()
        or not (OUT / "COMPLETE.json").is_file()
    ):
        raise RuntimeError(
            "Targeted i.i.d. run has not completed"
        )

    if not PAIRED.is_file():
        raise FileNotFoundError(
            f"Missing original cuts data: {PAIRED}"
        )

    cols = KEY + OBS + [
        "disorder_scheme",
        "f_plus_1e-03",
        "f_minus_1e-03",
        "psi_weak_fraction",
    ]

    iid = pd.read_csv(IID, usecols=cols)
    paired = pd.read_csv(PAIRED, usecols=cols)

    for frame in (iid, paired):
        frame["r"] = frame.r.round(6)
        frame["kappa"] = frame.kappa.round(6)

        if frame[KEY].duplicated().any():
            raise RuntimeError(
                "Duplicate realization keys in source data"
            )

    if set(iid.disorder_scheme) != {"iid"}:
        raise RuntimeError(
            "Target dataset is not independently sampled"
        )

    if len(iid) != 1540:
        raise RuntimeError(
            f"Unexpected target record count: {len(iid)}"
        )

    complete = json.loads(
        (OUT / "COMPLETE.json").read_text()
    )

    if int(complete["rows"]) != len(iid):
        raise RuntimeError(
            "Completion manifest disagrees with trial rows"
        )

    paired = paired.merge(
        iid[KEY],
        on=KEY,
        how="inner",
        validate="one_to_one",
    )

    if set(paired.disorder_scheme) != {"antithetic"}:
        raise RuntimeError(
            "Incorrect production ensemble"
        )

    match = paired.merge(
        iid,
        on=KEY,
        suffixes=("_paired", "_iid"),
        validate="one_to_one",
    )

    if len(match) != len(iid):
        raise RuntimeError(
            f"Matched only {len(match)}/{len(iid)} records"
        )

    if set(zip(iid.r, iid.kappa)) != ALL:
        raise RuntimeError(
            "Missing or unexpected parameter points"
        )

    if set(iid.N) != set(SIZE_GRID):
        raise RuntimeError(
            "Missing or unexpected system size"
        )

    groups = match.groupby(
        ["r", "kappa", "N"]
    )

    if len(groups) != 10 * len(SIZE_GRID):
        raise RuntimeError(
            "Incomplete parameter-size grid"
        )

    for (_, _, n), group in groups:
        if len(group) != TRIALS_BY_N[int(n)]:
            raise RuntimeError(
                f"Incorrect trial count at N={n}"
            )

    return iid, paired, match


def run():
    iid, paired, match = load_data()

    rows = []

    for (r, k, n), group in match.groupby(
        ["r", "kappa", "N"], sort=True
    ):
        row = dict(
            r=r,
            kappa=k,
            N=int(n),
            trials=int(len(group)),
        )

        for scheme in ("paired", "iid"):
            pinned = (
                group[f"f_plus_1e-03_{scheme}"]
                + group[f"f_minus_1e-03_{scheme}"]
            ) == 0

            row[f"fraction_pinned_{scheme}"] = float(
                pinned.mean()
            )

        for obs in OBS:
            for scheme in ("paired", "iid"):
                values = group[f"{obs}_{scheme}"]

                row[f"{obs}_{scheme}_mean"] = float(
                    values.mean()
                )

                row[f"{obs}_{scheme}_sem"] = float(
                    values.sem()
                )

            diff = (
                group[f"{obs}_paired"]
                - group[f"{obs}_iid"]
            )

            row[f"{obs}_paired_minus_iid_mean"] = float(
                diff.mean()
            )

            row[f"{obs}_paired_minus_iid_sem"] = float(
                diff.sem()
            )

        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        ["r", "kappa", "N"]
    )

    save_csv(
        comparison,
        "comparison_by_size.csv",
    )

    pin_rows = []

    for scheme, frame in (
        ("paired", paired),
        ("iid", iid),
    ):
        for (r, k, n), group in frame.groupby(
            ["r", "kappa", "N"]
        ):
            if (float(r), float(k)) not in MF_CHECK:
                continue

            pinned = group[
                (
                    group["f_plus_1e-03"]
                    + group["f_minus_1e-03"]
                ) == 0
            ]

            row = dict(
                scheme=scheme,
                r=r,
                kappa=k,
                N=int(n),
                count=len(pinned),
            )

            for obs in ("R1", "R2"):
                row[f"{obs}_mean"] = (
                    float(pinned[obs].mean())
                    if len(pinned)
                    else np.nan
                )

                row[f"{obs}_sem"] = (
                    float(pinned[obs].sem())
                    if len(pinned) > 1
                    else np.nan
                )

            pin_rows.append(row)

    pinned = pd.DataFrame(pin_rows)

    save_csv(
        pinned,
        "pinned_order_by_size.csv",
    )

    fit_rows = []

    for scheme, frame in (
        ("paired", paired),
        ("iid", iid),
    ):
        for r, k in sorted(
            MF_CHECK | REENTRANCE
        ):
            point = frame[
                np.isclose(frame.r, r)
                & np.isclose(frame.kappa, k)
            ]

            obslist = (
                ["R1", "R2"]
                if (r, k) in MF_CHECK
                else []
            )

            if (r, k) in REENTRANCE:
                obslist += ["D"]

            for obs in obslist:
                subset = point

                if obs in ("R1", "R2"):
                    subset = point[
                        (
                            point["f_plus_1e-03"]
                            + point["f_minus_1e-03"]
                        ) == 0
                    ]

                grouped = (
                    subset.groupby("N")[obs]
                    .agg(["mean", "sem", "size"])
                    .reset_index()
                )

                grouped = grouped[
                    grouped["size"] >= 3
                ]

                fit = weighted_fit(
                    1.0 / grouped.N.to_numpy(dtype=float),
                    grouped["mean"],
                    grouped["sem"],
                )

                fit_rows.append(
                    dict(
                        scheme=scheme,
                        r=r,
                        kappa=k,
                        observable=obs,
                        conditioned_on_pinned=(
                            obs in ("R1", "R2")
                        ),
                        **fit,
                    )
                )

    fits = pd.DataFrame(fit_rows)

    save_csv(
        fits,
        "extrapolations_1_over_N.csv",
    )

    plateau_rows = []
    plateau_fits = []

    for scheme, frame in (
        ("paired", paired),
        ("iid", iid),
    ):
        for r, k in sorted(
            RIDGE | {CONTROL}
        ):
            point = frame[
                np.isclose(frame.r, r)
                & np.isclose(frame.kappa, k)
            ].copy()

            point["dR1_squared"] = point.dR1**2

            grouped = (
                point.groupby("N", sort=True)
                .agg(
                    trials=("dR1", "size"),
                    mean_dR1=("dR1", "mean"),
                    sem_dR1=("dR1", "sem"),
                    mean_dR1_squared=(
                        "dR1_squared", "mean"
                    ),
                    sem_dR1_squared=(
                        "dR1_squared", "sem"
                    ),
                )
                .reset_index()
            )

            for record in grouped.to_dict(
                orient="records"
            ):
                plateau_rows.append(
                    dict(
                        scheme=scheme,
                        r=r,
                        kappa=k,
                        **record,
                    )
                )

            slope = loglog_slope(
                grouped.N,
                grouped.mean_dR1,
                grouped.sem_dR1,
            )

            sqfit = weighted_fit(
                1.0 / grouped.N.to_numpy(dtype=float),
                grouped.mean_dR1_squared,
                grouped.sem_dR1_squared,
            )

            a2 = sqfit["intercept"]
            se_a2 = sqfit["intercept_se"]

            amplitude = (
                float(np.sqrt(a2))
                if np.isfinite(a2) and a2 > 0
                else np.nan
            )

            amplitude_se = (
                float(se_a2 / (2 * amplitude))
                if (
                    np.isfinite(amplitude)
                    and np.isfinite(se_a2)
                )
                else np.nan
            )

            plateau_fits.append(
                dict(
                    scheme=scheme,
                    r=r,
                    kappa=k,
                    **slope,
                    A_squared=a2,
                    A_squared_se=se_a2,
                    C_over_N=sqfit["slope"],
                    A=amplitude,
                    A_se=amplitude_se,
                    squared_fit_status=sqfit["fit_status"],
                    squared_fit_reduced_chi2=(
                        sqfit["reduced_chi2"]
                    ),
                    squared_fit_zero_sem_sizes=(
                        sqfit["zero_sem_sizes"]
                    ),
                )
            )

    save_csv(
        pd.DataFrame(plateau_rows),
        "plateau_by_size.csv",
    )

    save_csv(
        pd.DataFrame(plateau_fits),
        "plateau_fits.csv",
    )

    frequency = match[KEY].copy()

    for scheme in ("paired", "iid"):
        for obs in (
            "peak_freq_hann",
            "psi_rate",
            "psi_weak_fraction",
            "R1",
            "dR1",
        ):
            frequency[f"{obs}_{scheme}"] = (
                match[f"{obs}_{scheme}"].to_numpy()
            )

        phase = frequency[
            f"psi_rate_{scheme}"
        ].to_numpy(dtype=float)

        peak = frequency[
            f"peak_freq_hann_{scheme}"
        ].to_numpy(dtype=float)

        weak = frequency[
            f"psi_weak_fraction_{scheme}"
        ].to_numpy(dtype=float)

        order = frequency[
            f"R1_{scheme}"
        ].to_numpy(dtype=float)

        fluct = frequency[
            f"dR1_{scheme}"
        ].to_numpy(dtype=float)

        n = frequency.N.to_numpy(dtype=float)

        is_control = (
            np.isclose(frequency.r, CONTROL[0])
            & np.isclose(frequency.kappa, CONTROL[1])
        )

        valid = (
            ~is_control
            & np.isfinite(phase)
            & np.isfinite(peak)
            & np.isfinite(weak)
            & np.isclose(weak, 0, atol=1e-12)
            & np.isfinite(order)
            & (
                order > np.maximum(
                    0.02,
                    1.0 / np.sqrt(n),
                )
            )
            & np.isfinite(fluct)
            & (fluct > 1e-9)
            & (np.abs(phase) > 1e-3)
            & (peak > 0)
        )

        ratio = np.full(
            len(frequency),
            np.nan,
        )

        ratio[valid] = (
            peak[valid]
            / (np.abs(phase[valid]) / np.pi)
        )

        frequency[
            f"frequency_valid_{scheme}"
        ] = valid

        frequency[
            f"f_over_abs_psi_rate_over_pi_{scheme}"
        ] = ratio

    save_csv(
        frequency,
        "frequency_ratios.csv",
    )

    report = dict(
        status=(
            "completed; scientific interpretation pending"
        ),
        i_i_d_records=int(len(iid)),
        matched_pairs=int(len(match)),
        size_comparisons=int(len(comparison)),
        pinned_size_rows=int(len(pinned)),
        extrapolation_rows=int(len(fits)),
        plateau_size_rows=int(len(plateau_rows)),
        plateau_fits=int(len(plateau_fits)),
        valid_frequency_records={
            scheme: int(
                frequency[
                    f"frequency_valid_{scheme}"
                ].sum()
            )
            for scheme in ("paired", "iid")
        },
        frequency_normalization=(
            "peak_freq_hann / (abs(psi_rate)/pi)"
        ),
        design_note=(
            "Matched schemes share the first N/2 "
            "Gaussian draws and all initial phases"
        ),
        zero_sem_note=(
            "Zero SEM floored to the smallest positive "
            "size SEM; constant all-zero-SEM series is "
            "descriptive, not a confidence interval"
        ),
    )

    (OUT / "analysis_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(json.dumps(report, indent=2))

    print(
        "Saved comparison_by_size.csv, "
        "pinned_order_by_size.csv, "
        "extrapolations_1_over_N.csv,"
    )

    print(
        "      plateau_by_size.csv, "
        "plateau_fits.csv, frequency_ratios.csv"
    )


if __name__ == "__main__":
    run()
