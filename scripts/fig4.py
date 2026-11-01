import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import SIM, THEORY, DOUBLE, R_COLORS, R_MARKERS, THEORY_COLOR, panel_label, save

R_SET = (4.5, 5.0, 6.0)
KAPPA_TEST = 6.0
FOLD_WINDOW = (3.5, 6.0)
RUN = 5
X_MAX = 0.0108

cuts = pd.read_csv(SIM / "cuts" / "trials.csv")
cuts["r"] = cuts.r.round(6)
cuts["kappa"] = cuts.kappa.round(6)
cuts["pinned"] = (cuts["f_plus_1e-03"] + cuts["f_minus_1e-03"]) == 0
roots = pd.read_csv(THEORY / "mf_roots_on_grid.csv")
roots["r"] = roots.r.round(6)
roots["kappa"] = roots.kappa.round(6)
folds = pd.read_csv(THEORY / "mf_folds.csv")
folds["r"] = folds.r.round(6)


def weighted_fit(x, y, s):
    design = np.column_stack((np.ones_like(x), x))
    weight = 1.0 / (s * s)
    normal = design.T @ (design * weight[:, None])
    covariance = np.linalg.inv(normal)
    coefficients = np.linalg.solve(normal, design.T @ (weight * y))
    return coefficients, np.sqrt(np.diag(covariance))


def first_pinned_run(group):
    group = group.sort_values("kappa")
    pinned = group.pinned.to_numpy()
    kappa = group.kappa.to_numpy()
    for index in range(len(pinned) - RUN + 1):
        if pinned[index:index + RUN].all():
            return kappa[index]
    return np.nan


def series_order(r, column):
    subset = cuts[(cuts.r == r) & (cuts.kappa == KAPPA_TEST) & cuts.pinned]
    stats = subset.groupby("N")[column].agg(["mean", "sem", "size"])
    return stats[stats["size"] >= 3]


def series_fold(r):
    subset = cuts[(cuts.r == r) & cuts.kappa.between(*FOLD_WINDOW)]
    onsets = subset.groupby(["N", "trial"]).apply(first_pinned_run, include_groups=False).dropna()
    stats = onsets.groupby(level=0).agg(["mean", "sem", "size"])
    return stats[stats["size"] >= 3]


def theory_order(r, column):
    rows = roots[(roots.r == r) & (roots.kappa == KAPPA_TEST) & (roots.n_unstable == 0)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one stable root at r={r:g}, kappa={KAPPA_TEST:g}")
    return float(rows[column].iloc[0])


def theory_fold(r):
    rows = folds[(folds.r == r) & (folds.kind == "lower")]
    if len(rows) != 1:
        raise RuntimeError(f"expected one lower fold at r={r:g}")
    return float(rows.kappa.iloc[0])


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.65), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.07)

specs = (
    (axes[0], lambda r: series_order(r, "R1"), lambda r: theory_order(r, "R1"), r"$\langle R_1\rangle$"),
    (axes[1], lambda r: series_order(r, "R2"), lambda r: theory_order(r, "R2"), r"$\langle R_2\rangle$"),
    (axes[2], series_fold, theory_fold, r"$\langle\kappa_\ell^{(N)}\rangle$"),
)

xfit = np.linspace(0.0, X_MAX, 100)
for (ax, series, theory, ylabel), letter in zip(specs, "abc"):
    for r in R_SET:
        stats = series(r)
        if len(stats) < 3:
            raise RuntimeError(f"insufficient system sizes for r={r:g} in panel {letter}")
        x = 1.0 / stats.index.to_numpy(dtype=float)
        y = stats["mean"].to_numpy(dtype=float)
        sem = stats["sem"].to_numpy(dtype=float)
        positive = sem[np.isfinite(sem) & (sem > 0)]
        if len(positive) == 0:
            raise RuntimeError(f"no positive SEM for r={r:g} in panel {letter}")
        s = np.where(np.isfinite(sem) & (sem > 0), sem, positive.min())
        coefficients, errors = weighted_fit(x, y, s)
        color = R_COLORS[r]
        ax.plot(xfit, coefficients[0] + coefficients[1] * xfit, color=color, lw=1.05, zorder=2)
        ax.errorbar(x, y, yerr=s, fmt=R_MARKERS[r], color=color, mfc="white", mew=0.9,
            ms=3.8, elinewidth=0.65, capsize=1.2, alpha=0.88, zorder=3)
        ax.plot(0.0, theory(r), marker="D", ms=7.0, mfc="none", mec=THEORY_COLOR,
                mew=1.0, ls="none", zorder=4, clip_on=False)
        ax.errorbar(0.0, coefficients[0], yerr=errors[0], fmt="*", color=color, ms=6.7,
                    mec=color, elinewidth=0.6, capsize=0, zorder=5, clip_on=False)
    ax.set_xlim(-0.0004, X_MAX)
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2), useMathText=True)
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda value: 1.0 / np.maximum(value, 1e-9),
                   lambda value: 1.0 / np.maximum(value, 1e-9)),
    )
    top.set_xticks([100, 200, 400, 800])
    top.set_xticklabels(["100", "200", "400", "800"])
    top.set_xlabel(r"$N$", labelpad=2)
    panel_label(ax, letter, x=-0.27, y=1.08)

for r in R_SET:
    stats = series_order(r, "R1")
    xx = 1.0 / stats.index.to_numpy(dtype=float)
    yy = stats["mean"].to_numpy(dtype=float)
    sem = stats["sem"].to_numpy(dtype=float)
    coefficients, _ = weighted_fit(xx, yy, sem)
    x_label = 0.0072
    offset = -0.005 if r == 4.5 else 0.004
    vertical_alignment = "top" if r == 4.5 else "bottom"
    axes[0].text(x_label, coefficients[0] + coefficients[1] * x_label + offset,
                 rf"$r={r:g}$", color=R_COLORS[r], fontsize=8,
                 ha="center", va=vertical_alignment)

axes[0].text(0.43, 0.97, rf"$\kappa={KAPPA_TEST:g}$", transform=axes[0].transAxes,
             ha="left", va="top", fontsize=8)
axes[1].text(0.43, 0.97, rf"$\kappa={KAPPA_TEST:g}$", transform=axes[1].transAxes,
             ha="left", va="top", fontsize=8)
axes[0].legend(
    handles=[
        Line2D([], [], color="0.35", marker="o", mfc="white", lw=0, ms=3.8, label="simulation"),
        Line2D([], [], color="0.35", lw=1.0, label=r"fit $a+C/N$"),
        Line2D([], [], color="0.35", marker="*", lw=0, ms=6.7, label=r"$N\to\infty$"),
        Line2D([], [], color=THEORY_COLOR, marker="D", mfc="none", lw=0, ms=6.5, label="mean field"),
    ],
    loc="center right", bbox_to_anchor=(0.99, 0.36), ncol=2,
    handlelength=1.4, handletextpad=0.35, columnspacing=0.7,
    labelspacing=0.25, borderaxespad=0.2,
)
axes[1].set_ylim(0.28, 0.67)

save(fig, "fig4")
