
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from figstyle import (SIM, THEORY, DOUBLE, N_COLORS, N_VALUES, FOLD_COLORS,
                      R_COLORS, R_MARKERS, panel_label, save)

R_MAIN = 4.5
R_COMPARE = 6.0
KAPPA_EXTRAP = (3.0, 6.0, 8.0)
EXTRAP_STYLE = {3.0: ("0.62", "o"), 6.0: ("#9365BA", "s"), 8.0: ("#70439C", "^")}
PINNED_POINTS = ((4.5, 4.6), (4.5, 6.0), (4.5, 8.0), (6.0, 6.0))
POINT_STYLE = {
    (4.5, 4.6): ("#BA9BD3", "v"),
    (4.5, 6.0): ("#9365BA", "s"),
    (4.5, 8.0): ("#70439C", "^"),
    (6.0, 6.0): ("black", "D"),
}
N_HYST = 3200
X_MAX = 0.0108
WILSON_Z = 1.96
HYST_R_VALUES = (4.25, 4.5, 5.0, 6.0)

cuts = pd.read_csv(SIM / "cuts" / "trials.csv", usecols=["r", "kappa", "N", "trial", "D", "f_plus_1e-03", "f_minus_1e-03"])
cuts["r"] = cuts.r.round(6)
cuts["kappa"] = cuts.kappa.round(6)
cuts["pinned"] = (cuts["f_plus_1e-03"] + cuts["f_minus_1e-03"]) == 0
hyst = pd.read_csv(SIM / "hysteresis" / "trials.csv", usecols=["r", "kappa", "N", "trial", "direction", "D"])
hyst["r"] = hyst.r.round(6)
hyst["kappa"] = hyst.kappa.round(6)
scan = pd.read_csv(SIM / "oscillation_scan" / "trials.csv", usecols=["r", "kappa", "N", "trial", "D"])
scan["r"] = scan.r.round(6)
scan["kappa"] = scan.kappa.round(6)
folds = pd.read_csv(THEORY / "mf_folds.csv", usecols=["r", "kind", "kappa"])
folds["r"] = folds.r.round(6)


def fold_at(r, kind):
    rows = folds[(folds.r == r) & (folds.kind == kind)]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {kind} fold at r={r:g}")
    return float(rows.kappa.iloc[0])


def fold_lines(ax, r, height=0.52):
    for kind in ("lower", "upper"):
        edge = fold_at(r, kind)
        if ax.get_xlim()[0] <= edge <= ax.get_xlim()[1]:
            ax.axvline(edge, ymin=0.0, ymax=height, color=FOLD_COLORS[kind], lw=1.05, zorder=1)


def top_axis(ax):
    top = ax.secondary_xaxis(
        "top",
        functions=(lambda v: 1.0 / np.maximum(v, 1e-9),
                   lambda v: 1.0 / np.maximum(v, 1e-9)),
    )
    top.set_xticks([100, 200, 400, 800])
    top.set_xticklabels(["100", "200", "400", "800"])
    top.set_xlabel(r"$N$", labelpad=2)


def wilson(k, n, z=WILSON_Z):
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return p, centre - half_width, centre + half_width


def weighted_fit(x, y, s):
    w = 1.0 / s ** 2
    design = np.c_[np.ones_like(x), x]
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    return cov @ (design.T @ (w * y)), np.sqrt(np.diag(cov))


def cut_profile(r):
    profile = cuts[cuts.r == r].groupby(["N", "kappa"]).D.mean()
    if set(profile.index.get_level_values("N")) != set(N_VALUES):
        raise RuntimeError(f"incomplete system sizes at r={r:g}")
    return profile


fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.9), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.07, hspace=0.07)
ax_a, ax_b, ax_c, ax_d = axes.flat

profile = cut_profile(R_MAIN)
for size in N_VALUES:
    curve = profile.xs(size).sort_index()
    ax_a.plot(curve.index, curve.to_numpy(), color=N_COLORS[size], lw=1.0, label=rf"$N={size}$")
ax_a.set_xlim(0.0, 8.0)
ax_a.set_ylim(0.0, 0.32)
fold_lines(ax_a, R_MAIN, height=0.65)
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$\langle D\rangle$")

n_handles = [Line2D([], [], color=N_COLORS[size], lw=1.0, label=rf"$N={size}$") for size in N_VALUES]
fold_handles = [Line2D([], [], color=FOLD_COLORS[kind], lw=1.05,
                       label=(r"lower fold $\kappa_\ell$" if kind == "lower" else r"upper fold $\kappa_u$"))
                for kind in ("upper", "lower")]
ax_a.legend(handles=n_handles + fold_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
            ncol=3, handlelength=1.25, handletextpad=0.32, labelspacing=0.18,
            columnspacing=0.55, borderaxespad=0.12, borderpad=0.15,
            title=rf"$r={R_MAIN:g}$", title_fontsize=8)
panel_label(ax_a, "a", x=-0.15)

xfit = np.linspace(0.0, X_MAX, 50)
for kappa in KAPPA_EXTRAP:
    stats = cuts[(cuts.r == R_MAIN) & (cuts.kappa == kappa)].groupby("N").D.agg(["mean", "sem"])
    if set(stats.index) != set(N_VALUES):
        raise RuntimeError(f"incomplete extrapolation system sizes at kappa={kappa:g}")
    x = 1.0 / stats.index.to_numpy(dtype=float)
    y, s = stats["mean"].to_numpy(), stats["sem"].to_numpy()
    color, marker = EXTRAP_STYLE[kappa]
    if kappa != 8.0:
        s = np.where(s > 0, s, np.min(s[s > 0]))
        coef, err = weighted_fit(x, y, s)
        ax_b.plot(xfit, coef[0] + coef[1] * xfit, color=color, lw=0.9, zorder=2)
    ax_b.errorbar(x, y, yerr=stats["sem"], fmt=marker,
                  color=color, mfc="white", mew=0.9, ms=3.8,
                  elinewidth=0.7, capsize=1.8, capthick=0.7,
                  zorder=3, label=rf"$\kappa={kappa:g}$")
    if kappa != 8.0:
        ax_b.errorbar(0.0, coef[0], yerr=err[0], fmt="*",
                      color=color, ms=7.5, elinewidth=0.7,
                      capsize=1.8, capthick=0.7,
                      zorder=4, clip_on=False)

ax_b.axhline(0.0, color="0.75", lw=0.5, zorder=0)
ax_b.set_xlim(-0.0004, X_MAX)
ax_b.set_ylim(-0.02, 0.37)
ax_b.set_xlabel(r"$1/N$")
ax_b.set_ylabel(r"$\langle D\rangle$")
ax_b.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2), useMathText=True)
top_axis(ax_b)
kappa_handles, _ = ax_b.get_legend_handles_labels()
handles = [kappa_handles[0],
           Line2D([], [], color="0.35", lw=0.9, label=r"fit $a+C/N$"),
           kappa_handles[1],
           Line2D([], [], color="0.35", marker="*", lw=0, ms=7.5, label=r"$N\to\infty$"),
           kappa_handles[2]]
ax_b.legend(handles=handles, loc="upper center", ncol=3, handlelength=1.5,
            labelspacing=0.25, columnspacing=0.8, borderaxespad=0.2,
            title=rf"$r={R_MAIN:g}$", title_fontsize=8)
panel_label(ax_b, "b", x=-0.15, y=1.08)

for point in PINNED_POINTS:
    subset = cuts[(cuts.r == point[0]) & (cuts.kappa == point[1])]
    data = subset.groupby("N").pinned.agg(["sum", "size"]).sort_index()
    if not np.array_equal(data.index.to_numpy(), np.asarray(N_VALUES)):
        raise RuntimeError(f"incomplete pinning size grid at {point}")
    p, lo, hi = wilson(data["sum"].to_numpy(dtype=float), data["size"].to_numpy(dtype=float))
    x = 1.0 / data.index.to_numpy(dtype=float)
    color, marker = POINT_STYLE[point]
    ax_c.errorbar(x, p, yerr=[np.maximum(p - lo, 0.0), np.maximum(hi - p, 0.0)], fmt=marker + "-",
                  color=color, mfc="white", mew=0.9, ms=4.0, lw=1.0,
                  elinewidth=0.75, capsize=1.6, zorder=3,
                  label=rf"$({point[0]:g},{point[1]:g})$")
ax_c.set_xlim(-0.0004, X_MAX)
ax_c.set_ylim(-0.04, 1.13)
ax_c.set_yticks(np.arange(0.0, 1.01, 0.2))
ax_c.set_xlabel(r"$1/N$")
ax_c.set_ylabel("Pinned fraction")
ax_c.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2), useMathText=True)
top_axis(ax_c)
ax_c.legend(loc="lower right", bbox_to_anchor=(0.95, 0.008), ncol=2,
            handlelength=1.0, columnspacing=0.35, handletextpad=0.22,
            labelspacing=0.12, borderaxespad=0.08, borderpad=0.10,
            title=r"$(r,\kappa)$; 95% Wilson intervals", title_fontsize=8)
panel_label(ax_c, "c", x=-0.15, y=1.08)

h = hyst[(hyst.r.isin(HYST_R_VALUES)) & (hyst.N == N_HYST) & hyst.kappa.between(3.0, 8.0)]
if set(h.r.unique()) != set(HYST_R_VALUES):
    raise RuntimeError("missing one or more hysteresis disorder strengths")
if set(h.direction) != {"up", "down"}:
    raise RuntimeError("both sweep directions are required")

up = h[h.direction == "up"]
down = h[h.direction == "down"]
keys = ["r", "kappa", "N", "trial"]
matched = up.merge(down, on=keys, validate="one_to_one", suffixes=("_up", "_down"))
if len(matched) != len(up) or len(matched) != len(down):
    raise RuntimeError("incomplete matched hysteresis realizations")

matched["difference"] = matched["D_up"] - matched["D_down"]
statistics = matched.groupby(["r", "kappa"]).difference.agg(["mean", "sem", "size"]).reset_index()

if (statistics["size"] < 2).any() or not np.isfinite(statistics[["mean", "sem"]].to_numpy()).all():
    raise RuntimeError("invalid paired-difference statistics")

plot_markers = {r: R_MARKERS[r] for r in HYST_R_VALUES}

ymin = np.inf
ymax = -np.inf

for i, r in enumerate(HYST_R_VALUES):
    data = statistics[statistics.r == r].sort_values("kappa")
    if data.empty:
        raise RuntimeError(f"missing paired-difference data for r={r:g}")

    k = data.kappa.to_numpy(dtype=float)
    mean = data["mean"].to_numpy(dtype=float)
    sem = data["sem"].to_numpy(dtype=float)

    ymin = min(ymin, float(np.min(mean - sem)))
    ymax = max(ymax, float(np.max(mean + sem)))

    ax_d.fill_between(k, mean - sem, mean + sem,
                      color=R_COLORS[r], alpha=0.25, lw=0, zorder=1)

    marker_start = 3.0 + 0.1 * i
    every = np.flatnonzero(np.isclose(
        np.mod(k - marker_start + 1e-9, 0.5), 0.0, atol=1e-6
    ))

    ax_d.plot(k, mean, color=R_COLORS[r], lw=1.1,
              marker=plot_markers[r], markevery=list(every),
              ms=3.6, mfc="white", mew=0.9, zorder=3 + i)

ypad = 0.08 * (ymax - ymin) if ymax > ymin else 1e-5
ylo = min(ymin - ypad, -1e-5)
yhi = max(ymax + ypad, 1e-5)

ax_d.axhline(0.0, color="0.55", lw=0.75, linestyle="--", zorder=2)
ax_d.set_xlim(3.0, 8.0)
ax_d.set_xticks(np.arange(3.0, 8.1, 1.0))
ax_d.set_ylim(ylo, yhi)
ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_d.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax_d.set_xlabel(r"$\kappa$")
ax_d.set_ylabel(r"$\langle D_{\mathrm{up}}-D_{\mathrm{down}}\rangle$")

legend_handles = [
    Line2D([], [], color=R_COLORS[r], lw=1.1,
           marker=plot_markers[r], markerfacecolor="white",
           markeredgewidth=0.9, markersize=3.6,
           label=rf"$r={r:g}$")
    for r in HYST_R_VALUES
]

ax_d.legend(handles=legend_handles, loc="upper right",
            bbox_to_anchor=(0.95, 0.99), ncol=2,
            handlelength=1.65, handletextpad=0.45,
            columnspacing=0.85, labelspacing=0.28,
            borderaxespad=0.25,
            title=rf"$N={N_HYST}$", title_fontsize=8)

panel_label(ax_d, "d", x=-0.15)

save(fig, "fig8")
