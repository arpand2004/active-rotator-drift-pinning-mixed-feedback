import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (SIM, THEORY, DOUBLE, OBS_COLORS, R_COLORS, R_MARKERS, THEORY_COLOR, UNSTABLE_DASH,
                      fold_marker, panel_label, save)

R_CUTS = (2.0, 4.0, 5.0, 6.0)
R_THEORY = (5.0, 6.0)
N_CUTS = 3200
BENCH_R = np.round(np.arange(0.0, 6.01, 0.5), 2)
MARK_EVERY = 0.5

grid_map = pd.read_csv(SIM / "map" / "summary.csv")
scan = pd.read_csv(SIM / "oscillation_scan" / "summary.csv")
roots = pd.read_csv(THEORY / "mf_roots_on_grid.csv")
folds = pd.read_csv(THEORY / "mf_folds.csv")
for frame in (grid_map, scan, roots, folds):
    frame["r"] = frame.r.round(6)
for frame in (grid_map, scan, roots):
    frame["kappa"] = frame.kappa.round(6)
scan = scan[scan.N == N_CUTS]


def gaussian(a, r):
    return np.exp(-0.5 * (a / r) ** 2) / (np.sqrt(2.0 * np.pi) * r)


def uncoupled_drift(r):
    if r == 0:
        return 1.0
    a = np.linspace(-1.0, 1.0, 200001)
    return np.trapezoid(np.sqrt(1.0 - a * a) * gaussian(a, r), a)


def uncoupled_nematic(r):
    if r == 0:
        return 0.0
    a = np.linspace(1e-9, 12.0 * r, 400001)
    inner = 1.0 - (2.0 / a ** 2) * (1.0 - np.sqrt(np.clip(1.0 - a * a, 0.0, None)))
    outer = 1.0 - 2.0 / a ** 2 + 2j * np.sqrt(np.clip(1.0 - 1.0 / a ** 2, 0.0, None)) / a
    return abs(np.trapezoid(2.0 * gaussian(a, r) * np.where(a < 1.0, inner, outer), a))


def fold_order(r, kind):
    row = folds[(folds.r == r) & (folds.kind == kind)].iloc[0]
    a = np.linspace(-8.0 * r, 8.0 * r, 3001)
    w = gaussian(a, r) * np.gradient(a)
    w[[0, -1]] *= 0.5
    w /= w.sum()
    v = a + complex(row.ReZ, row.ImZ)
    b = np.abs(v)
    h2 = 1.0 - 2.0 / b ** 2 + 2j * np.sqrt(1.0 - 1.0 / b ** 2) / b
    return row.kappa, row.R1, abs(np.sum(w * (v / b) ** 2 * h2))


def branch(r, nu, column):
    rows = roots[(roots.r == r) & (roots.n_unstable == nu)].sort_values("kappa")
    k0, r1, r2 = fold_order(r, "lower")
    return np.r_[k0, rows.kappa.to_numpy()], np.r_[r1 if column == "R1" else r2, rows[column].to_numpy()]


fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.9), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.06)
ax_a, ax_b, ax_c, ax_d = axes.flat

r_line = np.linspace(0.0, 6.0, 241)
bench = grid_map[(grid_map.kappa == 0.0) & grid_map.r.isin(BENCH_R)].sort_values("r")
ax_a.plot(r_line, [uncoupled_drift(r) for r in r_line], color=OBS_COLORS["D"], lw=1.2)
ax_a.plot(r_line, [uncoupled_nematic(r) for r in r_line], color=OBS_COLORS["R2"], lw=1.2)
for column, key, marker in (("D", "D", "^"), ("R2", "R2", "s"), ("R1", "R1", "o")):
    ax_a.errorbar(bench.r, bench[f"{column}_mean"], yerr=bench[f"{column}_sem"], fmt=marker, ms=3.8,
                  color=OBS_COLORS[key], mfc="white", mew=1.0, elinewidth=0.7, capsize=0)
ax_a.set_xlabel(r"$r$")
ax_a.set_ylabel(r"uncoupled observables ($\kappa=0$)")
ax_a.set_xlim(-0.15, 6.15)
ax_a.set_ylim(-0.03, 1.2)
ax_a.set_yticks(np.arange(0, 1.01, 0.2))

bench_handles = [
    Line2D([], [], linestyle="none", color="none", label=r"$N=200$"),
    Line2D([], [], color=OBS_COLORS["R2"], lw=1.2, label=r"$R_2^0$, exact"),
    Line2D([], [], color=OBS_COLORS["D"], lw=1.2, label=r"$D_0$, exact"),
    Line2D([], [], color=OBS_COLORS["R1"], marker="o", linestyle="none", mfc="white", mew=1.0, ms=4.0, label=r"$\langle R_1\rangle$"),
    Line2D([], [], color=OBS_COLORS["R2"], marker="s", linestyle="none", mfc="white", mew=1.0, ms=4.0, label=r"$\langle R_2\rangle$"),
    Line2D([], [], color=OBS_COLORS["D"], marker="^", linestyle="none", mfc="white", mew=1.0, ms=4.0, label=r"$\langle D\rangle$"),
]
left_legend = ax_a.legend(handles=bench_handles[:3], loc="upper right", bbox_to_anchor=(0.78, 1.0),
                          handlelength=1.5, handletextpad=0.4, labelspacing=0.30, borderaxespad=0.3)
ax_a.add_artist(left_legend)
ax_a.legend(handles=bench_handles[3:], loc="upper right", bbox_to_anchor=(1.0, 1.004),
            handlelength=1.5, handletextpad=0.4, labelspacing=0.39, borderaxespad=0.3)
panel_label(ax_a, "a", x=-0.17)

for ax, column, label in ((ax_b, "R1", r"$\langle R_1\rangle$"),
                          (ax_c, "R2", r"$\langle R_2\rangle$"),
                          (ax_d, "D", r"$\langle D\rangle$")):
    for r in R_CUTS:
        cut = scan[scan.r == r].sort_values("kappa")
        k, m, s = cut.kappa.to_numpy(), cut[f"{column}_mean"].to_numpy(), cut[f"{column}_sem"].to_numpy()
        ax.fill_between(k, m - s, m + s, color=R_COLORS[r], alpha=0.25, lw=0)
        every = np.flatnonzero(np.isclose(np.mod(k + 1e-9, MARK_EVERY), 0.0, atol=1e-6))
        ax.plot(k, m, color=R_COLORS[r], lw=1.1, marker=R_MARKERS[r], markevery=list(every), ms=3.6,
                mfc="white", mew=0.9)
    if column in ("R1", "R2"):
        for r in R_THEORY:
            for nu, style in ((0, "-"), (1, UNSTABLE_DASH)):
                k, v = branch(r, nu, column)
                ax.plot(k, v, color=THEORY_COLOR, lw=0.9, linestyle=style, zorder=5)
            k0, r1, r2 = fold_order(r, "lower")
            ax.plot(k0, r1 if column == "R1" else r2, **fold_marker("lower"))
    else:
        for r in R_THEORY:
            ax.plot([fold_order(r, "lower")[0], 8.0], [0.0, 0.0], color=THEORY_COLOR, lw=0.9, zorder=5)
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(label)

cut_handles = [
    Line2D([], [], color=R_COLORS[r], lw=1.2, marker=R_MARKERS[r], markerfacecolor="white",
           markeredgewidth=0.9, markersize=3.6, label=rf"$r={r:g}$") for r in R_CUTS
]
cut_handles.extend([
    Line2D([], [], color="none", linestyle="none", label=rf"$N={N_CUTS}$"),
    Line2D([], [], color=THEORY_COLOR, lw=0.9, label="MF stable"),
    Line2D([], [], color=THEORY_COLOR, lw=0.9, linestyle=UNSTABLE_DASH, label="MF unstable"),
    Line2D([], [], **{k: v for k, v in fold_marker("lower").items() if k != "zorder"},
           label=r"lower fold $\kappa_\ell$"),
])
ax_b.legend(handles=cut_handles, loc="lower right", ncol=2, fontsize=7.75, handlelength=1.65,
            handletextpad=0.45, columnspacing=0.85, labelspacing=0.28, borderaxespad=0.25)

for ax, letter in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d")):
    panel_label(ax, letter, x=-0.17)

save(fig, "fig3")
