import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from figstyle import SIM, DOUBLE, R_COLORS, R_MARKERS, GUIDE_COLOR, GUIDE_WIDTH, panel_label, save

R_BANDS = (2.0, 4.0, 4.5, 5.0, 6.0)
CONTROL_POINTS = ((4.24, 2.73), (4.5, 4.6), (6.0, 6.0))
R_SWEPT = (4.25, 4.5, 5.0, 6.0)
N_SCAN = 3200
N_HYST = 3200
K_RANGE = (3.0, 8.0)

CONTROL_COLORS = {
    (4.24, 2.73): R_COLORS[4.25],
    (4.5, 4.6): "0.55",
    (6.0, 6.0): "black",
}

THIN_X = Path([
    (-0.5, -0.38), (-0.38, -0.5), (0.0, -0.12),
    (0.38, -0.5), (0.5, -0.38), (0.12, 0.0),
    (0.5, 0.38), (0.38, 0.5), (0.0, 0.12),
    (-0.38, 0.5), (-0.5, 0.38), (-0.12, 0.0),
    (-0.5, -0.38),
], closed=True)

CONTROL_MARKERS = {
    (4.24, 2.73): THIN_X,
    (4.5, 4.6): R_MARKERS[4.25],
    (6.0, 6.0): R_MARKERS[5.0],
}

SCHEME_STYLE = {
    "antithetic": {"shift": 0.965, "mfc": "auto", "ls": "-", "lw": 0.95, "z": 4},
    "iid": {"shift": 1.035, "mfc": "white", "ls": (0, (2.2, 1.5)), "lw": 0.95, "z": 3},
}

scan = pd.read_csv(SIM / "oscillation_scan" / "trials.csv")
scan["r"] = scan.r.round(6)
scan["kappa"] = scan.kappa.round(6)

control = pd.read_csv(SIM / "disorder_check" / "trials.csv")
control["r"] = control.r.round(6)
control["kappa"] = control.kappa.round(6)

hyst = pd.read_csv(SIM / "hysteresis" / "trials.csv")
hyst["r"] = hyst.r.round(6)
hyst["kappa"] = hyst.kappa.round(6)

scan_stationary = scan[scan.N == N_SCAN]
scan_indep = scan[(scan.N == N_HYST) & scan.kappa.between(*K_RANGE)]
hyst = hyst[(hyst.N == N_HYST) & hyst.kappa.between(*K_RANGE) & hyst.r.isin(R_SWEPT)]

for point in CONTROL_POINTS:
    if control[(control.r == point[0]) & (control.kappa == point[1])].empty:
        raise RuntimeError(f"missing disorder-check data for (r, kappa)=({point[0]:g}, {point[1]:g})")

if set(hyst.direction.unique()) != {"up", "down"}:
    raise RuntimeError("both hysteresis sweep directions are required")

merge_keys = ["r", "kappa", "N", "trial"]
paired = hyst.merge(scan_indep[merge_keys + ["D"]], on=merge_keys,
                    suffixes=("", "_indep"), validate="many_to_one")
if len(paired) != len(hyst):
    raise RuntimeError("unmatched realizations between sweeps and independent initializations")

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.85), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.08, hspace=0.08)
ax_a, ax_b, ax_c, ax_d = axes.flat

shift = scan_stationary.assign(change=lambda d: d.R1_second - d.R1_first).groupby(["r", "kappa"]).change.mean()

yvals_a = []
for r in R_BANDS:
    if r not in shift.index.get_level_values("r"):
        raise RuntimeError(f"missing stationarity data at r={r:g}")
    cut = shift.xs(r).sort_index()
    y = 1e3 * cut.to_numpy(dtype=float)
    yvals_a.append(y)
    ax_a.plot(cut.index.to_numpy(dtype=float), y, color=R_COLORS[r], lw=1.0, label=rf"$r={r:g}$")

ax_a.axhline(0.0, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":")
all_a = np.concatenate(yvals_a)
pad_a = 0.12 * max(1e-6, float(all_a.max() - all_a.min()))

ax_a.set_xlim(0.0, 8.0)
ax_a.set_ylim(float(all_a.min() - pad_a), float(all_a.max() + pad_a))
ax_a.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$\langle R_1\rangle_{\mathrm{late}}-\langle R_1\rangle_{\mathrm{early}}\;(\times 10^{-3})$")
ax_a.legend(loc="upper right", ncol=2, handlelength=1.4, labelspacing=0.25, columnspacing=0.8,
            borderaxespad=0.2, title=rf"$N={N_SCAN}$", title_fontsize=8)
panel_label(ax_a, "a", x=-0.18)

n_values = np.array(sorted(control.N.unique()))
if len(n_values) == 0:
    raise RuntimeError("no disorder-check system sizes found")

for ax, column, ylabel in (
    (ax_b, "D", r"$\langle D\rangle$"),
    (ax_c, "dR1", r"$\langle\delta R_1\rangle$"),
):
    ymin = np.inf
    ymax = -np.inf

    for point in CONTROL_POINTS:
        color = CONTROL_COLORS[point]
        marker = CONTROL_MARKERS[point]

        for scheme in ("antithetic", "iid"):
            style = SCHEME_STYLE[scheme]
            stats = control[
                (control.r == point[0]) &
                (control.kappa == point[1]) &
                (control.disorder_scheme == scheme)
            ].groupby("N")[column].agg(["mean", "sem"]).sort_index()

            if stats.empty:
                raise RuntimeError(f"missing {scheme} disorder-check data for (r, kappa)=({point[0]:g}, {point[1]:g})")

            x = stats.index.to_numpy(dtype=float) * style["shift"]
            y = stats["mean"].to_numpy(dtype=float)
            s = stats["sem"].to_numpy(dtype=float)

            ymin = min(ymin, float(np.nanmin(y - np.nan_to_num(s, nan=0.0))))
            ymax = max(ymax, float(np.nanmax(y + np.nan_to_num(s, nan=0.0))))

            mfc = color if style["mfc"] == "auto" else style["mfc"]

            ax.errorbar(
                x, y, yerr=s,
                fmt="", marker=marker, linestyle=style["ls"],
                color=color, mfc=mfc, mec=color, mew=0.9,
                ms=4.1, lw=style["lw"], elinewidth=0.7,
                capsize=1.3, zorder=style["z"]
            )    

    pad = 0.10 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(float(n_values.min() * 0.82), float(n_values.max() * 1.24))
    ax.set_xticks(n_values)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin - pad, ymax + pad)

point_handles = [
    Line2D(
        [], [], color=CONTROL_COLORS[p],
        marker=CONTROL_MARKERS[p],
        mfc=CONTROL_COLORS[p] if p in ((4.24, 2.73), (6.0, 6.0)) else "white",
        mec=CONTROL_COLORS[p],
        mew=0.9, lw=0.95, ms=4.1,
        label=rf"$({p[0]:g},{p[1]:g})$"
    )
    for p in CONTROL_POINTS
]

scheme_handles = [
    Line2D([], [], color="0.55", marker="h", mfc="0.55", mec="0.55", ms=4.1, lw=0.95,
           label=r"$\pm$-paired Gaussian"),
    Line2D([], [], color="0.55", marker="h", mfc="white", mec="0.55", ms=4.1, lw=0.95,
           linestyle=(0, (2.2, 1.5)), label="i.i.d. Gaussian")
]

ax_b.legend(handles=point_handles + scheme_handles, loc="center right",
            bbox_to_anchor=(0.985, 0.55), ncol=2, fontsize=7.0,
            handlelength=1.6, handletextpad=0.4, labelspacing=0.25, columnspacing=0.7,
            borderaxespad=0.2, title=r"$(r,\kappa)$; 12 realizations/scheme", title_fontsize=8)

panel_label(ax_b, "b", x=-0.18)
panel_label(ax_c, "c", x=-0.18)

yvals_d = []
for r in R_SWEPT:
    data = paired[paired.r == r].copy()
    if data.empty:
        raise RuntimeError(f"missing sweep/init comparison data for r={r:g}")

    data["diff"] = data["D"] - data["D_indep"]
    grouped = data.groupby(["direction", "kappa"])["diff"].mean()

    for direction in ("up", "down"):
        curve = grouped.xs(direction).sort_index()
        x = curve.index.to_numpy(dtype=float)
        y = 1e3 * curve.to_numpy(dtype=float)
        yvals_d.append(y)

        ax_d.plot(
            x, y, color=R_COLORS[r],
            lw=1.05 if direction == "up" else 0.65,
            linestyle="-",
            alpha=1.0 if direction == "up" else 0.5,
            label=rf"$r={r:g}$" if direction == "up" else None
        )

ax_d.axhline(0.0, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":")
all_d = np.concatenate(yvals_d)
limit_d = 1.14 * max(0.05, float(np.max(np.abs(all_d))))

ax_d.set_xlim(*K_RANGE)
ax_d.set_ylim(-limit_d, limit_d)
ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_d.set_xlabel(r"$\kappa$")
ax_d.set_ylabel(r"$\langle D_{\mathrm{sweep}}-D_{\mathrm{indep}}\rangle\;(\times 10^{-3})$")

color_handles = [
    Line2D([], [], color=R_COLORS[r], lw=1.05, label=rf"$r={r:g}$")
    for r in R_SWEPT
]
style_handles = [
    Line2D([], [], color="0.25", lw=1.05, linestyle="-", label="up sweep"),
    Line2D([], [], color="0.25", lw=0.65, linestyle="-",
           alpha=0.5, label="down sweep"),
]

legend_colors = ax_d.legend(handles=color_handles, loc="upper left", ncol=2,
                            handlelength=1.6, handletextpad=0.4, labelspacing=0.25,
                            columnspacing=0.8, borderaxespad=0.2,
                            title=rf"$N={N_HYST}$", title_fontsize=8)
ax_d.add_artist(legend_colors)
ax_d.legend(handles=style_handles, loc="lower left", handlelength=1.6,
            handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)

panel_label(ax_d, "d", x=-0.18)

save(fig, "si_fig3")
