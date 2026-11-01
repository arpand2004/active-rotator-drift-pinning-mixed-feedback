import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from figstyle import SIM, DOUBLE, R_COLORS, R_MARKERS, GUIDE_COLOR, GUIDE_WIDTH, panel_label, save

EPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
N_EPS = 3200
POINTS = {
    (4.0, 2.95): (R_COLORS[4.0], "s"),
    (4.5, 3.3): (R_COLORS[4.5], R_MARKERS[4.5]),
    (5.0, 3.5): (R_COLORS[5.0], R_MARKERS[5.0]),
    (6.0, 3.85): (R_COLORS[6.0], R_MARKERS[6.0]),
    (4.5, 4.6): ("0.55", R_MARKERS[4.25]),
    (6.0, 6.0): ("black", R_MARKERS[5.0]),
}
WINDOW_CASES = {
    (2.0, 2.0): (R_COLORS[2.0], R_MARKERS[2.0]),
    (4.24, 2.73): (R_COLORS[4.25], R_MARKERS[4.0]),
    (4.5, 4.6): ("0.55", R_MARKERS[4.25]),
}

WINDOW_T = np.array([300.0, 600.0])
N_CONV = 800
DT_CONV = 0.02
TRANSIENT_CONV = 300.0
TRIALS_CONV = 16

THIN_X = Path([
    (-0.5, -0.38), (-0.38, -0.5), (0.0, -0.12),
    (0.38, -0.5), (0.5, -0.38), (0.12, 0.0),
    (0.5, 0.38), (0.38, 0.5), (0.0, 0.12),
    (-0.38, 0.5), (-0.5, 0.38), (-0.12, 0.0),
    (-0.5, -0.38),
], closed=True)

cuts = pd.read_csv(SIM / "cuts" / "trials.csv")
cuts["r"] = cuts.r.round(6)
cuts["kappa"] = cuts.kappa.round(6)
columns = [f"f_minus_{e:.0e}" for e in EPS]
if not set(columns + ["max_backward_excursion_turns"]).issubset(cuts.columns):
    raise RuntimeError("required negative-velocity or backward-excursion columns are missing")

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.05), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.125)

upper_a = 0.0
for (r, kappa), (color, marker) in POINTS.items():
    data = cuts[(cuts.r == r) & (cuts.kappa == kappa) & (cuts.N == N_EPS)][columns]
    if len(data) < 2:
        raise RuntimeError(f"insufficient threshold data for (r, kappa)=({r:g}, {kappa:g}), N={N_EPS}")
    mean = data.mean().to_numpy(dtype=float)
    sem = data.sem().to_numpy(dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(sem).all():
        raise RuntimeError(f"invalid threshold statistics at (r, kappa)=({r:g}, {kappa:g})")
    upper_a = max(upper_a, float(np.max(mean + sem)))
    ax_a.errorbar(EPS, mean, yerr=sem, fmt="none", color=color, ecolor=color,
                  elinewidth=0.7, capsize=1.4, zorder=2)
    shape = THIN_X if marker == R_MARKERS[4.0] else marker
    fill = color if (r, kappa) in ((4.0, 2.95), (6.0, 6.0)) else "white"
    ax_a.plot(EPS, mean, color=color, lw=0.9, marker=shape,
              ms=4.3 if shape is THIN_X else 4.0,
              mfc=fill, mec=color, mew=0.3 if shape is THIN_X else 0.8,
              zorder=3, label=rf"$({r:g},{kappa:g})$")

ax_a.axvline(1e-3, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":", zorder=1)
ax_a.set_xscale("log")
ax_a.set_xlim(8.0e-5, 1.3e-2)
ax_a.set_ylim(-0.004, max(0.091, 1.13 * upper_a))
ax_a.set_xticks(EPS)
ax_a.set_xticklabels([r"$10^{-4}$", r"$3\times10^{-4}$", r"$10^{-3}$",
                      r"$3\times10^{-3}$", r"$10^{-2}$"])
ax_a.set_xlabel(r"velocity threshold $\varepsilon$")
ax_a.set_ylabel(r"$\langle f_-(\varepsilon)\rangle$")
ax_a.text(1.13e-3, 0.14, "production\nthreshold",
          transform=ax_a.get_xaxis_transform(),
          fontsize=7.4, color=GUIDE_COLOR, ha="left", va="bottom",
          linespacing=0.85)
ax_a.legend(loc="upper right", ncol=2, handlelength=1.55, labelspacing=0.24,
            columnspacing=0.65, handletextpad=0.35, borderaxespad=0.2,
            frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
            title=rf"$(r,\kappa)$, $N={N_EPS}$", title_fontsize=8)
panel_label(ax_a, "a", x=-0.16)

excursion = cuts.max_backward_excursion_turns.to_numpy(dtype=float)
if len(excursion) == 0 or not np.isfinite(excursion).all() or np.any(excursion < 0):
    raise RuntimeError("invalid backward-excursion records")

ax_b.hist(excursion, bins=np.linspace(0.0, 1.1, 56), color="0.48", log=True)
ax_b.axvline(0.5, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":", zorder=2)
ax_b.axvline(1.0, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":", zorder=2)
ax_b.set_xlim(0.0, 1.1)
ax_b.set_xlabel("maximum backward excursion (turns)")
ax_b.set_ylabel("Number of runs")
ax_b.text(0.06, 0.96,
          f"{len(excursion):,} runs\n$N={int(cuts.N.min())}$–${int(cuts.N.max())}$",
          transform=ax_b.transAxes, fontsize=7.5, ha="left", va="top",
          linespacing=0.92)
ax_b.text(0.53, 0.63, "half-turn\nreference",
          transform=ax_b.get_xaxis_transform(), fontsize=7.4,
          color=GUIDE_COLOR, ha="left", va="center",
          linespacing=0.85)
ax_b.text(0.98, 0.30, "net backward\nrevolution",
          transform=ax_b.get_xaxis_transform(), fontsize=7.4,
          color=GUIDE_COLOR, ha="right", va="center",
          linespacing=0.85)
panel_label(ax_b, "b", x=-0.16)

conv = pd.read_csv(
    SIM / "convergence" / "trials.csv",
    usecols=["r", "kappa", "N", "trial", "dt", "T_transient", "T_measure", "min_omega"]
)
conv["r"] = conv.r.round(6)
conv["kappa"] = conv.kappa.round(6)
conv = conv[(conv.N == N_CONV) & np.isclose(conv.dt, DT_CONV)
            & np.isclose(conv.T_transient, TRANSIENT_CONV)
            & conv.T_measure.isin(WINDOW_T)]

for point, (color, marker) in WINDOW_CASES.items():
    case = conv[(conv.r == point[0]) & (conv.kappa == point[1])]
    pair = case.pivot(index="trial", columns="T_measure", values="min_omega")
    if len(pair) != TRIALS_CONV or not set(WINDOW_T).issubset(pair.columns):
        raise RuntimeError(f"incomplete convergence runs for (r,kappa)={point}")
    if not np.isfinite(pair[WINDOW_T].to_numpy(dtype=float)).all():
        raise RuntimeError(f"nonfinite convergence measurements for (r,kappa)={point}")
    values = -pair[WINDOW_T].min(axis=0).to_numpy(dtype=float)
    if np.any(values <= 0):
        raise RuntimeError(f"no negative fitted slope to compare for (r,kappa)={point}")
    normalized = values / values[0]
    shape = THIN_X if marker == R_MARKERS[4.0] else marker
    fill = color if shape is THIN_X else "white"
    ax_c.plot(WINDOW_T, normalized, color=color, lw=0.9, marker=shape,
              ms=4.3 if shape is THIN_X else 4.0,
              mfc=fill, mec=color, mew=0.3 if shape is THIN_X else 0.8,
              label=rf"$({point[0]:g},{point[1]:g})$")

guide_t = np.array([300.0, 600.0])
ax_c.plot(guide_t, (300.0 / guide_t) ** 2, color="0.12", lw=1.55,
          linestyle=(0, (1.2, 1.7)), zorder=10,
          path_effects=[pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()])
ax_c.set_xlim(280.0, 620.0)
ax_c.set_ylim(0.0, 1.08)
ax_c.set_xticks([300, 600])
ax_c.set_xlabel(r"fit window $T$")
ax_c.set_ylabel(r"$\frac{|\Omega_{\min}(T)|}{|\Omega_{\min}(300)|}$",
                fontsize=11.8, labelpad=4)
ax_c.legend(loc="upper right", handlelength=1.45, labelspacing=0.24,
            borderaxespad=0.2,
            title=rf"$(r,\kappa)$; $N={N_CONV}$", title_fontsize=8)
ax_c.plot([0.07, 0.20], [0.13, 0.13], color="0.12", lw=1.4,
          linestyle=(0, (1.2, 1.7)),
          transform=ax_c.transAxes, clip_on=False)
ax_c.text(0.23, 0.13, r"$T^{-2}$ guide", transform=ax_c.transAxes,
          fontsize=8, color="0.12", ha="left", va="center")
panel_label(ax_c, "c", x=-0.16)

save(fig, "si_fig6")
