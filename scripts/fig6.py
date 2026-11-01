import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import SIM, DOUBLE, R_COLORS, R_MARKERS, N_COLORS, N_VALUES, GUIDE_COLOR, GUIDE_WIDTH, panel_label, save

CONTROL = (4.0, 0.0)
RIDGE = ((4.0, 2.95), (4.5, 3.3), (5.0, 3.5), (6.0, 3.85))
R_PROFILE = 4.0

cuts = pd.read_csv(SIM / "cuts" / "trials.csv", usecols=["r", "kappa", "N", "dR1"])
cuts["r"] = cuts.r.round(6)
cuts["kappa"] = cuts.kappa.round(6)


def size_series(frame, r, kappa):
    data = frame[(frame.r == r) & (frame.kappa == kappa)]
    stats = data.groupby("N").dR1.agg(["mean", "sem", "size"]).sort_index()
    if not np.array_equal(stats.index.to_numpy(), np.array(N_VALUES)):
        raise RuntimeError(f"missing or unexpected system sizes at r={r}, kappa={kappa}")
    if not np.isfinite(stats[["mean", "sem"]].to_numpy()).all():
        raise RuntimeError(f"invalid amplitude statistics at r={r}, kappa={kappa}")
    if (stats["mean"] <= 0).any():
        raise RuntimeError(f"nonpositive amplitude at r={r}, kappa={kappa}")
    return stats


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.08)

control = size_series(cuts, *CONTROL)
n = control.index.to_numpy(dtype=float)
amplitude = np.exp(np.mean(np.log(control["mean"].to_numpy() * np.sqrt(n))))
n_line = np.array([70.0, 9000.0])

ax_a.plot(n_line, amplitude / np.sqrt(n_line), color="black", lw=1.2,
          linestyle=":", zorder=1)

ax_a.errorbar(n, control["mean"], yerr=control["sem"], fmt="o", color="0.25",
              mfc="0.25", ms=3.8, elinewidth=0.7, capsize=0, zorder=3)

for r, kappa in RIDGE:
    stats = size_series(cuts, r, kappa)
    ax_a.errorbar(
        stats.index, stats["mean"], yerr=stats["sem"],
        fmt=("s" if r == 4.0 else R_MARKERS[r]) + "-",
        color=R_COLORS[r],
        mfc=R_COLORS[r] if r == 4.0 else "white",
        mew=0.9, ms=3.8, lw=0.8,
        elinewidth=0.7, capsize=0, zorder=3
    )
ax_a.set_xscale("log")
ax_a.set_yscale("log")
ax_a.set_xlim(*n_line)
ax_a.set_ylim(1.5e-3, 0.21)
ax_a.set_xlabel(r"$N$")
ax_a.set_ylabel(r"$\langle\delta R_1\rangle$")

ridge_handles = [
    Line2D([], [], color=R_COLORS[r],
           marker="s" if r == 4.0 else R_MARKERS[r],
           mfc=R_COLORS[r] if r == 4.0 else "white",
           lw=0.8, ms=3.8, label=rf"$({r:g},{k:g})$")
    for r, k in RIDGE
]
control_handles = [Line2D([], [], color="0.25", marker="o", lw=0, ms=3.8,
                          label=rf"$({CONTROL[0]:g},{CONTROL[1]:g})$ control"),
                   Line2D([], [], color="black", lw=1.2, linestyle=":",
                          label=r"guide $\propto N^{-1/2}$")]
ridge_legend = ax_a.legend(handles=ridge_handles, loc="center", bbox_to_anchor=(0.65, 0.54),
                           ncol=2, title=r"$(r,\kappa)$", title_fontsize=8,
                           handlelength=1.6, labelspacing=0.35, columnspacing=0.9,
                           borderaxespad=0.2)
ax_a.add_artist(ridge_legend)
ax_a.legend(handles=control_handles, loc="lower left", bbox_to_anchor=(0.03, 0.04),
            handlelength=1.8, labelspacing=0.35, borderaxespad=0.2)
panel_label(ax_a, "a", x=-0.15)

profile = cuts[cuts.r == R_PROFILE].groupby(["N", "kappa"]).dR1.mean()
if set(profile.index.get_level_values("N")) != set(N_VALUES):
    raise RuntimeError(f"incomplete system-size coverage for r={R_PROFILE}")
reference_k = None
for size in N_VALUES:
    curve = profile.xs(size).sort_index()
    k = curve.index.to_numpy(dtype=float)
    if reference_k is None:
        reference_k = k
    elif not np.array_equal(k, reference_k):
        raise RuntimeError(f"inconsistent coupling grid for N={size}")
    ax_b.plot(k, curve.to_numpy(), color=N_COLORS[size], lw=1.0, label=rf"$N={size}$")
ax_b.set_xlim(0.0, 8.0)
ax_b.set_ylim(0.0, 0.060)
ax_b.set_xlabel(r"$\kappa$")
ax_b.set_ylabel(r"$\langle\delta R_1\rangle$")
ax_b.legend(loc="upper right", title=rf"$r={R_PROFILE:g}$", title_fontsize=8,
            handlelength=1.6, labelspacing=0.25)
panel_label(ax_b, "b", x=-0.15)

save(fig, "fig6")
