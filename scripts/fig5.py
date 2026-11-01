import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (SIM, THEORY, DOUBLE, OBS_COLORS, R_COLORS, R_MARKERS,
                      GUIDE_WIDTH, fold_marker, panel_label, save)

R_BANDS = (2.0, 4.0, 4.5, 5.0, 6.0)
N_SCAN = 3200
TRACE_N = 3200
TRACE_TRIAL = 0
TRACE_POINTS = ((4.5, 3.3), (2.0, 2.0))
TRACE_STYLE = {
    (4.5, 3.3): (R_COLORS[4.5], "s"),
    (2.0, 2.0): (R_COLORS[2.0], "^"),
}
SUPPORT_TRACE = SIM / "traces" / "figure_support" / "trace_r4p5_k3p3_N3200_trial0"
TIME_WINDOW = 40.0
MARK_EVERY = 0.5

scan = pd.read_csv(SIM / "oscillation_scan" / "summary.csv")
scan = scan[scan.N == N_SCAN].assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
cuts = pd.read_csv(SIM / "cuts" / "trials.csv")
cuts["r"] = cuts.r.round(6)
folds = pd.read_csv(THEORY / "mf_folds.csv")
folds["r"] = folds.r.round(6)


def trace_folder(r, kappa):
    if (r, kappa) == (4.5, 3.3):
        candidates = [SUPPORT_TRACE]
    else:
        candidates = sorted((SIM / "traces").glob("trace_[0-9][0-9][0-9][0-9]"))
    for folder in candidates:
        meta_file = folder / "observables.json"
        order_file = folder / "order_parameters.csv"
        diag_file = folder / "oscillator_diagnostics.csv"
        if not all(p.is_file() for p in (meta_file, order_file, diag_file)):
            continue
        meta = json.loads(meta_file.read_text())
        if (np.isclose(float(meta["r"]), r, rtol=0, atol=1e-8)
                and np.isclose(float(meta["kappa"]), kappa, rtol=0, atol=1e-8)
                and int(meta["N"]) == TRACE_N
                and int(meta["trial"]) == TRACE_TRIAL):
            return folder
    raise FileNotFoundError(
        f"Saved trace missing for (r, kappa)=({r:g}, {kappa:g}), N={TRACE_N}, trial={TRACE_TRIAL}. "
        f"Expected support trace at {SUPPORT_TRACE} for (4.5, 3.3)."
    )


def rotation_rate(order):
    t = order.tau.to_numpy(dtype=float)
    psi = order.psi1_unwrapped.to_numpy(dtype=float)
    if len(t) < 3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(psi)):
        raise RuntimeError("invalid collective-phase time series")
    t = t - t.mean()
    return float(np.abs(np.dot(t, psi - psi.mean()) / np.dot(t, t)))


fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.8), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.06)
ax_a, ax_b, ax_c, ax_d = axes.flat

for r in R_BANDS:
    cut = scan[scan.r == r].sort_values("kappa")
    if cut.empty:
        raise RuntimeError(f"no oscillation scan for r={r:g}, N={N_SCAN}")
    k = cut.kappa.to_numpy(dtype=float)
    m = cut.dR1_mean.to_numpy(dtype=float)
    s = cut.dR1_sem.to_numpy(dtype=float)
    ax_a.fill_between(k, np.maximum(0, m - s), m + s, color=R_COLORS[r], alpha=0.25, lw=0)
    every = np.flatnonzero(np.isclose(np.mod(k + 1e-9, MARK_EVERY), 0.0, atol=1e-6)).tolist()
    ax_a.plot(k, m, color=R_COLORS[r], lw=1.15, marker=R_MARKERS[r], markevery=every,
              ms=3.6, mfc="white", mew=0.9, label=rf"$r={r:g}$")

for kind in ("lower", "upper"):
    rows = folds[(folds.kind == kind) & folds.r.isin((4.5, 5.0, 6.0)) & folds.kappa.between(0, 8)]
    if not rows.empty:
        ax_a.plot(rows.kappa, np.full(len(rows), 0.002), clip_on=False,
                  **{**fold_marker(kind), "markersize": 4.5})
ax_a.set_xlim(0, 8)
ax_a.set_ylim(0, 0.10)
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$\langle\delta R_1\rangle$")
band_handles = ax_a.get_legend_handles_labels()[0]
band_handles.extend([
    Line2D([], [], **{**{k: v for k, v in fold_marker("upper").items() if k != "zorder"}, "markersize": 4.5},
           label=r"upper fold $\kappa_u$"),
    Line2D([], [], **{**{k: v for k, v in fold_marker("lower").items() if k != "zorder"}, "markersize": 4.5},
           label=r"lower folds $\kappa_\ell$")
])
ax_a.legend(handles=band_handles, loc="upper right", bbox_to_anchor=(1.0, 0.978),
            handlelength=1.6, labelspacing=0.24, borderaxespad=0.2,
            title=rf"$N={N_SCAN}$", title_fontsize=8)
panel_label(ax_a, "a", x=-0.17)

r0, k0 = TRACE_POINTS[0]
main_folder = trace_folder(r0, k0)
order_main = pd.read_csv(main_folder / "order_parameters.csv").sort_values("tau")
window = order_main[order_main.tau <= order_main.tau.iloc[0] + TIME_WINDOW]
if len(window) < 10:
    raise RuntimeError("representative trace does not cover the requested time window")
tau = window.tau.to_numpy(dtype=float) - window.tau.iloc[0]
ax_b.plot(tau, window.R1, color=OBS_COLORS["R1"], lw=1.1, label=r"$R_1(\tau)$")
ax_b.plot(tau, window.R2, color=OBS_COLORS["R2"], lw=1.1, label=r"$R_2(\tau)$")
ax_b.set_xlim(0, TIME_WINDOW)
ax_b.set_ylim(0, 1.05)
ax_b.set_xlabel(r"$\tau$")
ax_b.set_ylabel(r"$R_1$, $R_2$")
ax_b.legend(loc="upper right", ncol=2, fontsize=8.5, handlelength=1.6,
            columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
ax_b.text(0.02, 0.04, rf"$(r,\kappa)=({r0:g},{k0:g})$, $N={TRACE_N}$",
          transform=ax_b.transAxes, fontsize=8.5, ha="left", va="bottom")
panel_label(ax_b, "b", x=-0.17)

for r, kappa in TRACE_POINTS:
    folder = trace_folder(r, kappa)
    cells = pd.read_csv(folder / "oscillator_diagnostics.csv")
    order = pd.read_csv(folder / "order_parameters.csv")
    rate = rotation_rate(order)
    color, marker = TRACE_STYLE[(r, kappa)]
    ax_c.scatter(cells.a, cells.omega_lsq, s=9, marker=marker,
                 facecolors="none", edgecolors=color, linewidths=0.9,
                 alpha=0.78, rasterized=True,
                 label=rf"$(r,\kappa)=({r:g},{kappa:g})$")    
    ax_c.axhline(rate, color=color, lw=GUIDE_WIDTH, linestyle=":", zorder=0)
    ax_c.text(-13.9, rate + 0.015, rf"$|\dot\psi_1|={rate:.3f}$",
              color="0.45", fontsize=8.5,
              ha="left", va="bottom")  
ax_c.set_xlim(-16, 16)
ax_c.set_ylim(-0.08, 0.9)
ax_c.set_xlabel(r"$a_i$")
ax_c.set_ylabel(r"$\Omega_i$")
trace_handles = [
    Line2D([], [], linestyle="none", marker=TRACE_STYLE[p][1],
           markerfacecolor="none", markeredgecolor=TRACE_STYLE[p][0],
           markeredgewidth=1.1, markersize=5.5,
           label=rf"$(r,\kappa)=({p[0]:g},{p[1]:g})$")
    for p in TRACE_POINTS
]

ax_c.legend(handles=trace_handles, loc="center right",
            bbox_to_anchor=(1.0, 0.37), handletextpad=0.25,
            labelspacing=0.3, borderaxespad=0.2,
            title=rf"$N={TRACE_N}$", title_fontsize=8)
panel_label(ax_c, "c", x=-0.17)

shown = cuts[cuts.r.isin(R_BANDS)]
if shown.empty:
    raise RuntimeError("no cut data for frequency comparison")

x = shown.psi_rate.abs().to_numpy(dtype=float) / np.pi
y = shown.peak_freq_hann.to_numpy(dtype=float)
r1_values = shown.R1.to_numpy(dtype=float)

cloud = ax_d.scatter(
    x, y, c=r1_values, cmap="viridis",
    vmin=0.0, vmax=1.0, s=3.5,
    alpha=0.40, edgecolors="none",
    rasterized=True, zorder=3
)

end = max(0.4, np.ceil(np.max(x) * 20) / 20)
top = max(0.4, np.ceil(np.max(y) * 20) / 20)
line = np.array([0.0, min(end, top)])
harmonic_x = np.array([0.0, min(end, top / 2)])

ax_d.plot(line, line, color="0.25", lw=0.85, zorder=2)
ax_d.plot(harmonic_x, 2.0 * harmonic_x, color="0.25", lw=0.85,
          linestyle=":", zorder=2)
ax_d.set_xlim(0, end)
ax_d.set_ylim(0, top)
ax_d.set_xlabel(r"$|\dot\psi_1|/\pi$")
ax_d.set_ylabel(r"$f_{R_1}$")

ax_d.text(0.33, 0.955, "All trials; all simulated sizes",
          transform=ax_d.transAxes, fontsize=8.4, ha="center", va="top")

cax = ax_d.inset_axes([0.08, 0.76, 0.5, 0.035])
cb = fig.colorbar(cloud, cax=cax, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
cb.ax.tick_params(labelsize=7, length=2, pad=1)
cax.set_title(r"$\overline{R_1}$", fontsize=8, pad=2)

frequency_handles = [
    Line2D([], [], color="0.25", lw=0.85, label=r"$f_{R_1}=|\dot\psi_1|/\pi$"),
    Line2D([], [], color="0.25", lw=0.85, linestyle=":",
           label=r"$f_{R_1}=2|\dot\psi_1|/\pi$")
]
ax_d.legend(handles=frequency_handles, bbox_to_anchor=(0.65, 0.705),
            handlelength=1.5, handletextpad=0.35, labelspacing=0.35,
            borderaxespad=0.2)
panel_label(ax_d, "d", x=-0.19)

save(fig, "fig5")
