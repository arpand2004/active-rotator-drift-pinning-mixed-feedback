import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (SIM, THEORY, DOUBLE, OBS_COLORS, R_COLORS, R_MARKERS, GUIDE_COLOR, GUIDE_WIDTH,
                      fold_marker, panel_label, save)

R_BANDS = (2.0, 4.0, 4.25, 4.5, 5.0, 6.0)
N_SCAN = 3200
R_SLOWING = (5.0, 6.0)
N_SLOWING_MIN = 1600
DISTANCES = np.round(np.arange(0.05, 1.01, 0.05), 2)
FOLD_WINDOW = (3.5, 6.0)
RUN = 5
NEAR_FOLD = (4.5, 4.6)
TRACE_N = 3200
TRACE_TRIAL = 4
TIME_WINDOW = 120.0
MARK_EVERY = 0.5

scan = pd.read_csv(SIM / "oscillation_scan" / "trials.csv")
scan = scan[scan.N == N_SCAN].assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
cuts = pd.read_csv(SIM / "cuts" / "trials.csv").assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
cuts["pinned"] = (cuts["f_plus_1e-03"] + cuts["f_minus_1e-03"]) == 0
folds = pd.read_csv(THEORY / "mf_folds.csv")
folds["r"] = folds.r.round(6)


def first_pinned_run(group):
    group = group.sort_values("kappa")
    pinned, kappa = group.pinned.to_numpy(), group.kappa.to_numpy()
    for i in range(len(pinned) - RUN + 1):
        if pinned[i:i + RUN].all():
            return float(kappa[i])
    return np.nan


def moving_trace(r, kappa):
    for folder in sorted((SIM / "traces").glob("trace_[0-9][0-9][0-9][0-9]")):
        path = folder / "observables.json"
        if not path.is_file():
            continue
        meta = json.loads(path.read_text())
        if (np.isclose(float(meta["r"]), r, rtol=0, atol=1e-8)
                and np.isclose(float(meta["kappa"]), kappa, rtol=0, atol=1e-8)
                and int(meta["N"]) == TRACE_N and int(meta["trial"]) == TRACE_TRIAL
                and abs(float(meta["psi_rate"])) > 1e-3):
            if not (folder / "order_parameters.csv").is_file():
                raise FileNotFoundError(f"missing order-parameter series in {folder}")
            return folder
    raise FileNotFoundError(f"missing moving trace for (r,kappa)=({r:g},{kappa:g}), N={TRACE_N}, trial={TRACE_TRIAL}")


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.55), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.07)
ax_a, ax_b, ax_c = axes

for r in R_BANDS:
    subset = scan[scan.r == r]
    if subset.empty:
        raise RuntimeError(f"missing scan for r={r:g}, N={N_SCAN}")
    group = subset.groupby("kappa", sort=True)
    valid = group.psi_weak_fraction.max() == 0
    rate = group.psi_rate.apply(lambda v: np.mean(np.abs(v)) / np.pi).where(valid)
    err = group.psi_rate.apply(lambda v: np.std(np.abs(v), ddof=1) / np.sqrt(len(v)) / np.pi).where(valid)
    k = rate.index.to_numpy(dtype=float)
    if not np.isfinite(rate.dropna()).all() or not np.isfinite(err.dropna()).all():
        raise RuntimeError(f"invalid rotation-rate statistics for r={r:g}")
    ax_a.fill_between(k, (rate - err).to_numpy(dtype=float), (rate + err).to_numpy(dtype=float),
                      color=R_COLORS[r], alpha=0.25, lw=0)
    every = np.flatnonzero(np.isclose(np.mod(k + 1e-9, MARK_EVERY), 0, atol=1e-6)
                           & rate.notna().to_numpy()).tolist()
    ax_a.plot(k, rate.to_numpy(dtype=float), color=R_COLORS[r], lw=1.1,
              marker=R_MARKERS[r], markevery=every, ms=3.6, mfc="white", mew=0.9)

for kind in ("lower", "upper"):
    rows = folds[(folds.kind == kind) & folds.r.isin((4.5, 5.0, 6.0)) & (folds.kappa <= 8.0)]
    if not rows.empty:
        ax_a.plot(rows.kappa.to_numpy(), np.zeros(len(rows)), clip_on=False, **fold_marker(kind))
ax_a.set_xlim(0.0, 8.0)
ax_a.set_ylim(0.0, 0.6)
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$\langle|\dot\psi_1|\rangle/\pi$")

r_handles = [Line2D([], [], color=R_COLORS[r], marker=R_MARKERS[r], mfc="white",
                    markeredgewidth=0.9, lw=1.1, ms=3.6, label=rf"$r={r:g}$") for r in R_BANDS]
extra_handles = [Line2D([], [], color="none", linewidth=0, label=rf"$N={N_SCAN}$"),
                 Line2D([], [], **{k: v for k, v in fold_marker("upper").items() if k != "zorder"},
                        label=r"upper fold $\kappa_u$"),
                 Line2D([], [], **{k: v for k, v in fold_marker("lower").items() if k != "zorder"},
                        label=r"lower fold $\kappa_\ell$")]
empty_handles = [Line2D([], [], color="none", linewidth=0, label="") for _ in range(3)]
ax_a.legend(handles=r_handles + extra_handles + empty_handles, loc="upper center",
            bbox_to_anchor=(0.55, 0.995), ncol=2, handlelength=1.35,
            labelspacing=0.17, columnspacing=0.75, handletextpad=0.35,
            borderaxespad=0.10)
panel_label(ax_a, "a", x=-0.22)

slope_points = []
for r in R_SLOWING:
    window = cuts[(cuts.r == r) & (cuts.N >= N_SLOWING_MIN)]
    rows = []
    for (size, trial), group in window.groupby(["N", "trial"]):
        edge = first_pinned_run(group[group.kappa.between(*FOLD_WINDOW)])
        if np.isnan(edge):
            continue
        for d in DISTANCES:
            point = group[np.isclose(group.kappa, round(edge - d, 2), rtol=0, atol=1e-7)]
            if len(point) == 1 and point.psi_weak_fraction.iloc[0] == 0:
                rows.append((d, float(point.psi_rate.iloc[0]) ** 2))
    if not rows:
        raise RuntimeError(f"no near-fold rotation rates for r={r:g}")
    data = pd.DataFrame(rows, columns=["d", "w2"]).groupby("d", sort=True).w2.agg(["mean", "sem", "size"])
    if not np.array_equal(data.index.to_numpy(), DISTANCES):
        raise RuntimeError(f"incomplete near-fold distance grid for r={r:g}")
    if not np.isfinite(data[["mean", "sem"]].to_numpy()).all():
        raise RuntimeError(f"invalid near-fold errors for r={r:g}")
    slope_points.append(data)
    ax_b.errorbar(data.index.to_numpy(), data["mean"].to_numpy(), yerr=data["sem"].to_numpy(),
                  fmt=R_MARKERS[r] + "-", color=R_COLORS[r], mfc="white", mew=0.9,
                  ms=3.8, lw=0.9, elinewidth=0.7, capsize=0)

pooled = pd.concat(slope_points)
slope = np.sum(pooled.index.to_numpy() * pooled["mean"].to_numpy()) / np.sum(pooled.index.to_numpy() ** 2)
d_line = np.array([0.0, 1.0])
ax_b.plot(d_line, slope * d_line, color="black", lw=1.0, linestyle=":")
ax_b.set_xlim(0.0, 1.0)
ax_b.set_ylim(0.0, 0.34)
ax_b.set_yticks(np.arange(0.0, 0.31, 0.1))
ax_b.set_xlabel(r"$d=\kappa_\ell^{(N,j)}-\kappa$")
ax_b.set_ylabel(r"$\langle\dot\psi_1^{\,2}\rangle$")
slowing_handles = [Line2D([], [], color=R_COLORS[r], marker=R_MARKERS[r], mfc="white",
                          markeredgewidth=0.9, lw=0.9, ms=3.8, label=rf"$r={r:g}$") for r in R_SLOWING]
slowing_handles.append(Line2D([], [], color="black", lw=1.0,
                              linestyle=":", label=r"guide $\propto d$"))
ax_b.legend(handles=slowing_handles, loc="upper left", handlelength=1.8,
            labelspacing=0.3, title=rf"$N\geq{N_SLOWING_MIN}$", title_fontsize=8,
            borderaxespad=0.2)
panel_label(ax_b, "b", x=-0.22)

folder = moving_trace(*NEAR_FOLD)
order = pd.read_csv(folder / "order_parameters.csv").sort_values("tau")
window = order[order.tau <= order.tau.iloc[0] + TIME_WINDOW]
if len(window) < 10 or not np.isfinite(window[["tau", "R1", "R2"]].to_numpy()).all():
    raise RuntimeError("invalid or incomplete near-fold moving trace")
tau = window.tau.to_numpy(dtype=float) - float(window.tau.iloc[0])
ax_c.plot(tau, window.R1, color=OBS_COLORS["R1"], lw=1.0, label=r"$R_1(\tau)$")
ax_c.plot(tau, window.R2, color=OBS_COLORS["R2"], lw=1.0, label=r"$R_2(\tau)$")
ax_c.set_xlim(0.0, TIME_WINDOW)
ax_c.set_xlabel(r"$\tau$")
ax_c.set_ylabel(r"$R_1$, $R_2$")
ax_c.set_ylim(0.0, 1.15)
ax_c.set_yticks(np.arange(0.0, 1.01, 0.2))
ax_c.legend(loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0,
            title=rf"$(r,\kappa)=({NEAR_FOLD[0]:g},{NEAR_FOLD[1]:g})$, $N={TRACE_N}$",
            title_fontsize=8, borderaxespad=0.25)
panel_label(ax_c, "c", x=-0.22)

save(fig, "fig7")
