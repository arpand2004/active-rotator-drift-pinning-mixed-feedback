import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import DATA, THEORY, DOUBLE, R_COLORS, UNSTABLE_DASH, GUIDE_COLOR, GUIDE_WIDTH, fold_marker, panel_label, save

R_SHOWN = (4.5, 5.0, 6.0)
branches = pd.read_csv(THEORY / "mf_branches.csv")
roots = pd.read_csv(THEORY / "mf_roots_on_grid.csv")
folds = pd.read_csv(THEORY / "mf_folds.csv")
zero = pd.read_csv(DATA / "checks" / "zero_linearization.csv").sort_values("r")
for frame in (branches, roots, folds):
    frame["r"] = frame.r.round(6)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.45), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.07)
ax_a, ax_b, ax_c = axes

for r in R_SHOWN:
    selected = branches[branches.r == r]
    for _, loop in selected.groupby("branch"):
        loop = loop.sort_values("index")
        for nu, style in ((0, "-"), (1, UNSTABLE_DASH)):
            for part in runs(loop.n_unstable.to_numpy() == nu):
                ax_a.plot(loop.kappa.to_numpy()[part], loop.R2.to_numpy()[part], color=R_COLORS[r], lw=1.1, linestyle=style)
    unstable = roots[(roots.r == r) & (roots.n_unstable == 1)].sort_values("kappa")
    ax_b.plot(unstable.kappa, unstable.sigma_max, color=R_COLORS[r], lw=1.1,
              linestyle=UNSTABLE_DASH, label=rf"$r={r:g}$")
    lower = folds[(folds.r == r) & (folds.kind == "lower")].kappa.iloc[0]
    ax_b.plot(lower, 0.0, clip_on=False, **fold_marker("lower"))
    upper = folds[(folds.r == r) & (folds.kind == "upper")].kappa.iloc[0]
    if upper <= 8.0:
        ax_b.plot(upper, 0.0, clip_on=False, **fold_marker("upper"))

ax_a.set_xlim(3.8, 17.0)
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$R_2^{\mathrm{MF}}$")
color_handles = [
    Line2D([], [], color=R_COLORS[r], lw=1.1, label=rf"$r={r:g}$")
    for r in R_SHOWN
]
status_handles = [
    Line2D([], [], color="0.3", lw=1.1, label="stable"),
    Line2D([], [], color="0.3", lw=1.1, linestyle=UNSTABLE_DASH, label="unstable")
]
ax_a.legend(handles=color_handles + status_handles, loc="lower right",
            bbox_to_anchor=(0.99, 0.02), ncol=1, handlelength=1.8,
            handletextpad=0.45, labelspacing=0.28, borderaxespad=0.15)
panel_label(ax_a, "a", x=-0.22)

ax_b.set_xlim(4.0, 8.0)
ax_b.set_ylim(0.0, 2.3)
ax_b.set_yticks(np.arange(0.0, 2.01, 0.5))
ax_b.set_xlabel(r"$\kappa$")
ax_b.set_ylabel(r"$\sigma_{\max}$")
fold_handles = [
    Line2D([], [], **{k: v for k, v in fold_marker(kind).items() if k != "zorder"},
           label=(r"upper fold $\kappa_u$" if kind == "upper" else r"lower fold $\kappa_\ell$"))
    for kind in ("upper", "lower")
]
ax_b.legend(handles=fold_handles, loc="upper left", handlelength=1.4,
            handletextpad=0.38, labelspacing=0.25, borderaxespad=0.2)
panel_label(ax_b, "b", x=-0.22)

ax_c.plot(zero.r, zero.discriminant, color="black", lw=1.0, marker="o", ms=2.5)
ax_c.axhline(0.0, color=GUIDE_COLOR, lw=GUIDE_WIDTH, linestyle=":")
ax_c.set_xlim(3.8, 6.0)
ax_c.set_ylim(None, 0.005)
ax_c.set_xlabel(r"$r$")
ax_c.set_ylabel(r"response-Jacobian discriminant $\Delta_M$")
ax_c.text(0.35, 0.86, r"$Z=0$", transform=ax_c.transAxes,
          ha="left", va="top", fontsize=9)
panel_label(ax_c, "c", x=-0.26)

save(fig, "si_fig2")
