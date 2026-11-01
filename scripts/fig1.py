import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import SIM, THEORY, DOUBLE, FOLD_COLORS, panel_label, save

KAPPA_MAX = 8.0
STEP = 0.05
CMAP = "magma"

summary = pd.read_csv(SIM / "map" / "summary.csv")
folds = pd.read_csv(THEORY / "mf_folds.csv")
r_vals = np.sort(summary.r.round(6).unique())
k_vals = np.sort(summary.kappa.round(6).unique())
if not (np.allclose(np.diff(r_vals), STEP) and np.allclose(np.diff(k_vals), STEP)):
    raise RuntimeError("map grid is not uniform at the production spacing")
if summary.trials.nunique() != 1:
    raise RuntimeError("unequal realization counts across the map")
if len(summary) != len(r_vals) * len(k_vals):
    raise RuntimeError("map summary is incomplete or contains duplicate grid points")


def grid(column):
    table = summary.assign(r=summary.r.round(6), kappa=summary.kappa.round(6)).pivot(
        index="kappa", columns="r", values=column
    )
    return table.loc[k_vals, r_vals].to_numpy()


r_edges = np.r_[r_vals - STEP / 2, r_vals[-1] + STEP / 2]
k_edges = np.r_[k_vals - STEP / 2, k_vals[-1] + STEP / 2]
fold_table = folds.pivot_table(index="r", columns="kind", values="kappa").sort_index()

panels = [
    ("D_mean", r"$\langle D\rangle$", 1.0),
    ("R1_mean", r"$\langle R_1\rangle$", 1.0),
    ("R2_mean", r"$\langle R_2\rangle$", 1.0),
    ("dR1_mean", r"$\langle\delta R_1\rangle$", 0.065),
]

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.0), layout="constrained", sharex=True, sharey=True)
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.05, hspace=0.05)

for ax, (column, label, maximum), letter in zip(axes.flat, panels, "abcd"):
    values = grid(column)
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite values in {column}")
    mesh = ax.pcolormesh(
        r_edges, k_edges, values, cmap=CMAP, vmin=0.0, vmax=maximum,
        shading="flat", rasterized=True
    )
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.06)
    bar.set_label(label)
    bar.ax.tick_params(labelsize=8)
    ax.set_xlim(r_edges[0], r_edges[-1])
    ax.set_ylim(k_edges[0], KAPPA_MAX + STEP / 2)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks(np.arange(0, 9, 2))
    panel_label(ax, letter, x=-0.16)

for ax in axes[1]:
    ax.set_xlabel(r"$r$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\kappa$")

ax_d = axes[0, 0]
ax_d.annotate(
    "independent\ndrift",
    xy=(0.18, 0.12), xytext=(0.92, 1.95),
    color="#1A1A1A", fontsize=8, ha="center", va="center",
    linespacing=1.0, multialignment="center",
    arrowprops=dict(
        arrowstyle="-|>", color="#1A1A1A", lw=0.75,
        shrinkA=1.5, shrinkB=0, mutation_scale=7
    )
)
ax_d.text(0.9, 5.2, "coherent\ndrift", color="0.13", fontsize=8,
          ha="center", va="center", linespacing=1.0)
ax_d.text(4.95, 1.15, "antipodal\nlow drift", color="white", fontsize=8,
          ha="center", va="center", linespacing=1.0)
ax_d.text(5.15, 6.05, "coherent\npinning", color="white", fontsize=8,
          ha="center", va="center", linespacing=1.0)

ax_r1 = axes[0, 1]
for kind in ("lower", "upper"):
    curve = fold_table[kind]
    inside = curve <= KAPPA_MAX + 1e-9
    ax_r1.plot(curve.index[inside], curve[inside], color=FOLD_COLORS[kind], lw=1.6,
               solid_capstyle="round", zorder=5)

fold_handles = [
    Line2D([], [], color=FOLD_COLORS["upper"], lw=1.6, label=r"upper fold $\kappa_u(r)$"),
    Line2D([], [], color=FOLD_COLORS["lower"], lw=1.6, label=r"lower fold $\kappa_\ell(r)$"),
]
ax_r1.legend(handles=fold_handles, loc="upper left", bbox_to_anchor=(0.025, 0.99),
             fontsize=8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.24)

save(fig, "fig1")
