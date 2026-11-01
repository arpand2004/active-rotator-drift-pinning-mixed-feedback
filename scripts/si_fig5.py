import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import SIM, DOUBLE, N_COLORS, N_VALUES, panel_label, save

R_SHOWN = (2.0, 6.0)

cuts = pd.read_csv(SIM / "cuts" / "trials.csv").assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))


def weighted_fit(x, y, s):
    w = 1.0 / s ** 2
    design = np.c_[np.ones_like(x), x]
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    return cov @ (design.T @ (w * y)), np.sqrt(np.diag(cov))


def extrapolate(r):
    stats = cuts[cuts.r == r].groupby(["kappa", "N"]).D.agg(["mean", "sem"])
    out = []
    for kappa, group in stats.groupby(level=0):
        x = 1.0 / group.index.get_level_values(1).to_numpy(dtype=float)
        s = group["sem"].to_numpy()
        s = np.where(s > 0, s, np.min(s[s > 0]) if np.any(s > 0) else 1e-4)
        coef, err = weighted_fit(x, group["mean"].to_numpy(), s)
        out.append((kappa, coef[0], err[0]))
    return pd.DataFrame(out, columns=["kappa", "value", "error"]).sort_values("kappa")


fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.75), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.03, wspace=0.07)

for ax, r, x_range, y_range, letter in (
    (axes[0], 2.0, (0.0, 6.0), (0.0, 1.27), "a"),
    (axes[1], 6.0, (2.0, 8.0), (-0.02, 0.255), "b"),
):
    profile = cuts[cuts.r == r].groupby(["N", "kappa"]).D.mean()
    if set(profile.index.get_level_values("N")) != set(N_VALUES):
        raise RuntimeError(f"incomplete system sizes at r={r:g}")
    for size in N_VALUES:
        curve = profile.xs(size).sort_index()
        ax.plot(curve.index, curve.to_numpy(), color=N_COLORS[size], lw=1.0, zorder=3)

    limit = extrapolate(r)
    ax.fill_between(limit.kappa, limit.value - limit.error, limit.value + limit.error,
                    color="0.30", alpha=0.17, lw=0, zorder=1)
    ax.plot(limit.kappa, limit.value, color="0.12", lw=1.2, linestyle="--", zorder=4)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\langle D\rangle$")

    legend_handles = [
        Line2D([], [], color=N_COLORS[size], lw=1.0, label=rf"$N={size}$")
        for size in N_VALUES
    ]
    legend_handles.append(Line2D([], [], color="0.12", lw=1.2, linestyle="--",
                                 label=r"$N\to\infty$ (fit)"))
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
              ncol=3, handlelength=1.45, columnspacing=0.65,
              handletextpad=0.4, labelspacing=0.23, borderaxespad=0.15,
              title=rf"$r={r:g}$", title_fontsize=8)
    panel_label(ax, letter, x=-0.15)

axes[0].set_yticks(np.arange(0.0, 1.01, 0.2))
axes[1].set_yticks(np.arange(0.0, 0.201, 0.05))

save(fig, "si_fig5")
