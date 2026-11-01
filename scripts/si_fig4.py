import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from figstyle import DATA, THEORY, DOUBLE, R_COLORS, R_MARKERS, OBS_COLORS, panel_label, save

VALIDATION = DATA / "validations" / "iid_targeted_v3"
RIDGE = ((4.0, 2.95), (4.5, 3.3), (5.0, 3.5), (6.0, 3.85))
CONTROL = (4.0, 0.0)
R_STATIONARY = (4.5, 5.0, 6.0)
N_VALUES = (100, 200, 400, 800, 1600, 3200, 6400)
SCHEMES = ("paired", "iid")
SCHEME_NAME = {"paired": r"$\pm$-paired Gaussian", "iid": "i.i.d. Gaussian"}
SCHEME_STYLE = {
    "paired": dict(ls="-", mfc="auto", zorder=4),
    "iid": dict(ls=(0, (2.2, 1.5)), mfc="white", zorder=3),
}


def read_csv(name):
    path = VALIDATION / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation CSV: {path}")
    frame = pd.read_csv(path)
    for col in ("r", "kappa"):
        if col in frame:
            frame[col] = frame[col].round(6)
    return frame


plateau = read_csv("plateau_by_size.csv")
comparison = read_csv("comparison_by_size.csv")
extrap = read_csv("extrapolations_1_over_N.csv")
roots = pd.read_csv(THEORY / "mf_roots_on_grid.csv")
roots["r"] = roots.r.round(6)
roots["kappa"] = roots.kappa.round(6)


def select(frame, **conditions):
    result = frame
    for key, value in conditions.items():
        result = result[result[key] == value]
    return result


def expected_sizes(frame, label):
    result = frame.sort_values("N")
    if not np.array_equal(result.N.to_numpy(dtype=int), np.asarray(N_VALUES)):
        raise RuntimeError(f"Incomplete system-size series: {label}")
    return result


def fit_for(scheme, r, kappa, observable, pinned=False):
    sub = select(extrap, scheme=scheme, r=r, kappa=kappa, observable=observable,
                 conditioned_on_pinned=pinned)
    if len(sub) != 1:
        raise RuntimeError(f"Expected one extrapolation: {scheme}, {r}, {kappa}, {observable}")
    fit = sub.iloc[0]
    if not np.isfinite([fit.intercept, fit.intercept_se, fit.slope]).all():
        raise RuntimeError(f"Invalid extrapolation: {scheme}, {r}, {kappa}, {observable}")
    return fit


def root_for(r, observable):
    sub = roots[(roots.r == r) & (roots.kappa == 6.0) & (roots.n_unstable == 0)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected unique stable mean-field root for r={r}")
    return float(sub[observable].iloc[0])


def scheme_handles(color="0.55", include_mf=False):
    handles = [
        Line2D([], [], color=color, marker="h", mfc=color, mec=color, ms=4.1, lw=0.95,
               linestyle="-", label=SCHEME_NAME["paired"]),
        Line2D([], [], color=color, marker="h", mfc="white", mec=color, ms=4.1, lw=0.95,
               linestyle=(0, (2.2, 1.5)), label=SCHEME_NAME["iid"]),
    ]
    if include_mf:
        handles.append(Line2D([], [], color="black", marker="D", mfc="none", mec="black", mew=1.0,
                              ms=5.0, linestyle="none", label="mean field"))
    return handles


fig = plt.figure(figsize=(DOUBLE, 5.45))
ax_a = fig.add_axes([0.09, 0.565, 0.405, 0.370])
ax_b = fig.add_axes([0.59, 0.735, 0.375, 0.200])
ax_b_small = fig.add_axes([0.59, 0.558, 0.375, 0.095])
ax_c = fig.add_axes([0.09, 0.110, 0.405, 0.345])
ax_d = fig.add_axes([0.59, 0.110, 0.375, 0.345])

for r, k in (CONTROL, *RIDGE):
    color = "0.25" if k == 0 else R_COLORS[r]
    marker = "o" if k == 0 else ("s" if r == 4.0 else R_MARKERS[r])
    for scheme in SCHEMES:
        frame = expected_sizes(select(plateau, scheme=scheme, r=r, kappa=k), (r, k, scheme))
        style = SCHEME_STYLE[scheme]
        ax_a.errorbar(frame.N, frame.mean_dR1, yerr=frame.sem_dR1,
                      fmt=marker, color=color,
                      mfc=color if scheme == "paired" else "white", mec=color,
                      mew=0.9, ms=3.65, lw=0.9, linestyle=style["ls"],
                      elinewidth=0.65, capsize=0, zorder=style["zorder"])

control = expected_sizes(select(plateau, scheme="paired", r=4.0, kappa=0.0), "control")
scale = np.exp(np.mean(np.log(control.mean_dR1.to_numpy() * np.sqrt(control.N.to_numpy()))))
n_line = np.array([80.0, 8000.0])
ax_a.plot(n_line, scale / np.sqrt(n_line), color="black", linestyle=":", lw=0.9, zorder=1)
ax_a.set(xscale="log", yscale="log", xlim=(80, 8000), ylim=(0.0015, 0.24),
         xlabel=r"$N$", ylabel=r"$\langle\delta R_1\rangle$")

point_handles = []
for r, k in RIDGE:
    color = R_COLORS[r]
    marker = "s" if r == 4.0 else R_MARKERS[r]
    point_handles.append(Line2D([], [], marker=marker, color=color, mfc=color,
                                ms=3.7, lw=0.9, label=rf"$({r:g},{k:g})$"))

leg = ax_a.legend(handles=point_handles, loc="upper right", ncol=2,
                  title=r"$(r,\kappa)$", title_fontsize=8, fontsize=8,
                  alignment="center",
                  handlelength=1.1, handletextpad=0.25, columnspacing=0.7,
                  labelspacing=0.22, borderaxespad=0.15)
ax_a.add_artist(leg)

control_handle = Line2D([], [], marker="o", color="0.25", mfc="0.25",
                        lw=0.9, ms=3.7, label=r"$(4,0)$ control")

handles = [control_handle] + scheme_handles() + [
    Line2D([], [], color="black", lw=0.9, linestyle=":",
           label=r"guide $\propto N^{-1/2}$")
]

ax_a.legend(handles=handles, loc="lower left", fontsize=8,
            handlelength=1.5, labelspacing=0.22, borderaxespad=0.2)
panel_label(ax_a, "a", x=-0.18)

frame = expected_sizes(select(comparison, r=4.5, kappa=8.0), "reentrant drift")
x = 1.0 / frame.N.to_numpy(dtype=float)

for scheme in SCHEMES:
    color = OBS_COLORS["D"]
    style = SCHEME_STYLE[scheme]
    ax_b.errorbar(x, frame[f"D_{scheme}_mean"], yerr=frame[f"D_{scheme}_sem"],
                  fmt="o", color=color, mfc=color if scheme == "paired" else "white",
                  mec=color, mew=0.9, ms=3.9, linestyle=style["ls"],
                  lw=0.9, elinewidth=0.7, capsize=0, zorder=style["zorder"])

ax_b.set(xlim=(-0.00048, 0.0105), ylim=(-0.015, 0.345),
         xlabel=r"$1/N$", ylabel=r"$\langle D\rangle$")
ax_b.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2), useMathText=True)
ax_b.text(0.025, 0.95, r"$(r,\kappa)=(4.5,8)$",
          transform=ax_b.transAxes, ha="left", va="top", fontsize=8)

ax_b.legend(handles=scheme_handles(), loc="upper right",
            bbox_to_anchor=(0.99, 0.985), fontsize=8,
            handlelength=1.25, handletextpad=0.26, labelspacing=0.2,
            borderaxespad=0.0)
panel_label(ax_b, "b", x=-0.18)

for scheme in SCHEMES:
    moving = 1.0 - frame[f"fraction_pinned_{scheme}"].to_numpy(dtype=float)
    style = SCHEME_STYLE[scheme]

    ax_b_small.plot(frame.N, moving, color=OBS_COLORS["D"],
                    linestyle=style["ls"], marker="o", ms=3.25, mew=0.85,
                    mfc=OBS_COLORS["D"] if scheme == "paired" else "white",
                    mec=OBS_COLORS["D"], lw=0.85)

ax_b_small.set_xscale("log", base=2)
ax_b_small.set_xlim(80, 8000)
ax_b_small.set_ylim(-0.075, 1.075)
ax_b_small.set_xticks([100, 400, 1600, 6400])
ax_b_small.get_xaxis().set_major_formatter(ScalarFormatter())
ax_b_small.set_yticks([0, 0.5, 1])
ax_b_small.set_xlabel(r"$N$", fontsize=8, labelpad=1)
ax_b_small.set_ylabel(r"$f_{\mathrm{moving}}$", fontsize=8, labelpad=1)
ax_b_small.tick_params(labelsize=8, pad=1.5)

for ax, observable, letter, ylims in (
    (ax_c, "R1", "c", (0.68, 0.85)),
    (ax_d, "R2", "d", (0.26, 0.585)),
):
    for r in R_STATIONARY:
        color = R_COLORS[r]
        marker = R_MARKERS[r]

        for scheme, shift in (("paired", -0.065), ("iid", 0.065)):
            fit = fit_for(scheme, r, 6.0, observable, pinned=True)
            style = SCHEME_STYLE[scheme]

            ax.errorbar([r + shift], [fit.intercept], yerr=[fit.intercept_se],
                        fmt=marker, color=color, mec=color, mew=0.95, ms=4.35,
                        mfc=color if scheme == "paired" else "white",
                        elinewidth=0.65, capsize=0, zorder=style["zorder"])

        ax.plot(r, root_for(r, observable), marker="D", ms=5.6,
                color="black", mfc="none", mec="black", mew=1.0, ls="none", zorder=5)

    ax.set(xlim=(4.23, 6.27), ylim=ylims,
           xticks=list(R_STATIONARY), xlabel=r"$r$",
           ylabel=rf"$R_{{{1 if observable == 'R1' else 2},\infty}}$")
    panel_label(ax, letter, x=-0.18)

ax_c.text(0.05, 0.18, r"$\kappa=6$", transform=ax_c.transAxes,
          ha="left", va="top", fontsize=9)

ax_d.text(0.05, 0.08, r"$\kappa=6$", transform=ax_d.transAxes,
          ha="left", va="bottom", fontsize=9)

r_handles = [
    Line2D([], [], color=R_COLORS[r], marker=R_MARKERS[r], mfc=R_COLORS[r],
           ms=4.0, lw=0, label=rf"$r={r:g}$")
    for r in R_STATIONARY
]

leg_r = ax_c.legend(handles=r_handles, ncol=3, loc="lower left",
                    fontsize=8, handletextpad=0.3, columnspacing=0.65,
                    borderaxespad=0.15)
ax_c.add_artist(leg_r)

for ax in (ax_c, ax_d):
    ax.legend(handles=scheme_handles(include_mf=True), loc="upper right",
              fontsize=8, handlelength=1.3, handletextpad=0.35,
              labelspacing=0.24, borderaxespad=0.2)

fig.canvas.draw()
for text in fig.findobj(match=plt.Text):
    if text.get_fontsize() < 8:
        text.set_fontsize(8)

save(fig, "si_fig4")
