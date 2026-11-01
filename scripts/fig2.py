import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from figstyle import (THEORY, DOUBLE, R_COLORS, UNSTABLE_DASH, GUIDE_COLOR, GUIDE_WIDTH,
                      FOLD_COLORS, fold_marker, panel_label, save)

R_SHOWN = (4.5, 5.0, 6.0)
KAPPA_SIMULATED = 8.0
LABEL_R = 5.45

pseudo = pd.read_csv(THEORY / "mf_pseudo_arclength.csv")
folds = pd.read_csv(THEORY / "mf_folds.csv")


def quadrature(r, n=3001, span=8.0):
    a = np.linspace(-span * r, span * r, n)
    w = np.exp(-0.5 * (a / r) ** 2) * np.gradient(a)
    w[[0, -1]] *= 0.5
    return a, w / w.sum()


def unstable_directions(kappa, z, a, w):
    v = a + z
    b = np.abs(v)
    if np.any(b <= 1.0):
        raise RuntimeError("partially drifting stationary state; pinned-branch stability does not apply")
    theta = np.angle(v) + np.arcsin(1.0 / b)
    weight = w / np.sqrt(b * b - 1.0)
    c, s = np.cos(theta), np.sin(theta)
    g = kappa * np.array([[weight @ (c * c), weight @ (c * s)], [weight @ (c * s), weight @ (s * s)]])
    return int(np.count_nonzero(np.linalg.eigvalsh(g) > 1.0))


def fold_point(r, kind):
    row = folds[(folds.r == r) & (folds.kind == kind)].iloc[0]
    return np.array([row.kappa, row.ReZ, row.ImZ])


def split_loop(r):
    loop = pseudo[pseudo.r == r].sort_values("index")[["kappa", "ReZ", "ImZ"]].to_numpy()
    a, w = quadrature(r)
    nu = np.array([unstable_directions(k, complex(x, y), a, w) for k, x, y in loop])
    lower, upper = fold_point(r, "lower"), fold_point(r, "upper")
    i_lo = int(np.argmin(np.linalg.norm(loop - lower, axis=1)))
    i_up = int(np.argmin(np.linalg.norm(loop - upper, axis=1)))
    lo, hi = sorted((i_lo, i_up))
    pieces = {}
    for idx in (np.arange(lo, hi + 1), np.r_[np.arange(hi, len(loop)), np.arange(0, lo + 1)]):
        interior = nu[idx[2:-2]]
        if interior.min() != interior.max():
            raise RuntimeError(f"r={r}: stability changes away from the folds")
        arc = loop[idx]
        start, end = (lower, upper) if np.linalg.norm(arc[0] - lower) < np.linalg.norm(arc[0] - upper) else (upper, lower)
        pieces[int(interior[0])] = np.vstack([start, arc, end])
    if set(pieces) != {0, 1}:
        raise RuntimeError(f"r={r}: expected one stable and one unstable arc, found {sorted(pieces)}")
    return pieces


def r1(arc):
    return np.hypot(arc[:, 1], arc[:, 2]) / arc[:, 0]


loops = {r: split_loop(r) for r in R_SHOWN}

fig = plt.figure(figsize=(DOUBLE, 2.6), layout="constrained")
fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 1.0, 1.0])
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

for ax, plane in ((ax_a, "kappa"), (ax_b, "Z")):
    for r in R_SHOWN:
        for nu, style in ((0, "-"), (1, UNSTABLE_DASH)):
            arc = loops[r][nu]
            x, y = (arc[:, 0], r1(arc)) if plane == "kappa" else (arc[:, 1], arc[:, 2])
            ax.plot(x, y, color=R_COLORS[r], linestyle=style, lw=1.2)
        for kind in ("lower", "upper"):
            k, re, im = fold_point(r, kind)
            fx, fy = (k, np.hypot(re, im) / k) if plane == "kappa" else (re, im)
            ax.plot(fx, fy, **fold_marker(kind))

r_handles = [Line2D([], [], color=R_COLORS[r], lw=1.6, label=rf"$r={r:g}$") for r in R_SHOWN]
style_handles = [Line2D([], [], color="0.3", lw=1.2, label="stable"),
                 Line2D([], [], color="0.3", lw=1.2, linestyle=UNSTABLE_DASH, label="unstable"),
                 Line2D([], [], **fold_marker("upper"), label=r"upper fold $\kappa_u$"),
                 Line2D([], [], **fold_marker("lower"), label=r"lower fold $\kappa_l$")]
ax_a.legend(handles=r_handles + style_handles, loc="lower right", handlelength=1.9, borderaxespad=0.25,
            labelspacing=0.28)
ax_a.set_xlim(3.8, 17.0)
ax_a.set_ylim(0.44, 1.0)
ax_a.set_xlabel(r"$\kappa$")
ax_a.set_ylabel(r"$R_1^{\mathrm{MF}}$")
panel_label(ax_a, "a", x=-0.21)

ax_b.axhline(1.0, color="black", lw=0.9, linestyle=":")
ax_b.text(-9.0, 0.75, r"$\mathrm{Im}\,Z=1$", color=GUIDE_COLOR, fontsize=8, ha="right", va="top")
ax_b.legend(handles=r_handles, ncol=3, loc="lower center", columnspacing=0.45, handlelength=1.0,
            handletextpad=0.25, borderaxespad=0.25)
ax_b.set_xlim(-13.6, -0.6)
ax_b.set_ylim(-3.2, 10.9)
ax_b.set_yticks(np.arange(0, 11, 2))
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_xlabel(r"$\mathrm{Re}\,Z$")
ax_b.set_ylabel(r"$\mathrm{Im}\,Z$")
panel_label(ax_b, "b", x=-0.25)

f = folds.pivot_table(index="r", columns="kind", values="kappa").sort_index()
rv = f.index.to_numpy()
ax_c.fill_between(rv, f["lower"], f["upper"], color="0.91", lw=0, zorder=0)

rr = np.linspace(3.8, 6.0, 200)
ax_c.plot(rr, rr ** 2 / 2, color="black", lw=0.9, linestyle="-.", zorder=2)
ax_c.axhline(KAPPA_SIMULATED, color="black", lw=0.9, linestyle=":", zorder=2)

for kind, col in (("lower", "lower"), ("upper", "upper")):
    ax_c.plot(rv, f[col], color=FOLD_COLORS[kind], lw=1.4, zorder=3)

kl, ku = np.interp(LABEL_R, rv, f["lower"]), np.interp(LABEL_R, rv, f["upper"])
ax_c.annotate(r"$\kappa_u(r)$", (LABEL_R, ku), xytext=(4, -3), textcoords="offset points",
              color=FOLD_COLORS["upper"], fontsize=9, ha="left", va="top")
ax_c.annotate(r"$\kappa_l(r)$", (LABEL_R, kl), xytext=(4, 3), textcoords="offset points",
              color=FOLD_COLORS["lower"], fontsize=9, ha="left", va="bottom")

ax_c.text(4.98, 6.55, "stable pinned\nbranch exists",
          fontsize=8, ha="center", va="center", color="0.25", linespacing=0.9)

guide_handles = [
    Line2D([], [], color="black", lw=0.9, linestyle="-.", label=r"$\kappa=r^2/2$"),
    Line2D([], [], color="black", lw=0.9, linestyle=":", label="simulated range")
]
ax_c.legend(handles=guide_handles, loc="upper left", bbox_to_anchor=(0.03, 0.99),
            handlelength=1.5, borderaxespad=0.25, labelspacing=0.3)

ax_c.set_xlim(3.8, 6.0)
ax_c.set_ylim(3.0, 22.0)
ax_c.set_yticks(np.arange(4, 21, 2))
ax_c.set_xlabel(r"$r$")
ax_c.set_ylabel(r"$\kappa$")
panel_label(ax_c, "c", x=-0.24)

fig.canvas.draw()
height_reference = ax_b.get_position()
fig.set_layout_engine(None)
for ax in (ax_a, ax_c):
    bounds = ax.get_position()
    ax.set_position([bounds.x0, height_reference.y0, bounds.width, height_reference.height])

save(fig, "fig2")
