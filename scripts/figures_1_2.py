from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11.2,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
})


omega = 0.5

a_pos_locked = 1.5
a_neg_locked = -1.5
a_crit_pos = 1.0
a_crit_neg = -1.0
a_drift = 0.5

N_ts = 30
K_ratio = 1.2
K = K_ratio * omega

dt = 0.01
T = 100.0

T_plot_locked = 20.0
T_plot_crit = 30.0
T_plot_drift = 30.0

steps = int(T / dt)

rng = np.random.default_rng(12345)


c_pos_stable = "#1f4e79"
c_pos_unstable = "#85add8"

c_neg_stable = "#b24a1a"
c_neg_unstable = "#efb08f"

c_crit_pos = c_pos_stable
c_crit_neg = c_neg_stable

c_drift = "#4d4d4d"

traj_colors = plt.cm.viridis(np.linspace(0.12, 0.92, N_ts))


def flow(a, theta):
    return 1.0 - a * np.sin(theta)


def fixed_points(a):
    if a > 1:
        theta_s = np.arcsin(1.0 / a)
        theta_u = np.pi - theta_s
        return theta_s, theta_u

    if a < -1:
        alpha = np.arcsin(1.0 / a)
        theta_s = np.pi - alpha
        theta_u = 2.0 * np.pi + alpha
        return theta_s, theta_u

    return None, None


def add_phase_angle_labels(ax, r=1.16, fs=12.2):
    angle_specs = [
        (0.0, r"$0$", "left", "center"),
        (np.pi / 2.0, r"$\pi/2$", "center", "bottom"),
        (np.pi, r"$\pi$", "right", "center"),
        (3.0 * np.pi / 2.0, r"$3\pi/2$", "center", "top"),
    ]

    for th, label, ha, va in angle_specs:
        x = r * np.cos(th)
        y = r * np.sin(th)
        ax.text(x, y, label, ha=ha, va=va, fontsize=fs, color="0.30")


def draw_unit_circle(ax):
    phi = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(np.cos(phi), np.sin(phi), color="0.72", lw=2.0)

    ax.axhline(0, color="0.90", lw=1.0, zorder=0)
    ax.axvline(0, color="0.90", lw=1.0, zorder=0)

    add_phase_angle_labels(ax, r=1.16, fs=12.2)

    ax.set_aspect("equal")
    ax.set_xlim(-1.28, 1.28)
    ax.set_ylim(-1.28, 1.28)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_flow_arrows(ax, a, color, radius=1.0, n_arrows=24,
                     delta=0.22, lw=1.9, alpha=1.0):
    angles = np.linspace(0, 2.0 * np.pi, n_arrows, endpoint=False)

    for th in angles:
        f = flow(a, th)

        if abs(f) < 0.035:
            continue

        sgn = np.sign(f)
        th1 = th - 0.5 * delta * sgn
        th2 = th + 0.5 * delta * sgn

        x1, y1 = radius * np.cos(th1), radius * np.sin(th1)
        x2, y2 = radius * np.cos(th2), radius * np.sin(th2)

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                lw=lw,
                color=color,
                alpha=alpha,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=11,
            ),
            zorder=3,
        )


def draw_fixed_point(ax, theta, color, label=None, size=145):
    x, y = np.cos(theta), np.sin(theta)

    ax.scatter(
        [x], [y],
        s=size,
        facecolor=color,
        edgecolor="white",
        linewidth=1.8,
        zorder=8,
        label=label,
    )

    ax.scatter(
        [x], [y],
        s=size + 35,
        facecolor="none",
        edgecolor="black",
        linewidth=0.7,
        zorder=7,
    )


def set_phase_ticks(ax):
    ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def simulate_coupled_wrapped(a, theta0, omega=0.5, K_ratio=1.2, dt=0.02, T=45.0):
    N = len(theta0)
    steps = int(T / dt)
    t_arr = np.linspace(0.0, T, steps)

    A = a * omega
    K = K_ratio * omega

    theta = np.array(theta0, dtype=float)
    hist = np.zeros((N, steps))
    hist[:, 0] = np.mod(theta, 2.0 * np.pi)

    fc_factor = N / (N - 1) if N > 1 else 1.0

    for step in range(1, steps):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        mean_sin = np.mean(sin_theta)
        mean_cos = np.mean(cos_theta)

        coupling = fc_factor * (
            mean_sin * cos_theta
            -
            mean_cos * sin_theta
        )

        dtheta = omega - A * sin_theta + K * coupling
        theta = theta + dt * dtheta

        hist[:, step] = np.mod(theta, 2.0 * np.pi)

    return t_arr, hist


fig1, axes = plt.subplots(
    1, 3,
    figsize=(15.2, 5.35),
    gridspec_kw={"wspace": 0.03}
)

ax1, ax2, ax3 = axes


draw_unit_circle(ax1)

draw_flow_arrows(
    ax1,
    a_pos_locked,
    c_pos_stable,
    radius=1.035,
    n_arrows=26,
    delta=0.22,
    lw=1.85,
    alpha=0.95,
)

draw_flow_arrows(
    ax1,
    a_neg_locked,
    c_neg_stable,
    radius=0.965,
    n_arrows=26,
    delta=0.22,
    lw=1.85,
    alpha=0.95,
)

theta_s_pos, theta_u_pos = fixed_points(a_pos_locked)
theta_s_neg, theta_u_neg = fixed_points(a_neg_locked)

draw_fixed_point(ax1, theta_s_pos, c_pos_stable)
draw_fixed_point(ax1, theta_u_pos, c_pos_unstable)

draw_fixed_point(ax1, theta_s_neg, c_neg_stable)
draw_fixed_point(ax1, theta_u_neg, c_neg_unstable)

ax1.set_title(r"Two fixed points: $|a|>1$")


draw_unit_circle(ax2)

draw_flow_arrows(
    ax2,
    a_crit_pos,
    c_pos_stable,
    radius=1.035,
    n_arrows=26,
    delta=0.21,
    lw=1.85,
    alpha=0.95,
)

draw_flow_arrows(
    ax2,
    a_crit_neg,
    c_neg_stable,
    radius=0.965,
    n_arrows=26,
    delta=0.21,
    lw=1.85,
    alpha=0.95,
)

theta_c_pos = np.pi / 2.0
theta_c_neg = 3.0 * np.pi / 2.0

draw_fixed_point(ax2, theta_c_pos, c_crit_pos, size=155)
draw_fixed_point(ax2, theta_c_neg, c_crit_neg, size=155)

ax2.set_title(r"Critical fixed point: $|a|=1$")


draw_unit_circle(ax3)

draw_flow_arrows(
    ax3,
    a_drift,
    c_drift,
    radius=1.0,
    n_arrows=30,
    delta=0.26,
    lw=2.0,
    alpha=0.95,
)

sample_angles = np.linspace(0, 2*np.pi, 20, endpoint=False)

ax3.scatter(
    np.cos(sample_angles),
    np.sin(sample_angles),
    s=24,
    facecolor="white",
    edgecolor="0.45",
    linewidth=0.8,
    zorder=4,
)

ax3.set_title(r"No fixed point: $|a|<1$")


handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_pos_stable,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a>1$: Stable FP, Q1",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_pos_unstable,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a>1$: Unstable FP, Q2",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_neg_stable,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a<-1$: Stable FP, Q3",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_neg_unstable,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a<-1$: Unstable FP, Q4",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_crit_pos,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a=1$: Critical FP at $\pi/2$",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=c_crit_neg,
        markeredgecolor="black",
        markeredgewidth=0.8,
        markersize=9.8,
        label=r"$a=-1$: Critical FP at $3\pi/2$",
    ),
]

fig1.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.015),
    columnspacing=1.8,
    handletextpad=0.55,
    borderaxespad=0.0,
    alignment="center",
)

fig1.tight_layout(rect=[0, 0.11, 1, 1.00])

fig1.savefig(OUTPUT_DIR / "figure_1.pdf", dpi=600, bbox_inches="tight")
fig1.savefig(OUTPUT_DIR / "figure_1.png", dpi=600, bbox_inches="tight")
plt.close(fig1)


theta0_common = rng.uniform(0.0, 2.0 * np.pi, N_ts)

t_locked, hist_locked = simulate_coupled_wrapped(
    a_pos_locked,
    theta0_common,
    omega=omega,
    K_ratio=K_ratio,
    dt=dt,
    T=T,
)

t_crit, hist_crit = simulate_coupled_wrapped(
    a_crit_pos,
    theta0_common,
    omega=omega,
    K_ratio=K_ratio,
    dt=dt,
    T=T,
)

t_drift, hist_drift = simulate_coupled_wrapped(
    a_drift,
    theta0_common,
    omega=omega,
    K_ratio=K_ratio,
    dt=dt,
    T=T,
)

theta_s_locked, theta_u_locked = fixed_points(a_pos_locked)
theta_c = np.pi / 2.0

mask_locked = t_locked <= T_plot_locked
mask_crit = t_crit <= T_plot_crit
mask_drift = t_drift <= T_plot_drift

text_box = dict(
    boxstyle="round,pad=0.25",
    facecolor="white",
    edgecolor="none",
    alpha=0.82,
)

x_text_locked = T_plot_locked * 0.90
x_text_crit = T_plot_crit * 0.92

fig2, axes2 = plt.subplots(
    1, 3,
    figsize=(15.3, 5.0),
    sharey=True,
    gridspec_kw={
        "width_ratios": [0.28, 0.36, 0.36],
        "wspace": 0.08,
    },
)

bx1, bx2, bx3 = axes2


for i in range(N_ts):
    bx1.plot(
        t_locked[mask_locked],
        hist_locked[i, mask_locked],
        lw=1.12,
        alpha=0.93,
        color=traj_colors[i],
    )

bx1.axhline(theta_s_locked, color=c_pos_stable, lw=2.2, ls="--")
bx1.axhline(theta_u_locked, color=c_pos_unstable, lw=2.0, ls="--")

bx1.text(
    x_text_locked,
    theta_s_locked + 0.14,
    "stable",
    color=c_pos_stable,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=text_box,
)

bx1.text(
    x_text_locked,
    theta_u_locked + 0.14,
    "unstable",
    color=c_pos_unstable,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=text_box,
)

bx1.set_title(r"Locked ($a=1.5$)")
bx1.set_xlabel(r"$t$")
bx1.set_ylabel(r"$\theta_i\;(\mathrm{mod}\;2\pi)$")
bx1.set_xlim(0, T_plot_locked)
bx1.set_ylim(0, 2*np.pi)
set_phase_ticks(bx1)


for i in range(N_ts):
    bx2.plot(
        t_crit[mask_crit],
        hist_crit[i, mask_crit],
        lw=1.12,
        alpha=0.93,
        color=traj_colors[i],
    )

bx2.axhline(theta_c, color=c_crit_pos, lw=2.2, ls="--")

bx2.text(
    x_text_crit,
    theta_c + 0.14,
    "critical",
    color=c_crit_pos,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox=text_box,
)

bx2.set_title(r"Critical ($a=1$)")
bx2.set_xlabel(r"$t$")
bx2.set_xlim(0, T_plot_crit)
bx2.set_ylim(0, 2*np.pi)
set_phase_ticks(bx2)


for i in range(N_ts):
    bx3.plot(
        t_drift[mask_drift],
        hist_drift[i, mask_drift],
        lw=1.12,
        alpha=0.93,
        color=traj_colors[i],
    )

bx3.set_title(r"Drifting ($a=0.5$)")
bx3.set_xlabel(r"$t$")
bx3.set_xlim(0, T_plot_drift)
bx3.set_ylim(0, 2*np.pi)
set_phase_ticks(bx3)


for ax in axes2:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.tight_layout(rect=[0, 0.02, 1, 1.00])

fig2.savefig(OUTPUT_DIR / "figure_2.pdf", dpi=600, bbox_inches="tight")
fig2.savefig(OUTPUT_DIR / "figure_2.png", dpi=600, bbox_inches="tight")
plt.close(fig2)