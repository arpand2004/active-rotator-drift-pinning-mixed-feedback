from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SIM = DATA / "simulations"
THEORY = DATA / "theory"
FIGURES = ROOT / "figures"

SINGLE = 3.37
DOUBLE = 6.69

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "savefig.dpi": 800,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OBS_COLORS = {"R1": "#1F5FA6", "R2": "#D9731A", "D": "#4F7F86", "dR1": "#6B3FA0"}
OBS_CMAPS = {"D": "bone", "R1": "Blues_r", "R2": "Oranges_r", "dR1": "Purples_r"}

R_COLORS = {2.0: "#E6CFA3", 4.0: "#D2A86C", 4.25: "#C8954F", 4.5: "#C08540", 5.0: "#82461C", 6.0: "#2B180A"}
R_MARKERS = {2.0: "^", 4.0: "X", 4.25: "v", 4.5: "s", 5.0: "D", 6.0: "o"}

N_VALUES = (100, 200, 400, 800, 1600, 3200, 6400)
N_COLORS = dict(zip(N_VALUES, plt.get_cmap("YlGnBu")(np.linspace(0.35, 1.0, len(N_VALUES)))))

THEORY_COLOR = "black"
UNSTABLE_DASH = (0, (3.5, 2.0))
GUIDE_COLOR = "0.45"
FOLD_COLORS = {"lower": "#78D64B", "upper": "#128A43"}
GUIDE_WIDTH = 0.7


def fold_marker(kind):
    return dict(marker="o", markersize=3.8, markerfacecolor="white", markeredgecolor=FOLD_COLORS[kind],
                markeredgewidth=1.1, linestyle="none", zorder=6)


def halo(width=1.6):
    return [pe.withStroke(linewidth=width, foreground="white")]


def panel_label(ax, letter, x=-0.2, y=1.0):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=800)
    plt.close(fig)
