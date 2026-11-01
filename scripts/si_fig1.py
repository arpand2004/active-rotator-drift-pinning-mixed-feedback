import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import SIM, SINGLE, panel_label, save

summary = pd.read_csv(SIM / "map" / "summary.csv")
if np.any(summary.R2_mean < 2.0 * summary.R1_mean ** 2 - 1.0 - 1e-12):
    raise RuntimeError("two-harmonic bound violated")

fig, ax = plt.subplots(figsize=(SINGLE, 2.9), layout="constrained")
points = ax.scatter(summary.R1_mean, summary.R2_mean, c=summary.kappa, s=2, cmap="viridis", vmin=0.0, vmax=8.0,
                    linewidths=0, rasterized=True)
r1 = np.linspace(0.0, 1.0, 400)
ax.plot(r1, np.maximum(0.0, 2.0 * r1 ** 2 - 1.0), color="black", lw=1.0)
ax.text(0.46, 0.025, r"$\langle R_2\rangle\geq\max(0,\,2\langle R_1\rangle^2-1)$",
        fontsize=8, ha="center", va="bottom")
ax.text(0.5, 0.94, r"$N=200$", transform=ax.transAxes,
        fontsize=8, ha="center", va="top")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel(r"$\langle R_1\rangle$")
ax.set_ylabel(r"$\langle R_2\rangle$")
bar = fig.colorbar(points, ax=ax, pad=0.02)
bar.set_label(r"$\kappa$")
save(fig, "si_fig1")
