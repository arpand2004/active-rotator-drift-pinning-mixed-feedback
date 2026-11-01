import numpy as np
import pandas as pd
from figstyle import SIM, THEORY
from tablefmt import compact, write
 
CUTS = {4.5: (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.6),
        5.0: (4.25, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0),
        6.0: (4.3, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)}
N_LARGEST = 6400
MIN_PINNED = 3
FOLD_MARGIN = 0.15
 
cuts = pd.read_csv(SIM / "cuts" / "trials.csv").assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
cuts["pinned"] = (cuts["f_plus_1e-03"] + cuts["f_minus_1e-03"]) == 0
roots = pd.read_csv(THEORY / "mf_roots_on_grid.csv").assign(kappa=lambda d: d.kappa.round(6))
folds = pd.read_csv(THEORY / "mf_folds.csv")
 
 
def fit(x, y, s):
    w = 1.0 / s ** 2
    design = np.c_[np.ones_like(x), x]
    cov = np.linalg.inv(design.T @ (design * w[:, None]))
    coef = cov @ (design.T @ (w * y))
    chi2 = np.sum(w * (y - design @ coef) ** 2) / max(len(x) - 2, 1)
    return coef[0], np.sqrt(cov[0, 0]), coef[1], np.sqrt(cov[1, 1]), chi2
 
 
rows = []
for r, kappas in CUTS.items():
    for kappa in kappas:
        theory = roots[(roots.r == r) & (roots.kappa == kappa) & (roots.n_unstable == 0)]
        if len(theory) != 1:
            raise RuntimeError(f"no unique stable root at r={r}, kappa={kappa}")
        point = cuts[(cuts.r == r) & (cuts.kappa == kappa)]
        counts = point.groupby("N").pinned.agg(["sum", "size"])
        row = dict(r=r, kappa=kappa, pinned_largest=int(counts.loc[N_LARGEST, "sum"]),
                   trials_largest=int(counts.loc[N_LARGEST, "size"]))
        for name in ("R1", "R2"):
            stats = point[point.pinned].groupby("N")[name].agg(["mean", "sem", "size"])
            stats = stats[stats["size"] >= MIN_PINNED]
            if len(stats) < 3:
                raise RuntimeError(f"fewer than three sizes with pinned realizations at r={r}, kappa={kappa}")
            n = stats.index.to_numpy(dtype=float)
            s = stats["sem"].to_numpy()
            s = np.where(s > 0, s, np.min(s[s > 0]))
            row[f"{name}_MF"] = theory[name].iloc[0]
            row[f"{name}_sizes"] = ",".join(str(int(v)) for v in n)
            for label, x in (("invN", 1.0 / n), ("invsqrtN", 1.0 / np.sqrt(n))):
                a, ea, c, ec, chi2 = fit(x, stats["mean"].to_numpy(), s)
                row.update({f"{name}_inf_{label}": a, f"{name}_inf_{label}_err": ea, f"{name}_C_{label}": c,
                            f"{name}_C_{label}_err": ec, f"{name}_chi2_{label}": chi2})
        rows.append(row)
table = pd.DataFrame(rows)
 
near_fold = {(r, k) for r, ks in CUTS.items() for k in ks
             if min(abs(k - folds[(folds.r == r) & (folds.kind == kind)].kappa.iloc[0]) for kind in ("lower", "upper")) < FOLD_MARGIN}
away = table[[(r, k) not in near_fold for r, k in zip(table.r, table.kappa)]]
stats = {}
for name in ("R1", "R2"):
    z = (away[f"{name}_inf_invN"] - away[f"{name}_MF"]) / away[f"{name}_inf_invN_err"]
    stats[name] = (100 * np.mean(np.abs(z) < 2), table[f"{name}_chi2_invN"].median(), table[f"{name}_chi2_invsqrtN"].median())
caption = (r"Stationary mean-field theory versus simulation along the cuts $r=4.5$, 5 and 6. "
           r"$R_i^{\mathrm{MF}}$ are stable-branch values. $R_{i,\infty}$ are extrapolations to $N\to\infty$ of the "
           r"realization mean over pinned realizations (every oscillator with $|\Omega_i|\le10^{-3}$), from weighted fits "
           r"$a+C/N$ and $a+C/\sqrt{N}$ over all sizes $N=100$--6400 with at least three pinned realizations; "
           r"parentheses give the $1\sigma$ fit uncertainty in the last digits, and $C_{R_i}$ is the coefficient of the "
           r"$1/N$ fit. The last column is the number of pinned realizations at $N=6400$. "
           rf"Daggers mark couplings within {FOLD_MARGIN:g} of a fold, where the pinned subset is selected by the disorder "
           r"realization and agreement deteriorates. Away from the folds the $1/N$ extrapolation agrees with theory "
           rf"within $2\sigma$ for {stats['R1'][0]:.0f}\% ($R_1$) and {stats['R2'][0]:.0f}\% ($R_2$) of entries. "
           rf"Median $\chi^2$ per degree of freedom: {stats['R1'][1]:.2f} ($1/N$) and {stats['R1'][2]:.2f} ($1/\sqrt{{N}}$) for "
           rf"$R_1$; {stats['R2'][1]:.2f} and {stats['R2'][2]:.2f} for $R_2$; the data do not distinguish the two ans\"atze.")
lines = [r"\begin{table*}", rf"\caption{{{caption}}}", r"\label{tab:mf-comparison}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{cc|cccc|cccc|c}", r"\hline\hline",
         r"$r$ & $\kappa$ & $R_1^{\mathrm{MF}}$ & $R_{1,\infty}^{(1/N)}$ & $R_{1,\infty}^{(1/\sqrt{N})}$ & $C_{R_1}$ & "
         r"$R_2^{\mathrm{MF}}$ & $R_{2,\infty}^{(1/N)}$ & $R_{2,\infty}^{(1/\sqrt{N})}$ & $C_{R_2}$ & pinned \\", r"\hline"]
previous = None
for _, row in table.iterrows():
    r_cell = f"{row.r:g}" if row.r != previous else ""
    if previous is not None and row.r != previous:
        lines.append(r"\hline")
    previous = row.r
    mark = r"$^\dagger$" if (row.r, row.kappa) in near_fold else ""
    cells = [r_cell, f"{row.kappa:g}{mark}"]
    for name in ("R1", "R2"):
        cells += [f"{row[f'{name}_MF']:.4f}",
                  compact(row[f"{name}_inf_invN"], row[f"{name}_inf_invN_err"]),
                  compact(row[f"{name}_inf_invsqrtN"], row[f"{name}_inf_invsqrtN_err"]),
                  compact(row[f"{name}_C_invN"], row[f"{name}_C_invN_err"])]
    cells.append(f"{row.pinned_largest}/{row.trials_largest}")
    lines.append(" & ".join(cells) + r" \\")
lines += [r"\hline\hline", r"\end{tabular}", r"\end{table*}", ""]
write("table_s2", "\n".join(lines), table)
 
z1 = (table.R1_inf_invN - table.R1_MF) / table.R1_inf_invN_err
z2 = (table.R2_inf_invN - table.R2_MF) / table.R2_inf_invN_err
print(f"rows {len(table)}; |z|<2 (1/N): R1 {np.mean(np.abs(z1) < 2):.2f}, R2 {np.mean(np.abs(z2) < 2):.2f}; "
      f"max |z| R1 {np.abs(z1).max():.1f} at {tuple(table.loc[np.abs(z1).idxmax(), ['r', 'kappa']])}, "
      f"R2 {np.abs(z2).max():.1f} at {tuple(table.loc[np.abs(z2).idxmax(), ['r', 'kappa']])}")
for name in ("R1", "R2"):
    print(f"{name}: median chi2/dof 1/N {table[f'{name}_chi2_invN'].median():.2f}, 1/sqrtN {table[f'{name}_chi2_invsqrtN'].median():.2f}")
print("near-fold rows:", sorted(near_fold))
 