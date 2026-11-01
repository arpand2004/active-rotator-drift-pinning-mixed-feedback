import numpy as np
import pandas as pd
from figstyle import SIM
from tablefmt import compact, sci, write
 
OBS = ("D", "R1", "R2", "dR1")
REFERENCE = dict(dt=0.02, T_transient=300.0, T_measure=300.0)
CHANGES = [("convergence", dict(dt=0.01), r"$\delta\tau=0.01$"),
           ("convergence", dict(T_transient=600.0), r"$T_{\mathrm{tr}}=600$"),
           ("convergence", dict(T_measure=600.0), r"$T_{\mathrm{ms}}=600$"),
           ("convergence", dict(T_transient=600.0, T_measure=600.0), r"$T_{\mathrm{tr}}=T_{\mathrm{ms}}=600$"),
           ("precision", dict(dt=0.01), r"$\delta\tau=0.01$"),
           ("precision", dict(dt=0.002), r"$\delta\tau=0.002$")]
KEY = ["r", "kappa", "trial"]
 
 
def select(frame, settings):
    mask = np.ones(len(frame), dtype=bool)
    for column, value in {**REFERENCE, **settings}.items():
        if column in frame:
            mask &= np.isclose(frame[column], value)
    return frame[mask]
 
 
paired_rows = []
for mode, settings, label in CHANGES:
    data = pd.read_csv(SIM / mode / "trials.csv").assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
    base, test = select(data, {}), select(data, settings)
    merged = base.merge(test, on=KEY, suffixes=("_ref", "_test"), validate="one_to_one")
    if len(merged) != len(base) or len(merged) == 0:
        raise RuntimeError(f"{mode} {settings}: unpaired realizations")
    row = dict(mode=mode, change=label, N=int(merged.N_ref.iloc[0]), realizations=int(merged.groupby(["r", "kappa"]).size().iloc[0]),
               points=int(merged.groupby(["r", "kappa"]).ngroups))
    for name in OBS:
        row[f"max_abs_{name}"] = (merged[f"{name}_test"] - merged[f"{name}_ref"]).groupby([merged.r, merged.kappa]).mean().abs().max()
    paired_rows.append(row)
paired = pd.DataFrame(paired_rows)
 
control = pd.read_csv(SIM / "disorder_check" / "trials.csv").assign(r=lambda d: d.r.round(6), kappa=lambda d: d.kappa.round(6))
scheme_rows = []
for (r, kappa, n), group in control.groupby(["r", "kappa", "N"]):
    row = dict(r=r, kappa=kappa, N=int(n))
    for name in ("D", "dR1"):
        stats = group.groupby("disorder_scheme")[name].agg(["mean", "sem"])
        a, i = stats.loc["antithetic"], stats.loc["iid"]
        row.update({f"{name}_anti": a["mean"], f"{name}_anti_err": a["sem"], f"{name}_iid": i["mean"], f"{name}_iid_err": i["sem"],
                    f"{name}_z": (a["mean"] - i["mean"]) / np.hypot(a["sem"], i["sem"]) if np.hypot(a["sem"], i["sem"]) > 0 else 0.0})
    scheme_rows.append(row)
schemes = pd.DataFrame(scheme_rows)
 
points = ", ".join(sorted({f"({r:g},{k:g})" for r, k in zip(pd.read_csv(SIM / "convergence" / "trials.csv").r, pd.read_csv(SIM / "convergence" / "trials.csv").kappa)}))
caption = (r"Numerical controls. (A) Largest change of the realization-averaged observables over the test points "
           rf"$(r,\kappa)\in\{{{points}\}}$ when one numerical setting is changed from the production values "
           r"$\delta\tau=0.02$, $T_{\mathrm{tr}}=T_{\mathrm{ms}}=300$; comparisons are paired, with identical disorder and "
           r"initial phases. (B) Realization means and standard errors for $\pm$-paired and i.i.d. Gaussian disorder sampling "
           r"(12 realizations each); entries equal to zero are stationary pinned states.")
lines = [r"\begin{table}", rf"\caption{{{caption}}}", r"\label{tab:controls}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llcccc}", r"\hline\hline",
         r"\multicolumn{6}{l}{(A) Largest paired change of the realization mean over the test points} \\", r"\hline",
         r"test & change & $|\Delta\langle D\rangle|$ & $|\Delta\langle R_1\rangle|$ & $|\Delta\langle R_2\rangle|$ & "
         r"$|\Delta\langle\delta R_1\rangle|$ \\", r"\hline"]
for _, row in paired.iterrows():
    lines.append(" & ".join([f"{row['mode']} ($N={int(row.N)}$)", row.change] + [sci(row[f"max_abs_{o}"]) for o in OBS]) + r" \\")
lines += [r"\hline", r"\multicolumn{6}{l}{(B) $\pm$-paired versus i.i.d. Gaussian disorder (12 realizations each)} \\", r"\hline",
          r"$(r,\kappa)$ & $N$ & $\langle D\rangle$ $\pm$ & $\langle D\rangle$ i.i.d. & $\langle\delta R_1\rangle$ $\pm$ & "
          r"$\langle\delta R_1\rangle$ i.i.d. \\", r"\hline"]
for _, row in schemes.iterrows():
    lines.append(" & ".join([f"$({row.r:g},{row.kappa:g})$", f"{int(row.N)}",
                             compact(row.D_anti, row.D_anti_err), compact(row.D_iid, row.D_iid_err),
                             compact(row.dR1_anti, row.dR1_anti_err), compact(row.dR1_iid, row.dR1_iid_err)]) + r" \\")
lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}", ""]
write("table_s1", "\n".join(lines), pd.concat([paired.assign(block="A"), schemes.assign(block="B")], ignore_index=True))
print(paired[["mode", "change", "N", "realizations", "points"] + [f"max_abs_{o}" for o in OBS]].to_string(index=False))
print(schemes[["r", "kappa", "N", "D_z", "dR1_z"]].round(1).to_string(index=False))