import math
from figstyle import ROOT
 
TABLES = ROOT / "tables"
 
 
def compact(value, error, digits=2):
    if abs(value) < 1e-9 and abs(error) < 1e-9:
        return "0"
    if not (math.isfinite(value) and math.isfinite(error)) or error <= 0:
        return f"{value:.4f}"
    exponent = math.floor(math.log10(error)) - (digits - 1)
    decimals = max(0, -exponent)
    scaled = round(error / 10 ** exponent)
    if scaled >= 10 ** digits:
        scaled //= 10
        decimals = max(0, decimals - 1)
    return f"{value:.{decimals}f}({scaled})"
 
 
def sci(value):
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    return rf"${value / 10 ** exponent:.1f}\times10^{{{exponent}}}$"
 
 
def write(stem, latex, frame):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{stem}.tex").write_text(latex)
    frame.to_csv(TABLES / f"{stem}.csv", index=False, float_format="%.10g")