"""Exploratory robustness checks for the pre-registered hypotheses (not part of the registration)."""
import numpy as np
import pandas as pd

from hart.compare import loss_frames
from hart.hypotheses import HYPOTHESES, one_sided_p
from hart.robustness import acerbi_szekely_z2, power_projection, sign_counts
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.losses import pooled_average
from volgate.evaluate.stats import dm_test
from volgate.risk.dist import alpha_tag
from hart.compare import forecasts

MAIN = 0.025
PAIRS = [(n, lo, hi) for n, lo, hi in HYPOTHESES] + [("S1", "r2_mlp", "r0"), ("S2", "r2_linear", "r0")]
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
T = P["tables"] / "robustness"
T.mkdir(parents=True, exist_ok=True)


def dm_row(L, name, lo, hi, **extra):
    s, p = dm_test(L[lo], L[hi])
    return {**extra, "pair": name, "lower": lo, "higher": hi, "n": len(L),
            "diff": L[lo].mean() - L[hi].mean(), "dm_stat": s, "p_one_sided": one_sided_p(s, p)}


per_alpha = {a: loss_frames(P["processed"], cfg["assets"], a) for a in cfg["alphas"]}

# 1. By tail level (pairs whose methods exist at that level).
rows = [dm_row(per["ALL"], n, lo, hi, alpha=a) for a, per in per_alpha.items()
        for n, lo, hi in PAIRS if {lo, hi} <= set(per["ALL"].columns)]
pd.DataFrame(rows).to_csv(T / "by_alpha.csv", index=False)

# 2. By test year at the main level.
per = per_alpha[MAIN]
rows = []
for year in sorted(set(per["ALL"].index.year)):
    sub = {k: v[v.index.year == year] for k, v in per.items() if k != "ALL"}
    L = pooled_average(sub)
    rows += [dm_row(L, n, lo, hi, year=year) for n, lo, hi in PAIRS]
pd.DataFrame(rows).to_csv(T / "by_year.csv", index=False)

# 3. Per-asset win counts and by sector.
assets = {k: v for k, v in per.items() if k != "ALL"}
rows = [{"pair": n, "lower": lo, "higher": hi, "wins": sign_counts(assets, lo, hi)[0], "assets": len(assets)}
        for n, lo, hi in PAIRS]
pd.DataFrame(rows).to_csv(T / "asset_wins.csv", index=False)
sectors = cfg.get("sectors", {})
if sectors:
    rows = []
    for sector, members in sectors.items():
        sub = {k: assets[k] for k in members if k in assets}
        for n, lo, hi in PAIRS:
            rows.append({"sector": sector, "pair": n, "wins": sign_counts(sub, lo, hi)[0], "assets": len(sub)})
    pd.DataFrame(rows).to_csv(T / "sector_wins.csv", index=False)

# 4. Mechanism: learned gate scale (log g) vs gate's loss gap to HAR, per asset and year.
gate = pd.read_csv(P["processed"] / "gate" / alpha_tag(MAIN) / "gate.csv", parse_dates=["date"])
rows = []
for asset, L in assets.items():
    g = gate[gate.asset == asset].set_index("date")["g"].reindex(L.index)
    for year in sorted(set(L.index.year)):
        m = L.index.year == year
        rows.append({"asset": asset, "year": year, "mean_log_g": float(np.log(g[m]).mean()),
                     "gap_gate_minus_r0": float((L["gate"] - L["r0"])[m].mean()),
                     "gap_gatev2_minus_r0": float((L["gate_v2"] - L["r0"])[m].mean())})
mech = pd.DataFrame(rows)
mech.to_csv(T / "mechanism.csv", index=False)
mech_summary = {"corr_abs_log_g_vs_gap": float(np.corrcoef(mech.mean_log_g.abs(), mech.gap_gate_minus_r0)[0, 1]),
                "corr_log_g_vs_gap": float(np.corrcoef(mech.mean_log_g, mech.gap_gate_minus_r0)[0, 1]),
                "mean_log_g": float(mech.mean_log_g.mean()), "cells": len(mech)}
pd.Series(mech_summary).to_csv(T / "mechanism_summary.csv", header=["value"])

# 5. Acerbi-Szekely Z2 per method and asset at the main level (reject if Z2 < -0.70, AS 2014 5% threshold).
rows = []
for asset in cfg["assets"]:
    y, fc = forecasts(P["processed"], asset, MAIN)
    d = assets[asset].index
    for name, (v, e) in fc.items():
        rows.append({"asset": asset, "method": name,
                     "z2": acerbi_szekely_z2(y.loc[d], v.loc[d], e.loc[d], MAIN)})
z2 = pd.DataFrame(rows)
z2.to_csv(T / "as_z2.csv", index=False)
z2_summary = z2.assign(reject=z2.z2 < -0.70).groupby("method").agg(mean_z2=("z2", "mean"), rejections=("reject", "sum"))
z2_summary.to_csv(T / "as_z2_summary.csv")

# 6. Power projection for pooled DM (dates x assets differential panels).
rows = []
for n, lo, hi in PAIRS:
    D = pd.DataFrame({k: v[lo] - v[hi] for k, v in assets.items()})
    rows.append({"pair": n, "lower": lo, "higher": hi, **power_projection(D)})
pd.DataFrame(rows).to_csv(T / "power.csv", index=False)

for f in ("by_alpha", "by_year", "asset_wins", "power"):
    print(f"--- {f}")
    print(pd.read_csv(T / f"{f}.csv").round(4).to_string(index=False))
print("--- mechanism", mech_summary)
print("--- AS Z2"); print(z2_summary.round(3).to_string())
