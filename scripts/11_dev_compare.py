import numpy as np
import pandas as pd

from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.losses import pooled_average
from volgate.evaluate.stats import dm_test
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

START, END = "2023-01-01", "2026-09-30"
REFS = ["r0", "r1", "min_score", "mean"]
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    r2 = {k: pd.read_csv(P["processed"] / "r2" / t / f"{k}.csv", parse_dates=["date"])
          for k in ("mlp", "linear")}
    gate_file = P["processed"] / "gate" / t / "gate_v2.csv"
    gate = pd.read_csv(gate_file, parse_dates=["date"]) if gate_file.exists() else None
    per = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[START:END]
        r1 = pd.read_csv(P["processed"] / "r1" / f"{asset}.csv", index_col="date", parse_dates=True)
        co = pd.read_csv(P["processed"] / "combos" / t / f"{asset}.csv", index_col="date", parse_dates=True)
        fc = {"r0": (df[f"har__var_{t}"], df[f"har__es_{t}"]),
              "r1": (r1[f"var_{t}"], r1[f"es_{t}"])}
        for c in ("mean", "min_score", "relative_score"):
            fc[c] = (co[f"{c}__var"], co[f"{c}__es"])
        for k, g in r2.items():
            gi = g[g.asset == asset].set_index("date")
            fc[f"r2_{k}"] = (gi["var"], gi["es"])
        if gate is not None:
            gi = gate[gate.asset == asset].set_index("date")
            fc["gate_v2"] = (gi["var"], gi["es"])
        dates = df.index
        for v, _ in fc.values():
            dates = dates.intersection(v.dropna().index)
        y = df.loc[dates, "log_return"].to_numpy()
        per[asset] = pd.DataFrame({k: fz0_loss(y, v.loc[dates].to_numpy(), e.loc[dates].to_numpy(), a)
                                   for k, (v, e) in fc.items()}, index=dates)
    per["ALL"] = pooled_average({k: v for k, v in per.items()})
    for asset, L in per.items():
        for m in L.columns:
            row = {"alpha": a, "asset": asset, "method": m, "n": len(L), "mean_fz0": L[m].mean()}
            for ref in REFS:
                s, p = dm_test(L[m], L[ref]) if m != ref else (np.nan, np.nan)
                row[f"dm_vs_{ref}"], row[f"p_vs_{ref}"] = s, p
            rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "compare.csv", index=False)
show = table[(table.asset == "ALL") & (table.alpha == 0.025)].sort_values("mean_fz0")
print(show.drop(columns=["alpha", "asset"]).round(4).to_string(index=False))
