import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

TEST_START, TEST_END = "2023-01-01", "2026-09-30"
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    gates = {f.stem: pd.read_csv(f, parse_dates=["date"])
             for f in sorted((P["processed"] / "gate" / t).glob("*.csv"))}
    pooled = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[TEST_START:TEST_END]
        combos = pd.read_csv(P["processed"] / "combos" / t / f"{asset}.csv", index_col="date",
                             parse_dates=True).loc[TEST_START:TEST_END]
        fc = {m: (df[f"{m}__var_{t}"], df[f"{m}__es_{t}"]) for m in MODELS}
        for c in ("mean", "median", "min_score", "relative_score", "previous_best"):
            fc[c] = (combos[f"{c}__var"], combos[f"{c}__es"])
        for name, g in gates.items():
            gi = g[g.asset == asset].set_index("date")
            fc[name] = (gi["var"], gi["es"])
        common = df.index
        for v, _ in fc.values():
            common = common.intersection(v.dropna().index)
        y = df.loc[common, "log_return"].to_numpy()
        for name, (v, e) in fc.items():
            L = fz0_loss(y, v.loc[common].to_numpy(), e.loc[common].to_numpy(), a)
            rows.append({"alpha": a, "asset": asset, "method": name, "n": len(L), "mean_fz0": L.mean()})
            pooled.setdefault(name, []).append(L)
    for name, Ls in pooled.items():
        L = np.concatenate(Ls)
        rows.append({"alpha": a, "asset": "ALL", "method": name, "n": len(L), "mean_fz0": L.mean()})
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "test_fz0.csv", index=False)
print(table[table.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0").round(4))
