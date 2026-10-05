import pandas as pd

from volgate.combine.baselines import mean_combo, median_combo, rolling_combination
from volgate.combine.data import MODELS, load_asset, pairs
from volgate.config import REPO_ROOT, load_config
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
for a in cfg["alphas"]:
    out_dir = P["processed"] / "combos" / alpha_tag(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset)
        V, E = pairs(df, a)
        y = df["log_return"].to_numpy()
        out = pd.DataFrame(index=df.index)
        out["mean__var"], out["mean__es"] = mean_combo(V, E)
        out["median__var"], out["median__es"] = median_combo(V, E)
        for method in ("min_score", "relative_score", "previous_best"):
            v, e, W = rolling_combination(y, V, E, a, method, cfg["refit_every"], 250)
            out[f"{method}__var"], out[f"{method}__es"] = v, e
            if method != "previous_best":
                for i, m in enumerate(MODELS):
                    out[f"{method}__wq_{m}"] = W[:, i]
        out.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")
        print(f"{alpha_tag(a)} {asset} done", flush=True)
