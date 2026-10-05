import numpy as np
import pandas as pd

from volgate.config import REPO_ROOT, load_config
from volgate.models.garch import GarchT
from volgate.models.simple import EWMA, HAR, VIXModel
from volgate.models.walkforward import walk_forward
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
out_dir = P["processed"] / "base_forecasts"
out_dir.mkdir(parents=True, exist_ok=True)
models = [EWMA(), GarchT("garch_t"), GarchT("egarch_t"), GarchT("gjr_t"), HAR(), VIXModel()]

rows = []
for asset in cfg["assets"]:
    panel = pd.read_csv(P["processed"] / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    parts = []
    for m in models:
        wf = walk_forward(panel, m, cfg["oos_start"], cfg["refit_every"], cfg["alphas"])
        y = panel["log_return"].reindex(wf.index).to_numpy()
        for a in cfg["alphas"]:
            t = alpha_tag(a)
            ok = np.isfinite(wf[f"var_{t}"]) & np.isfinite(wf[f"es_{t}"])
            rows.append({
                "asset": asset, "model": m.name, "alpha": a, "n": int(ok.sum()),
                "hit_rate": float(np.mean(y[ok] <= wf[f"var_{t}"][ok])),
                "mean_fz0": float(fz0_loss(y[ok], wf[f"var_{t}"][ok], wf[f"es_{t}"][ok], a).mean()),
                "fallback_blocks": int(wf.groupby("refit_date")["fallback"].first().sum()),
            })
        parts.append(wf.drop(columns="refit_date").add_prefix(f"{m.name}__"))
        print(f"{asset} {m.name} done")
    pd.concat(parts, axis=1).to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")

summary = pd.DataFrame(rows)
summary.to_csv(P["tables"] / "base_model_summary.csv", index=False)
print(summary[summary["alpha"] == 0.025].to_string(index=False))
