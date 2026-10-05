import pandas as pd

from hart.harx import walk_forward_harx
from volgate.config import REPO_ROOT, load_config

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
out_dir = P["processed"] / "r1"
out_dir.mkdir(parents=True, exist_ok=True)
for asset in cfg["assets"]:
    panel = pd.read_csv(P["processed"] / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    wf = walk_forward_harx(panel, cfg["oos_start"], cfg["refit_every"], cfg["alphas"])
    wf.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")
    print(f"{asset} done", flush=True)
