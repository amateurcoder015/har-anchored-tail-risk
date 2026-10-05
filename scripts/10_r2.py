import pandas as pd

from hart.r2 import build_r2_long, run_r2
from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
anchor, lam_c = cfg["r2"]["anchor"], cfg["r2"]["lam_c"]
lam_mean = cfg["r2"].get("lam_mean", 0.0)
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
for a in cfg["alphas"]:
    t = alpha_tag(a)
    if anchor == "r0":
        anchors = {k: f[[f"har__var_{t}", f"har__es_{t}"]].set_axis(["var", "es"], axis=1)
                   for k, f in frames.items()}
    else:
        anchors = {k: pd.read_csv(P["processed"] / "r1" / f"{k}.csv", index_col="date",
                                  parse_dates=True)[[f"var_{t}", f"es_{t}"]].set_axis(["var", "es"], axis=1)
                   for k in frames}
    long = build_r2_long(frames, anchors, a)
    out_dir = P["processed"] / "r2" / t
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("mlp", "linear"):
        run_r2(long, kind, a, FOLDS, lam_c=lam_c, lam_mean=lam_mean).to_csv(out_dir / f"{kind}.csv", index=False,
                                                          date_format="%Y-%m-%d")
        print(f"{t} {kind} done", flush=True)
