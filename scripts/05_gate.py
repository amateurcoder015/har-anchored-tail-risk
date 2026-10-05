import time

import pandas as pd

from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS, VARIANTS, build_pooled, run_variant
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
main = cfg.get("main_variant", "gate")
variants = cfg.get("gate_variants", list(VARIANTS))
runs = [(0.025, v) for v in variants] + [(a, main) for a in cfg["alphas"] if a != 0.025]
selections = []
for a, name in runs:
    t0 = time.time()
    out_dir = P["processed"] / "gate" / alpha_tag(a)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, sel = run_variant(build_pooled(frames, a), VARIANTS[name], a, FOLDS)
    pred.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    selections.append(sel.assign(alpha=a, variant=name))
    print(f"{alpha_tag(a)} {name} done in {time.time() - t0:.0f}s", flush=True)
pd.concat(selections)[["alpha", "variant", "fold", "lam", "val_loss"]].to_csv(
    P["tables"] / "gate_selection.csv", index=False)
