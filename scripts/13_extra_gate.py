"""Exploratory: run the original gate at the tail levels it was not pre-registered for."""
from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS, VARIANTS, build_pooled, run_variant
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
for a in cfg["alphas"]:
    out = P["processed"] / "gate" / alpha_tag(a) / "gate.csv"
    if out.exists():
        continue
    pred, _ = run_variant(build_pooled(frames, a), VARIANTS["gate"], a, FOLDS)
    pred.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"{alpha_tag(a)} gate done", flush=True)
