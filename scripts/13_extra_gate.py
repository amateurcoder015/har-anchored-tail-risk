"""Exploratory: run the original gate at the tail levels it was not pre-registered for.

Output goes to processed/gate_extra/ so the registered pipeline (which reads processed/gate/)
is unchanged.
"""
from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS, VARIANTS, build_pooled, run_variant
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
for a in cfg["alphas"]:
    if (P["processed"] / "gate" / alpha_tag(a) / "gate.csv").exists():
        continue  # produced by the registered pipeline
    out = P["processed"] / "gate_extra" / alpha_tag(a) / "gate.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        continue
    pred, _ = run_variant(build_pooled(frames, a), VARIANTS["gate"], a, FOLDS)
    pred.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"{alpha_tag(a)} gate done", flush=True)
