import pandas as pd

from volgate.combine.data import load_asset
from volgate.evaluate.losses import pooled_average
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

START, END = "2023-01-01", "2026-09-30"


def forecasts(processed, asset, alpha, gate_dirs=("gate",)):
    """VaR/ES forecasts of R0, R1, baselines, R2 variants and every gate file for one asset."""
    t = alpha_tag(alpha)
    df = load_asset(processed, asset)
    r1 = pd.read_csv(processed / "r1" / f"{asset}.csv", index_col="date", parse_dates=True)
    co = pd.read_csv(processed / "combos" / t / f"{asset}.csv", index_col="date", parse_dates=True)
    fc = {"r0": (df[f"har__var_{t}"], df[f"har__es_{t}"]), "r1": (r1[f"var_{t}"], r1[f"es_{t}"])}
    for c in ("mean", "min_score", "relative_score"):
        fc[c] = (co[f"{c}__var"], co[f"{c}__es"])
    for f in sorted((processed / "r2" / t).glob("*.csv")):
        g = pd.read_csv(f, parse_dates=["date"])
        gi = g[g.asset == asset].set_index("date")
        fc[f"r2_{f.stem}"] = (gi["var"], gi["es"])
    for f in sorted(f for d in gate_dirs for f in (processed / d / t).glob("*.csv")):
        g = pd.read_csv(f, parse_dates=["date"])
        gi = g[g.asset == asset].set_index("date")
        fc[f.stem] = (gi["var"], gi["es"])
    return df["log_return"], fc


def loss_frames(processed, assets, alpha, start=START, end=END, gate_dirs=("gate",)):
    """Per-asset FZ0 loss frames on dates common to all methods, plus pooled 'ALL'."""
    per = {}
    for asset in assets:
        y, fc = forecasts(processed, asset, alpha, gate_dirs)
        dates = y.loc[start:end].dropna().index
        for v, _ in fc.values():
            dates = dates.intersection(v.dropna().index)
        yv = y.loc[dates].to_numpy()
        per[asset] = pd.DataFrame({k: fz0_loss(yv, v.loc[dates].to_numpy(), e.loc[dates].to_numpy(), alpha)
                                   for k, (v, e) in fc.items()}, index=dates)
    per["ALL"] = pooled_average({k: v for k, v in per.items()})
    return per
