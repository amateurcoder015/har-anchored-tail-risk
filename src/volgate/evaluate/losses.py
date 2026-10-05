from pathlib import Path

import pandas as pd

from volgate.combine.data import MODELS, load_asset
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

BASELINES = ["mean", "median", "min_score", "relative_score", "previous_best"]


def collect_forecasts(processed: Path, asset: str, alpha: float, start: str, end: str):
    """VaR/ES forecasts of base models, baselines and every gate file for one asset."""
    processed = Path(processed)
    t = alpha_tag(alpha)
    df = load_asset(processed, asset).loc[start:end]
    fc = {m: df[[f"{m}__var_{t}", f"{m}__es_{t}"]].set_axis(["var", "es"], axis=1) for m in MODELS}
    combos = pd.read_csv(processed / "combos" / t / f"{asset}.csv", index_col="date",
                         parse_dates=True).loc[start:end]
    for c in BASELINES:
        fc[c] = combos[[f"{c}__var", f"{c}__es"]].set_axis(["var", "es"], axis=1)
    for f in sorted((processed / "gate" / t).glob("*.csv")):
        g = pd.read_csv(f, parse_dates=["date"])
        fc[f.stem] = g[g["asset"] == asset].set_index("date")[["var", "es"]]
    return fc, df


def aligned(fc: dict, y: pd.Series):
    dates = y.dropna().index
    for v in fc.values():
        dates = dates.intersection(v.dropna().index)
    return dates, y.loc[dates].to_numpy(), {k: (v.loc[dates, "var"].to_numpy(),
                                               v.loc[dates, "es"].to_numpy()) for k, v in fc.items()}


def loss_frame(fc_aligned: dict, y, alpha: float, dates=None) -> pd.DataFrame:
    return pd.DataFrame({k: fz0_loss(y, v, e, alpha) for k, (v, e) in fc_aligned.items()},
                        index=dates)


def pooled_average(per_asset: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-asset mean loss per date, on dates where every asset is present."""
    common = None
    for L in per_asset.values():
        common = L.index if common is None else common.intersection(L.index)
    return sum(L.loc[common] for L in per_asset.values()) / len(per_asset)
