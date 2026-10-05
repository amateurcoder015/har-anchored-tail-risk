from pathlib import Path

import numpy as np
import pandas as pd

from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

MODELS = ["ewma", "garch_t", "egarch_t", "gjr_t", "har", "vix"]


def load_asset(processed: Path, asset: str) -> pd.DataFrame:
    processed = Path(processed)
    panel = pd.read_csv(processed / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    base = pd.read_csv(processed / "base_forecasts" / f"{asset}.csv", index_col="date",
                       parse_dates=True)
    return panel.join(base, how="inner")


def pairs(df: pd.DataFrame, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    t = alpha_tag(alpha)
    V = df[[f"{m}__var_{t}" for m in MODELS]].to_numpy(dtype=float)
    E = df[[f"{m}__es_{t}" for m in MODELS]].to_numpy(dtype=float)
    return V, E


def variances(df: pd.DataFrame) -> np.ndarray:
    return df[[f"{m}__sigma2" for m in MODELS]].to_numpy(dtype=float)


def model_losses(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    V, E = pairs(df, alpha)
    y = np.broadcast_to(df["log_return"].to_numpy(dtype=float)[:, None], V.shape)
    ok = np.isfinite(V) & np.isfinite(E) & np.isfinite(y)
    L = np.full(V.shape, np.nan)
    L[ok] = fz0_loss(y[ok], V[ok], E[ok], alpha)
    return pd.DataFrame(L, index=df.index, columns=MODELS)
