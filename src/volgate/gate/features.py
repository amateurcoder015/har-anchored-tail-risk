import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, model_losses

LOSS_FEATURES = [f"fz_{m}" for m in MODELS]
RV_FEATURES = ["rv5", "rv22", "rv66", "vov30"]
VIX_FEATURES = ["vix", "vix_chg5", "vrp"]
EXPIRY_FEATURES = ["own_is_expiry", "own_days_to_expiry", "nifty_is_expiry", "nifty_days_to_expiry"]
RETURN_FEATURES = ["ret_neg_lag1", "absret_lag1"]


def feature_columns(use_vix: bool, use_expiry: bool) -> list[str]:
    cols = LOSS_FEATURES + RV_FEATURES + RETURN_FEATURES
    if use_vix:
        cols = cols + VIX_FEATURES
    if use_expiry:
        cols = cols + EXPIRY_FEATURES
    return cols


def gate_features(df: pd.DataFrame, alpha: float, window: int = 60) -> pd.DataFrame:
    """State features for day t built from data before t (expiry calendar is known ahead)."""
    f = pd.DataFrame(index=df.index)
    L = model_losses(df, alpha)
    for m in MODELS:
        f[f"fz_{m}"] = L[m].shift(1).rolling(window, min_periods=window).mean()
    gk = df["gk_var"].shift(1)
    for w in (5, 22, 66):
        f[f"rv{w}"] = np.sqrt(gk.rolling(w).mean())
    f["vov30"] = f["rv5"].rolling(30).std()
    r1 = df["log_return"].shift(1)
    f["ret_neg_lag1"] = (r1 < 0).astype(float).where(r1.notna())
    f["absret_lag1"] = r1.abs()
    vix = df["vix"].shift(1)
    f["vix"] = vix
    f["vix_chg5"] = vix - vix.shift(5)
    f["vrp"] = (vix / 100.0) ** 2 / 252.0 - gk.rolling(22).mean()
    for c in EXPIRY_FEATURES:
        f[c] = df[c].astype(float)
    return f
