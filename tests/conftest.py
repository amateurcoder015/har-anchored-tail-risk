import numpy as np
import pandas as pd
import pytest

from volgate.data.panel import gk_variance


def make_panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic GARCH(1,1)-t returns with OHLC, GK variance and a VIX-like series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, name="date")
    omega, a, b = 2e-6, 0.08, 0.90
    z = rng.standard_t(6, n) / np.sqrt(6 / 4)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = omega / (1 - a - b)
    for t in range(n):
        if t > 0:
            s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "adj_close": close, "volume": 1000, "log_return": r}, index=dates)
    df["gk_var"] = gk_variance(df)
    df["vix"] = 100 * np.sqrt(252 * s2) * np.exp(rng.normal(0, 0.05, n))
    return df


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()


from scipy import stats  # noqa: E402

from volgate.combine.data import MODELS  # noqa: E402
from volgate.risk.dist import alpha_tag  # noqa: E402

_FACTORS = [1.0, 0.6, 1.5, 0.9, 1.2, 2.0]


def make_combo_frame(n: int = 700, seed: int = 0) -> pd.DataFrame:
    """make_panel plus base-forecast columns for six models with mis-scaled variances."""
    df = make_panel(n, seed)
    true_s2 = df["log_return"].rolling(20, min_periods=1).var().bfill().to_numpy() + 1e-6
    for m, f in zip(MODELS, _FACTORS):
        s2 = true_s2 * f
        df[f"{m}__sigma2"] = s2
        df[f"{m}__fallback"] = False
        for a in (0.01, 0.025, 0.05):
            q = stats.norm.ppf(a)
            df[f"{m}__var_{alpha_tag(a)}"] = np.sqrt(s2) * q
            df[f"{m}__es_{alpha_tag(a)}"] = -np.sqrt(s2) * stats.norm.pdf(q) / a
    for c in ["own_is_expiry", "own_days_to_expiry", "nifty_is_expiry", "nifty_days_to_expiry"]:
        df[c] = 0.0
    return df


@pytest.fixture
def combo_frame() -> pd.DataFrame:
    return make_combo_frame()
