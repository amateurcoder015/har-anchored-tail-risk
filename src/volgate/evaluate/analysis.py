import numpy as np
import pandas as pd
import statsmodels.api as sm


def hac_ols(y, X: pd.DataFrame) -> pd.DataFrame:
    """OLS with Newey-West errors; constant regressor columns are dropped."""
    X = X.loc[:, X.std() > 0]
    n = len(X)
    lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    res = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues})


def expiry_did(d: pd.Series, is_expiry: pd.Series, switch_date: str = "2025-09-01") -> pd.DataFrame:
    post = (d.index >= pd.Timestamp(switch_date)).astype(float)
    e = is_expiry.reindex(d.index).astype(float).to_numpy()
    X = pd.DataFrame({"expiry": e, "post": post, "expiry_x_post": e * post}, index=d.index)
    return hac_ols(d.to_numpy(), X)


def state_regression(d: pd.Series, states: pd.DataFrame) -> pd.DataFrame:
    s = states.reindex(d.index)
    z = (s - s.mean()) / s.std()
    return hac_ols(d.to_numpy(), z)
