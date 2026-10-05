import numpy as np
import pandas as pd

from volgate.models.base import FitResult

_FLOOR = 1e-10


def _std_resid(r: np.ndarray, s2: np.ndarray, burn: int) -> np.ndarray:
    z = r[burn:] / np.sqrt(s2[burn:])
    return z[np.isfinite(z)]


class EWMA:
    """RiskMetrics EWMA; tails by filtered historical simulation."""

    name = "ewma"

    def __init__(self, lam: float = 0.94, init_var: float = 1e-4, burn: int = 22):
        self.lam, self.init_var, self.burn = lam, init_var, burn

    def _recursion(self, r: np.ndarray) -> np.ndarray:
        s2 = np.empty(len(r))
        s2[0] = self.init_var  # fixed prior: uses no data
        for t in range(1, len(r)):
            s2[t] = self.lam * s2[t - 1] + (1 - self.lam) * r[t - 1] ** 2
        return s2

    def fit(self, train: pd.DataFrame) -> FitResult:
        r = train["log_return"].to_numpy()
        z = _std_resid(r, self._recursion(r), self.burn)
        return FitResult(params={"lam": self.lam}, z=z, nu=None, converged=True)

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        return self._recursion(data["log_return"].to_numpy())


class HAR:
    """HAR on log Garman-Klass variance; tails from empirical standardized returns."""

    name = "har"

    @staticmethod
    def _design(gk: pd.Series) -> np.ndarray:
        g = gk.clip(lower=_FLOOR)
        return np.column_stack([
            np.ones(len(g)),
            np.log(g).shift(1),
            np.log(g.rolling(5).mean()).shift(1),
            np.log(g.rolling(22).mean()).shift(1),
        ])

    def fit(self, train: pd.DataFrame) -> FitResult:
        X = self._design(train["gk_var"])
        y = np.log(train["gk_var"].clip(lower=_FLOOR)).to_numpy()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        s2e = float(np.var(y[ok] - X[ok] @ beta, ddof=X.shape[1]))
        fit = FitResult(params={"beta": beta, "s2e": s2e}, z=None, nu=None, converged=True)
        s2 = self.filter(fit, train)
        fit.z = _std_resid(train["log_return"].to_numpy(), s2, 0)
        return fit

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        X = self._design(data["gk_var"])
        return np.exp(X @ fit.params["beta"] + 0.5 * fit.params["s2e"])


class VIXModel:
    """sigma2_t = beta * (VIX_{t-1}/100)^2 / 252, beta by OLS through the origin on r^2."""

    name = "vix"

    @staticmethod
    def _implied(data: pd.DataFrame) -> np.ndarray:
        return ((data["vix"].shift(1) / 100.0) ** 2 / 252.0).to_numpy()

    def fit(self, train: pd.DataFrame) -> FitResult:
        x = self._implied(train)
        r = train["log_return"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(r)
        beta = max(float(np.sum(x[ok] * r[ok] ** 2) / np.sum(x[ok] ** 2)), 1e-6)
        z = r[ok] / np.sqrt(beta * x[ok])
        return FitResult(params={"beta": beta}, z=z, nu=None, converged=True)

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        return fit.params["beta"] * self._implied(data)
