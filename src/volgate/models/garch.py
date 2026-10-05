import numpy as np
import pandas as pd
from arch import arch_model

from volgate.models.base import FitResult

_SPECS = {
    "garch_t": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
    "gjr_t": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    "egarch_t": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
}


class GarchT:
    """GARCH-family model with Student-t errors, fitted on returns x 100."""

    def __init__(self, name: str):
        self.spec = _SPECS[name]
        self.name = name

    def _model(self, r: np.ndarray):
        return arch_model(r * 100.0, mean="Zero", dist="t", rescale=False, **self.spec)

    def fit(self, train: pd.DataFrame) -> FitResult:
        res = self._model(train["log_return"].to_numpy()).fit(disp="off", show_warning=False)
        params = res.params.to_dict()
        converged = res.convergence_flag == 0 and params["nu"] > 2.0
        return FitResult(params=params, z=None, nu=float(params["nu"]), converged=bool(converged))

    def filter(self, fit: FitResult, data: pd.DataFrame) -> np.ndarray:
        fixed = self._model(data["log_return"].to_numpy()).fix(np.array(list(fit.params.values())))
        return (np.asarray(fixed.conditional_volatility) / 100.0) ** 2
