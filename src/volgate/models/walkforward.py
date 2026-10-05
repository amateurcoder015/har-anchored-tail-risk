import numpy as np
import pandas as pd

from volgate.models.base import tail_var_es
from volgate.risk.dist import alpha_tag


def walk_forward(data: pd.DataFrame, model, oos_start: str, refit_every: int,
                 alphas: list[float]) -> pd.DataFrame:
    """Expanding-window refit every `refit_every` rows from `oos_start`.

    A block starting at row k is fitted on rows < k. Forecast for row t uses
    rows < t through the model's filter. A failed or unconverged fit reuses
    the previous fit and sets `fallback`.
    """
    data = data.sort_index()
    oos = np.flatnonzero(data.index >= pd.Timestamp(oos_start))
    if oos.size == 0:
        raise ValueError(f"no rows on or after {oos_start}")
    blocks, prev = [], None
    for k in range(oos[0], len(data), refit_every):
        end = min(k + refit_every, len(data))
        fallback = False
        try:
            fit = model.fit(data.iloc[:k])
            if not fit.converged:
                raise RuntimeError("fit did not converge")
        except Exception as exc:
            if prev is None:
                raise RuntimeError(f"{model.name}: first fit failed") from exc
            fit, fallback = prev, True
        prev = fit
        s2 = model.filter(fit, data.iloc[:end])[k:end]
        block = pd.DataFrame({"sigma2": s2}, index=data.index[k:end])
        block["refit_date"] = data.index[k]
        block["fallback"] = fallback
        sigma = np.sqrt(s2)
        for a in alphas:
            v, e = tail_var_es(fit, sigma, a)
            block[f"var_{alpha_tag(a)}"] = v
            block[f"es_{alpha_tag(a)}"] = e
        blocks.append(block)
    return pd.concat(blocks)
