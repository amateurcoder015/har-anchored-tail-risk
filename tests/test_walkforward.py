import numpy as np
import pandas as pd
import pytest

from volgate.models.base import FitResult
from volgate.models.garch import GarchT
from volgate.models.simple import EWMA, HAR, VIXModel
from volgate.models.walkforward import walk_forward

ALPHAS = [0.01, 0.025]


class Flaky:
    name = "flaky"

    def __init__(self, fail_on=(), unconverged_on=()):
        self.calls = 0
        self.fail_on, self.unconverged_on = set(fail_on), set(unconverged_on)

    def fit(self, train):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("boom")
        return FitResult(params={"n": len(train)}, z=np.linspace(-3, 3, 401), nu=None,
                         converged=self.calls not in self.unconverged_on)

    def filter(self, fit, data):
        return np.full(len(data), 1e-4 * fit.params["n"] / 400)


def test_blocks_columns_and_refit_dates(panel):
    oos = panel.index[400].strftime("%Y-%m-%d")
    out = walk_forward(panel, Flaky(), oos, 21, ALPHAS)
    assert out.index[0] == panel.index[400] and out.index[-1] == panel.index[-1]
    assert {"sigma2", "refit_date", "fallback", "var_a010", "es_a025"} <= set(out.columns)
    assert out["refit_date"].nunique() == int(np.ceil(200 / 21))
    assert (out["es_a025"] <= out["var_a025"]).all()


@pytest.mark.parametrize("kw", [{"fail_on": [2]}, {"unconverged_on": [2]}])
def test_failed_or_unconverged_fit_falls_back(panel, kw):
    oos = panel.index[400].strftime("%Y-%m-%d")
    out = walk_forward(panel, Flaky(**kw), oos, 21, ALPHAS)
    blocks = out.groupby("refit_date")
    flags = blocks["fallback"].first().to_numpy()
    assert flags.tolist() == [False, True] + [False] * (len(flags) - 2)
    s = blocks["sigma2"].first().to_numpy()
    assert s[1] == s[0]  # block 2 reuses block 1 params


def test_first_fit_failure_raises(panel):
    with pytest.raises(RuntimeError, match="first fit"):
        walk_forward(panel, Flaky(fail_on=[1]), panel.index[400].strftime("%Y-%m-%d"), 21, ALPHAS)


@pytest.mark.parametrize("model", [EWMA(), HAR(), VIXModel(), GarchT("garch_t"),
                                   GarchT("egarch_t"), GarchT("gjr_t")])
def test_no_lookahead(panel, model):
    oos = panel.index[400].strftime("%Y-%m-%d")
    base = walk_forward(panel, model, oos, 21, ALPHAS)
    j = 450
    pert = panel.copy()
    cols = pert.columns.get_indexer(["log_return", "gk_var", "vix", "open", "high", "low", "close"])
    pert.iloc[j, cols] *= 3
    out = walk_forward(pert, model, oos, 21, ALPHAS)
    dj = panel.index[j]
    pd.testing.assert_frame_equal(base.loc[:dj], out.loc[:dj])
    assert not base.loc[panel.index[j + 1]].equals(out.loc[panel.index[j + 1]])
