import numpy as np
import pandas as pd

from conftest import make_combo_frame
from hart.r2 import R2_FEATURES, build_r2_long, run_r2
from volgate.gate.walkforward import Fold

TOY = [Fold("2019-06-30", "2019-07-01", "2019-12-31", "2020-01-01", "2020-06-30")]


def _inputs():
    frames = {"a": make_combo_frame(650, 0), "b": make_combo_frame(650, 1).iloc[:500]}
    anchors = {k: f[["har__var_a025", "har__es_a025"]].set_axis(["var", "es"], axis=1)
               for k, f in frames.items()}
    anchors["a"].iloc[200, 0] = np.nan
    return frames, anchors


def test_build_long_drops_nan_and_has_features():
    long = build_r2_long(*_inputs(), 0.025)
    assert not long.isna().any().any()
    assert set(R2_FEATURES) <= set(long.columns)
    assert pd.Timestamp(_inputs()[0]["a"].index[200]) not in set(long.loc[long.asset == "a", "date"])


def test_run_r2_outputs_only_test_rows_and_present_assets():
    long = build_r2_long(*_inputs(), 0.025)
    long["ret_neg_lag1"] = 1.0  # constant feature in training
    for kind in ("mlp", "linear"):
        pred = run_r2(long, kind, 0.025, TOY, seeds=(0,), epochs=3, patience=2, hidden=4)
        assert pred["date"].min() >= pd.Timestamp("2020-01-01")
        assert set(pred["asset"]) == {"a"}  # asset b ends before the test fold
        assert (pred["es"] <= pred["var"]).all() and np.isfinite(pred["var"]).all()
        assert (pred[["cv", "cs"]].abs() <= 0.25).all().all()
