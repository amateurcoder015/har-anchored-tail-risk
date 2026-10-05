import numpy as np
import pandas as pd

from conftest import make_combo_frame
from volgate.gate.walkforward import FOLDS, Fold, GateConfig, VARIANTS, build_pooled, run_variant

FAST = dict(epochs=3, patience=2, hidden=4)
TOY_FOLDS = [Fold("2019-06-30", "2019-07-01", "2019-12-31", "2020-01-01", "2020-06-30")]


def _frames():
    a = make_combo_frame(650, seed=0)
    b = make_combo_frame(650, seed=1).iloc[5:]  # different date set
    return {"a": a, "b": b}


def test_variants_and_folds():
    assert set(VARIANTS) == {"gate", "gate_novix", "gate_noexpiry", "gate_noscale",
                             "gate_variance", "gate_qlike", "gate_v2"}
    assert FOLDS[-1].test_end == "2026-09-30"


def test_build_pooled_keeps_asset_dates_and_drops_nan():
    frames = _frames()
    long = build_pooled(frames, 0.025)
    assert set(long["asset"]) == {"a", "b"}
    assert not long.isna().any().any()
    assert set(long.loc[long.asset == "b", "date"]) <= set(frames["b"].index)


def test_run_variant_outputs_test_rows_only_and_valid_pairs():
    long = build_pooled(_frames(), 0.025)
    cfg = GateConfig("t", **FAST)
    pred, sel = run_variant(long, cfg, 0.025, TOY_FOLDS, lam_grid=(0.0, 1.0), seeds=(0, 1))
    assert pred["date"].min() >= pd.Timestamp("2020-01-01")
    assert pred["date"].max() <= pd.Timestamp("2020-06-30")
    assert (pred["es"] <= pred["var"]).all() and (pred["var"] < 0).all()
    assert len(sel) == 2


def test_variance_variants_run():
    long = build_pooled(_frames(), 0.025)
    for name in ("gate_variance", "gate_qlike"):
        cfg = GateConfig(name, mode="variance", loss=VARIANTS[name].loss, use_scale=False, **FAST)
        pred, _ = run_variant(long, cfg, 0.025, TOY_FOLDS, lam_grid=(0.0,), seeds=(0,))
        assert (pred["es"] <= pred["var"]).all()


def test_constant_feature_does_not_break_scaling():
    long = build_pooled(_frames(), 0.025)
    long["own_is_expiry"] = 0.0  # zero std in training
    pred, _ = run_variant(long, GateConfig("t", **FAST), 0.025, TOY_FOLDS, lam_grid=(0.0,), seeds=(0,))
    assert np.isfinite(pred["var"]).all()


def test_test_period_returns_do_not_change_earlier_predictions():
    frames = _frames()
    cfg = GateConfig("t", **FAST)
    base, _ = run_variant(build_pooled(frames, 0.025), cfg, 0.025, TOY_FOLDS, (0.0,), (0,))
    j = frames["a"].index.get_loc(pd.Timestamp(base["date"].iloc[len(base) // 4]))
    pert = {k: v.copy() for k, v in frames.items()}
    pert["a"].iloc[j, pert["a"].columns.get_loc("log_return")] = -0.5
    out, _ = run_variant(build_pooled(pert, 0.025), cfg, 0.025, TOY_FOLDS, (0.0,), (0,))
    cut = frames["a"].index[j]
    m = (base.asset == "a") & (base.date <= cut)
    assert m.sum() > 10 and len(out) == len(base)
    pd.testing.assert_frame_equal(base[m].reset_index(drop=True), out[m].reset_index(drop=True))
