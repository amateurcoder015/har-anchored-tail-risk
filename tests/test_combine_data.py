import numpy as np
import pandas as pd

from volgate.combine.data import MODELS, load_asset, model_losses, pairs, variances
from volgate.risk.fz import fz0_loss


def test_pairs_and_variances_shapes(combo_frame):
    V, E = pairs(combo_frame, 0.025)
    assert V.shape == E.shape == (len(combo_frame), 6)
    assert np.all(E <= V)
    assert variances(combo_frame).shape == (len(combo_frame), 6)


def test_model_losses_match_fz0_and_propagate_nan(combo_frame):
    df = combo_frame.copy()
    df.iloc[5, df.columns.get_loc("vix__var_a025")] = np.nan
    L = model_losses(df, 0.025)
    assert list(L.columns) == MODELS
    y = df["log_return"].iloc[3]
    expected = fz0_loss(y, df["ewma__var_a025"].iloc[3], df["ewma__es_a025"].iloc[3], 0.025)
    assert L["ewma"].iloc[3] == expected
    assert np.isnan(L["vix"].iloc[5]) and np.isfinite(L["ewma"].iloc[5])


def test_load_asset_inner_joins(tmp_path, combo_frame):
    (tmp_path / "panel").mkdir()
    (tmp_path / "base_forecasts").mkdir()
    panel_cols = [c for c in combo_frame.columns if "__" not in c]
    base_cols = [c for c in combo_frame.columns if "__" in c]
    combo_frame[panel_cols].to_csv(tmp_path / "panel" / "x.csv")
    combo_frame[base_cols].iloc[100:].to_csv(tmp_path / "base_forecasts" / "x.csv")
    df = load_asset(tmp_path, "x")
    assert len(df) == len(combo_frame) - 100
    assert isinstance(df.index, pd.DatetimeIndex)
