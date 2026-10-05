import numpy as np
import pandas as pd
import pytest

from volgate.data.panel import apply_corrections, build_asset_frame, gk_variance, outlier_table


def _raw(n=6):
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    close = 100 * np.exp(np.arange(n) * 0.01)
    return pd.DataFrame({
        "open": close * 0.99, "high": close * 1.02, "low": close * 0.97,
        "close": close, "adj_close": close, "volume": 1000,
    }, index=idx)


def test_gk_known_value_and_floor():
    df = pd.DataFrame({"open": [100.0, 100.0], "high": [110.0, 100.0],
                       "low": [100.0, 100.0], "close": [105.0, 101.0]})
    gk = gk_variance(df)
    assert gk.iloc[0] == pytest.approx(0.0036224, rel=1e-4)
    assert gk.iloc[1] == 1e-10  # high == low gives negative GK -> floored


def test_build_frame_uses_adj_close_and_drops_first_row():
    raw = _raw()
    raw["adj_close"] = raw["close"] * 0.5  # constant factor: returns unchanged
    vix = pd.Series(15.0, index=raw.index)
    df, stats = build_asset_frame(raw, vix)
    assert len(df) == 5 and stats["rows"] == 5
    assert df["log_return"].iloc[0] == pytest.approx(0.01)


def test_flat_no_trade_rows_are_dropped():
    raw = _raw()
    d = raw.index[3]
    raw.loc[d, ["open", "high", "low", "close", "adj_close"]] = raw.loc[raw.index[2], "close"]
    raw.loc[d, "volume"] = 0
    df, stats = build_asset_frame(raw, pd.Series(15.0, index=raw.index))
    assert stats["flat_dropped"] == 1
    assert d not in df.index
    assert (df["log_return"] != 0).all()


def test_vix_forward_fill_limited_to_one_day():
    raw = _raw(6)
    vix = pd.Series([10.0, 11.0, 12.0], index=raw.index[:3])  # missing last three days
    df, stats = build_asset_frame(raw, vix)
    assert df.loc[raw.index[3], "vix"] == 12.0  # filled once
    assert np.isnan(df.loc[raw.index[4], "vix"])
    assert stats["vix_filled"] == 1 and stats["vix_missing"] == 2


def test_outlier_table_and_corrections():
    raw = _raw()
    raw.loc[raw.index[3]:, ["adj_close", "close"]] *= 0.5  # -69% jump
    df, _ = build_asset_frame(raw, pd.Series(15.0, index=raw.index))
    table = outlier_table({"x": df}, 0.10)
    assert list(table["date"]) == [raw.index[3].strftime("%Y-%m-%d")]
    corr = pd.DataFrame({"asset": ["x"], "date": table["date"], "action": ["drop"], "note": ["test"]})
    fixed = apply_corrections({"x": df}, corr)
    assert raw.index[3] not in fixed["x"].index
    with pytest.raises(ValueError, match="unknown action"):
        apply_corrections({"x": df}, corr.assign(action="fix"))


def test_excluded_session_dropped_before_returns():
    raw = _raw(6)
    special = raw.index[3]
    df, stats = build_asset_frame(raw, pd.Series(15.0, index=raw.index), exclude_dates=[special])
    assert special not in df.index
    assert stats["special_dropped"] == 1
    # return on the day after the special session spans two days of price change
    assert df.loc[raw.index[4], "log_return"] == pytest.approx(0.02)


def test_nse_calendar_is_union_minus_excluded():
    from volgate.data.panel import nse_calendar
    a = pd.DataFrame(index=pd.DatetimeIndex(["2021-11-01", "2021-11-03", "2021-11-04"]))
    b = pd.DataFrame(index=pd.DatetimeIndex(["2021-11-02", "2021-11-03"]))
    cal = nse_calendar({"a": a, "b": b}, ["2021-11-04"])
    assert list(cal.strftime("%Y-%m-%d")) == ["2021-11-01", "2021-11-02", "2021-11-03"]
