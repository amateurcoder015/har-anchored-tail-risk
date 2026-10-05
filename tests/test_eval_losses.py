import numpy as np
import pandas as pd

from volgate.evaluate.losses import aligned, loss_frame, pooled_average


def test_aligned_uses_common_dates():
    idx = pd.bdate_range("2023-01-02", periods=5)
    y = pd.Series(np.linspace(-0.02, 0.02, 5), index=idx)
    a = pd.DataFrame({"var": -0.03, "es": -0.04}, index=idx)
    b = a.copy()
    b.iloc[1] = np.nan
    dates, yv, fc = aligned({"a": a, "b": b}, y)
    assert len(dates) == 4 and idx[1] not in dates
    L = loss_frame(fc, yv, 0.025, dates)
    assert list(L.columns) == ["a", "b"] and len(L) == 4 and (L.index == dates).all()


def test_pooled_average_requires_all_assets():
    i1 = pd.bdate_range("2023-01-02", periods=4)
    x = pd.DataFrame({"m": [1.0, 2.0, 3.0, 4.0]}, index=i1)
    y = pd.DataFrame({"m": [3.0, 4.0, 5.0]}, index=i1[1:])
    p = pooled_average({"x": x, "y": y})
    assert list(p.index) == list(i1[1:])
    assert p["m"].tolist() == [2.5, 3.5, 4.5]
