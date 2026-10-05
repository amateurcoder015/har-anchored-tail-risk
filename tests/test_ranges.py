import numpy as np
import pandas as pd
import pytest

from hart.ranges import RANGE_FLOOR, parkinson, rogers_satchell


def _df():
    return pd.DataFrame({"open": [100.0, 100.0], "high": [110.0, 100.0],
                         "low": [95.0, 100.0], "close": [105.0, 100.0]})


def test_parkinson_known_value_and_floor():
    p = parkinson(_df())
    assert p.iloc[0] == pytest.approx(np.log(110 / 95) ** 2 / (4 * np.log(2)))
    assert p.iloc[1] == RANGE_FLOOR


def test_rogers_satchell_known_value_and_floor():
    r = rogers_satchell(_df())
    expected = np.log(110 / 105) * np.log(110 / 100) + np.log(95 / 105) * np.log(95 / 100)
    assert r.iloc[0] == pytest.approx(expected)
    assert r.iloc[1] == RANGE_FLOOR
