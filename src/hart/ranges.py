import numpy as np
import pandas as pd

RANGE_FLOOR = 1e-10


def parkinson(df: pd.DataFrame) -> pd.Series:
    return (np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))).clip(lower=RANGE_FLOOR)


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = (np.log(df[k]) for k in ("high", "low", "open", "close"))
    return ((h - c) * (h - o) + (l - c) * (l - o)).clip(lower=RANGE_FLOOR)
