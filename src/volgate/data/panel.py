from pathlib import Path

import numpy as np
import pandas as pd

_GK_FLOOR = 1e-10


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True).sort_index()


def gk_variance(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return (0.5 * hl**2 - (2 * np.log(2) - 1) * co**2).clip(lower=_GK_FLOOR)


def nse_calendar(frames: dict[str, pd.DataFrame], exclude_dates=()) -> pd.DatetimeIndex:
    """Common NSE trading calendar: union of all asset dates minus special sessions."""
    cal = pd.DatetimeIndex([])
    for df in frames.values():
        cal = cal.union(df.index)
    return cal.difference(pd.DatetimeIndex(exclude_dates))


def build_asset_frame(raw: pd.DataFrame, vix_close: pd.Series, drop_flat: bool = True,
                      exclude_dates=()) -> tuple[pd.DataFrame, dict]:
    """Exclude special sessions (e.g. Diwali Muhurat) before computing returns."""
    df = raw.sort_index().dropna(subset=["open", "high", "low", "close", "adj_close"]).copy()
    special = df.index.isin(pd.DatetimeIndex(exclude_dates))
    df = df[~special]
    n_flat = 0
    if drop_flat:
        flat = ((df["high"] == df["low"]) & (df["open"] == df["close"])
                & (df["high"] == df["close"]) & (df["volume"] == 0))
        n_flat = int(flat.sum())
        df = df[~flat]
    df["log_return"] = np.log(df["adj_close"]).diff()
    df["gk_var"] = gk_variance(df)
    vix_close = vix_close.dropna()
    union = df.index.union(vix_close.index)
    df["vix"] = vix_close.reindex(union).ffill(limit=1).reindex(df.index)
    df = df.iloc[1:]
    stats = {
        "rows": len(df),
        "flat_dropped": n_flat,
        "special_dropped": int(special.sum()),
        "vix_filled": int((~df.index.isin(vix_close.index) & df["vix"].notna()).sum()),
        "vix_missing": int(df["vix"].isna().sum()),
    }
    return df, stats


def outlier_table(frames: dict[str, pd.DataFrame], threshold: float) -> pd.DataFrame:
    rows = []
    for asset, df in frames.items():
        close_ret = np.log(df["close"]).diff()
        for d in df.index[df["log_return"].abs() > threshold]:
            rows.append({"asset": asset, "date": d.strftime("%Y-%m-%d"),
                         "log_return": df.at[d, "log_return"], "close_return": close_ret.get(d)})
    return pd.DataFrame(rows, columns=["asset", "date", "log_return", "close_return"])


def apply_corrections(frames: dict[str, pd.DataFrame],
                      corrections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in frames.items()}
    for row in corrections.itertuples(index=False):
        if row.action == "keep":
            continue
        if row.action != "drop":
            raise ValueError(f"unknown action {row.action!r}")
        d = pd.Timestamp(row.date)
        if d not in out[row.asset].index:
            raise KeyError(f"{row.asset} {row.date} not in data")
        out[row.asset] = out[row.asset].drop(index=d)
    return out
