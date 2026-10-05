import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_raw(tickers: dict[str, str], start: str, end: str, out_dir: Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for key, ticker in tickers.items():
        df = yf.download(
            ticker, start=start, end=end_exclusive, auto_adjust=False,
            actions=False, progress=False, multi_level_index=False,
        )
        if df is None or df.empty:
            raise RuntimeError(f"no data for {ticker}")
        df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))[_COLUMNS]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = "date"
        path = out_dir / f"{key}.csv"
        df.to_csv(path, date_format="%Y-%m-%d")
        rows.append({
            "key": key, "ticker": ticker, "file": path.name, "rows": len(df),
            "first": df.index.min().strftime("%Y-%m-%d"),
            "last": df.index.max().strftime("%Y-%m-%d"),
            "sha256": sha256_file(path),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "MANIFEST.csv", index=False)
    return manifest


def verify_manifest(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    manifest = pd.read_csv(out_dir / "MANIFEST.csv")
    return [
        row.file for row in manifest.itertuples()
        if sha256_file(out_dir / row.file) != row.sha256
    ]
