import pandas as pd
import pytest

from volgate.data import download as dl


def _fake_download(ticker, start, end, **kwargs):
    idx = pd.bdate_range(start, end, inclusive="left")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Adj Close": 1.4, "Volume": 10},
        index=idx,
    )


def test_download_writes_files_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", _fake_download)
    manifest = dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
    df = pd.read_csv(tmp_path / "x.csv", index_col="date", parse_dates=True)
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df.index.max() == pd.Timestamp("2024-01-05")  # end is inclusive
    assert manifest.loc[0, "rows"] == 5
    assert dl.verify_manifest(tmp_path) == []


def test_verify_manifest_detects_change(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", _fake_download)
    dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
    with open(tmp_path / "x.csv", "a") as f:
        f.write("tampered\n")
    assert dl.verify_manifest(tmp_path) == ["x.csv"]


def test_empty_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        dl.download_raw({"x": "X.NS"}, "2024-01-01", "2024-01-05", tmp_path)
