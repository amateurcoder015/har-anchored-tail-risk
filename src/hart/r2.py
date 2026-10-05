import numpy as np
import pandas as pd

from hart.correction import CorrectionNet
from volgate.gate.features import gate_features
from volgate.gate.net import train_gate
from volgate.risk.fz import fz0_loss

R2_FEATURES = ["vov30", "vix", "vix_chg5", "vrp", "anchor_fz60", "ret_neg_lag1", "absret_lag1"]


def build_r2_long(frames, anchors, alpha):
    parts = []
    for asset, df in frames.items():
        g = gate_features(df, alpha)
        an = anchors[asset].reindex(df.index)
        f = g[["vov30", "vix", "vix_chg5", "vrp", "ret_neg_lag1", "absret_lag1"]].copy()
        y = df["log_return"]
        ok = an["var"].notna() & an["es"].notna() & y.notna()
        L = pd.Series(np.nan, index=df.index)
        L[ok] = fz0_loss(y[ok].to_numpy(), an.loc[ok, "var"].to_numpy(), an.loc[ok, "es"].to_numpy(), alpha)
        f["anchor_fz60"] = L.shift(1).rolling(60, min_periods=60).mean()
        f["VA"], f["EA"], f["y"] = an["var"], an["es"], y
        f.insert(0, "date", df.index)
        f.insert(0, "asset", asset)
        parts.append(f.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True).dropna().reset_index(drop=True)


def _batch(rows, mean, std, assets):
    return {"X": (rows[R2_FEATURES].to_numpy(float) - mean) / std,
            "a": rows["asset"].map(assets).to_numpy(int),
            "VA": rows["VA"].to_numpy(float), "EA": rows["EA"].to_numpy(float),
            "y": rows["y"].to_numpy(float)}


def run_r2(long, kind, alpha, folds, lam_c=1.0, seeds=(0, 1, 2, 3, 4), epochs=300, patience=20,
           hidden=16, lam_mean=0.0):
    assets = {a: i for i, a in enumerate(sorted(long["asset"].unique()))}
    preds = []
    for k, fold in enumerate(folds, start=1):
        d = long["date"]
        tr_rows = long[d <= fold.train_end]
        va_rows = long[(d >= fold.val_start) & (d <= fold.val_end)]
        te_rows = long[(d >= fold.test_start) & (d <= fold.test_end)]
        if te_rows.empty:
            continue
        mean = tr_rows[R2_FEATURES].to_numpy(float).mean(axis=0)
        std = tr_rows[R2_FEATURES].to_numpy(float).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        tr, va, te = (_batch(r, mean, std, assets) for r in (tr_rows, va_rows, te_rows))
        outs = []
        for s in seeds:
            net = CorrectionNet(len(R2_FEATURES), len(assets), kind=kind, hidden=hidden,
                                alpha=alpha, lam_c=lam_c, lam_mean=lam_mean, seed=s)
            train_gate(net, tr, va, epochs=epochs, patience=patience, seed=s)
            outs.append(net.forward(te))
        preds.append(pd.DataFrame({
            "asset": te_rows["asset"].to_numpy(), "date": te_rows["date"].to_numpy(), "fold": k,
            "var": np.mean([o["var"] for o in outs], axis=0),
            "es": np.mean([o["es"] for o in outs], axis=0),
            "cv": np.mean([o["cv"] for o in outs], axis=0),
            "cs": np.mean([o["cs"] for o in outs], axis=0)}))
    return pd.concat(preds, ignore_index=True)
