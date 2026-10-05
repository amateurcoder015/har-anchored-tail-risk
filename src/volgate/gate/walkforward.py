from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volgate.combine.data import MODELS, pairs, variances
from volgate.gate.features import feature_columns, gate_features
from volgate.gate.net import GateNet, train_gate


@dataclass(frozen=True)
class GateConfig:
    name: str
    mode: str = "pairs"
    loss: str = "fz0"
    use_scale: bool = True
    use_vix: bool = True
    use_expiry: bool = True
    hidden: int = 16
    emb: int = 4
    dropout: float = 0.1
    wd: float = 1e-4
    lr: float = 1e-3
    epochs: int = 300
    batch: int = 256
    patience: int = 20
    lam_g: float = 0.0


VARIANTS = {c.name: c for c in [
    GateConfig("gate"),
    GateConfig("gate_novix", use_vix=False),
    GateConfig("gate_noexpiry", use_expiry=False),
    GateConfig("gate_noscale", use_scale=False),
    GateConfig("gate_variance", mode="variance", use_scale=False),
    GateConfig("gate_qlike", mode="variance", loss="qlike", use_scale=False),
    GateConfig("gate_v2", lam_g=10.0),
]}


@dataclass(frozen=True)
class Fold:
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


FOLDS = [
    Fold("2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    Fold("2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    Fold("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    Fold("2024-12-31", "2025-01-01", "2025-12-31", "2026-01-01", "2026-09-30"),
]


def build_pooled(frames: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    """Long frame over assets: features, base pairs and variances, target; NaN rows dropped."""
    parts = []
    for asset, df in frames.items():
        f = gate_features(df, alpha)
        V, E = pairs(df, alpha)
        S2 = variances(df)
        for i, m in enumerate(MODELS):
            f[f"V__{m}"], f[f"E__{m}"], f[f"S2__{m}"] = V[:, i], E[:, i], S2[:, i]
        f["y"] = df["log_return"].to_numpy()
        f.insert(0, "date", df.index)
        f.insert(0, "asset", asset)
        parts.append(f.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True).dropna().reset_index(drop=True)


def _batch(rows, cols, mean, std, assets):
    return {
        "X": (rows[cols].to_numpy(float) - mean) / std,
        "a": rows["asset"].map(assets).to_numpy(int),
        "V": rows[[f"V__{m}" for m in MODELS]].to_numpy(float),
        "E": rows[[f"E__{m}" for m in MODELS]].to_numpy(float),
        "S2": rows[[f"S2__{m}" for m in MODELS]].to_numpy(float),
        "y": rows["y"].to_numpy(float),
    }


def _fit_nu(rows) -> float:
    s2 = rows[[f"S2__{m}" for m in MODELS]].to_numpy(float).mean(axis=1)
    z = rows["y"].to_numpy(float) / np.sqrt(s2)
    return float(max(stats.t.fit(z, floc=0)[0], 2.5))


def _net(cfg, n_feat, n_assets, alpha, nu, lam, seed):
    return GateNet(n_feat, n_assets, len(MODELS), hidden=cfg.hidden, emb=cfg.emb,
                   dropout=cfg.dropout, mode=cfg.mode, loss=cfg.loss, use_scale=cfg.use_scale,
                   alpha=alpha, nu=nu, lam_eq=lam, wd=cfg.wd, lam_g=cfg.lam_g, seed=seed)


def run_variant(long, cfg, alpha, folds, lam_grid=(0.0, 0.1, 1.0, 10.0), seeds=(0, 1, 2, 3, 4)):
    """Per fold: choose lam_eq on validation with the first seed, then average seeds on test."""
    cols = feature_columns(cfg.use_vix, cfg.use_expiry)
    assets = {a: i for i, a in enumerate(sorted(long["asset"].unique()))}
    preds, selection = [], []
    for k, fold in enumerate(folds, start=1):
        d = long["date"]
        tr_rows = long[d <= fold.train_end]
        va_rows = long[(d >= fold.val_start) & (d <= fold.val_end)]
        te_rows = long[(d >= fold.test_start) & (d <= fold.test_end)]
        mean = tr_rows[cols].to_numpy(float).mean(axis=0)
        std = tr_rows[cols].to_numpy(float).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        tr, va, te = (_batch(r, cols, mean, std, assets) for r in (tr_rows, va_rows, te_rows))
        nu = _fit_nu(tr_rows) if cfg.mode == "variance" else None
        best_lam, best_val = None, np.inf
        for lam in lam_grid:
            net = _net(cfg, len(cols), len(assets), alpha, nu, lam, seeds[0])
            info = train_gate(net, tr, va, lr=cfg.lr, epochs=cfg.epochs, batch=cfg.batch,
                              patience=cfg.patience, seed=seeds[0])
            selection.append({"fold": k, "lam": lam, "val_loss": info["best_val"]})
            if info["best_val"] < best_val:
                best_lam, best_val = lam, info["best_val"]
        outs = []
        for s in seeds:
            net = _net(cfg, len(cols), len(assets), alpha, nu, best_lam, s)
            train_gate(net, tr, va, lr=cfg.lr, epochs=cfg.epochs, batch=cfg.batch,
                       patience=cfg.patience, seed=s)
            outs.append(net.forward(te))
        out = pd.DataFrame({"asset": te_rows["asset"].to_numpy(), "date": te_rows["date"].to_numpy(),
                            "fold": k, "lam": best_lam,
                            "var": np.mean([o["var"] for o in outs], axis=0),
                            "es": np.mean([o["es"] for o in outs], axis=0)})
        wq = np.mean([o["wq"] for o in outs], axis=0)
        for i, m in enumerate(MODELS):
            out[f"wq_{m}"] = wq[:, i]
        out["g"] = np.mean([o.get("g", np.ones(len(te_rows))) for o in outs], axis=0)
        preds.append(out)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(selection)
