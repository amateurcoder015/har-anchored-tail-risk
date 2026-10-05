# Plan 2: R2 Bounded State Correction and Development Runs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement R2 (MLP and linear bounded corrections of an anchor VaR/ES pair, trained on FZ0), run it on the development universe within the 3-revision budget, choose the anchor, and log everything in `docs/devlog.md`.

**Architecture:** `src/hart/correction.py` holds `CorrectionNet` (NumPy, hand-written gradients, duck-type compatible with `volgate.gate.net.train_gate`). `src/hart/r2.py` builds the pooled correction dataset from panels and anchor forecasts and runs the yearly walk-forward (reusing `volgate.gate.walkforward.FOLDS`). `scripts/10_r2.py` writes forecasts; `scripts/11_dev_compare.py` writes the development comparison table.

**Tech Stack:** NumPy, pandas, ported `volgate` (FZ0, gate features, training loop, folds, DM).

**Spec:** `docs/superpowers/specs/2026-10-06-har-anchored-design.md` sections 2 (R2, anchor choice), 4, 8.

## Global Constraints

- VaR = VaR_A·exp(c_v); ES = VaR + (ES_A − VaR_A)·exp(c_s); c = 0.25·tanh(f(x)).
- Loss: mean FZ0 + λ_c·mean(c_v² + c_s²) + weight decay 1e-4. λ_c starts at 1.0 (Run 1); any change is a logged revision.
- R2-MLP: hidden 16, ReLU, dropout 0.1, asset embedding 4. R2-linear: f = Wx + b, no hidden layer, no embedding.
- Correction features x_t (through t−1, standardized on training rows): vov30, vix, vix_chg5, vrp, anchor trailing 60-day mean FZ0 (`anchor_fz60`), ret_neg_lag1, absret_lag1.
- Folds = predecessor `FOLDS` (tests 2023, 2024, 2025, 2026-01..09); Adam lr 1e-3, batch 256, max 300 epochs, patience 20, 5 seeds averaged.
- Heads initialised at zero: an untrained correction returns the anchor exactly.
- Development budget: at most 3 revisions after Run 1, each logged.

## Review Focus

1. Anchor with NaN forecasts on some days → rows dropped from the pooled dataset, no NaN in training (Task 2 test).
2. ES spacing of the anchor equal to zero on a row (ES_A = VaR_A) → corrected ES equals corrected VaR, still ≤ VaR, no division by zero (Task 1 test).
3. Linear mode with an embedding size given → embedding ignored, gradients still correct (Task 1 test).
4. Features constant in a training fold → std set to 1 (Task 2 test).
5. A fold with no test rows for an asset → predictions only for assets present (Task 2 test).

---

### Task 1: CorrectionNet

**Files:** Create `src/hart/correction.py`; test `tests/test_correction.py`.

**Interfaces:**
- `CorrectionNet(n_features, n_assets, kind="mlp"|"linear", hidden=16, emb=4, dropout=0.1, alpha=0.025, lam_c=1.0, wd=1e-4, bound=0.25, seed=0)`; attribute `p`.
- `forward(b, train=False, rng=None) -> dict` with `var, es, cv, cs`; batch `b` keys `X, a, VA, EA, y`.
- `loss_and_grad(b, train=False, rng=None) -> (float, dict)`; `data_loss(b) -> float` (mean FZ0 only).
- Compatible with `volgate.gate.net.train_gate(net, tr, va, ...)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
import pytest
from scipy import stats

from hart.correction import CorrectionNet
from volgate.gate.net import train_gate

A = 0.025


def _batch(n=80, F=4, n_assets=2, seed=0):
    rng = np.random.default_rng(seed)
    s = np.exp(rng.normal(-4, 0.3, n))
    q = stats.norm.ppf(A)
    VA = s * q
    EA = -s * stats.norm.pdf(q) / A
    return {"X": rng.normal(size=(n, F)), "a": rng.integers(0, n_assets, n), "VA": VA, "EA": EA,
            "y": rng.normal(0, 0.03, n)}


@pytest.mark.parametrize("kind", ["mlp", "linear"])
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_gradients_match_finite_differences(kind, dropout):
    b = _batch()
    net = CorrectionNet(4, 2, kind=kind, hidden=5, emb=2, dropout=dropout, lam_c=0.7, wd=1e-3, seed=1)
    rng = np.random.default_rng(2)
    for k in net.p:
        net.p[k] = net.p[k] + rng.normal(0, 0.3, net.p[k].shape)
    out = net.forward(b)
    hits = (b["y"] <= out["var"]).sum()
    assert 0 < hits < len(b["y"])

    def loss(grad=False):
        res = net.loss_and_grad(b, train=dropout > 0, rng=np.random.default_rng(7))
        return res if grad else res[0]

    _, g = loss(grad=True)
    h = 1e-6
    for k, v in net.p.items():
        for idx in np.ndindex(v.shape):
            old = v[idx]
            v[idx] = old + h
            lp = loss()
            v[idx] = old - h
            lm = loss()
            v[idx] = old
            assert g[k][idx] == pytest.approx((lp - lm) / (2 * h), rel=1e-4, abs=1e-6), (k, idx)


@pytest.mark.parametrize("kind", ["mlp", "linear"])
def test_untrained_equals_anchor(kind):
    b = _batch()
    out = CorrectionNet(4, 2, kind=kind).forward(b)
    np.testing.assert_allclose(out["var"], b["VA"])
    np.testing.assert_allclose(out["es"], b["EA"])


def test_bounds_and_ordering_with_extreme_params():
    b = _batch()
    b["EA"][0] = b["VA"][0]  # zero ES spacing on one row
    net = CorrectionNet(4, 2, seed=0)
    for k in net.p:
        net.p[k] = net.p[k] + 50.0
    out = net.forward(b)
    assert np.all(np.abs(out["cv"]) <= 0.25 + 1e-12) and np.all(np.abs(out["cs"]) <= 0.25 + 1e-12)
    assert np.all(out["es"] <= out["var"]) and np.all(out["var"] < 0)
    assert out["es"][0] == out["var"][0]


def test_linear_ignores_embedding():
    net = CorrectionNet(4, 2, kind="linear", emb=3)
    assert "emb" not in net.p and "W1" not in net.p


def test_learns_state_dependent_scale():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    sd = np.where(x == 1, 0.012, 0.01)  # anchor assumes 0.01; true risk 20% higher when x = 1
    y = rng.normal(0, sd)
    q = stats.norm.ppf(A)
    b = {"X": x[:, None], "a": np.zeros(n, int), "VA": np.full(n, 0.01 * q),
         "EA": np.full(n, -0.01 * stats.norm.pdf(q) / A), "y": y}
    tr = {k: v[:3000] for k, v in b.items()}
    va = {k: v[3000:] for k, v in b.items()}
    net = CorrectionNet(1, 1, emb=0, dropout=0.0, lam_c=0.0, seed=0)
    base = net.data_loss(va)
    info = train_gate(net, tr, va, lr=1e-2, epochs=100, patience=20, seed=0)
    assert info["best_val"] < base
    cv = net.forward(va)["cv"]
    assert cv[va["X"][:, 0] == 1].mean() > cv[va["X"][:, 0] == 0].mean() + 0.1
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/hart/correction.py`**

```python
import numpy as np

_DECAYED = ("W1", "Wo", "emb")


class CorrectionNet:
    """Bounded multiplicative correction of an anchor VaR/ES pair.

    VaR = VA*exp(cv); ES = VaR + (EA - VA)*exp(cs); c = bound*tanh(f(x)).
    kind="mlp": f = relu([x, emb(a)] W1 + b1) Wo + bo; kind="linear": f = x Wo + bo.
    """

    def __init__(self, n_features, n_assets, kind="mlp", hidden=16, emb=4, dropout=0.1,
                 alpha=0.025, lam_c=1.0, wd=1e-4, bound=0.25, seed=0):
        if kind not in ("mlp", "linear"):
            raise ValueError(f"bad kind {kind!r}")
        self.kind, self.F, self.alpha = kind, n_features, alpha
        self.dropout = dropout if kind == "mlp" else 0.0
        self.lam_c, self.wd, self.bound = lam_c, wd, bound
        rng = np.random.default_rng(seed)
        if kind == "mlp":
            d = n_features + emb
            self.p = {"W1": rng.normal(0, np.sqrt(2.0 / d), (d, hidden)), "b1": np.zeros(hidden),
                      "emb": rng.normal(0, 0.1, (n_assets, emb)),
                      "Wo": np.zeros((hidden, 2)), "bo": np.zeros(2)}
        else:
            self.p = {"Wo": np.zeros((n_features, 2)), "bo": np.zeros(2)}

    def forward(self, b, train=False, rng=None):
        p = self.p
        c = {}
        if self.kind == "mlp":
            z = np.concatenate([b["X"], p["emb"][b["a"]]], axis=1)
            a1 = z @ p["W1"] + p["b1"]
            h = np.maximum(a1, 0.0)
            if train and self.dropout > 0:
                mask = (rng.random(h.shape) >= self.dropout) / (1.0 - self.dropout)
            else:
                mask = np.ones_like(h)
            hd = h * mask
            c.update(z=z, a1=a1, mask=mask, inp=hd)
        else:
            c["inp"] = b["X"]
        th = np.tanh(c["inp"] @ p["Wo"] + p["bo"])
        cvs = self.bound * th
        c["th"], c["cv"], c["cs"] = th, cvs[:, 0], cvs[:, 1]
        c["D"] = b["EA"] - b["VA"]
        c["var"] = b["VA"] * np.exp(c["cv"])
        c["es"] = c["var"] + c["D"] * np.exp(c["cs"])
        return c

    def _fz(self, c, y):
        v, e, n = c["var"], c["es"], len(y)
        hit = (y <= v).astype(float)
        loss = np.mean(-hit * (v - y) / (self.alpha * e) + v / e + np.log(-e) - 1.0)
        dv = (-hit / (self.alpha * e) + 1.0 / e) / n
        de = (hit * (v - y) / (self.alpha * e**2) - v / e**2 + 1.0 / e) / n
        return loss, dv, de

    def data_loss(self, b):
        return float(self._fz(self.forward(b), b["y"])[0])

    def loss_and_grad(self, b, train=False, rng=None):
        p = self.p
        c = self.forward(b, train, rng)
        n = len(b["y"])
        loss, dv, de = self._fz(c, b["y"])
        dcv = (dv + de) * c["var"] + 2.0 * self.lam_c * c["cv"] / n
        dcs = de * c["D"] * np.exp(c["cs"]) + 2.0 * self.lam_c * c["cs"] / n
        dpre = np.column_stack([dcv, dcs]) * self.bound * (1.0 - c["th"] ** 2)
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"], g["bo"] = c["inp"].T @ dpre, dpre.sum(axis=0)
        if self.kind == "mlp":
            da1 = (dpre @ p["Wo"].T) * c["mask"] * (c["a1"] > 0)
            g["W1"], g["b1"] = c["z"].T @ da1, da1.sum(axis=0)
            np.add.at(g["emb"], b["a"], (da1 @ p["W1"].T)[:, self.F:])
        reg = self.lam_c * np.mean(c["cv"] ** 2 + c["cs"] ** 2)
        for k in _DECAYED:
            if k in p:
                reg += 0.5 * self.wd * np.sum(p[k] ** 2)
                g[k] = g[k] + self.wd * p[k]
        return float(loss + reg), g
```

- [ ] **Step 4: Run** — Expected: 8 passed; full suite passes.
- [ ] **Step 5: Commit** `feat: add bounded VaR/ES correction network with verified gradients`.

---

### Task 2: Pooled dataset and walk-forward driver

**Files:** Create `src/hart/r2.py`; test `tests/test_r2.py`.

**Interfaces:**
- `R2_FEATURES = ["vov30", "vix", "vix_chg5", "vrp", "anchor_fz60", "ret_neg_lag1", "absret_lag1"]`
- `build_r2_long(frames: dict[str, pd.DataFrame], anchors: dict[str, pd.DataFrame], alpha) -> pd.DataFrame` — `anchors[asset]` has `var, es` indexed by date; output columns `asset, date, y, VA, EA` + features; NaN rows dropped.
- `run_r2(long, kind, alpha, folds, lam_c=1.0, seeds=(0,1,2,3,4), epochs=300, patience=20, hidden=16) -> pd.DataFrame` with `asset, date, fold, var, es, cv, cs`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
import pandas as pd

from conftest import make_combo_frame
from hart.r2 import R2_FEATURES, build_r2_long, run_r2
from volgate.gate.walkforward import Fold

TOY = [Fold("2019-06-30", "2019-07-01", "2019-12-31", "2020-01-01", "2020-06-30")]


def _inputs():
    frames = {"a": make_combo_frame(650, 0), "b": make_combo_frame(650, 1).iloc[:500]}
    anchors = {k: f[["har__var_a025", "har__es_a025"]].set_axis(["var", "es"], axis=1)
               for k, f in frames.items()}
    anchors["a"].iloc[200, 0] = np.nan
    return frames, anchors


def test_build_long_drops_nan_and_has_features():
    long = build_r2_long(*_inputs(), 0.025)
    assert not long.isna().any().any()
    assert set(R2_FEATURES) <= set(long.columns)
    assert pd.Timestamp(_inputs()[0]["a"].index[200]) not in set(long.loc[long.asset == "a", "date"])


def test_run_r2_outputs_only_test_rows_and_present_assets():
    long = build_r2_long(*_inputs(), 0.025)
    long["ret_neg_lag1"] = 1.0  # constant feature in training
    for kind in ("mlp", "linear"):
        pred = run_r2(long, kind, 0.025, TOY, seeds=(0,), epochs=3, patience=2, hidden=4)
        assert pred["date"].min() >= pd.Timestamp("2020-01-01")
        assert set(pred["asset"]) == {"a"}  # asset b ends before the test fold
        assert (pred["es"] <= pred["var"]).all() and np.isfinite(pred["var"]).all()
        assert (pred[["cv", "cs"]].abs() <= 0.25).all().all()
```

- [ ] **Step 2: Run** — Expected: module missing.

- [ ] **Step 3: Implement `src/hart/r2.py`**

```python
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
           hidden=16):
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
                                alpha=alpha, lam_c=lam_c, seed=s)
            train_gate(net, tr, va, epochs=epochs, patience=patience, seed=s)
            outs.append(net.forward(te))
        preds.append(pd.DataFrame({
            "asset": te_rows["asset"].to_numpy(), "date": te_rows["date"].to_numpy(), "fold": k,
            "var": np.mean([o["var"] for o in outs], axis=0),
            "es": np.mean([o["es"] for o in outs], axis=0),
            "cv": np.mean([o["cv"] for o in outs], axis=0),
            "cs": np.mean([o["cs"] for o in outs], axis=0)}))
    return pd.concat(preds, ignore_index=True)
```

- [ ] **Step 4: Run** — Expected: 2 passed; full suite passes.
- [ ] **Step 5: Commit** `feat: add R2 pooled dataset and walk-forward driver`.

---

### Task 3: Development runs, anchor choice, devlog

**Files:** Create `scripts/10_r2.py`, `scripts/11_dev_compare.py`; modify `Makefile`, `docs/devlog.md`, `README.md`.

**Interfaces:**
- `scripts/10_r2.py`: reads `anchor` (`r0` or `r1`) and `lam_c` from the config (`r2: {anchor, lam_c}`), writes `data/{universe}/processed/r2/{tag}/{kind}.csv` for kind in mlp, linear at every α.
- `scripts/11_dev_compare.py`: for every α, pooled and per-asset mean FZ0 and DM (method vs R0, vs R1, vs Taylor min-score, vs equal weights) for R0, R1, R2-MLP, R2-linear, mean, min_score, relative_score, gate_v2 (if present); writes `results/{universe}/tables/compare.csv` and prints the pooled α = 2.5% table.

- [ ] **Step 1: Add to `configs/dev.yaml`** `r2: {anchor: <r0 or r1 from Plan 1 devlog Run 0>, lam_c: 1.0}`.

- [ ] **Step 2: `scripts/10_r2.py`**

```python
import pandas as pd

from hart.r2 import build_r2_long, run_r2
from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.gate.walkforward import FOLDS
from volgate.risk.dist import alpha_tag

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
anchor, lam_c = cfg["r2"]["anchor"], cfg["r2"]["lam_c"]
frames = {a: load_asset(P["processed"], a) for a in cfg["assets"]}
for a in cfg["alphas"]:
    t = alpha_tag(a)
    if anchor == "r0":
        anchors = {k: f[[f"har__var_{t}", f"har__es_{t}"]].set_axis(["var", "es"], axis=1)
                   for k, f in frames.items()}
    else:
        anchors = {k: pd.read_csv(P["processed"] / "r1" / f"{k}.csv", index_col="date",
                                  parse_dates=True)[[f"var_{t}", f"es_{t}"]].set_axis(["var", "es"], axis=1)
                   for k in frames}
    long = build_r2_long(frames, anchors, a)
    out_dir = P["processed"] / "r2" / t
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("mlp", "linear"):
        run_r2(long, kind, a, FOLDS, lam_c=lam_c).to_csv(out_dir / f"{kind}.csv", index=False,
                                                          date_format="%Y-%m-%d")
        print(f"{t} {kind} done", flush=True)
```

- [ ] **Step 3: `scripts/11_dev_compare.py`**

```python
import numpy as np
import pandas as pd

from volgate.combine.data import load_asset
from volgate.config import REPO_ROOT, load_config
from volgate.evaluate.losses import pooled_average
from volgate.evaluate.stats import dm_test
from volgate.risk.dist import alpha_tag
from volgate.risk.fz import fz0_loss

START, END = "2023-01-01", "2026-09-30"
REFS = ["r0", "r1", "min_score", "mean"]
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    r2 = {k: pd.read_csv(P["processed"] / "r2" / t / f"{k}.csv", parse_dates=["date"])
          for k in ("mlp", "linear")}
    gate_file = P["processed"] / "gate" / t / "gate_v2.csv"
    gate = pd.read_csv(gate_file, parse_dates=["date"]) if gate_file.exists() else None
    per = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[START:END]
        r1 = pd.read_csv(P["processed"] / "r1" / f"{asset}.csv", index_col="date", parse_dates=True)
        co = pd.read_csv(P["processed"] / "combos" / t / f"{asset}.csv", index_col="date", parse_dates=True)
        fc = {"r0": (df[f"har__var_{t}"], df[f"har__es_{t}"]),
              "r1": (r1[f"var_{t}"], r1[f"es_{t}"])}
        for c in ("mean", "min_score", "relative_score"):
            fc[c] = (co[f"{c}__var"], co[f"{c}__es"])
        for k, g in r2.items():
            gi = g[g.asset == asset].set_index("date")
            fc[f"r2_{k}"] = (gi["var"], gi["es"])
        if gate is not None:
            gi = gate[gate.asset == asset].set_index("date")
            fc["gate_v2"] = (gi["var"], gi["es"])
        dates = df.index
        for v, _ in fc.values():
            dates = dates.intersection(v.dropna().index)
        y = df.loc[dates, "log_return"].to_numpy()
        per[asset] = pd.DataFrame({k: fz0_loss(y, v.loc[dates].to_numpy(), e.loc[dates].to_numpy(), a)
                                   for k, (v, e) in fc.items()}, index=dates)
    per["ALL"] = pooled_average({k: v for k, v in per.items()})
    for asset, L in per.items():
        for m in L.columns:
            row = {"alpha": a, "asset": asset, "method": m, "n": len(L), "mean_fz0": L[m].mean()}
            for ref in REFS:
                s, p = dm_test(L[m], L[ref]) if m != ref else (np.nan, np.nan)
                row[f"dm_vs_{ref}"], row[f"p_vs_{ref}"] = s, p
            rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "compare.csv", index=False)
show = table[(table.asset == "ALL") & (table.alpha == 0.025)].sort_values("mean_fz0")
print(show.drop(columns=["alpha", "asset"]).round(4).to_string(index=False))
```

- [ ] **Step 4: Makefile targets** `r2: uv run python scripts/10_r2.py`, `compare: uv run python scripts/11_dev_compare.py`; also run `make gate` for the development universe (gate_v2 only, per `configs/dev.yaml`).

- [ ] **Step 5: Run 1** `make gate r2 compare`. Record in `docs/devlog.md` under "Run 1": anchor, λ_c, pooled α = 2.5% mean FZ0 and DM vs anchor for R2-MLP and R2-linear, and the per-α summary.

- [ ] **Step 6: Revisions (budget 3).** Only if Run 1 shows a specific, explainable defect (for example corrections stuck at zero, or saturated at the bound). Each revision: one change, its reason written in the devlog before running, rerun `make r2 compare`, record results. No revision is required.

- [ ] **Step 7: Freeze development.** Devlog "Development frozen" entry: final anchor, λ_c, list of runs. README development-results paragraph (from `compare.csv`), labelled as development results.

- [ ] **Step 8: Run** `uv run pytest -q`; commit scripts, Makefile, config, devlog, README, `results/dev/tables/*.csv`.
