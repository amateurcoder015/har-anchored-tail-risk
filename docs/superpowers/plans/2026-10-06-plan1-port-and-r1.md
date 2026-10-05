# Plan 1: Port Pipeline, Range Estimators and R1 (HAR-X FZ Regression) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working repo with the predecessor pipeline and the 20-asset development universe, plus R1 (HAR-X joint VaR/ES regression fitted on FZ0) and a first development comparison of R1 against R0 (HAR).

**Architecture:** Copy the predecessor's tested `volgate` package, tests and scripts unchanged. New code lives in `src/hart/`: `ranges.py` (Parkinson, Rogers–Satchell; Garman–Klass reused), `harx.py` (state vector, FZ0 objective with analytic gradient, fit, predict, walk-forward). Scripts `08_r1.py` and `09_dev_report.py` produce development forecasts and the R0-vs-R1 table.

**Tech Stack:** Python 3.12, uv, NumPy, pandas, SciPy (L-BFGS-B), the ported `volgate` package.

**Spec:** `docs/superpowers/specs/2026-10-06-har-anchored-design.md` (sections 2 R0/R1, 3, 4, 6, 8).

## Global Constraints

- Data 2015-01-01 to 2026-09-30; base and R1 forecasts out-of-sample from 2020-01-01; development test period 2023-01-01 to 2026-09-30.
- Refit every 21 trading days on an expanding window (rows strictly before the block).
- α ∈ {0.01, 0.025, 0.05}; primary 0.025; column tags `a010, a025, a050`.
- R1: VaR_t = −exp(a + b'z_t), ES_t = VaR_t·(1 + exp(c)); 12 parameters; minimise mean FZ0.
- Every forecast for day t uses only data before t. ES < VaR < 0.
- Development universe = the predecessor's 20 assets; their frozen raw files are copied, with a combined SHA-256 manifest.
- All design changes after the first full development run are logged in `docs/devlog.md` (budget: 3 revisions).

## Review Focus

1. A day with high = low or a negative Rogers–Satchell value → floored at 1e-10, logs stay finite (Task 2 test).
2. An R1 fit that fails or returns non-finite parameters → previous parameters reused and the block flagged; failed first fit raises (Task 3 test).
3. Training window with very few VaR hits (α = 1%) → fit still returns finite parameters with ES < VaR < 0 (Task 3 test).
4. VIX missing on a day → R1 forecast NaN for the next day, no crash (Task 3 test).
5. Running a script without `VOLGATE_CONFIG` set → Makefile defaults to `configs/dev.yaml`, so development and fresh outputs never mix (Task 1 check).

---

### Task 1: Port predecessor code and development data

**Files:** copy from `../state-dependent-gate-trained-on-FZ-loss`: `pyproject.toml`, `uv.lock`, `src/volgate/`, `tests/`, `scripts/0[1-7]_*.py`, `configs/expiry_rules.yaml`, `configs/default.yaml` (renamed below). Create `configs/dev.yaml`, `data/dev/raw/` (20 assets + India VIX + `MANIFEST.csv`), `data/dev/audit/corrections.csv`, `Makefile`, `NOTICE.md`.

- [ ] **Step 1: Copy code** with `git -C ../state-dependent-gate-trained-on-FZ-loss archive HEAD src tests scripts configs/expiry_rules.yaml pyproject.toml uv.lock | tar -x`. Rename the package metadata name in `pyproject.toml` to `har-anchored-tail-risk` and add `"src/hart"` to `[tool.hatch.build.targets.wheel] packages`.
- [ ] **Step 2: Copy data.** `data/raw/*.csv` (8 assets + india_vix) and `data/holdout/raw/*.csv` (12 assets) from the predecessor into `data/dev/raw/`. The two India VIX files are byte-identical (checked with `cmp`). Combine both `MANIFEST.csv` files, keeping one `india_vix` row. Combine `data/audit/corrections.csv` and `data/holdout/audit/corrections.csv` into `data/dev/audit/corrections.csv`.
- [ ] **Step 3: Write `configs/dev.yaml`**: keys as the predecessor's `default.yaml`; `assets` = all 20 (nifty50 `^NSEI`, banknifty `^NSEBANK`, adanient, tatasteel, dlf, hindunilvr, nestleind, sunpharma, reliance, infy, icicibank, itc, lt, maruti, bhartiartl, hindalco, drreddy, ongc, titan, bajfinance with `.NS` tickers); `contracts` nifty50 NIFTY, banknifty BANKNIFTY, others STOCK; same `special_sessions`; `main_variant: gate_v2`; `gate_variants: [gate_v2]`; paths `data/dev/raw`, `data/dev/audit`, `data/dev/processed`, `results/dev/tables`, `results/dev/figures`, `configs/expiry_rules.yaml`.
- [ ] **Step 4: Point the default config at dev.** In `src/volgate/config.py` change the fallback path from `configs/default.yaml` to `configs/dev.yaml`; in `tests/test_config.py` replace `default.yaml` with `dev.yaml` in the env-var test. Delete `configs/default.yaml` if copied.
- [ ] **Step 5: Makefile** with `export VOLGATE_CONFIG ?= configs/dev.yaml` and targets `prepare base combos gate table evaluate r1 devreport test` calling `scripts/02..09` (`r1` → `08_r1.py`, `devreport` → `09_dev_report.py`).
- [ ] **Step 6: `NOTICE.md`** credits the predecessor repository for `src/volgate`, its tests and scripts 01–07.
- [ ] **Step 7: Verify.** `uv sync && uv run pytest -q` — Expected: 105 passed (predecessor suite). `uv run python -c "from volgate.data.download import verify_manifest; print(verify_manifest('data/dev/raw'))"` — Expected: `[]`.
- [ ] **Step 8: Run development pipeline.** `make prepare base combos` — Expected: 20-asset data summary; base and combination forecasts written under `data/dev/processed/`.
- [ ] **Step 9: Commit** code, configs, raw data, audit, Makefile, NOTICE, `results/dev/tables/*.csv`.

---

### Task 2: Range estimators

**Files:** Create `src/hart/__init__.py` (empty), `src/hart/ranges.py`; test `tests/test_ranges.py`.

**Interfaces:** `parkinson(df) -> pd.Series`, `rogers_satchell(df) -> pd.Series` (both floored at 1e-10, inputs `open, high, low, close`); `RANGE_FLOOR = 1e-10`.

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ranges.py -q` — Expected: `No module named 'hart'`.
- [ ] **Step 3: Implement**

```python
import numpy as np
import pandas as pd

RANGE_FLOOR = 1e-10


def parkinson(df: pd.DataFrame) -> pd.Series:
    return (np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))).clip(lower=RANGE_FLOOR)


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = (np.log(df[k]) for k in ("high", "low", "open", "close"))
    return ((h - c) * (h - o) + (l - c) * (l - o)).clip(lower=RANGE_FLOOR)
```

- [ ] **Step 4: Run** — Expected: 2 passed; full suite passes.
- [ ] **Step 5: Commit** `feat: add Parkinson and Rogers-Satchell range estimators`.

---

### Task 3: R1 — HAR-X joint VaR/ES regression

**Files:** Create `src/hart/harx.py`; test `tests/test_harx.py`.

**Interfaces:**
- Consumes: `gk_variance` (volgate.data.panel), `parkinson`, `rogers_satchell`, `fz0_loss`, `alpha_tag`.
- Produces:
  - `harx_state(df) -> pd.DataFrame` — 10 columns `{gk,pk,rs}_{d,w,m}` and `vix2`, each the log of a quantity measured through t−1
  - `harx_loss_grad(theta, Z, y, alpha) -> (float, np.ndarray)` — mean FZ0 and its gradient; `theta = [a, b(10), c]`, Z already standardized
  - `harx_predict(theta, Z) -> (var, es)`
  - `fit_harx(Z, y, alpha, init=None, n_starts=3, seed=0) -> (theta, success: bool)`
  - `walk_forward_harx(df, oos_start, refit_every, alphas) -> pd.DataFrame` with columns `var_{tag}, es_{tag}, fallback_{tag}, refit_date`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hart.harx import fit_harx, harx_loss_grad, harx_predict, harx_state, walk_forward_harx

A = 0.025


def _sim(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 10))
    b0 = np.zeros(10)
    b0[0], b0[3], b0[9] = 0.6, 0.3, 0.4
    log_s2 = -8.0 + Z @ b0
    y = np.exp(0.5 * log_s2) * rng.standard_normal(n)
    q = stats.norm.ppf(A)
    true = np.concatenate([[np.log(-q) - 4.0], 0.5 * b0, [np.log(stats.norm.pdf(q) / A / -q - 1)]])
    return Z, y, true


def test_gradient_matches_finite_differences():
    Z, y, true = _sim(2000)
    theta = true + np.random.default_rng(1).normal(0, 0.05, true.size)
    _, g = harx_loss_grad(theta, Z, y, A)
    h = 1e-6
    for i in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += h
        tm[i] -= h
        num = (harx_loss_grad(tp, Z, y, A)[0] - harx_loss_grad(tm, Z, y, A)[0]) / (2 * h)
        assert g[i] == pytest.approx(num, rel=1e-4, abs=1e-6)


def test_fit_recovers_parameters_and_orders_pair():
    Z, y, true = _sim()
    theta, ok = fit_harx(Z, y, A)
    assert ok
    np.testing.assert_allclose(theta[1:11], true[1:11], atol=0.06)
    v, e = harx_predict(theta, Z)
    assert np.all(e < v) and np.all(v < 0)


def test_fit_works_with_few_hits():
    Z, y, _ = _sim(1200, seed=3)
    theta, ok = fit_harx(Z, y, 0.01)
    v, e = harx_predict(theta, Z)
    assert np.all(np.isfinite(theta)) and np.all(e < v) and np.all(v < 0)


def test_state_columns_and_lookahead(panel):
    s = harx_state(panel)
    assert list(s.columns) == [f"{e}_{h}" for e in ("gk", "pk", "rs") for h in ("d", "w", "m")] + ["vix2"]
    j = 300
    pert = panel.copy()
    pert.iloc[j, pert.columns.get_indexer(["open", "high", "low", "close", "vix"])] *= 1.5
    out = harx_state(pert)
    pd.testing.assert_frame_equal(s.iloc[: j + 1], out.iloc[: j + 1])
    assert not s.iloc[j + 1].equals(out.iloc[j + 1])


def test_walk_forward_no_lookahead_and_nan_vix(panel):
    oos = panel.index[400].strftime("%Y-%m-%d")
    base = walk_forward_harx(panel, oos, 50, [A])
    assert (base["es_a025"] < base["var_a025"]).all()
    j = 470
    pert = panel.copy()
    pert.iloc[j, pert.columns.get_indexer(["log_return", "open", "high", "low", "close", "vix"])] *= 1.5
    out = walk_forward_harx(pert, oos, 50, [A])
    pd.testing.assert_frame_equal(base.loc[: panel.index[j]], out.loc[: panel.index[j]])
    p2 = panel.copy()
    p2.iloc[500, p2.columns.get_loc("vix")] = np.nan
    r = walk_forward_harx(p2, oos, 50, [A])
    assert np.isnan(r.loc[panel.index[501], "var_a025"])


def test_walk_forward_fallback_and_first_failure(panel, monkeypatch):
    import hart.harx as hx
    calls = {"n": 0}
    real = hx.fit_harx

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return np.full(12, np.nan), False
        return real(*args, **kwargs)

    monkeypatch.setattr(hx, "fit_harx", flaky)
    out = walk_forward_harx(panel, panel.index[400].strftime("%Y-%m-%d"), 50, [A])
    flags = out.groupby("refit_date")["fallback_a025"].first().tolist()
    assert flags[:3] == [False, True, False]
    calls["n"] = 1  # next call is the 2nd -> fails on the first block
    with pytest.raises(RuntimeError, match="first fit"):
        walk_forward_harx(panel, panel.index[400].strftime("%Y-%m-%d"), 50, [A])
```

- [ ] **Step 2: Run** — Expected: module missing.
- [ ] **Step 3: Implement `src/hart/harx.py`**

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from hart.ranges import RANGE_FLOOR, parkinson, rogers_satchell
from volgate.data.panel import gk_variance
from volgate.risk.dist import alpha_tag

ESTIMATORS = {"gk": gk_variance, "pk": parkinson, "rs": rogers_satchell}
STATE_COLUMNS = [f"{e}_{h}" for e in ESTIMATORS for h in ("d", "w", "m")] + ["vix2"]


def harx_state(df: pd.DataFrame) -> pd.DataFrame:
    """Log range variances (daily, 5-day, 22-day means) and log VIX variance, through t-1."""
    s = pd.DataFrame(index=df.index)
    for name, fn in ESTIMATORS.items():
        v = fn(df).clip(lower=RANGE_FLOOR).shift(1)
        s[f"{name}_d"] = np.log(v)
        s[f"{name}_w"] = np.log(v.rolling(5).mean())
        s[f"{name}_m"] = np.log(v.rolling(22).mean())
    s["vix2"] = np.log((df["vix"].shift(1) / 100.0) ** 2 / 252.0)
    return s


def harx_predict(theta, Z):
    u = theta[0] + Z @ theta[1:-1]
    var = -np.exp(u)
    return var, var * (1.0 + np.exp(theta[-1]))


def harx_loss_grad(theta, Z, y, alpha):
    """Mean FZ0 with VaR = -exp(u), ES = k*VaR, k = 1 + exp(c); analytic gradient."""
    u = theta[0] + Z @ theta[1:-1]
    v = -np.exp(u)
    k = 1.0 + np.exp(theta[-1])
    hit = (y <= v).astype(float)
    r = y / v
    L = -hit * (1.0 - r) / (alpha * k) + 1.0 / k + u + np.log(k) - 1.0
    dL_du = 1.0 - hit * r / (alpha * k)
    dL_dk = hit * (1.0 - r) / (alpha * k**2) - 1.0 / k**2 + 1.0 / k
    n = len(y)
    g = np.empty_like(theta)
    g[0] = dL_du.mean()
    g[1:-1] = Z.T @ dL_du / n
    g[-1] = dL_dk.mean() * (k - 1.0)
    return float(L.mean()), g


def _init(Z, y, alpha):
    X = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(X, np.log(y**2 + 1e-12), rcond=None)
    sd = np.exp(0.5 * (X @ beta))
    zq = y / sd
    q = np.quantile(zq, alpha)
    es = zq[zq <= q].mean()
    ratio = max(es / q - 1.0, 0.05)
    return np.concatenate([[np.log(-q) + 0.5 * beta[0]], 0.5 * beta[1:], [np.log(ratio)]])


def fit_harx(Z, y, alpha, init=None, n_starts=3, seed=0):
    rng = np.random.default_rng(seed)
    base = _init(Z, y, alpha) if init is None else np.asarray(init, dtype=float)
    starts = [base] + [base + rng.normal(0, 0.1, base.size) for _ in range(n_starts - 1)]
    best = None
    for s in starts:
        res = minimize(harx_loss_grad, s, args=(Z, y, alpha), jac=True, method="L-BFGS-B")
        if np.all(np.isfinite(res.x)) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return np.full(base.size, np.nan), False
    return best.x, True


def walk_forward_harx(df, oos_start, refit_every, alphas):
    """Expanding-window refit every `refit_every` rows; Z standardized on the training rows."""
    df = df.sort_index()
    S = harx_state(df).to_numpy(dtype=float)
    y = df["log_return"].to_numpy(dtype=float)
    ok = np.isfinite(S).all(axis=1) & np.isfinite(y)
    first = int(np.flatnonzero(df.index >= pd.Timestamp(oos_start))[0])
    out = pd.DataFrame(index=df.index[first:])
    for a in alphas:
        t = alpha_tag(a)
        var = np.full(len(df), np.nan)
        es = np.full(len(df), np.nan)
        fb = np.zeros(len(df), dtype=bool)
        prev = None
        for k in range(first, len(df), refit_every):
            end = min(k + refit_every, len(df))
            tr = ok[:k]
            mu, sd = S[:k][tr].mean(axis=0), S[:k][tr].std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            theta, success = fit_harx((S[:k][tr] - mu) / sd, y[:k][tr], a,
                                      init=None if prev is None else prev[0])
            if not success:
                if prev is None:
                    raise RuntimeError("R1: first fit failed")
                theta, mu, sd = prev
                fb[k:end] = True
            prev = (theta, mu, sd)
            var[k:end], es[k:end] = harx_predict(theta, (S[k:end] - mu) / sd)
        out[f"var_{t}"], out[f"es_{t}"] = var[first:], es[first:]
        out[f"fallback_{t}"] = fb[first:]
    out["refit_date"] = [df.index[first + ((i // refit_every) * refit_every)] for i in range(len(out))]
    return out
```

Note on `init` for later blocks: the previous block's θ is in the previous block's standardized units; it is only a starting value and the fit is re-run, so a small mismatch affects speed, not the result.

- [ ] **Step 4: Run** `uv run pytest tests/test_harx.py -q` — Expected: 6 passed; full suite passes.
- [ ] **Step 5: Commit** `feat: add R1 HAR-X joint VaR/ES regression fitted on FZ0`.

---

### Task 4: Development run of R1 and R0-vs-R1 report

**Files:** Create `scripts/08_r1.py`, `scripts/09_dev_report.py`, `docs/devlog.md`; modify `README.md`.

**Interfaces:** Produces `data/dev/processed/r1/{asset}.csv` (index date; `var_{tag}, es_{tag}, fallback_{tag}, refit_date`) and `results/dev/tables/r0_vs_r1.csv` (`alpha, asset, n, fz0_r0, fz0_r1, dm_stat, dm_p, hit_r0, hit_r1, fallback_blocks_r1`; `asset = "ALL"` rows use the cross-asset average loss per date).

- [ ] **Step 1: `scripts/08_r1.py`**

```python
from volgate.config import REPO_ROOT, load_config
from hart.harx import walk_forward_harx
import pandas as pd

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
out_dir = P["processed"] / "r1"
out_dir.mkdir(parents=True, exist_ok=True)
for asset in cfg["assets"]:
    panel = pd.read_csv(P["processed"] / "panel" / f"{asset}.csv", index_col="date", parse_dates=True)
    wf = walk_forward_harx(panel, cfg["oos_start"], cfg["refit_every"], cfg["alphas"])
    wf.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")
    print(f"{asset} done", flush=True)
```

- [ ] **Step 2: `scripts/09_dev_report.py`**

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
cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
rows = []
for a in cfg["alphas"]:
    t = alpha_tag(a)
    per = {}
    for asset in cfg["assets"]:
        df = load_asset(P["processed"], asset).loc[START:END]
        r1 = pd.read_csv(P["processed"] / "r1" / f"{asset}.csv", index_col="date",
                         parse_dates=True).loc[START:END]
        d = df.index.intersection(r1.dropna(subset=[f"var_{t}"]).index)
        y = df.loc[d, "log_return"].to_numpy()
        v0, e0 = df.loc[d, f"har__var_{t}"].to_numpy(), df.loc[d, f"har__es_{t}"].to_numpy()
        v1, e1 = r1.loc[d, f"var_{t}"].to_numpy(), r1.loc[d, f"es_{t}"].to_numpy()
        L = pd.DataFrame({"r0": fz0_loss(y, v0, e0, a), "r1": fz0_loss(y, v1, e1, a)}, index=d)
        per[asset] = L
        s, p = dm_test(L["r1"], L["r0"])
        rows.append({"alpha": a, "asset": asset, "n": len(d), "fz0_r0": L.r0.mean(),
                     "fz0_r1": L.r1.mean(), "dm_stat": s, "dm_p": p,
                     "hit_r0": np.mean(y <= v0), "hit_r1": np.mean(y <= v1),
                     "fallback_blocks_r1": int(r1.groupby("refit_date")[f"fallback_{t}"].first().sum())})
    L = pooled_average(per)
    s, p = dm_test(L["r1"], L["r0"])
    rows.append({"alpha": a, "asset": "ALL", "n": len(L), "fz0_r0": L.r0.mean(), "fz0_r1": L.r1.mean(),
                 "dm_stat": s, "dm_p": p, "hit_r0": np.nan, "hit_r1": np.nan, "fallback_blocks_r1": np.nan})
table = pd.DataFrame(rows)
table.to_csv(P["tables"] / "r0_vs_r1.csv", index=False)
print(table[table.asset == "ALL"].round(4).to_string(index=False))
```

- [ ] **Step 3: Run** `make r1 devreport`. Expected: 20 `done` lines; a pooled table with one row per α. Sanity: R1 hit rates between 0.5α and 2α for most assets; fallback blocks ≤ 5 per asset.
- [ ] **Step 4: `docs/devlog.md`** — create with a header explaining the budget (3 revisions after the first full run) and entry "Run 0 (baseline design)" listing the R1 specification and the pooled R0/R1 FZ0 and DM at each α copied from `r0_vs_r1.csv`.
- [ ] **Step 5: README** status → "Plan 1 complete: development pipeline and R1"; add a short development-results paragraph citing `results/dev/tables/r0_vs_r1.csv`, stating these are development (not held-out) numbers.
- [ ] **Step 6: Run** `uv run pytest -q`; **commit** scripts, devlog, README, `results/dev/tables/*.csv`.
