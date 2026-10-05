# Design: HAR-Anchored Tail-Risk Forecasting with Bounded State Corrections

Date: 2026-10-06
Status: Draft for review
Target venue: IEEE CIFEr (conference), with fallbacks described in section 7
Predecessor: https://github.com/amateurcoder015/state-dependent-gate-trained-on-FZ-loss

## 1. Background and goal

The predecessor study combined six VaR/ES models with a state-dependent neural gate trained on the FZ0 loss. On a development universe (8 NSE assets) and a pre-registered held-out universe (12 NSE stocks) the gate did not significantly beat equal weights or Taylor (2020) combinations. Two lessons carry over:

- A HAR model on range-based variance was the strongest single method on the held-out stocks and the only member of the pooled Model Confidence Set.
- Free multiplicative scaling learned from data overfit volatility regimes; shrinking it toward 1 removed the failure.

This project builds on the strongest model instead of choosing among weak ones. It asks, in steps, whether (1) a better econometric anchor and (2) a small, bounded, state-dependent correction of that anchor improve joint VaR/ES forecasts on NSE stocks never used in either study.

Success criteria:
- Every number in the paper is produced by code in this repo from frozen raw data.
- Each hypothesis in section 5 is tested once, on the fresh universe, with the decision rule fixed before the fresh data is downloaded.
- Results are reported whatever the outcome.

## 2. Method ladder

All methods forecast the pair (VaR_t, ES_t) at tail level α for day t using information up to t−1, with ES_t ≤ VaR_t < 0.

### R0: HAR baseline

Identical to the predecessor's `har` model: HAR on log Garman–Klass variance, refitted every 21 trading days on an expanding window, VaR/ES from empirical quantiles of standardized returns in the estimation window.

### R1: HAR-X joint VaR/ES regression

State vector z_t (all measured at t−1):
- For each of three range estimators — Garman–Klass, Parkinson, Rogers–Satchell — the log of the daily value and of its 5-day and 22-day means (9 terms).
- log((VIX_{t−1}/100)^2 / 252).

Model:
- VaR_t = −exp(a + b'z_t)
- ES_t = VaR_t · (1 + exp(c))

So VaR follows a log-linear HAR-X equation and ES is a constant multiple of VaR above 1 (12 parameters). The exponential links guarantee ES_t < VaR_t < 0. Parameters minimise mean FZ0 loss over the estimation window, refitted every 21 trading days on an expanding window. Starting values come from an OLS HAR-X regression of log squared returns on z_t; the fit uses several starts and keeps the lowest loss. Fit failures reuse the previous parameters and are counted.

### R2: bounded state correction of the anchor

Given an anchor pair (VaR^A_t, ES^A_t):
- VaR_t = VaR^A_t · exp(c_v,t)
- ES_t = VaR_t + (ES^A_t − VaR^A_t) · exp(c_s,t)
- c_t = 0.25 · tanh(f(x_t)), so each correction multiplies VaR or the ES spacing by a factor between exp(−0.25) ≈ 0.78 and exp(0.25) ≈ 1.28.

The training loss is mean FZ0 + λ_c · mean(c_v² + c_s²), with λ_c fixed during development and recorded in the development log. Two versions of f:
- R2-MLP: one hidden layer of 16 units, ReLU, dropout 0.1, asset embedding of size 4.
- R2-linear: f(x) = Wx + b.

Correction state x_t (all at t−1, standardized on the training window): vol-of-vol (std of 5-day Garman–Klass volatility over 30 days), India VIX level, 5-day VIX change, variance risk premium, the anchor's trailing 60-day mean FZ0 loss, lagged return sign and lagged absolute return.

Training: pooled across assets, walk-forward by year as in the predecessor (train from the first usable row, validate on the next year, test on the year after), early stopping on validation FZ0, average of 5 seeds.

Anchor choice: R1 if its pooled mean FZ0 on the development test period is lower than R0's, otherwise R0. The choice is made once, at the end of development, and recorded in the development log.

### Comparison methods

Equal-weight mean, median, Taylor (2020) minimum-score and relative-score combinations and `gate_v2` over the predecessor's six base models (EWMA, GARCH-t, EGARCH-t, GJR-t, HAR, VIX), computed by the ported predecessor code.

## 3. Data

- Daily OHLC from Yahoo Finance, 2015-01-01 to 2026-09-30, frozen with a SHA-256 manifest; India VIX as before.
- Same cleaning as the predecessor: adjusted-close log returns, no-trade rows dropped, Diwali Muhurat sessions excluded, VIX forward-filled at most one day, outliers (|r| > 10%) reviewed with a written rule.
- Development universe: the 20 assets of the predecessor (8 development + 12 held-out). Their frozen raw files are copied from the predecessor repository with their manifests.
- Fresh universe: about 25 NSE single-stock F&O constituents, never used before. Selection rule, fixed in the pre-registration before any download: traded in F&O through the whole window, not among the 20 development assets, at most 3 per sector, taken in order from the current NSE F&O list sorted by sector then by name.
- Test period for both universes: 2023-01-01 to 2026-09-30. Base forecasts and R1 start out-of-sample on 2020-01-01.

## 4. Development protocol

- All design work (R1 specification, λ_c, correction features, anchor choice) uses only the development universe.
- Design budget: at most 3 revisions after the first full development run. Each run is logged in `docs/devlog.md` with date, change, reason and pooled development results.
- After development: freeze the code at a tagged commit, write the pre-registration (assets, hypotheses, decision rules, commit hash), push it, then download the fresh universe and run once. Only crash fixes are allowed after the download, and each is listed with its reason.

## 5. Pre-registered hypotheses (fresh universe, α = 2.5%, pooled cross-asset average loss)

- H1: R1 has lower mean FZ0 than R0.
- H2: R2-MLP has lower mean FZ0 than its anchor.
- H3: R2-MLP has lower mean FZ0 than Taylor minimum score and than the equal-weight mean (both comparisons must hold).
- H4: R2-linear has lower mean FZ0 than its anchor.

Test: Diebold–Mariano with Newey–West variance and HLN correction, one-sided. Multiple testing: Holm correction across H1–H4 at family-wise level 0.05 (for H3, the larger of its two p-values is used).

Reported without decision rules: α = 1% and 5%, per-asset results, Model Confidence Set (10%), Kupiec, Christoffersen, DQ and McNeil–Frey backtests, results by year, and the development-universe results.

## 6. Repository layout

```
pyproject.toml, Makefile, configs/{dev.yaml, fresh.yaml, expiry_rules.yaml}
src/volgate/     ported from the predecessor (data, models, risk, combine, gate, evaluate)
src/hart/        new code: range estimators, HAR-X FZ regression (R1), bounded correction (R2), pipeline glue
scripts/         numbered stages
tests/           pytest, including leakage tests for every new module
data/dev/raw, data/fresh/raw   frozen raw data with manifests
docs/devlog.md, docs/preregistration/, docs/superpowers/{specs,plans}/
results/{dev,fresh}/tables, results/{dev,fresh}/figures
```

The ported package keeps its name and git history reference; README credits the predecessor repository.

## 7. Outcomes and fallbacks

- H2 or H4 supported: method paper (anchored correction).
- Only H1 supported: econometrics paper (HAR-X joint VaR/ES regression for Indian stocks).
- Nothing supported: second pre-registered null result; together with the predecessor this supports a replication paper on tail-risk combination in Indian equities.

## 8. Testing and integrity

- Unit tests for range estimators (known values), R1 loss and constraints (ES < VaR < 0, recovery of known parameters on simulated data), R2 corrections (bounds, gradients against finite differences, zero correction at initialization).
- Leakage tests for every new forecast: perturbing inputs at day t changes no output dated t or earlier.
- No hardcoded status strings in reports.
- Independent review of each plan before merge.

## 9. Plans

1. Port the predecessor pipeline and data; range estimators; R1 with tests; development run of R0 vs R1.
2. R2 (MLP and linear) with tests; development runs within the design budget; devlog; anchor choice.
3. Freeze, pre-register, select and download the fresh universe, run once, evaluate, report.
