# HAR-Anchored Tail-Risk Forecasting

Research code for forecasting daily Value-at-Risk (VaR) and Expected Shortfall (ES) for Indian equities, and for testing whether adaptive combinations of risk models beat a simple range-based HAR model.

> **Status:** pre-registered test on 27 never-used NSE stocks complete. Two of four pre-registered hypotheses supported (H3, H4: level drift); H1 and H2 (HAR beats combinations) not supported.

## Question

Can forecast combinations or learned, state-dependent corrections beat a HAR model on range-based variance for joint VaR/ES forecasting? If adaptive methods fail, is the cause a learned level shift ("level drift"), as forecast-combination and location-shift theory would predict?

## What has been found so far

**Predecessor project** ([state-dependent-gate-trained-on-FZ-loss](https://github.com/amateurcoder015/state-dependent-gate-trained-on-FZ-loss)): a neural gate over six risk models, trained on the FZ0 joint VaR/ES loss, did not beat equal weights or Taylor (2020) combinations in a pre-registered held-out test on 12 stocks. HAR on range-based variance was the only member of the 10% Model Confidence Set.

**This project, development universe** (the predecessor's 20 assets, test period 2023-01 to 2026-09, full log in [docs/devlog.md](docs/devlog.md)). Pooled mean FZ0 loss, lower is better:

| Method | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| R0: HAR on Garman–Klass variance | -2.9765 | **-3.2589** | -3.4787 |
| R2-MLP: bounded correction of HAR, with level control | -2.9716 | -3.2572 | **-3.4799** |
| R2-linear | -2.9662 | -3.2542 | -3.4795 |
| gate_v2 (predecessor gate with scale shrinkage) | -2.9629 | -3.2543 | -3.4763 |
| Taylor minimum score | -2.9631 | -3.2495 | -3.4796 |
| Equal-weight mean | -2.9568 | -3.2404 | -3.4675 |
| R1: HAR-X fitted directly on FZ0 | -2.9137 | -3.2369 | -3.4680 |

- No method beats HAR significantly. HAR beats the equal-weight mean significantly at α = 2.5% (DM 2.20, p = 0.028).
- R2's first version learned a level shift (VaR about 6% smaller on average) and lost to HAR. Penalising the average correction (revision 1) brought it level with HAR.
- R1 is worse than HAR at every level, in line with the known lower accuracy of FZ-loss estimation compared with maximum likelihood.

These are development results; the methods were tuned on this data.

## Pre-registered test: results

Hypotheses, asset list and code were fixed in [docs/preregistration/2026-10-06-fresh.md](docs/preregistration/2026-10-06-fresh.md) and tagged `prereg-fresh` before any fresh data was downloaded. The study was run once. The only post-download decision was the pre-registered outlier rule: one row dropped (VEDL 2026-04-30, the unadjusted Vedanta demerger ex-date); all 27 stocks met the coverage rule.

Pooled over 27 stocks, α = 2.5%, test period 2023-01 to 2026-09 (`results/fresh/tables/hypotheses.csv`):

| Hypothesis | Mean FZ0 (lower vs higher) | DM | One-sided p | Holm p | Result |
|---|---|---|---|---|---|
| H1: HAR < equal-weight mean | -3.1808 vs -3.1752 | -0.53 | 0.300 | 0.600 | not supported |
| H2: HAR < Taylor minimum score | -3.1808 vs -3.1799 | -0.09 | 0.463 | 0.600 | not supported |
| H3: HAR < original gate (no level control) | -3.1808 vs -3.1460 | -3.47 | 0.0003 | 0.001 | **supported** |
| H4: gate with level control < original gate | -3.1743 vs -3.1460 | -3.19 | 0.0007 | 0.002 | **supported** |

What this shows:

- **Learned gates without level control fail, and level control fixes most of it.** The original gate is significantly worse than HAR and than the same gate with scale shrinkage. It also has the worst exception clustering: the DQ test rejects for 11 of 27 stocks, against 2 for the gate with level control (`backtest_rejections.csv`). This replicates the predecessor's development and held-out results and this project's development results, now on stocks chosen and registered in advance.
- **HAR does not beat simple combinations.** HAR, equal weights, Taylor's combinations, previous-best selection and the level-controlled gate are statistically indistinguishable and all sit in the 10% Model Confidence Set (`mcs.csv`). The single GARCH-family models, EWMA, the VIX model, the median combination and the original gate are excluded.
- **Secondary results:**
  - R2-MLP, a bounded and level-controlled correction of HAR, has the lowest pooled loss at α = 2.5% and 5%, and beats HAR on 17 of 27 stocks. Its pooled advantage is not significant (one-sided p = 0.21).
  - R1 is significantly worse than HAR (DM 3.57, two-sided p ≈ 0.0004; `secondary.csv`).

Pooled mean FZ0 by tail level (`compare.csv`):

| Method | α = 1% | α = 2.5% | α = 5% |
|---|---|---|---|
| R2-MLP | -2.9107 | **-3.1844** | **-3.3970** |
| R2-linear | -2.9090 | -3.1824 | -3.3967 |
| R0: HAR | **-2.9166** | -3.1808 | -3.3948 |
| Taylor minimum score | -2.9065 | -3.1799 | -3.3954 |
| Equal-weight mean | -2.9106 | -3.1752 | -3.3881 |
| gate_v2 (level control) | -2.9037 | -3.1743 | -3.3920 |
| R1 | -2.8380 | -3.1479 | -3.3772 |
| Original gate | | -3.1460 | |

### Disclosures

- The confirmatory family above replaced an earlier one (R1 and R2 as the main contributions) after development and before the fresh download. The earlier family was evaluated on the fresh data for transparency; none of it is supported (`results/fresh/tables/robustness/original_family.csv`).
- The fresh stocks were chosen by sector and size, not by the mechanical rule in the design spec.
- Pooled tests use the 919 dates common to all 27 stocks.

Details: post-registration notes in [the pre-registration](docs/preregistration/2026-10-06-fresh.md).

## Robustness checks (exploratory, not pre-registered)

Files in `results/fresh/tables/robustness/`. None of these tests is corrected for multiple comparisons.

- **Tail levels** (`by_alpha.csv`). H3 holds at α = 1%, 2.5% and 5% (one-sided p 0.010, 0.0003, 0.005). H4 holds at 2.5% and 5% (0.0007, 0.004) and is borderline at 1% (0.057). The original gate's 1% and 5% runs were made after the results commit.
- **Years** (`by_year.csv`). H3 is significant in 2023, 2024 and 2026 and has the same sign in 2025. H4 is significant in 2024 and 2026, has the same sign in 2025 and the opposite sign (not significant) in 2023. H1 looks significant in 2023 and 2025 but reverses in 2026; this is not evidence for H1.
- **Stocks** (`asset_wins.csv`). The level-controlled gate beats the original gate on 25 of 27 stocks; HAR beats the original gate on 22 of 27.
- **Mechanism** (`mechanism.csv`). Across 108 stock-years, the original gate's mean log scale factor correlates −0.45 with its loss gap to HAR (more shrinkage, bigger loss); average shrinkage is about 6%. This is descriptive: cells are not independent and the scale enters the forecast directly.
- **ES backtest** (`as_z2_summary.csv`). Acerbi–Székely Z2 with 5% critical values simulated for each stock's sample length (about −0.37 for 919 days). Rejections out of 27: original gate 9, R1 10, HAR 5, R2-linear 5, R2-MLP 4, Taylor minimum score 4, gate with level control 3, equal weights 0.
- **Power** (`power.csv`). With the observed effect sizes and cross-stock correlation, no number of additional stocks would make H1, H2 or R2-MLP vs HAR significant (the achievable DM statistic is capped below 1.1). A further "more stocks" test is therefore not worth running.

## Pre-registered test (design)

Hypotheses, the asset list and the code commit are fixed in [docs/preregistration/2026-10-06-fresh.md](docs/preregistration/2026-10-06-fresh.md), pushed before any fresh data was downloaded. At α = 2.5%, pooled, one-sided DM tests with Holm correction:

- H1: HAR < equal-weight mean
- H2: HAR < Taylor minimum-score combination
- H3: HAR < original gate (no level control)
- H4: gate with level control < gate without it

## Positioning

- To our knowledge this is the first pre-registered evaluation of VaR/ES forecast combination (no prior example found in OpenAlex or web searches).
- The level-drift explanation applies existing theory (Claeskens et al. 2016; Elliott and Liao 2026; Clements and Hendry on location shifts) to learned VaR/ES corrections; it is not presented as new theory.
- Range-based tail-risk models (Taylor 2020) and HAR are established; the contribution is pre-registered evidence for Indian equities, a tested explanation of why adaptive combinations fail there, and an open, leakage-tested benchmark.

## Methods

- **R0:** HAR on log Garman–Klass variance, VaR/ES from empirical standardized residuals, refitted every 21 trading days.
- **R1:** VaR = −exp(a + b'z), ES = VaR·(1 + exp(c)), z = log daily, weekly and monthly Garman–Klass, Parkinson and Rogers–Satchell variances plus log VIX variance; fitted by minimising FZ0.
- **R2:** VaR = VaR_HAR·exp(c_v), ES = VaR + (ES_HAR − VaR_HAR)·exp(c_s), c = 0.25·tanh(f(x)); f is an MLP or linear in vol-of-vol, India VIX, VIX change, variance risk premium, HAR's trailing loss and lagged returns; trained on FZ0 with penalties on the size and on the average of the corrections.
- **Comparisons:** six base models (EWMA, GARCH-t, EGARCH-t, GJR-t, HAR, India VIX), equal-weight mean, median, Taylor (2020) minimum and relative score, and the predecessor's gates.

## Reproducing

```bash
uv sync
uv run pytest -q                      # unit, gradient and look-ahead tests
make prepare base combos gate table evaluate r1 r2 compare hypotheses   # development universe
VOLGATE_CONFIG=configs/fresh.yaml make prepare base combos gate table evaluate r1 r2 compare hypotheses robustness
# raw data in data/*/raw is frozen; `make download` refuses to overwrite it
```

## Layout

```
configs/                dev.yaml, fresh.yaml, expiry_rules.yaml
src/volgate/            pipeline ported from the predecessor (see NOTICE.md)
src/hart/               range estimators, R1, R2, comparison and hypothesis tests
scripts/                numbered pipeline stages
data/dev, data/fresh    frozen raw data and outlier decisions
docs/                   design spec, plans, development log, pre-registration
results/dev, results/fresh   generated tables
```

## Credit

`src/volgate/` and scripts 01–07 come from the predecessor project; see [NOTICE.md](NOTICE.md).
