# Development log

All runs use only the development universe (the 20 assets of the predecessor study), test period 2023-01-01 to 2026-09-30. Budget: at most 3 design revisions after Run 1 (spec section 4). Bug fixes are listed separately and do not count as revisions.

## Run 0 — R1 against R0 (2026-10-06)

R1 as specified: VaR = -exp(a + b'z), ES = VaR·(1 + exp(c)); z = log daily, 5-day and 22-day Garman–Klass, Parkinson and Rogers–Satchell variances plus log VIX variance; fitted on FZ0 every 21 days on an expanding window.

**Bug fix 1 (before the numbers below).** The first run gave pooled mean FZ0 of -1.84 for R1 at α = 1% because Rogers–Satchell is exactly zero on trend days (open = low, close = high). The 1e-10 floor became log ≈ -23 and the next day's VaR collapsed to about zero (16 such days across 20 assets; DLF 2026-08-25 loss 18,858). Fix: each daily term is bounded below at one tenth of its own 22-day mean (`hart.harx.harx_state`, test `test_zero_range_day_does_not_explode_state`).

Pooled results after the fix (`results/dev/tables/r0_vs_r1.csv`; DM is R1 minus R0, positive favours R0):

| α | FZ0 R0 | FZ0 R1 | DM | p |
|---|---|---|---|---|
| 0.01 | -2.9765 | -2.9137 | 1.75 | 0.080 |
| 0.025 | -3.2589 | -3.2369 | 1.37 | 0.172 |
| 0.05 | -3.4787 | -3.4680 | 1.04 | 0.297 |

R1 has lower loss than R0 for 4 of 20 assets at α = 2.5%.

**Anchor choice (spec section 2):** R1's pooled development loss is not lower than R0's, so R2 is anchored on **R0**. R1 stays in the study unchanged and is tested as H1.

## Run 1 — R2 anchored on R0, λ_c = 1 (2026-10-06)

Pooled mean FZ0 (`results/dev/tables/compare.csv`; DM vs R0, positive favours R0):

| Method | α = 1% | α = 2.5% | α = 5% | DM vs R0 (2.5%) | p |
|---|---|---|---|---|---|
| R0 (HAR) | -2.9765 | -3.2589 | -3.4787 | | |
| R2-linear | -2.9634 | -3.2538 | -3.4779 | 0.99 | 0.323 |
| R2-MLP | -2.9638 | -3.2506 | -3.4762 | 0.76 | 0.450 |
| gate_v2 | -2.9629 | -3.2543 | -3.4763 | 0.87 | 0.386 |
| Taylor min score | -2.9631 | -3.2495 | -3.4796 | 1.04 | 0.300 |
| Equal-weight mean | -2.9568 | -3.2404 | -3.4675 | 2.20 | 0.028 |

R2-MLP beats R0 on 6 of 20 assets, R2-linear on 5. Mean correction on test rows: R2-MLP c_v = -0.059 (std 0.087), c_s = -0.016; R2-linear c_v = -0.012.

**Diagnosis.** R2-MLP mostly learns a level shift (VaR about 6% smaller on average), the same failure as the predecessor's scale head. HAR is refitted every 21 days, so its level is already estimated; the correction should only respond to state.

## Revision 1 (planned before running)

Change: add a penalty λ_m·(mean(c_v)² + mean(c_s)²) over each batch, λ_m = 100, so the average correction is pushed to zero and only state-dependent deviations remain. Everything else unchanged (anchor R0, λ_c = 1).
