# HAR-Anchored Tail-Risk Forecasting

Research code for forecasting daily Value-at-Risk (VaR) and Expected Shortfall (ES) for Indian equities. The method starts from a HAR model on range-based volatility and adds (1) a better econometric anchor fitted directly to a joint VaR/ES loss and (2) a small, bounded, state-dependent correction of that anchor.

> **Status:** design stage. No code or results yet.

This project follows [state-dependent-gate-trained-on-FZ-loss](https://github.com/amateurcoder015/state-dependent-gate-trained-on-FZ-loss), where a neural combination gate did not beat simple combinations in a pre-registered test, and HAR on range-based variance was the strongest single model.

## Plan

- **R0:** HAR on Garman–Klass variance (baseline).
- **R1:** HAR-X joint VaR/ES regression on three range estimators and India VIX, fitted by minimising the FZ0 loss.
- **R2:** bounded state correction of the better anchor (MLP and linear versions), trained on FZ0.

The methods are developed on 20 NSE assets used before, then tested once on about 25 NSE stocks never used in either project, with hypotheses and decision rules published on GitHub before the fresh data is downloaded.

Design: [docs/superpowers/specs/2026-10-06-har-anchored-design.md](docs/superpowers/specs/2026-10-06-har-anchored-design.md)
