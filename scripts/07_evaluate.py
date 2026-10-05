import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from volgate.combine.data import MODELS, load_asset  # noqa: E402
from volgate.config import REPO_ROOT, load_config  # noqa: E402
from volgate.evaluate.analysis import expiry_did, state_regression  # noqa: E402
from volgate.evaluate.latex import to_latex  # noqa: E402
from volgate.evaluate.losses import aligned, collect_forecasts, loss_frame, pooled_average  # noqa: E402
from volgate.evaluate.stats import (christoffersen, dm_test, dq_test, kupiec,  # noqa: E402
                                    mcneil_frey, mcs_table)
from volgate.gate.features import gate_features  # noqa: E402
from volgate.risk.dist import alpha_tag  # noqa: E402

START, END, MAIN = "2023-01-01", "2026-09-30", 0.025
cfg = load_config()
GATE = cfg.get("main_variant", "gate")
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
TEX, FIG = P["tables"] / "tex", P.get("figures", REPO_ROOT / "results" / "figures")
TEX.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

dm_rows, mcs_rows, bt_rows, mean_rows = [], [], [], []
losses_main = {}
for a in cfg["alphas"]:
    per_asset = {}
    for asset in cfg["assets"]:
        fc, df = collect_forecasts(P["processed"], asset, a, START, END)
        dates, y, fca = aligned(fc, df["log_return"])
        per_asset[asset] = loss_frame(fca, y, a, dates)
        for name, (v, e) in fca.items():
            hits = y <= v
            cc = christoffersen(hits, a)
            mf = mcneil_frey(y, v, e)
            bt_rows.append({"alpha": a, "asset": asset, "method": name, "n": len(y),
                            "hit_rate": hits.mean(), "kupiec_p": kupiec(hits, a)[1],
                            "cc_p": cc["p_cc"], "dq_p": dq_test(hits, v, a)[1],
                            "mf_mean": mf[0], "mf_p": mf[1], "mf_n": mf[2]})
    per_asset["ALL"] = pooled_average(dict(per_asset))
    if a == MAIN:
        losses_main = per_asset
    for asset, L in per_asset.items():
        for other in L.columns.drop(GATE):
            s, p = dm_test(L[GATE], L[other])
            dm_rows.append({"alpha": a, "asset": asset, "other": other, "dm_stat": s, "p": p})
        t = mcs_table(L)
        for name, r in t.iterrows():
            mcs_rows.append({"alpha": a, "asset": asset, "method": name,
                             "pvalue": r["pvalue"], "in_mcs": bool(r["in_mcs"])})
        for name in L.columns:
            mean_rows.append({"alpha": a, "asset": asset, "method": name, "mean_fz0": L[name].mean()})
    print(f"alpha {a} done", flush=True)

dm, mcs, bt, means = (pd.DataFrame(r) for r in (dm_rows, mcs_rows, bt_rows, mean_rows))
for name, t in [("dm_gate", dm), ("mcs", mcs), ("backtests", bt)]:
    t.to_csv(P["tables"] / f"{name}.csv", index=False)

# Expiry and state analyses at the main alpha (gate minus equal-weight mean).
did_rows, panels, weights_rows = [], [], []
gate = pd.read_csv(P["processed"] / "gate" / alpha_tag(MAIN) / f"{GATE}.csv", parse_dates=["date"])
for asset in cfg["assets"]:
    L = losses_main[asset]
    full = load_asset(P["processed"], asset)
    df = full.reindex(L.index)
    st = gate_features(full, MAIN).reindex(L.index)
    panels.append(pd.DataFrame({"d": L[GATE] - L["mean"], "d_gate": L["gate"] - L["mean"], "expiry": df["own_is_expiry"],
                                "asset": asset, "vov30": st["vov30"], "vix": st["vix"],
                                "vrp": st["vrp"],
                                "loss_dispersion": st[[f"fz_{m}" for m in MODELS]].std(axis=1)}))
    g = gate[gate.asset == asset].set_index("date").reindex(L.index)
    g["expiry"] = df["own_is_expiry"].to_numpy()
    g["post"] = g.index >= pd.Timestamp("2025-09-01")
    w = g.groupby(["post", "expiry"])[[f"wq_{m}" for m in MODELS] + ["g"]].mean()
    weights_rows.append(w.assign(asset=asset).reset_index())
panel = pd.concat(panels)
nifty = panel[panel.asset == "nifty50"]
stocks = panel[~panel.asset.isin(["nifty50", "banknifty"])].groupby(level=0).mean(numeric_only=True)
for name, p in [("nifty50", nifty), ("stocks_avg", stocks)]:
    if p.empty:
        continue
    did_rows.append(expiry_did(p["d"], p["expiry"]).assign(sample=name).reset_index(names="term"))
pd.concat(did_rows).to_csv(P["tables"] / "expiry_did.csv", index=False)
avg = panel.groupby(level=0).mean(numeric_only=True)
STATES = ["vov30", "vix", "vrp", "loss_dispersion"]
pd.concat([state_regression(avg["d"], avg[STATES]).reset_index(names="term").assign(method=GATE),
           state_regression(avg["d_gate"], avg[STATES]).reset_index(names="term").assign(method="gate")]
          ).drop_duplicates(subset=["term", "method"]).to_csv(P["tables"] / "state_regression.csv", index=False)
pd.concat(weights_rows).to_csv(P["tables"] / "gate_weights_expiry.csv", index=False)

# Gate behaviour by test year at the main alpha (scale factor and hit rates).
year_rows = []
for asset in cfg["assets"]:
    fc, df = collect_forecasts(P["processed"], asset, MAIN, START, END)
    dates, y, fca = aligned(fc, df["log_return"])
    L = losses_main[asset]
    g = gate[gate.asset == asset].set_index("date").reindex(dates)["g"]
    year_rows.append(pd.DataFrame({
        "year": dates.year, "g": g.to_numpy(),
        "gate_hit": y <= fca[GATE][0], "mean_hit": y <= fca["mean"][0],
        "fz0_gate": L[GATE].to_numpy(),
        **({"fz0_gate_noscale": L["gate_noscale"].to_numpy()} if "gate_noscale" in L else {}),
        "fz0_mean": L["mean"].to_numpy()}))
by_year = pd.concat(year_rows).groupby("year").mean()
by_year.to_csv(P["tables"] / "gate_by_year.csv")

# LaTeX: main results (pooled).
pooled = means[means.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0")
pooled.columns = [f"FZ0 a={c:g}" for c in pooled.columns]
m25 = mcs[(mcs.asset == "ALL") & (mcs.alpha == MAIN)].set_index("method")
d25 = dm[(dm.asset == "ALL") & (dm.alpha == MAIN)].set_index("other")
pooled["MCS p (2.5%)"] = m25["pvalue"]
pooled[f"DM {GATE} vs (2.5%)"] = d25["dm_stat"]
pooled["DM p (2.5%)"] = d25["p"]
pooled.index.name = "method"
pooled = pooled.sort_values("FZ0 a=0.025")
pooled.to_csv(P["tables"] / "main_results.csv")
(TEX / "main_results.tex").write_text(to_latex(
    pooled, "Pooled test-period FZ0 loss, 2023-01 to 2026-09", "tab:main", digits=4,
    note=f"Lower is better. DM: {GATE} minus row method, HLN-corrected; negative favours the gate. "
         "Pooled series is the cross-asset average loss per date."))
b25 = bt[bt.alpha == MAIN].assign(
    kupiec=lambda x: x.kupiec_p < 0.05, cc=lambda x: x.cc_p < 0.05,
    dq=lambda x: x.dq_p < 0.05, mf=lambda x: x.mf_p < 0.05)
rej = b25.groupby("method")[["hit_rate", "kupiec", "cc", "dq", "mf"]].agg(
    {"hit_rate": "mean", "kupiec": "sum", "cc": "sum", "dq": "sum", "mf": "sum"})
rej.index.name = "method"
rej.to_csv(P["tables"] / "backtest_rejections.csv")
(TEX / "backtests.tex").write_text(to_latex(
    rej, f"Backtest rejections at 5\\% across {len(cfg['assets'])} assets, alpha = 2.5\\%", "tab:backtests", digits=3,
    note="Counts of assets where each test rejects; hit\\_rate is the mean of per-asset hit rates. McNeil-Frey uses residuals scaled by |ES|."))

# Figure: monthly mean gate weights and scale.
gm = gate.set_index("date")[[f"wq_{m}" for m in MODELS] + ["g"]].resample("ME").mean()
fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].stackplot(gm.index, gm[[f"wq_{m}" for m in MODELS]].T.to_numpy(), labels=MODELS)
ax[0].set_ylabel("VaR weight")
ax[0].legend(loc="upper left", ncol=3, fontsize=8)
ax[1].plot(gm.index, gm["g"])
ax[1].set_ylabel("scale g")
fig.tight_layout()
fig.savefig(FIG / "gate_weights.png", dpi=200)
print(pooled.round(4).to_string())
