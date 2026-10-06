"""Generate LaTeX tables and PDF figures for the paper from result CSVs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volgate.config import REPO_ROOT  # noqa: E402

R = REPO_ROOT / "results"
OUT = REPO_ROOT / "paper"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6b68"
NAMES = {"r0": "HAR (R0)", "r1": "HAR-X FZ (R1)", "r2_mlp": "R2-MLP", "r2_linear": "R2-linear",
         "mean": "Equal weights", "min_score": "Taylor min.\\ score", "relative_score": "Taylor rel.\\ score",
         "gate": "Gate (no level control)", "gate_v2": "Gate (level control)"}
ORDER = ["r2_mlp", "r2_linear", "r0", "min_score", "relative_score", "mean", "gate_v2", "r1", "gate"]


def f4(x):
    return "--" if pd.isna(x) else f"{x:.4f}"


def pval(p):
    return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write(name, body):
    (OUT / "tables" / f"{name}.tex").write_text(body)


# Table: pooled mean FZ0 by method and alpha, development and fresh universes.
rows = []
for uni in ("dev", "fresh"):
    t = pd.read_csv(R / uni / "tables" / "compare.csv")
    piv = t[t.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0")
    for m in ORDER:
        if m in piv.index:
            rows.append((uni, m, piv.loc[m]))
lines = []
for m in ORDER:
    cells = []
    for uni in ("dev", "fresh"):
        piv = pd.read_csv(R / uni / "tables" / "compare.csv")
        piv = piv[piv.asset == "ALL"].pivot(index="method", columns="alpha", values="mean_fz0")
        cells += [f4(piv.loc[m, a]) if m in piv.index and a in piv.columns else "--" for a in (0.01, 0.025, 0.05)]
    lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
write("fz0", "\n".join(lines) + "\n")

# Table: pre-registered hypotheses (fresh).
h = pd.read_csv(R / "fresh" / "tables" / "hypotheses.csv")
text = {"H1": "HAR $<$ equal weights", "H2": "HAR $<$ Taylor min.\\ score",
        "H3": "HAR $<$ gate (no level control)", "H4": "Gate (level control) $<$ gate (no level control)"}
lines = [f"{r.hypothesis} & {text[r.hypothesis]} & {r.dm_stat:.2f} & {pval(r.p)} & {pval(r.p_holm)} & "
         f"{'Yes' if r.reject else 'No'} \\\\" for r in h.itertuples()]
write("hypotheses", "\n".join(lines) + "\n")

# Table: robustness of H3/H4 by tail level and year.
ba = pd.read_csv(R / "fresh" / "tables" / "robustness" / "by_alpha.csv")
by = pd.read_csv(R / "fresh" / "tables" / "robustness" / "by_year.csv")
lines = []
for label, df, key in (("$\\alpha$", ba, "alpha"), ("Year", by, "year")):
    for k in sorted(df[key].unique()):
        cell = []
        for pair in ("H3", "H4"):
            r = df[(df[key] == k) & (df.pair == pair)].iloc[0]
            cell += [f"{r.dm_stat:.2f}", pval(r.p_one_sided)]
        kk = f"{k:g}" if key == "alpha" else str(k)
        lines.append(f"{kk} & " + " & ".join(cell) + r" \\")
    lines.append(r"\midrule")
write("robustness", "\n".join(lines[:-1]) + "\n")

# Table: backtests (fresh, alpha 2.5%): DQ rejections and AS Z2 rejections.
bt = pd.read_csv(R / "fresh" / "tables" / "backtest_rejections.csv").set_index("method")
z2 = pd.read_csv(R / "fresh" / "tables" / "robustness" / "as_z2_summary.csv").set_index("method")
bt = bt.rename(index={"har": "r0"})
lines = []
for m in ["r0", "mean", "min_score", "gate_v2", "gate", "r1", "r2_mlp"]:
    hit = f"{bt.loc[m, 'hit_rate']:.3f}" if m in bt.index else "--"
    kup = str(int(bt.loc[m, "kupiec"])) if m in bt.index else "--"
    dq = str(int(bt.loc[m, "dq"])) if m in bt.index else "--"
    zr = str(int(z2.loc[m, "rejections"])) if m in z2.index else "--"
    lines.append(f"{NAMES[m]} & {hit} & {kup} & {dq} & {zr} \\\\")
write("backtests", "\n".join(lines) + "\n")

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

# Figure: mechanism (learned scale vs loss gap), one point per stock-year.
mech = pd.read_csv(R / "fresh" / "tables" / "robustness" / "mechanism.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.axhline(0, color=MUTED, lw=0.6)
ax.axvline(0, color=MUTED, lw=0.6)
ax.scatter(mech.mean_log_g, mech.gap_gate_minus_r0, s=12, color=BLUE, edgecolor="white", linewidth=0.6)
b = np.polyfit(mech.mean_log_g, mech.gap_gate_minus_r0, 1)
xs = np.linspace(mech.mean_log_g.min(), mech.mean_log_g.max(), 50)
ax.plot(xs, np.polyval(b, xs), color=INK, lw=1)
ax.set_xlabel("Mean log scale factor of the gate, log g")
ax.set_ylabel("FZ0 gap: gate minus HAR")
fig.tight_layout()
fig.savefig(OUT / "figures" / "mechanism.pdf")

# Figure: H3/H4 loss differences by year (negative = hypothesis direction).
fig, ax = plt.subplots(figsize=(3.4, 2.2))
years = sorted(by.year.unique())
x = np.arange(len(years))
for i, (pair, col, lab) in enumerate((("H3", BLUE, "HAR minus gate"), ("H4", ORANGE, "Level-controlled gate minus gate"))):
    d = by[by.pair == pair].set_index("year").loc[years, "diff"]
    ax.bar(x + (i - 0.5) * 0.36, d, width=0.34, color=col, label=lab, edgecolor="white", linewidth=1)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xticks(x, [str(y) for y in years])
ax.set_ylabel("Mean FZ0 difference")
ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "figures" / "by_year.pdf")
print("paper assets written")

# ---- Extended assets -------------------------------------------------------------------------
import yaml  # noqa: E402

fresh_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "fresh.yaml"))
dev_cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "dev.yaml"))

# Table: fresh universe by sector.
lines = []
for sector, members in fresh_cfg["sectors"].items():
    tick = ", ".join(fresh_cfg["assets"][m].replace(".NS", "").replace("&", "\\&") for m in members)
    label = {"banks": "Private banks", "it": "IT services", "auto": "Automobiles", "pharma": "Pharma",
             "metals": "Metals", "fmcg": "FMCG", "power_oil_psu": "Power and oil PSUs",
             "cement": "Cement and building", "paints_chem": "Paints and chemicals"}.get(sector, sector)
    lines.append(f"{label} & {tick} \\\\")
write("universe", "\n".join(lines) + "\n")

# Table: per-stock mean FZ0 at 2.5% (fresh).
c = pd.read_csv(R / "fresh" / "tables" / "compare.csv")
c = c[(c.alpha == 0.025) & (c.asset != "ALL")].pivot(index="asset", columns="method", values="mean_fz0")
cols = ["r0", "mean", "min_score", "gate_v2", "gate"]
lines = []
for asset in fresh_cfg["assets"]:
    row = c.loc[asset, cols]
    best = row.idxmin()
    cells = [(f"\\textbf{{{v:.3f}}}" if m == best else f"{v:.3f}") for m, v in row.items()]
    name = fresh_cfg["assets"][asset].replace(".NS", "").replace("&", "\\&")
    lines.append(f"{name} & " + " & ".join(cells) + r" \\")
write("per_stock", "\n".join(lines) + "\n")

# Table: Model Confidence Set p-values, pooled, 2.5% (fresh).
m = pd.read_csv(R / "fresh" / "tables" / "mcs.csv")
m = m[(m.alpha == 0.025) & (m.asset == "ALL")].sort_values("pvalue", ascending=False)
mcs_names = {**NAMES, "har": "HAR (R0)", "ewma": "EWMA", "garch_t": "GARCH-$t$", "egarch_t": "EGARCH-$t$",
             "gjr_t": "GJR-GARCH-$t$", "vix": "India VIX model", "median": "Median",
             "previous_best": "Previous best"}
lines = [f"{mcs_names.get(r.method, r.method)} & {r.pvalue:.3f} & {'Yes' if r.in_mcs else 'No'} \\\\"
         for r in m.itertuples()]
write("mcs", "\n".join(lines) + "\n")

# Table: original hypothesis family and power projection (fresh).
of = pd.read_csv(R / "fresh" / "tables" / "robustness" / "original_family.csv")
of_text = {"orig_H1": "R1 $<$ HAR", "orig_H2": "R2-MLP $<$ HAR", "orig_H3a": "R2-MLP $<$ Taylor min.\\ score",
           "orig_H3b": "R2-MLP $<$ equal weights", "orig_H4": "R2-linear $<$ HAR"}
write("original_family", "\n".join(f"{of_text[r.pair]} & {r.dm_stat:.2f} & {pval(r.p_one_sided)} \\\\"
                                    for r in of.itertuples()) + "\n")
pw = pd.read_csv(R / "fresh" / "tables" / "robustness" / "power.csv")
pw_text = {"H1": "H1: HAR vs equal weights", "H2": "H2: HAR vs Taylor min.\\ score", "H3": "H3: HAR vs gate",
           "H4": "H4: level control vs gate", "S1": "R2-MLP vs HAR", "S2": "R2-linear vs HAR"}
write("power", "\n".join(
    f"{pw_text[r.pair]} & {r.rho:.3f} & {r.t_at_N:.2f} & {r.t_limit:.2f} & "
    f"{'--' if pd.isna(r.n_required) else int(r.n_required)} \\\\" for r in pw.itertuples()) + "\n")

# Table: data summary for both universes.
lines = []
for uni, cfg_u in (("Development", dev_cfg), ("Fresh", fresh_cfg)):
    ds = pd.read_csv(R / ("dev" if uni == "Development" else "fresh") / "tables" / "data_summary.csv")
    oa = pd.read_csv(R / ("dev" if uni == "Development" else "fresh") / "tables" / "outlier_audit.csv")
    lines.append(f"{uni} & {len(ds)} & {int(ds.rows_after_corrections.median())} & {int(ds.flat_dropped.sum())} & "
                 f"{int(ds.special_dropped.sum())} & {len(oa)} & {int((oa.action == 'drop').sum())} \\\\")
write("data_summary", "\n".join(lines) + "\n")

# Figure: learned scale over time, uncontrolled vs level-controlled gate (fresh, 2.5%).
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for name, col, lab in (("gate", BLUE, "No level control"), ("gate_v2", ORANGE, "Level control")):
    g = pd.read_csv(REPO_ROOT / "data" / "fresh" / "processed" / "gate" / "a025" / f"{name}.csv",
                    parse_dates=["date"]).groupby("date")["g"].mean().resample("ME").mean()
    ax.plot(g.index, g.values, color=col, lw=2, label=lab)
ax.axhline(1.0, color=MUTED, lw=0.6)
ax.set_ylabel("Mean scale factor $g$")
ax.set_ylim(top=1.06)
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.5)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "figures" / "scale.pdf")
print("extended paper assets written")
