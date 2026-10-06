"""Flowcharts and result figures for the presentation (dark theme to match the deck)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "img"
BG, PANEL, INK, MUTED = "#0f2a22", "#163a2f", "#f2f5f0", "#9fb8ad"
LIME, TEAL, ORANGE, RED = "#b6e35a", "#4fd1c5", "#f5a35c", "#ef6f6c"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})


def canvas(w=13.33, h=7.5):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=TEAL, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=PANEL, ec=color, lw=2.2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.9, body, ha="center", va="top", fontsize=fs - 2, color=MUTED,
                linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, lw=2,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", fontsize=11, color=color)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


# 1. How a daily forecast is made and checked.
fig, ax = canvas()
steps = [("Prices up to\ntoday", "open, high, low,\nclose for each stock", TEAL),
         ("Six risk models", "EWMA, GARCH x3,\nHAR, India VIX", TEAL),
         ("Combine", "average, Taylor's\nweights, or a gate", LIME),
         ("Tomorrow's\nVaR and ES", "e.g. VaR -3.0%\nES -4.2%", ORANGE),
         ("Tomorrow\nhappens", "real return\nobserved", TEAL),
         ("Score", "FZ0 loss:\nlower = better", LIME)]
x = 0.4
for i, (t, b, c) in enumerate(steps):
    box(ax, x, 3.6, 1.85, 2.1, t, b, c, fs=15)
    if i < len(steps) - 1:
        arrow(ax, x + 1.85, 4.65, x + 2.2, 4.65)
    x += 2.2
ax.text(0.5, 2.3, "Repeated every trading day from 2023 to September 2026, for every stock.", fontsize=16, color=MUTED)
ax.text(0.5, 1.7, "Models only ever see data up to the evening before: no peeking at tomorrow.", fontsize=16, color=MUTED)
save(fig, "flow_daily.png")

# 2. What the gate does.
fig, ax = canvas()
for j, m in enumerate(["EWMA", "GARCH", "EGARCH", "GJR-GARCH", "HAR", "VIX model"]):
    box(ax, 0.4, 5.65 - j * 0.95, 2.2, 0.75, m, color=TEAL, fs=14)
    arrow(ax, 2.6, 6.03 - j * 0.95, 5.2, 3.9, color=MUTED)
box(ax, 0.4 + 4.8, 4.9, 2.6, 1.6, "Market state", "volatility, India VIX,\nrecent model errors", LIME, 15)
arrow(ax, 6.5, 4.9, 6.5, 4.35, color=LIME)
box(ax, 5.2, 2.4, 2.6, 1.95, "Gate network", "decides\n1) weights per model\n2) scale factor g", ORANGE, 15)
arrow(ax, 7.8, 3.4, 9.0, 3.4, color=ORANGE)
box(ax, 9.0, 2.4, 3.9, 1.95, "Final VaR and ES", "= g x (weighted mix)\ng = 1: no change\ng = 0.9: 10% less risk", ORANGE, 15)
ax.text(0.5, 0.3, "Level drift: the gate learns a g below 1 from calm training years and keeps using it later.",
        fontsize=16, color=RED)
save(fig, "flow_gate.png")

# 3. Study design.
fig, ax = canvas()
stages = [("1  Develop", "20 stocks already\nused before.\nDesign and tune\nmethods here.", TEAL),
          ("2  Write it down", "4 hypotheses,\ntest rules, 27 new\nstock names, exact\ncode version", LIME),
          ("3  Publish", "pushed to GitHub\n(tag prereg-fresh)\nBEFORE any new\ndata exists", LIME),
          ("4  Download", "27 never-used\nNSE stocks,\n3 per sector", TEAL),
          ("5  Run once", "no tuning\nafterwards;\nreport everything", ORANGE)]
x = 0.4
for i, (t, b, c) in enumerate(stages):
    box(ax, x, 3.3, 2.3, 2.9, t, b, c, fs=16)
    if i < len(stages) - 1:
        arrow(ax, x + 2.3, 4.75, x + 2.6, 4.75)
    x += 2.6
ax.text(0.5, 1.9, "Why: it proves results were not picked or tuned after seeing the test data.", fontsize=16, color=MUTED)
ax.text(0.5, 1.3, "Only post-download decision: the pre-agreed outlier rule removed 1 row (Vedanta demerger day).",
        fontsize=16, color=MUTED)
save(fig, "flow_study.png")

# 4. Evolution of the three projects.
fig, ax = canvas()
proj = [("Project 1", "volatality-modelling", "Neural blend of 4 models\nvolatility only, 8 assets,\n1 test year\n\nLost to simple EWMA;\nchecks not real", RED),
        ("Project 2", "state-dependent gate", "VaR + ES, FZ0 score,\n6 models refitted monthly,\nreal tests\n\nGate trails simple blends;\nscale factor fell to 0.82", ORANGE),
        ("Project 3", "har-anchored-tail-risk", "Why do gates fail?\nPre-registered test,\n27 untouched stocks\n\nConfirmed: level drift;\nlevel control fixes it", LIME)]
x = 0.5
for i, (t, s, b, c) in enumerate(proj):
    box(ax, x, 1.6, 3.8, 4.6, t, s + "\n\n" + b, c, fs=21)
    if i < 2:
        arrow(ax, x + 3.8, 3.9, x + 4.4, 3.9)
    x += 4.4
save(fig, "flow_projects.png")


# Result figures (dark versions of the paper figures).
def dark_axes(w=11, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


R = ROOT / "results" / "fresh" / "tables" / "robustness"
mech = pd.read_csv(R / "mechanism.csv")
fig, ax = dark_axes()
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
ax.scatter(mech.mean_log_g, mech.gap_gate_minus_r0, s=50, color=TEAL, edgecolor=BG)
b = np.polyfit(mech.mean_log_g, mech.gap_gate_minus_r0, 1)
xs = np.linspace(mech.mean_log_g.min(), mech.mean_log_g.max(), 50)
ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=2.5)
ax.set_xlabel("How much the gate shrank risk (log of scale factor g; left = more shrinking)", fontsize=13)
ax.set_ylabel("Extra loss vs HAR (higher = worse)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "mechanism.png", dpi=150, facecolor=BG); plt.close(fig)

fig, ax = dark_axes()
for name, col, lab in (("gate", ORANGE, "Gate without level control"), ("gate_v2", LIME, "Gate with level control")):
    g = pd.read_csv(ROOT / "data" / "fresh" / "processed" / "gate" / "a025" / f"{name}.csv",
                    parse_dates=["date"]).groupby("date")["g"].mean().resample("ME").mean()
    ax.plot(g.index, g.values, color=col, lw=3, label=lab)
ax.axhline(1.0, color=MUTED, lw=1, ls="--")
ax.text(ax.get_xlim()[0], 1.004, "  g = 1: no shrinking", color=MUTED, fontsize=12)
ax.set_ylabel("Average scale factor g", fontsize=13)
ax.legend(frameon=False, fontsize=12, labelcolor=INK, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "scale.png", dpi=150, facecolor=BG); plt.close(fig)
print("figures done")

# Crop every figure to its content plus a small margin, so it displays as large as possible.
from PIL import Image, ImageChops
bg_rgb = tuple(int(BG[i:i + 2], 16) for i in (1, 3, 5))
for f in OUT.glob("*.png"):
    im = Image.open(f).convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, bg_rgb)).convert("L").point(lambda v: 255 if v > 18 else 0)
    l, t, r, b = diff.getbbox()
    pad = 30
    im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))).save(f)
print("cropped")
