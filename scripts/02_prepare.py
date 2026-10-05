import warnings

import pandas as pd

from volgate.config import REPO_ROOT, load_config
from volgate.data.download import verify_manifest
from volgate.data.expiry import asset_expiry_features, load_rules, unverified
from volgate.data.panel import apply_corrections, build_asset_frame, load_raw, nse_calendar, outlier_table

cfg = load_config()
P = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
bad = verify_manifest(P["raw"])
if bad:
    raise SystemExit(f"raw data changed since download: {bad}")

vix = load_raw(P["raw"] / "india_vix.csv")["close"]
frames, stats = {}, []
for asset in cfg["assets"]:
    frames[asset], s = build_asset_frame(load_raw(P["raw"] / f"{asset}.csv"), vix,
                                         exclude_dates=cfg.get("special_sessions", []))
    stats.append({"asset": asset, **s})

P["tables"].mkdir(parents=True, exist_ok=True)
corrections = pd.read_csv(P["audit"] / "corrections.csv", dtype=str)
audit = outlier_table(frames, cfg["outlier_threshold"]).merge(
    corrections, on=["asset", "date"], how="left")
audit["action"] = audit["action"].fillna("unreviewed")
audit.to_csv(P["tables"] / "outlier_audit.csv", index=False)
n_unreviewed = int((audit["action"] == "unreviewed").sum())
if n_unreviewed:
    warnings.warn(f"{n_unreviewed} outliers not reviewed; see results/tables/outlier_audit.csv")
frames = apply_corrections(frames, corrections)

rules = load_rules(P["expiry_rules"])
for item in unverified(rules):
    warnings.warn(f"unverified expiry rule: {item}")

out_dir = P["processed"] / "panel"
out_dir.mkdir(parents=True, exist_ok=True)
calendar = nse_calendar(frames, cfg.get("special_sessions", []))
for asset, df in frames.items():
    df = df.join(asset_expiry_features(df.index, rules[cfg["contracts"][asset]], calendar, "own"))
    df = df.join(asset_expiry_features(df.index, rules["NIFTY"], calendar, "nifty"))
    df.to_csv(out_dir / f"{asset}.csv", date_format="%Y-%m-%d")

summary = pd.DataFrame(stats)
summary["rows_after_corrections"] = [len(frames[a]) for a in summary["asset"]]
summary.to_csv(P["tables"] / "data_summary.csv", index=False)
print(summary.to_string(index=False))
