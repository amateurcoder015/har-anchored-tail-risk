from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
_FREQ = {v: f"W-{k}" for k, v in WEEKDAYS.items()}


@dataclass(frozen=True)
class ExpiryRule:
    kind: str
    weekday: int
    start: pd.Timestamp
    end: pd.Timestamp | None
    source: str


def load_rules(path: Path) -> dict[str, list[ExpiryRule]]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    out = {}
    for contract, items in raw.items():
        rules = []
        for it in items:
            if it["weekday"] not in WEEKDAYS:
                raise ValueError(f"bad weekday {it['weekday']!r} in {contract}")
            if it["kind"] not in ("weekly", "monthly"):
                raise ValueError(f"bad kind {it['kind']!r} in {contract}")
            if not it.get("source"):
                raise ValueError(f"missing source in {contract}")
            rules.append(ExpiryRule(
                it["kind"], WEEKDAYS[it["weekday"]], pd.Timestamp(it["start"]),
                pd.Timestamp(it["end"]) if it["end"] else None, it["source"]))
        out[contract] = rules
    return out


def unverified(rules: dict[str, list[ExpiryRule]]) -> list[str]:
    return [f"{c}: {r.kind} {r.start.date()}..{r.end.date() if r.end else 'open'}"
            for c, rs in rules.items() for r in rs if r.source == "UNVERIFIED"]


def _candidates(rule: ExpiryRule, lo: pd.Timestamp, hi: pd.Timestamp) -> list[pd.Timestamp]:
    if rule.kind == "weekly":
        return list(pd.date_range(lo, hi, freq=_FREQ[rule.weekday]))
    out = []
    for period in pd.period_range(lo, hi, freq="M"):
        month_end = period.end_time.normalize()
        d = month_end - pd.Timedelta(days=(month_end.weekday() - rule.weekday) % 7)
        if lo <= d <= hi:
            out.append(d)
    return out


def expiry_dates(rules: list[ExpiryRule], trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Scheduled expiries; a non-trading expiry moves to the previous trading day.

    Days after the last trading day are approximated by business days so that
    days-to-expiry is defined at the end of the sample.
    """
    future = pd.bdate_range(trading_days[-1] + pd.Timedelta(days=1), periods=40)
    cal = trading_days.union(future)
    out = set()
    for rule in rules:
        lo = max(rule.start, cal[0])
        hi = min(rule.end if rule.end is not None else cal[-1], cal[-1])
        for d in _candidates(rule, lo, hi):
            pos = cal.searchsorted(d, side="right") - 1
            if pos >= 0:
                out.add(cal[pos])
    return pd.DatetimeIndex(sorted(out))


def expiry_features(dates: pd.DatetimeIndex, expiries: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    future = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=40)
    cal = dates.union(future).union(expiries)
    pos_dates = cal.get_indexer(dates)
    exp_pos = np.sort(cal.get_indexer(expiries))
    nxt = np.searchsorted(exp_pos, pos_dates, side="left")
    has_next = nxt < len(exp_pos)
    dte = np.where(has_next, exp_pos[np.minimum(nxt, len(exp_pos) - 1)] - pos_dates, np.nan)
    return pd.DataFrame({
        f"{prefix}_is_expiry": dates.isin(expiries).astype(int),
        f"{prefix}_days_to_expiry": dte,
    }, index=dates)


def asset_expiry_features(asset_days: pd.DatetimeIndex, rules: list[ExpiryRule],
                          calendar: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    """Expiry features on the common NSE calendar, restricted to one asset's days."""
    feats = expiry_features(calendar, expiry_dates(rules, calendar), prefix)
    return feats.reindex(asset_days)
