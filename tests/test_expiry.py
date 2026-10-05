import pandas as pd
import pytest
import yaml

from volgate.data.expiry import ExpiryRule, expiry_dates, expiry_features, load_rules, unverified


def _rule(kind, wd, start, end=None):
    return ExpiryRule(kind, wd, pd.Timestamp(start), pd.Timestamp(end) if end else None, "test")


def test_weekly_thursday_moves_to_previous_day_on_holiday():
    days = pd.bdate_range("2024-01-01", "2024-01-31").drop(pd.Timestamp("2024-01-25"))
    exp = expiry_dates([_rule("weekly", 3, "2024-01-01", "2024-01-31")], days)
    assert list(exp.strftime("%Y-%m-%d")) == ["2024-01-04", "2024-01-11", "2024-01-18", "2024-01-24"]


def test_monthly_last_tuesday():
    days = pd.bdate_range("2025-09-01", "2025-10-31")
    exp = expiry_dates([_rule("monthly", 1, "2025-09-01")], days)
    assert "2025-09-30" in exp.strftime("%Y-%m-%d")
    assert "2025-10-28" in exp.strftime("%Y-%m-%d")


def test_features_days_to_expiry():
    days = pd.bdate_range("2024-01-01", "2024-01-12")
    exp = expiry_dates([_rule("weekly", 3, "2024-01-01")], days)
    f = expiry_features(days, exp, "own")
    assert f.loc["2024-01-04", "own_is_expiry"] == 1
    assert f.loc["2024-01-04", "own_days_to_expiry"] == 0
    assert f.loc["2024-01-03", "own_days_to_expiry"] == 1
    assert f.loc["2024-01-05", "own_days_to_expiry"] == 4  # Fri -> next Thu
    assert f.loc["2024-01-12", "own_days_to_expiry"] == 4  # counts business days after sample end
    assert f["own_days_to_expiry"].notna().all()  # beyond-sample expiry still found


def test_load_rules_and_unverified(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"X": [
        {"kind": "weekly", "weekday": "THU", "start": "2024-01-01", "end": None, "source": "UNVERIFIED"},
        {"kind": "monthly", "weekday": "TUE", "start": "2024-01-01", "end": None, "source": "https://x"},
    ]}))
    rules = load_rules(p)
    assert rules["X"][0].weekday == 3
    assert len(unverified(rules)) == 1


def test_load_rules_rejects_bad_weekday(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"X": [
        {"kind": "weekly", "weekday": "SAT", "start": "2024-01-01", "end": None, "source": "s"}]}))
    with pytest.raises(ValueError, match="weekday"):
        load_rules(p)


def test_repo_rules_file_loads():
    from volgate.config import REPO_ROOT
    rules = load_rules(REPO_ROOT / "configs" / "expiry_rules.yaml")
    assert set(rules) == {"NIFTY", "BANKNIFTY", "STOCK"}


def test_asset_expiry_features_use_common_calendar():
    from volgate.data.expiry import asset_expiry_features
    # Muhurat 2021-11-04 (Thu) excluded from calendar: weekly expiry moves to Wed 2021-11-03
    cal = pd.bdate_range("2021-11-01", "2021-11-12").drop(pd.Timestamp("2021-11-04"))
    asset_days = cal.drop(pd.Timestamp("2021-11-02"))  # asset missing a day the calendar has
    f = asset_expiry_features(asset_days, [_rule("weekly", 3, "2021-01-01")], cal, "own")
    assert f.loc["2021-11-03", "own_is_expiry"] == 1
    assert f.loc["2021-11-01", "own_days_to_expiry"] == 2  # counted on the common calendar
    assert list(f.index) == list(asset_days)
