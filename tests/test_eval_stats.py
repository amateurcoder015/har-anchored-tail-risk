import numpy as np
import pandas as pd
import pytest

from volgate.evaluate.stats import (christoffersen, dm_test, dq_test, kupiec, mcneil_frey,
                                    mcs_table, nw_variance)


def test_nw_variance_iid_close_to_sample_variance():
    d = np.random.default_rng(0).normal(size=20000)
    assert nw_variance(d) == pytest.approx(1.0, rel=0.05)


def test_dm_detects_lower_loss_and_handles_identical():
    rng = np.random.default_rng(1)
    l2 = rng.normal(size=1000)
    stat, p = dm_test(l2 - 0.2 + rng.normal(0, 0.5, 1000), l2)
    assert stat < 0 and p < 0.01
    s, p = dm_test(l2, l2)
    assert np.isnan(s) and np.isnan(p)


def test_kupiec_exact_rate_and_zero_hits():
    hits = np.zeros(400, bool)
    hits[::40] = True  # 10 hits = 2.5%
    lr, p = kupiec(hits, 0.025)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    lr0, p0 = kupiec(np.zeros(400, bool), 0.025)
    assert np.isfinite(lr0) and p0 < 0.001


def test_christoffersen_flags_clustering():
    clustered = np.zeros(400, bool)
    clustered[100:110] = True
    spread = np.zeros(400, bool)
    spread[::40] = True
    assert christoffersen(clustered, 0.025)["p_ind"] < 0.01
    assert christoffersen(spread, 0.025)["p_ind"] > 0.1
    assert np.isfinite(christoffersen(np.zeros(400, bool), 0.025)["lr_cc"])


def test_dq_flags_clustered_hits():
    var = np.full(500, -0.02)
    clustered = np.zeros(500, bool)
    clustered[200:213] = True
    assert dq_test(clustered, var, 0.025)[1] < 0.01
    iid = np.random.default_rng(3).random(500) < 0.025
    assert dq_test(iid, var, 0.025)[1] > 0.01


def test_mcneil_frey_detects_too_shallow_es():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(5000)
    var = np.full(5000, -1.96)
    good, bad = np.full(5000, -2.338), np.full(5000, -2.1)
    assert mcneil_frey(y, var, good)[1] > 0.05
    assert mcneil_frey(y, var, bad)[1] < 0.01
    m, p, n = mcneil_frey(y, np.full(5000, -10.0), np.full(5000, -11.0))
    assert n == 0 and np.isnan(m) and np.isnan(p)


def test_mcs_table_excludes_clearly_worse():
    rng = np.random.default_rng(5)
    L = pd.DataFrame({"a": rng.normal(0, 1, 800), "b": rng.normal(1.0, 1, 800)})
    t = mcs_table(L, reps=300)
    assert bool(t.loc["a", "in_mcs"]) and not bool(t.loc["b", "in_mcs"])


def test_mcs_table_handles_identical_methods():
    rng = np.random.default_rng(6)
    a = rng.normal(0, 1, 600)
    L = pd.DataFrame({"a": a, "a_copy": a, "b": rng.normal(1.0, 1, 600)})
    t = mcs_table(L, reps=200)
    assert list(t.index) == ["a", "a_copy", "b"]
    assert t.loc["a", "pvalue"] == t.loc["a_copy", "pvalue"]
    assert bool(t.loc["a_copy", "in_mcs"]) and not bool(t.loc["b", "in_mcs"])
