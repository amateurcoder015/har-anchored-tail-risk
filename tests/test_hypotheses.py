import numpy as np
import pandas as pd
import pytest

from hart.hypotheses import HYPOTHESES, evaluate_hypotheses, holm, one_sided_p


def test_one_sided_p():
    assert one_sided_p(-2.0, 0.04) == pytest.approx(0.02)
    assert one_sided_p(2.0, 0.04) == pytest.approx(0.98)
    assert np.isnan(one_sided_p(np.nan, np.nan))


def test_holm_step_down():
    t = holm({"a": 0.01, "b": 0.04, "c": 0.03}, level=0.05)
    assert t.loc["a", "p_holm"] == pytest.approx(0.03)
    assert t.loc["c", "p_holm"] == pytest.approx(0.06)
    assert t.loc["b", "p_holm"] == pytest.approx(0.06)  # monotone: max with previous
    assert t["reject"].tolist() == [True, False, False]


def test_holm_nan_never_rejects():
    t = holm({"a": 0.001, "b": np.nan}, level=0.05)
    assert bool(t.loc["a", "reject"]) and not bool(t.loc["b", "reject"])


def test_evaluate_hypotheses_direction():
    rng = np.random.default_rng(0)
    n = 900
    base = rng.normal(0, 1, n)
    L = pd.DataFrame({"r0": base - 0.3, "mean": base, "min_score": base,
                      "gate": base + 0.3, "gate_v2": base})
    t = evaluate_hypotheses(L)
    assert list(t.index) == [h[0] for h in HYPOTHESES]
    assert t["reject"].all()
    L2 = L.assign(r0=base + 0.3)
    assert not evaluate_hypotheses(L2).loc["H1", "reject"]
