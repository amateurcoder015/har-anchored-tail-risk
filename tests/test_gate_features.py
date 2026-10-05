import pandas as pd

from volgate.gate.features import feature_columns, gate_features


def test_columns_and_selection(combo_frame):
    f = gate_features(combo_frame, 0.025)
    assert set(feature_columns(True, True)) == set(f.columns)
    assert "vix" not in feature_columns(False, True)
    assert "own_is_expiry" not in feature_columns(True, False)
    assert f.iloc[100:].notna().all().all()


def test_no_lookahead(combo_frame):
    base = gate_features(combo_frame, 0.025)
    j = 400
    pert = combo_frame.copy()
    cols = [c for c in pert.columns
            if c in ("log_return", "gk_var", "vix") or ("__" in c and "fallback" not in c)]
    pert.iloc[j, pert.columns.get_indexer(cols)] *= 3
    out = gate_features(pert, 0.025)
    pd.testing.assert_frame_equal(base.iloc[: j + 1], out.iloc[: j + 1])
    assert not base.iloc[j + 1].equals(out.iloc[j + 1])
