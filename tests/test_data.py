import numpy as np
import pandas as pd

from rul.data import (
    SENSORS,
    ConditionScaler,
    add_test_rul,
    add_train_rul,
    make_windows,
    split_engines,
    window_summary,
)
from rul.metrics import nasa_score


def toy_engine(engine, n_cycles, subset="FD001"):
    df = pd.DataFrame({"engine": engine, "unit": engine % 1000, "cycle": np.arange(1, n_cycles + 1),
                       "subset": subset, "condition": 0})
    for s in SENSORS:
        df[s] = np.arange(n_cycles, dtype=float)
    return df


def test_train_rul_counts_down_and_is_capped():
    df = add_train_rul(toy_engine(1001, 200), cap=125)
    assert df["RUL"].iloc[-1] == 0
    assert df["RUL"].max() == 125
    assert df["RUL"].iloc[-10] == 9


def test_test_rul_uses_true_rul_at_last_cycle():
    df = add_test_rul(toy_engine(1001, 50), np.array([30.0]), cap=125)
    assert df["RUL_true"].iloc[-1] == 30
    assert df["RUL_true"].iloc[0] == 79


def test_windows_have_right_shape_and_label():
    df = add_train_rul(toy_engine(1001, 40))
    X, y, eng, _ = make_windows(df, ["s2", "s3"], window=30)
    assert X.shape == (40, 30, 2)
    assert y[-1] == 0
    np.testing.assert_array_equal(X[-1, :, 0], np.arange(10, 40))


def test_short_engine_is_front_padded():
    df = add_train_rul(toy_engine(1001, 5))
    X, *_ = make_windows(df, ["s2"], window=30, last_only=True)
    assert X.shape == (1, 30, 1)
    assert (X[0, :26, 0] == 0).all()
    np.testing.assert_array_equal(X[0, -5:, 0], np.arange(5))


def test_validation_split_has_no_engine_overlap():
    df = pd.concat([toy_engine(1000 + i, 10) for i in range(1, 21)])
    fit, val = split_engines(df, val_frac=0.2, seed=0)
    assert set(fit.engine).isdisjoint(set(val.engine))
    assert val.engine.nunique() == 4


def test_scaler_gives_zero_mean_per_condition():
    df = pd.concat([toy_engine(1001, 30), toy_engine(1002, 30)])
    df.loc[df.engine == 1002, "condition"] = 1
    df.loc[df.engine == 1002, "s2"] += 100
    out = ConditionScaler().fit(df, ["s2"]).transform(df)
    assert abs(out.groupby("condition")["s2"].mean()).max() < 1e-9


def test_window_summary_slope():
    X = np.arange(30, dtype=np.float32).reshape(1, 30, 1)
    feats = window_summary(X)
    assert feats.shape == (1, 3)
    assert abs(feats[0, 2] - 1.0) < 1e-6


def test_nasa_score_penalises_late_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [50]) == 0
