"""Loading, labelling, normalising and windowing the NASA C-MAPSS turbofan data.

Each raw file has one row per engine per flight cycle:
    unit, cycle, 3 operational settings, 21 sensor readings.
Training engines run until failure; test engines stop some time before failure,
and the RUL_FD00x.txt file gives the true remaining cycles at that last point.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle", "op1", "op2", "op3"] + SENSORS
SUBSETS = [1, 2, 3, 4]

# The six flight conditions in C-MAPSS (altitude-ish, Mach-ish, throttle).
# FD001 and FD003 only use the first one; FD002 and FD004 use all six.
CONDITIONS = [
    (0.0, 0.0, 100.0),
    (10.0, 0.25, 100.0),
    (20.0, 0.7, 100.0),
    (25.0, 0.62, 60.0),
    (35.0, 0.84, 100.0),
    (42.0, 0.84, 100.0),
]

# Piecewise-linear RUL target: an engine far from failure looks healthy, so its
# label is capped. 125 cycles is the standard choice in the C-MAPSS literature.
RUL_CAP = 125


def read_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def load_subset(raw_dir: Path, fd: int, split: str) -> pd.DataFrame:
    """Load train or test data for one sub-dataset, with a globally unique engine id."""
    df = read_file(Path(raw_dir) / f"{split}_FD00{fd}.txt")
    df["subset"] = f"FD00{fd}"
    df["engine"] = fd * 1000 + df["unit"]
    return df


def assign_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Map the three noisy operational settings to one of the 6 flight conditions."""
    keys = list(zip(df["op1"].round(0), df["op2"].round(2), df["op3"].round(0)))
    lookup = {c: i for i, c in enumerate(CONDITIONS)}
    cond = [lookup.get(k, -1) for k in keys]
    if -1 in cond:
        raise ValueError("Found an operating setting that matches no known condition")
    df = df.copy()
    df["condition"] = cond
    return df


def add_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Training engines run to failure, so RUL = last cycle - current cycle."""
    df = df.copy()
    last = df.groupby("engine")["cycle"].transform("max")
    df["RUL"] = (last - df["cycle"]).clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_at_end: np.ndarray, cap: int = RUL_CAP) -> pd.DataFrame:
    """Test engines stop early; rul_at_end[i] is the true RUL at unit i+1's last cycle."""
    df = df.copy()
    end = pd.Series(rul_at_end, index=np.arange(1, len(rul_at_end) + 1))
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL_true"] = df["unit"].map(end) + (last - df["cycle"])
    df["RUL"] = df["RUL_true"].clip(upper=cap)
    return df


def load_all(raw_dir: Path, subsets=SUBSETS):
    """Return (train, test) DataFrames for the chosen sub-datasets, labelled."""
    train, test = [], []
    for fd in subsets:
        tr = add_train_rul(assign_condition(load_subset(raw_dir, fd, "train")))
        te = assign_condition(load_subset(raw_dir, fd, "test"))
        rul_end = np.loadtxt(Path(raw_dir) / f"RUL_FD00{fd}.txt")
        te = add_test_rul(te, np.atleast_1d(rul_end))
        train.append(tr)
        test.append(te)
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def informative_sensors(train: pd.DataFrame, threshold: float = 1e-3) -> list[str]:
    """Drop sensors that do not change within a flight condition (they carry no wear signal)."""
    within_std = train.groupby("condition")[SENSORS].std().mean()
    return [s for s in SENSORS if within_std[s] >= threshold]


class ConditionScaler:
    """Z-score each sensor separately per flight condition, using training data only.

    Raw readings jump whenever the flight condition changes; normalising within a
    condition removes that jump so the slow wear trend becomes visible.
    """

    def fit(self, df: pd.DataFrame, features: list[str]):
        self.features = list(features)
        grouped = df.groupby("condition")[self.features]
        self.mean_ = grouped.mean()
        self.std_ = grouped.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mu = self.mean_.loc[df["condition"]].to_numpy()
        sd = self.std_.loc[df["condition"]].to_numpy()
        df[self.features] = (df[self.features].to_numpy() - mu) / sd
        return df


def make_windows(df: pd.DataFrame, features: list[str], window: int, last_only: bool = False):
    """Turn each engine's history into fixed-length windows of `window` cycles.

    A window ending at cycle t is labelled with the RUL at cycle t. Engines with
    fewer cycles than the window are padded at the front by repeating their first
    row. With last_only=True, only the final window per engine is returned (used
    for the official test set, where RUL is known only at the last cycle).
    """
    xs, ys, engines, subsets = [], [], [], []
    for eng, g in df.sort_values(["engine", "cycle"]).groupby("engine", sort=False):
        values = g[features].to_numpy(dtype=np.float32)
        labels = g["RUL"].to_numpy(dtype=np.float32)
        pad = np.repeat(values[:1], window - 1, axis=0)
        padded = np.vstack([pad, values])
        # windows[i] covers original cycles i-window+1 .. i
        windows = np.lib.stride_tricks.sliding_window_view(padded, (window, len(features)))[:, 0]
        if last_only:
            windows, labels = windows[-1:], labels[-1:]
        xs.append(windows)
        ys.append(labels)
        engines.append(np.full(len(labels), eng))
        subsets.append(np.full(len(labels), g["subset"].iloc[0]))
    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys),
        np.concatenate(engines),
        np.concatenate(subsets),
    )


def split_engines(train: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Hold out whole engines (not rows) for validation, per sub-dataset.

    Splitting by row would leak: neighbouring cycles of the same engine would sit
    on both sides of the split and the validation score would look too good.
    """
    rng = np.random.default_rng(seed)
    val = []
    for _, g in train.groupby("subset"):
        engines = g["engine"].unique()
        n_val = max(1, int(round(len(engines) * val_frac)))
        val.extend(rng.choice(engines, size=n_val, replace=False))
    is_val = train["engine"].isin(val)
    return train[~is_val].copy(), train[is_val].copy()


def window_summary(X: np.ndarray) -> np.ndarray:
    """Flatten windows for the tree baseline: last value, mean and linear slope per sensor."""
    w = X.shape[1]
    t = np.arange(w, dtype=np.float32)
    t -= t.mean()
    slope = ((X - X.mean(axis=1, keepdims=True)) * t[None, :, None]).sum(axis=1) / (t**2).sum()
    return np.hstack([X[:, -1, :], X.mean(axis=1), slope])
