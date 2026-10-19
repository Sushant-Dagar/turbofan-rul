"""Exploratory data analysis: dataset size, flat sensors, and how sensors drift before failure.

Run:  python eda.py
Writes reports/eda_summary.json and two charts to reports/figures/.
"""

import json
from pathlib import Path

import numpy as np

from rul import plotting
from rul.data import ConditionScaler, SENSORS, informative_sensors, load_all

RAW = Path("data/raw")
OUT = Path("reports")
FIG = OUT / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plotting.setup()
    train, test = load_all(RAW)

    features = informative_sensors(train)
    dropped = [s for s in SENSORS if s not in features]
    scaled = ConditionScaler().fit(train, features).transform(train)

    # How strongly each sensor tracks remaining life (Pearson r with capped RUL)
    corr = scaled[features].corrwith(scaled["RUL"]).sort_values()

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "total_rows": int(len(train) + len(test)),
        "train_engines": int(train["engine"].nunique()),
        "test_engines": int(test["engine"].nunique()),
        "per_subset": {
            s: {
                "train_rows": int((train["subset"] == s).sum()),
                "train_engines": int(train.loc[train["subset"] == s, "engine"].nunique()),
                "conditions": int(train.loc[train["subset"] == s, "condition"].nunique()),
                "median_life_cycles": float(
                    train[train["subset"] == s].groupby("engine")["cycle"].max().median()
                ),
            }
            for s in sorted(train["subset"].unique())
        },
        "sensors_dropped_flat": dropped,
        "sensors_used": features,
        "corr_with_rul": {k: round(float(v), 3) for k, v in corr.items()},
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "corr_with_rul"}, indent=2))

    # Chart 1: correlation of each sensor with RUL (one series, one hue)
    order = corr.reindex(corr.abs().sort_values().index)
    fig, ax = plotting.plt.subplots(figsize=(6.4, 4.6))
    ax.barh(order.index, order.values, color=plotting.BLUE, height=0.7)
    ax.axvline(0, color=plotting.AXIS, linewidth=1)
    for y, v in enumerate(order.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color=plotting.INK_2)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson correlation with remaining useful life")
    ax.set_title("Which sensors track engine wear")
    ax.grid(axis="y", visible=False)
    plotting.save(fig, FIG / "sensor_correlation.png")

    # Chart 2: average drift of the 4 strongest sensors, aligned at failure (FD001)
    top = list(corr.abs().sort_values(ascending=False).index[:4])
    fd1 = scaled[scaled["subset"] == "FD001"].copy()
    fd1["to_failure"] = fd1.groupby("engine")["cycle"].transform("max") - fd1["cycle"]
    fd1 = fd1[fd1["to_failure"] <= 200]
    fig, axes = plotting.plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for ax, s in zip(axes.ravel(), top):
        g = fd1.groupby("to_failure")[s]
        x = -g.mean().index.to_numpy()
        ax.fill_between(x, g.quantile(0.25).to_numpy(), g.quantile(0.75).to_numpy(),
                        color=plotting.BLUE, alpha=0.18, linewidth=0)
        ax.plot(x, g.mean().to_numpy(), color=plotting.BLUE)
        ax.set_title(f"Sensor {s}", fontsize=9.5)
        ax.set_ylabel("z-score")
    for ax in axes[1]:
        ax.set_xlabel("Cycles before failure")
    fig.suptitle("Sensors drift as engines approach failure (FD001, mean and IQR)",
                 fontsize=11, fontweight="bold", color=plotting.INK)
    plotting.save(fig, FIG / "sensor_drift_before_failure.png")


if __name__ == "__main__":
    main()
