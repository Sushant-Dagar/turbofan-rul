"""Train and evaluate remaining-useful-life (RUL) models on all four C-MAPSS sub-datasets.

Pipeline
  1. Load FD001-FD004 (160,359 training rows), label RUL capped at 125 cycles
  2. Drop flat sensors, z-score each sensor per flight condition (train stats only)
  3. Hold out 20% of engines (not rows) for validation
  4. Build 30-cycle sliding windows
  5. Fit a mean baseline, a Random Forest on window summaries, and a TensorFlow LSTM
  6. Score all three on NASA's official test engines (RMSE, MAE, PHM08 score)

Run:  python train.py            (full run, about 10-20 min on a laptop CPU)
      python train.py --quick    (2 epochs, small forest; checks everything works)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rul import plotting
from rul.data import (
    RUL_CAP,
    ConditionScaler,
    informative_sensors,
    load_all,
    make_windows,
    split_engines,
    window_summary,
)
from rul.metrics import all_metrics

RAW = Path("data/raw")
OUT = Path("reports")
FIG = OUT / "figures"
MODELS = Path("models")


def build_lstm(window: int, n_features: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def score_by_subset(y, pred, subsets) -> dict:
    out = {"ALL": all_metrics(y, pred)}
    for s in sorted(set(subsets)):
        m = subsets == s
        out[s] = all_metrics(y[m], pred[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 2

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    for d in (OUT, FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    plotting.setup()
    t0 = time.time()

    # 1-2. Load, label, select sensors, normalise per flight condition
    train_df, test_df = load_all(RAW)
    features = informative_sensors(train_df)
    fit_df, val_df = split_engines(train_df, val_frac=0.2, seed=args.seed)
    scaler = ConditionScaler().fit(fit_df, features)  # validation engines never seen here
    fit_df, val_df, test_s = (scaler.transform(d) for d in (fit_df, val_df, test_df))
    print(f"Rows: {len(train_df):,} train, {len(test_df):,} test | "
          f"engines: {fit_df.engine.nunique()} fit, {val_df.engine.nunique()} val, "
          f"{test_df.engine.nunique()} test | sensors used: {len(features)}")

    # 4. Windows
    W = args.window
    Xtr, ytr, _, _ = make_windows(fit_df, features, W)
    Xva, yva, _, sva = make_windows(val_df, features, W)
    Xte, yte, ete, ste = make_windows(test_s, features, W, last_only=True)
    print(f"Windows: {len(Xtr):,} train, {len(Xva):,} val, {len(Xte)} test (one per test engine)")

    results, preds = {}, {}

    # 5a. Mean baseline: always predict the average training label
    base = np.full_like(yte, ytr.mean())
    preds["Mean baseline"] = base
    results["Mean baseline"] = score_by_subset(yte, base, ste)

    # 5b. Random Forest on window summaries (last value, mean, slope per sensor)
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=30 if args.quick else 150, min_samples_leaf=5, max_features=0.5,
        max_samples=0.3, n_jobs=-1, random_state=args.seed,
    )
    rf.fit(window_summary(Xtr), ytr)
    rf_val = rf.predict(window_summary(Xva))
    p = np.clip(rf.predict(window_summary(Xte)), 0, RUL_CAP)
    preds["Random Forest"] = p
    results["Random Forest"] = score_by_subset(yte, p, ste)
    print(f"Random Forest done ({time.time() - t0:.0f}s)")

    # 5c. LSTM in TensorFlow/Keras; target scaled to 0-1 for stable training
    model = build_lstm(W, len(features))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    hist = model.fit(
        Xtr, ytr / RUL_CAP, validation_data=(Xva, yva / RUL_CAP),
        epochs=args.epochs, batch_size=256, callbacks=callbacks, verbose=2,
    )
    lstm_val = model.predict(Xva, batch_size=2048, verbose=0).ravel() * RUL_CAP
    p = np.clip(model.predict(Xte, verbose=0).ravel() * RUL_CAP, 0, RUL_CAP)
    preds["LSTM (TensorFlow)"] = p
    results["LSTM (TensorFlow)"] = score_by_subset(yte, p, ste)
    model.save(MODELS / "lstm.keras")

    # 6. Save metrics, predictions, charts
    metrics = {
        "config": {"window": W, "rul_cap": RUL_CAP, "epochs_run": len(hist.history["loss"]),
                   "seed": args.seed, "sensors_used": features, "quick": args.quick},
        "data": {"train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
                 "train_windows": int(len(Xtr)), "val_windows": int(len(Xva)),
                 "test_engines": int(len(Xte))},
        "validation_rmse": {"Random Forest": round(float(np.sqrt(np.mean((rf_val - yva) ** 2))), 2),
                            "LSTM (TensorFlow)": round(float(np.sqrt(np.mean((lstm_val - yva) ** 2))), 2)},
        "test": results,
        "runtime_seconds": round(time.time() - t0),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"engine": ete, "subset": ste, "true_rul": yte,
                  **{f"pred_{k}": v for k, v in preds.items()}}).to_csv(
        OUT / "test_predictions.csv", index=False)

    make_charts(hist.history, results, yte, preds["LSTM (TensorFlow)"], ste)

    print("\nTest RMSE (cycles) by sub-dataset")
    print(pd.DataFrame({m: {s: r[s]["rmse"] for s in r} for m, r in results.items()}).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s")


def make_charts(history, results, yte, lstm_pred, ste):
    plt = plotting.plt

    # Training curve: RMSE in cycles per epoch
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ep = np.arange(1, len(history["rmse"]) + 1)
    tr = np.array(history["rmse"]) * RUL_CAP
    va = np.array(history["val_rmse"]) * RUL_CAP
    ax.plot(ep, tr, color="#2a78d6", label="Training")
    ax.plot(ep, va, color="#eb6834", label="Validation (held-out engines)")
    ax.text(ep[-1], tr[-1], "  Training", va="center", fontsize=8, color=plotting.INK_2)
    ax.text(ep[-1], va[-1], "  Validation", va="center", fontsize=8, color=plotting.INK_2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title("LSTM learning curve")
    ax.legend(loc="upper right")
    ax.set_xlim(1, ep[-1] + max(2, len(ep) * 0.18))
    plotting.save(fig, FIG / "lstm_learning_curve.png")

    # Test RMSE by sub-dataset and model (grouped bars, value labels)
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    models = list(plotting.MODEL_COLORS)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.26
    x = np.arange(len(subsets))
    for i, m in enumerate(models):
        vals = [results[m][s]["rmse"] for s in subsets]
        pos = x + (i - 1) * width
        ax.bar(pos, vals, width - 0.02, color=plotting.MODEL_COLORS[m], label=m)
        for xp, v in zip(pos, vals):
            ax.text(xp, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5, color=plotting.INK_2)
    ax.set_xticks(x, subsets)
    ax.set_ylabel("Test RMSE (cycles, lower is better)")
    ax.set_title("Prediction error on NASA's test engines")
    ax.legend(ncols=3, loc="upper left")
    ax.set_ylim(0, max(results["Mean baseline"][s]["rmse"] for s in subsets) * 1.2)
    ax.grid(axis="x", visible=False)
    plotting.save(fig, FIG / "test_rmse_by_subset.png")

    # Predicted vs true RUL for the LSTM, one panel per sub-dataset
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), sharey=True)
    for ax, s in zip(axes, subsets):
        m = ste == s
        ax.plot([0, RUL_CAP], [0, RUL_CAP], color=plotting.AXIS, linewidth=1)
        ax.scatter(yte[m], lstm_pred[m], s=12, color="#2a78d6", alpha=0.7,
                   edgecolors=plotting.SURFACE, linewidths=0.5)
        ax.set_title(f"{s}  (RMSE {results['LSTM (TensorFlow)'][s]['rmse']:.1f})", fontsize=9.5)
        ax.set_xlabel("True RUL")
        ax.set_xlim(-5, RUL_CAP + 5)
        ax.set_ylim(-5, RUL_CAP + 5)
    axes[0].set_ylabel("Predicted RUL")
    fig.suptitle("LSTM predictions vs truth (diagonal = perfect)", fontsize=11,
                 fontweight="bold", color=plotting.INK)
    plotting.save(fig, FIG / "lstm_pred_vs_true.png")


if __name__ == "__main__":
    main()
