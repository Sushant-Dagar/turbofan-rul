import numpy as np


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 scoring function (Saxena et al., 2008). Lower is better.

    It is asymmetric: predicting failure too late (d > 0) is penalised more than
    predicting it too early, because a late prediction means an engine flies
    past its safe point, while an early one only costs an extra inspection.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def all_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(rmse(y_true, y_pred), 2),
        "mae": round(mae(y_true, y_pred), 2),
        "nasa_score": round(nasa_score(y_true, y_pred), 1),
        "n": int(len(y_true)),
    }
