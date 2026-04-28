"""Evaluation metrics for ETA prediction.

Provides MAE, RMSE, MAPE, pinball (quantile) loss, and coverage checks
for P10/P50/P90 quantile predictions.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Rows where ``y_true == 0`` are excluded to avoid division by zero.
    """
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile *q* in (0, 1)."""
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def coverage(y_true: np.ndarray, y_low: np.ndarray, y_high: np.ndarray) -> float:
    """Fraction of actuals that fall inside [y_low, y_high]."""
    inside = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(inside))


def evaluate(
    y_true: np.ndarray,
    p50: np.ndarray,
    p10: np.ndarray | None = None,
    p90: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a full evaluation report.

    Parameters
    ----------
    y_true : array-like
        Ground-truth ETA values (seconds or minutes — consistent units).
    p50 : array-like
        Median (P50) predictions.
    p10, p90 : array-like, optional
        Lower and upper quantile predictions for coverage/pinball.

    Returns
    -------
    dict with metric name → value.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p50 = np.asarray(p50, dtype=np.float64)

    report: dict[str, float] = {
        "mae": mae(y_true, p50),
        "rmse": rmse(y_true, p50),
        "mape": mape(y_true, p50),
    }

    if p10 is not None:
        p10 = np.asarray(p10, dtype=np.float64)
        report["pinball_p10"] = pinball_loss(y_true, p10, 0.1)

    if p90 is not None:
        p90 = np.asarray(p90, dtype=np.float64)
        report["pinball_p90"] = pinball_loss(y_true, p90, 0.9)

    if p10 is not None and p90 is not None:
        report["coverage_p10_p90"] = coverage(y_true, p10, p90)

    return report
