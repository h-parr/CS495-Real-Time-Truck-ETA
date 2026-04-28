"""ETA model module.

Trains and evaluates quantile regression models to predict truck ETA.
Supports:
  - HistGradientBoostingRegressor baseline (scikit-learn)
  - LightGBM quantile regressor (P10 / P50 / P90)
  - Model persistence with joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from features import FEATURE_COLS, build_features
from metrics import evaluate

# ── Defaults ─────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
QUANTILES = (0.1, 0.5, 0.9)
RANDOM_STATE = 42


# ── Label derivation ─────────────────────────────────────────

def add_eta_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``eta_remaining_s``: seconds from current ping to trip end."""
    trip_end = df.groupby("TripId")["Timestamp"].transform("max")
    df["eta_remaining_s"] = (trip_end - df["Timestamp"]).dt.total_seconds()
    return df


# ── Train / test split (trip-level) ──────────────────────────

def trip_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on TripId to prevent data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(df, groups=df["TripId"]))

    df_test = df.iloc[test_idx]
    df_train_val = df.iloc[train_val_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["TripId"]))

    return df_train_val.iloc[train_idx], df_train_val.iloc[val_idx], df_test


# ── Baseline ─────────────────────────────────────────────────

def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingRegressor:
    """HistGradientBoosting baseline (squared_error)."""
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ── LightGBM quantile models ────────────────────────────────

def train_lgb_quantile(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.5,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM quantile regressor at level ``alpha``."""
    if not HAS_LGB:
        raise ImportError("lightgbm is required for quantile models")

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return model


# ── Pipeline ─────────────────────────────────────────────────

def run_pipeline(input_csv: str | Path) -> None:
    """Full training pipeline: features → split → train → evaluate → save."""
    print("Loading segmented trips …")
    df = pd.read_csv(input_csv, parse_dates=["Timestamp"])
    df.columns = df.columns.str.strip()

    print("Building features …")
    df = build_features(df)
    df = add_eta_label(df)

    # Drop rows with NaN in features or label
    df = df.dropna(subset=FEATURE_COLS + ["eta_remaining_s"]).reset_index(drop=True)

    # Convert label to minutes for interpretability
    df["eta_remaining_min"] = df["eta_remaining_s"] / 60.0

    print("Splitting (trip-level) …")
    train_df, val_df, test_df = trip_split(df)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["eta_remaining_min"].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["eta_remaining_min"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["eta_remaining_min"].values

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    # ── Baseline ──
    print("\nTraining HistGradientBoosting baseline …")
    baseline = train_baseline(X_train, y_train)
    bl_pred = baseline.predict(X_test)
    bl_report = evaluate(y_test, bl_pred)
    print(f"  Baseline MAE:  {bl_report['mae']:.2f} min")
    print(f"  Baseline RMSE: {bl_report['rmse']:.2f} min")

    # ── LightGBM quantile ──
    if HAS_LGB:
        models: dict[float, lgb.LGBMRegressor] = {}
        for q in QUANTILES:
            print(f"\nTraining LightGBM quantile α={q} …")
            models[q] = train_lgb_quantile(X_train, y_train, X_val, y_val, alpha=q)

        p10 = models[0.1].predict(X_test)
        p50 = models[0.5].predict(X_test)
        p90 = models[0.9].predict(X_test)

        lgb_report = evaluate(y_test, p50, p10, p90)
        print("\n── LightGBM Results ──")
        for k, v in lgb_report.items():
            print(f"  {k:<20} {v:.4f}")

        improvement = (bl_report["mae"] - lgb_report["mae"]) / bl_report["mae"] * 100
        print(f"\n  MAE improvement over baseline: {improvement:.1f}%")

        # ── Persist ──
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for q, m in models.items():
            out = MODEL_DIR / f"lgb_q{int(q*100):02d}.pkl"
            joblib.dump(m, out)
            print(f"  Saved {out}")
        joblib.dump(baseline, MODEL_DIR / "baseline_hgb.pkl")
        print(f"  Saved {MODEL_DIR / 'baseline_hgb.pkl'}")
    else:
        print("\nlightgbm not installed — skipping quantile models.")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train ETA models.")
    parser.add_argument(
        "--input",
        default=str(_root / "data" / "segmented_trips.csv"),
        help="Path to segmented trips CSV",
    )
    args = parser.parse_args()
    run_pipeline(args.input)
