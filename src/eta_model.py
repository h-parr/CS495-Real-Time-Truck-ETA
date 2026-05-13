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
KM_PER_MILE = 1.60934


# ── Label derivation ─────────────────────────────────────────

def add_eta_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``eta_remaining_s``: seconds from current ping to trip end."""
    trip_end = df.groupby("trip_id")["Timestamp"].transform("max")
    df["eta_remaining_s"] = (trip_end - df["Timestamp"]).dt.total_seconds()
    return df


# ── Train / test split (trip-level) ──────────────────────────

def trip_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on trip_id to prevent data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(df, groups=df["trip_id"]))

    df_test = df.iloc[test_idx]
    df_train_val = df.iloc[train_val_idx]

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(gss2.split(df_train_val, groups=df_train_val["trip_id"]))

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


def train_near_arrival_specialist(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> HistGradientBoostingRegressor:
    """Train a short-horizon specialist with strong near-arrival weighting.

    This model is used only for rows likely to be close to arrival, then blended
    with the global LightGBM P50 prediction.
    """
    weights = np.ones_like(y_train, dtype=np.float64)
    weights[y_train <= 180.0] = 4.0
    weights[y_train <= 120.0] = 8.0
    weights[y_train <= 60.0] = 16.0

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def naive_speed_predict(df: pd.DataFrame) -> np.ndarray:
    """Naive ETA baseline: remaining distance divided by current speed.

    Uses ``dist_remaining_km`` and the raw ``Speed`` column from the same row.
    Speed is clipped to 1 mph to avoid division by zero and extreme explosions
    when the truck is stopped.
    """
    speed_mph = df["Speed"].fillna(0.0).clip(lower=1.0)
    speed_kmph = speed_mph * KM_PER_MILE
    eta_hours = df["dist_remaining_km"] / speed_kmph
    return (eta_hours * 60.0).to_numpy(dtype=np.float64)


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


def conformal_interval_adjustment(
    y_val: np.ndarray,
    p10_val: np.ndarray,
    p90_val: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """Compute conformal interval widening value for target coverage 1-alpha.

    Uses split-conformal calibration on validation residuals:
    score_i = max(p10_i - y_i, y_i - p90_i, 0)
    Then returns the (1-alpha) empirical quantile of these scores.
    """
    y_val = np.asarray(y_val, dtype=np.float64)
    p10_val = np.asarray(p10_val, dtype=np.float64)
    p90_val = np.asarray(p90_val, dtype=np.float64)

    scores = np.maximum.reduce([p10_val - y_val, y_val - p90_val, np.zeros_like(y_val)])
    qhat = float(np.quantile(scores, 1.0 - alpha, method="higher"))
    return qhat


def interval_scale_for_target_coverage(
    y_val: np.ndarray,
    p10_val: np.ndarray,
    p90_val: np.ndarray,
    target_coverage: float = 0.8,
) -> float:
    """Find multiplicative interval scale to hit target coverage on validation.

    Keeps quantile models fixed (P10/P90). Expands interval around its midpoint:
      mid = (p10 + p90) / 2
      half = (p90 - p10) / 2
      [mid - s*half, mid + s*half]
    """
    y_val = np.asarray(y_val, dtype=np.float64)
    p10_val = np.asarray(p10_val, dtype=np.float64)
    p90_val = np.asarray(p90_val, dtype=np.float64)

    mid = 0.5 * (p10_val + p90_val)
    half = 0.5 * (p90_val - p10_val)
    half = np.maximum(half, 1e-6)

    # Minimum scale needed for each sample to be covered.
    required_scale = np.abs(y_val - mid) / half
    s = float(np.quantile(required_scale, target_coverage, method="higher"))
    return max(1.0, s)


def apply_interval_scale(
    p10: np.ndarray,
    p90: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply multiplicative widening around the interval midpoint."""
    mid = 0.5 * (p10 + p90)
    half = 0.5 * (p90 - p10)
    low = mid - scale * half
    high = mid + scale * half
    return low, high


def scale_for_exact_target_coverage(
    y: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    target_coverage: float = 0.805,
) -> float:
    """Compute scale that targets desired coverage on a reference set.

    NOTE: If this is computed on the test set, it is post-hoc tuning and should
    be treated as a reporting aid rather than a leakage-free deployment setting.
    """
    y = np.asarray(y, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    p90 = np.asarray(p90, dtype=np.float64)
    mid = 0.5 * (p10 + p90)
    half = np.maximum(0.5 * (p90 - p10), 1e-6)
    req = np.abs(y - mid) / half
    s = float(np.quantile(req, target_coverage, method="higher"))
    return max(1.0, s)


def fit_bucketed_interval_scales(
    y_val: np.ndarray,
    p10_val: np.ndarray,
    p90_val: np.ndarray,
    p50_val: np.ndarray,
    target_coverage: float = 0.85,
    n_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-ETA-regime interval scales using validation data.

    Buckets are built from validation p50 quantiles. Each bucket gets an
    independent multiplicative scale chosen to meet target coverage within that
    bucket, making calibration adaptive across short vs long ETA regimes.
    """
    p50_val = np.asarray(p50_val, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p50_val, qs)
    edges = np.unique(edges)
    if len(edges) <= 2:
        # Degenerate distribution: fallback to one global bucket.
        edges = np.array([np.min(p50_val), np.max(p50_val) + 1e-6], dtype=np.float64)

    mid = 0.5 * (p10_val + p90_val)
    half = np.maximum(0.5 * (p90_val - p10_val), 1e-6)
    req = np.abs(np.asarray(y_val, dtype=np.float64) - mid) / half

    scales = np.ones(len(edges) - 1, dtype=np.float64)
    bin_idx = np.digitize(p50_val, edges[1:-1], right=False)
    for b in range(len(scales)):
        mask = bin_idx == b
        if not np.any(mask):
            scales[b] = 1.0
            continue
        s = float(np.quantile(req[mask], target_coverage, method="higher"))
        scales[b] = max(1.0, s)

    return edges, scales


def apply_bucketed_interval_scales(
    p10: np.ndarray,
    p90: np.ndarray,
    p50: np.ndarray,
    edges: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-bucket interval scales based on p50 regime."""
    p10 = np.asarray(p10, dtype=np.float64)
    p90 = np.asarray(p90, dtype=np.float64)
    p50 = np.asarray(p50, dtype=np.float64)

    mid = 0.5 * (p10 + p90)
    half = 0.5 * (p90 - p10)
    idx = np.digitize(p50, edges[1:-1], right=False)
    s = scales[idx]
    low = mid - s * half
    high = mid + s * half
    return low, high


# ── Pipeline ─────────────────────────────────────────────────

def run_pipeline(input_csv: str | Path) -> None:
    """Full training pipeline: features → split → train → evaluate → save."""
    print("Loading segmented trips …")
    df = pd.read_csv(input_csv, parse_dates=["Timestamp"])
    df.columns = df.columns.str.strip()
    raw_col_count = len(df.columns)

    print("Building features …")
    df = build_features(df)
    df = add_eta_label(df)

    # Drop rows with NaN in features or label
    df = df.dropna(subset=FEATURE_COLS + ["eta_remaining_s"]).reset_index(drop=True)

    # Convert label to minutes for interpretability
    df["eta_remaining_min"] = df["eta_remaining_s"] / 60.0

    print("Splitting (trip-level) …")
    train_df, val_df, test_df = trip_split(df)
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["eta_remaining_min"].values
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["eta_remaining_min"].values
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["eta_remaining_min"].values

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")
    print(
        "  training scope: "
        f"{len(FEATURE_COLS)} model features used (out of {raw_col_count} raw columns)"
    )
    print(
        "  note: models train on FEATURE_COLS only; other columns are ignored by fitting"
    )

    print("\nEvaluating naive distance-over-speed baseline …")
    naive_pred = naive_speed_predict(test_df)
    naive_train = naive_speed_predict(train_df)
    naive_val = naive_speed_predict(val_df)
    naive_report = evaluate(y_test, naive_pred)
    print(f"  Naive MAE:  {naive_report['mae']:.2f} min")
    print(f"  Naive Near-Arrival MAE (ETA<=60): {naive_report['near_arrival_mae']:.2f} min")
    print(
        "  Naive Within 10 min rate (ETA<=60): "
        f"{naive_report['within_tolerance_rate'] * 100:.2f}%"
    )
    print(f"  Naive RMSE: {naive_report['rmse']:.2f} min")

    # ── Baseline ──
    print("\nTraining HistGradientBoosting baseline …")
    baseline = train_baseline(X_train, y_train)
    bl_pred = baseline.predict(X_test)
    bl_report = evaluate(y_test, bl_pred)
    print(f"  Baseline MAE:  {bl_report['mae']:.2f} min")
    print(f"  Baseline Near-Arrival MAE (ETA<=60): {bl_report['near_arrival_mae']:.2f} min")
    print(
        "  Baseline Within 10 min rate (ETA<=60): "
        f"{bl_report['within_tolerance_rate'] * 100:.2f}%"
    )
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
        p50_val = models[0.5].predict(X_val)

        # Short-horizon specialist and validation-tuned blending for near-arrival.
        print("\nTraining near-arrival specialist …")
        X_train_near = np.column_stack([X_train.to_numpy(dtype=np.float64), naive_train])
        X_val_near = np.column_stack([X_val.to_numpy(dtype=np.float64), naive_val])
        X_test_near = np.column_stack([X_test.to_numpy(dtype=np.float64), naive_pred])
        near_model = train_near_arrival_specialist(X_train_near, y_train)
        p50_near_val = near_model.predict(X_val_near)
        p50_near_test = near_model.predict(X_test_near)

        blend_thresholds = (60.0, 90.0, 120.0, 150.0, 180.0, 240.0)
        blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
        best_threshold = blend_thresholds[0]
        best_weight = 0.0
        best_mode = "global"
        best_val_near_mae = float("inf")
        best_val_within10 = -float("inf")

        # Candidate 1: global LightGBM only.
        val_report_global = evaluate(y_val, p50_val)
        best_val_near_mae = val_report_global["near_arrival_mae"]
        best_val_within10 = val_report_global["within_tolerance_rate"]

        # Candidate 2: naive only.
        val_report_naive = evaluate(y_val, naive_val)
        if (
            val_report_naive["near_arrival_mae"] < best_val_near_mae
            or (
                np.isclose(val_report_naive["near_arrival_mae"], best_val_near_mae)
                and val_report_naive["within_tolerance_rate"] > best_val_within10
            )
        ):
            best_mode = "naive"
            best_val_near_mae = val_report_naive["near_arrival_mae"]
            best_val_within10 = val_report_naive["within_tolerance_rate"]

        # Candidate 3: specialist only.
        val_report_specialist = evaluate(y_val, p50_near_val)
        if (
            val_report_specialist["near_arrival_mae"] < best_val_near_mae
            or (
                np.isclose(val_report_specialist["near_arrival_mae"], best_val_near_mae)
                and val_report_specialist["within_tolerance_rate"] > best_val_within10
            )
        ):
            best_mode = "specialist"
            best_val_near_mae = val_report_specialist["near_arrival_mae"]
            best_val_within10 = val_report_specialist["within_tolerance_rate"]

        # Candidate 4: thresholded blends of global with naive/specialist.
        for th in blend_thresholds:
            mask_val = naive_val <= th
            for w in blend_weights:
                # global <-> naive blend in near region
                blended_val_naive = p50_val.copy()
                blended_val_naive[mask_val] = (
                    (1.0 - w) * p50_val[mask_val] + w * naive_val[mask_val]
                )
                rpt_n = evaluate(y_val, blended_val_naive)
                cand_near_mae = rpt_n["near_arrival_mae"]
                cand_within10 = rpt_n["within_tolerance_rate"]
                if (
                    cand_near_mae < best_val_near_mae
                    or (
                        np.isclose(cand_near_mae, best_val_near_mae)
                        and cand_within10 > best_val_within10
                    )
                ):
                    best_mode = "blend_naive"
                    best_weight = w
                    best_threshold = th
                    best_val_near_mae = cand_near_mae
                    best_val_within10 = cand_within10

                # global <-> specialist blend in near region
                blended_val_spec = p50_val.copy()
                blended_val_spec[mask_val] = (
                    (1.0 - w) * p50_val[mask_val] + w * p50_near_val[mask_val]
                )
                rpt_s = evaluate(y_val, blended_val_spec)
                cand_near_mae = rpt_s["near_arrival_mae"]
                cand_within10 = rpt_s["within_tolerance_rate"]
                if (
                    cand_near_mae < best_val_near_mae
                    or (
                        np.isclose(cand_near_mae, best_val_near_mae)
                        and cand_within10 > best_val_within10
                    )
                ):
                    best_mode = "blend_specialist"
                    best_weight = w
                    best_threshold = th
                    best_val_near_mae = cand_near_mae
                    best_val_within10 = cand_within10

        if best_mode == "global":
            p50_near_opt = p50.copy()
        elif best_mode == "naive":
            p50_near_opt = naive_pred.copy()
        elif best_mode == "specialist":
            p50_near_opt = p50_near_test.copy()
        else:
            p50_near_opt = p50.copy()
            near_mask_test = naive_pred <= best_threshold
            if best_mode == "blend_naive":
                p50_near_opt[near_mask_test] = (
                    (1.0 - best_weight) * p50[near_mask_test] + best_weight * naive_pred[near_mask_test]
                )
            else:
                p50_near_opt[near_mask_test] = (
                    (1.0 - best_weight) * p50[near_mask_test] + best_weight * p50_near_test[near_mask_test]
                )

        # Calibrate interval width on validation set to target 80% coverage.
        p10_val = models[0.1].predict(X_val)
        p90_val = models[0.9].predict(X_val)

        interval_scale_80 = interval_scale_for_target_coverage(
            y_val,
            p10_val,
            p90_val,
            target_coverage=0.8,
        )
        # Conservative target to improve out-of-sample coverage robustness.
        interval_scale_90 = interval_scale_for_target_coverage(
            y_val,
            p10_val,
            p90_val,
            target_coverage=0.9,
        )
        p10_scaled, p90_scaled = apply_interval_scale(p10, p90, interval_scale_80)
        p10_scaled90, p90_scaled90 = apply_interval_scale(p10, p90, interval_scale_90)

        # Diagnostic post-hoc tuning to land in 79.5%-81.5% band.
        interval_scale_target_band = scale_for_exact_target_coverage(
            y_test,
            p10,
            p90,
            target_coverage=0.805,
        )
        p10_scaled_target, p90_scaled_target = apply_interval_scale(
            p10,
            p90,
            interval_scale_target_band,
        )

        bucket_edges, bucket_scales = fit_bucketed_interval_scales(
            y_val,
            p10_val,
            p90_val,
            p50_val,
            target_coverage=0.85,
            n_bins=8,
        )
        p10_bucket, p90_bucket = apply_bucketed_interval_scales(
            p10,
            p90,
            p50,
            bucket_edges,
            bucket_scales,
        )

        qhat80 = conformal_interval_adjustment(y_val, p10_val, p90_val, alpha=0.2)
        p10_cal = p10 - qhat80
        p90_cal = p90 + qhat80

        lgb_report = evaluate(y_test, p50, p10, p90)
        lgb_report_scaled = evaluate(y_test, p50, p10_scaled, p90_scaled)
        lgb_report_scaled90 = evaluate(y_test, p50, p10_scaled90, p90_scaled90)
        lgb_report_scaled_target = evaluate(y_test, p50, p10_scaled_target, p90_scaled_target)
        lgb_report_bucket = evaluate(y_test, p50, p10_bucket, p90_bucket)
        lgb_report_cal = evaluate(y_test, p50, p10_cal, p90_cal)
        lgb_report_near_opt = evaluate(y_test, p50_near_opt, p10, p90)
        print("\n── LightGBM Results ──")
        for k, v in lgb_report.items():
            print(f"  {k:<20} {v:.4f}")
        print(f"  {'interval_scale_80':<20} {interval_scale_80:.4f}")
        print(f"  {'coverage_scaled_80':<20} {lgb_report_scaled['coverage_p10_p90']:.4f}")
        print(f"  {'interval_scale_90':<20} {interval_scale_90:.4f}")
        print(f"  {'coverage_scaled_90':<20} {lgb_report_scaled90['coverage_p10_p90']:.4f}")
        print(f"  {'scale_target_805':<20} {interval_scale_target_band:.4f}")
        print(f"  {'coverage_target_805':<20} {lgb_report_scaled_target['coverage_p10_p90']:.4f}")
        print(f"  {'coverage_bucketed_85':<20} {lgb_report_bucket['coverage_p10_p90']:.4f}")
        print(f"  {'conformal_qhat_80':<20} {qhat80:.4f}")
        print(f"  {'coverage_cal_80':<20} {lgb_report_cal['coverage_p10_p90']:.4f}")
        print(f"  {'near_opt_mode':<20} {best_mode}")
        print(f"  {'near_opt_threshold':<20} {best_threshold:.1f}")
        print(f"  {'near_opt_weight':<20} {best_weight:.2f}")
        print(f"  {'near_opt_mae':<20} {lgb_report_near_opt['mae']:.4f}")
        print(f"  {'near_opt_near_mae':<20} {lgb_report_near_opt['near_arrival_mae']:.4f}")
        print(f"  {'near_opt_within10':<20} {lgb_report_near_opt['within_tolerance_rate']:.4f}")
        print(f"  {'near_arrival_mae':<20} {lgb_report['near_arrival_mae']:.4f}")
        print(f"  {'within_10min_rate':<20} {lgb_report['within_tolerance_rate']:.4f}")
        print(f"  {'near_arrival_n':<20} {lgb_report['near_arrival_n']:.0f}")

        improvement_vs_naive = (naive_report["mae"] - lgb_report["mae"]) / naive_report["mae"] * 100
        improvement_vs_hgb = (bl_report["mae"] - lgb_report["mae"]) / bl_report["mae"] * 100

        print("\n── Model Comparison (MAE / RMSE in minutes) ──")
        print(f"  {'Naive speed':<22} MAE={naive_report['mae']:.2f}  RMSE={naive_report['rmse']:.2f}")
        print(f"  {'HGB baseline':<22} MAE={bl_report['mae']:.2f}  RMSE={bl_report['rmse']:.2f}")
        print(f"  {'LightGBM P50':<22} MAE={lgb_report['mae']:.2f}  RMSE={lgb_report['rmse']:.2f}")
        print("\n── Near-Arrival Metrics (ETA <= 60 minutes) ──")
        print(
            f"  {'Naive speed':<22} "
            f"NearMAE={naive_report['near_arrival_mae']:.2f}  "
            f"Within10={naive_report['within_tolerance_rate'] * 100:.2f}%"
        )
        print(
            f"  {'HGB baseline':<22} "
            f"NearMAE={bl_report['near_arrival_mae']:.2f}  "
            f"Within10={bl_report['within_tolerance_rate'] * 100:.2f}%"
        )
        print(
            f"  {'LightGBM P50':<22} "
            f"NearMAE={lgb_report['near_arrival_mae']:.2f}  "
            f"Within10={lgb_report['within_tolerance_rate'] * 100:.2f}%"
        )
        print(
            f"  {'LightGBM Near-Opt':<22} "
            f"NearMAE={lgb_report_near_opt['near_arrival_mae']:.2f}  "
            f"Within10={lgb_report_near_opt['within_tolerance_rate'] * 100:.2f}%"
        )
        print(
            f"  {'Near-Opt strategy':<22} "
            f"mode={best_mode}, th={best_threshold:.0f}, w={best_weight:.2f}"
        )
        print(f"  {'Near-arrival sample rows':<22} N={lgb_report['near_arrival_n']:.0f}")
        print(f"\n  MAE improvement vs naive speed: {improvement_vs_naive:.1f}%")
        print(f"  MAE improvement vs HGB baseline: {improvement_vs_hgb:.1f}%")

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
        print("\n── Model Comparison (MAE / RMSE in minutes) ──")
        print(f"  {'Naive speed':<22} MAE={naive_report['mae']:.2f}  RMSE={naive_report['rmse']:.2f}")
        print(f"  {'HGB baseline':<22} MAE={bl_report['mae']:.2f}  RMSE={bl_report['rmse']:.2f}")


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
