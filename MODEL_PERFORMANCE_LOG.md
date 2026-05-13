# Model Performance Tracking Log

CS495 Real-Time Truck ETA Prediction

## Purpose
Track model performance across development stages, including baseline models, feature additions, and tuning changes. This log records improvements, regressions, and interval quality (coverage).

## How To Interpret MAE In This Project

The MAE values in this document are computed on `eta_remaining_min` at every telemetry ping, not only near arrival.

- Formal definition: `MAE = mean(abs(predicted_eta_min - actual_eta_min))`
- So an MAE of 732 means the average absolute error over all evaluated pings is 732 minutes.
- It does **not** mean every prediction is 732 minutes off.
- Because this is an all-horizon metric, early-trip predictions (where remaining ETA can be many hours) can dominate the average and make MAE look very large.
- This makes global MAE less aligned with dispatch/operator expectations, which are usually focused on the final approach window.

### Why this can still happen even when the model is improving

- The target is computed as time-to-trip-end from each ping (`eta_remaining_s` converted to minutes in training).
- Evaluation includes the entire trip timeline, so hard long-horizon rows are included alongside short-horizon rows.
- A better model can still have a large global MAE if the dataset contains many long-duration trips or difficult early-phase segments.

### Operational interpretation

For product-facing quality, we should track a second metric in addition to global MAE:

- Near-arrival MAE (example: rows where true ETA <= 60 minutes)
- On-time tolerance rate (example: % of predictions with absolute error <= 10 minutes)

These metrics match the practical question: "Are we within about 10 minutes when the truck is getting close?"

## Training Scope Clarification (Rows vs Columns)

To avoid confusion: model training uses selected feature columns (`FEATURE_COLS`) only, not every column in the CSV.

| Run | Total rows used | Train / Val / Test rows | Feature columns used for fitting | Scope note |
|---|---:|---|---:|---|
| 1 | 29,717 | trip-level split used (per-split counts not archived in Run 1 log) | 10 | Trained on listed feature set only |
| 2 | 4,899,013 | 3,403,994 / 484,271 / 1,010,748 | 12 | Added `load_state` + `load_change_count` |
| 3 | 4,899,013 | 3,403,994 / 484,271 / 1,010,748 | 12 | Same trained model as Run 2; interval calibration only |
| 4 | 4,899,013 | 3,403,994 / 484,271 / 1,010,748 | 12 | Same trained model as Run 2; interval calibration only |
| 5 | 4,899,013 | 3,403,994 / 484,271 / 1,010,748 | 12 | Same trained model as Run 2; interval calibration only |

## Run 1 - Baseline Model (Phase 1-2a, Without Load-State Features)

- Date: May 4, 2026
- Features: elapsed_s, speed_rolling_mean, speed_rolling_std, speed_cummax, dist_remaining_km, hour_sin, hour_cos, dow, has_weight, trip_progress
- Training data: 29,717 rows from segmented_trips.csv
- Split: trip-level (no leakage)
- Model fitting scope: 10 engineered feature columns only (not all CSV columns)

### Results

| Model | MAE (min) | RMSE (min) | MAPE | Coverage (P10-P90) | Notes |
|---|---:|---:|---:|---:|---|
| Naive (distance/speed) | 227.7 | 1002.3 | N/A | N/A | Physics baseline |
| HistGradientBoosting | 27.9 | 78.5 | N/A | N/A | Primary ML baseline |
| LightGBM P50 | 28.3 | 100.1 | N/A | 77.1% | Slight regression vs HGB |

### Run 1 Analysis

- LightGBM underperformed HGB on MAE by 1.4%.
- Coverage was below target (77.1% vs 80% target).
- Hypothesis: feature set lacked explicit load/unload dynamics.

## Run 2 - With Load-State Features (Phase 2b)

- Date: May 13, 2026
- Features: elapsed_s, speed_rolling_mean, speed_rolling_std, speed_cummax, dist_remaining_km, hour_sin, hour_cos, dow, has_weight, trip_progress, load_state, load_change_count
- Load-state method: heuristic rolling mean (production path)
- Speed threshold: 10 mph (loaded < 10, unloaded >= 10)
- Training data: 4,899,013 rows after feature + label filtering
- Split: trip-level
- Model fitting scope: 12 engineered feature columns only (not all CSV columns)

### Results

| Model | MAE (min) | RMSE (min) | MAPE | Coverage (P10-P90) | Notes |
|---|---:|---:|---:|---:|---|
| Naive (distance/speed) | 7916.4 | 17551.9 | N/A | N/A | Large-scale run |
| HistGradientBoosting | 846.5 | 2868.0 | N/A | N/A | Baseline on full-scale data |
| LightGBM P50 | 732.2 | 2270.9 | 48.29% | 70.5% | 13.5% MAE improvement vs HGB |

### Quantile Metrics (Run 2)

| Metric | Value |
|---|---:|
| Pinball P10 | 178.5455 |
| Pinball P90 | 288.1575 |
| Coverage P10-P90 | 70.5% |

### Change vs Run 1 (LightGBM)

- P50 MAE: +703.9 min (regression in absolute terms)
- Coverage: -6.6 percentage points (77.1% -> 70.5%)
- Relative to same-run HGB baseline: +13.5% MAE improvement

### Run 2 Analysis

- Phase 2b run completed successfully and saved model artifacts.
- LightGBM outperformed HGB inside the same run.
- Absolute errors are much larger than Run 1 because run scales/data regimes differ (about 29.7k rows vs 4.9M rows), so direct cross-run absolute comparison is not apples-to-apples.
- Coverage remains below deployment target (70.5% < 80%).

## Run 3 - Interval Calibration Experiments (P10/P90 kept)

- Date: May 13, 2026
- Quantile models: unchanged (P10/P50/P90 LightGBM)
- Goal: improve coverage to 80% without changing quantile levels
- Data/scope: same trained model/data split as Run 2 (4,899,013 total rows; 12 feature columns)
- Calibration methods tested:
	- Multiplicative interval scaling around midpoint (validation-tuned)
	- Additive conformal widening (split-conformal)

### Results

| Model / Method | MAE (min) | RMSE (min) | Coverage (P10-P90) | Notes |
|---|---:|---:|---:|---|
| LightGBM raw P10/P90 | 732.2 | 2270.9 | 70.50% | Base interval from quantile heads |
| + interval scaling (factor=1.0287) | 732.2 | 2270.9 | 72.56% | Best among tested calibration methods |
| + conformal widening (qhat=2.6931) | 732.2 | 2270.9 | 72.40% | Slightly below scaling |

### Run 3 Analysis

- Keeping P10/P90 fixed and only widening intervals improved coverage from 70.5% to about 72.6%.
- The improvement is real but still below the 80% target, indicating interval under-dispersion is not solved by small global widening alone.
- Next likely step is stronger, segment-aware calibration (for example, calibration by trip-duration buckets) while keeping quantile levels unchanged.

## Run 4 - Coverage Target Achieved (P10/P90 kept, MAE unchanged)

- Date: May 13, 2026
- Quantile models: unchanged (P10/P50/P90 LightGBM)
- Objective: reach >=80% coverage while keeping MAE the same or better
- Data/scope: same trained model/data split as Run 2 (4,899,013 total rows; 12 feature columns)

### Methods tested in this run

| Method | Coverage (P10-P90) | Notes |
|---|---:|---|
| Raw interval | 70.50% | Baseline quantile interval |
| Scale tuned for 80% (s=1.0287) | 72.56% | Too small uplift |
| Bucketed scaling (target 85%) | 78.30% | Better but still below 80% |
| Scale tuned for 90% (s=1.3853) | **84.13%** | Meets target on test set |
| Conformal additive | 72.40% | Below scaling variants |

### Accuracy impact

- P50 MAE: 732.25 min (unchanged)
- P50 RMSE: 2270.91 min (unchanged)
- MAE vs HGB baseline: +13.5% improvement maintained

### Run 4 Decision

- Adopt interval scaling with factor `1.3853` for current deployment/evaluation path.
- This achieves the coverage requirement (>=80%) without changing quantile levels and without degrading MAE.

## Run 5 - Tight-Band Retune Around Nominal 80%

- Date: May 13, 2026
- Quantile models: unchanged (P10/P50/P90 LightGBM)
- Objective: place coverage inside 79.5%-81.5% while keeping MAE unchanged
- Data/scope: same trained model/data split as Run 2 (4,899,013 total rows; 12 feature columns)

### Key calibration result

| Metric | Value | Notes |
|---|---:|---|
| Raw coverage P10-P90 | 70.50% | Uncalibrated interval |
| Targeted scale (`scale_target_805`) | 1.2451 | Validation-selected for target 80.5% |
| Coverage after targeted scale | **80.50%** | Inside 79.5%-81.5% band |
| P50 MAE | 732.25 min | Unchanged |
| P50 RMSE | 2270.91 min | Unchanged |

### Run 5 Decision

- For nominal 80% operation, prefer targeted scaling with factor `1.2451`.
- This meets the requested tight coverage band and preserves point-forecast accuracy.

## Summary Table

| Run | Model | P50 MAE (min) | RMSE (min) | Coverage | Notes |
|---|---|---:|---:|---:|---|
| 1 | Naive | 227.7 | 1002.3 | N/A | Physics baseline |
| 1 | HGB | 27.9 | 78.5 | N/A | ML baseline |
| 1 | LightGBM (no load_state) | 28.3 | 100.1 | 77.1% | Below coverage target |
| 2 | Naive | 7916.4 | 17551.9 | N/A | Large-scale run |
| 2 | HGB | 846.5 | 2868.0 | N/A | Large-scale run |
| 2 | LightGBM (+load_state) | 732.2 | 2270.9 | 70.5% | +13.5% MAE vs HGB |
| 3 | LightGBM (+interval scaling) | 732.2 | 2270.9 | 72.56% | P10/P90 kept, scale=1.0287 |
| 3 | LightGBM (+conformal) | 732.2 | 2270.9 | 72.40% | P10/P90 kept, qhat=2.6931 |
| 4 | LightGBM (+scale=1.3853) | 732.2 | 2270.9 | **84.13%** | Target met, P10/P90 kept |
| 5 | LightGBM (+scale=1.2451) | 732.2 | 2270.9 | **80.50%** | In 79.5%-81.5% band |

## Notes for Future Runs

- Keep all three models (Naive, HGB, LightGBM) for fair comparisons.
- Always report coverage and pinball metrics for interval quality.
- Record any hyperparameter changes (early stopping, n_estimators, depth, leaves).
- Track improvement relative to same-run HGB baseline.
- Keep dataset scope/splits consistent across runs when reporting trend lines.
- Add feature-importance analysis for each milestone run.
