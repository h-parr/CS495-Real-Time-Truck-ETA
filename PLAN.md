# PLAN.md

> **Single source of truth** for scope, goals, architecture, and progress.
> Update this file as the project evolves.

---

## Project Overview

**Title:** Real-Time Truck ETA Prediction  
**Author:** Harri  
**Course:** CS 495 — Capstone Project  
**Date:** April 27, 2026

### Description

A machine learning pipeline that ingests raw **GPS/telematics** pings from commercial trucks, segments them into discrete trips, engineers time and movement features, and trains a quantile regression model to predict **estimated time of arrival (ETA)** with uncertainty bounds (P10/P50/P90). Predictions update continuously at every new telemetry ping per truck.

### Objectives

- Build a quantile ETA model with MAE < 10 minutes on the P50 (median) estimate
- Deliver calibrated uncertainty: ≥ 80% of actual arrivals fall inside the P10–P90 interval
- Build a streaming inference loop that updates ETA at every telemetry ping per (VIN, TripId)
- Backtest the model on replayed historical trip streams and demonstrate stable predictions
- Deploy an interactive Streamlit dashboard showing live ETA and confidence band on a map

---

## Environment Setup

### Requirements

- Python 3.13
- pip + virtualenv (via `venv`)
- GNU Make (for Makefile targets)
- Git + Git LFS (large CSV files tracked via LFS)

### Dependencies (`requirements.txt`)

```
numpy>=2.0
pandas>=2.2
scikit-learn>=1.5
xgboost>=2.1
lightgbm>=4.4
ruptures>=1.1
matplotlib>=3.9
seaborn>=0.13
joblib>=1.4
pyarrow>=16.0
```

Dev extras (installed by `make install-dev`): `pytest`, `pytest-cov`, `ruff`

---

### Quick Start (macOS)

```bash
git clone https://github.com/h-parr/CS495-Real-Time-Truck-ETA.git
cd CS495-Real-Time-Truck-ETA
make install          # creates .venv with Python 3.13 and installs requirements
source .venv/bin/activate
make test             # verify everything works
```

Or manually without Make:

```bash
brew install python@3.13   # if not already installed
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Quick Start (Linux)

```bash
git clone https://github.com/h-parr/CS495-Real-Time-Truck-ETA.git
cd CS495-Real-Time-Truck-ETA
make install
source .venv/bin/activate
make test
```

Or manually without Make:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3.13 python3.13-venv make -y
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Quick Start (Windows)

```powershell
git clone https://github.com/h-parr/CS495-Real-Time-Truck-ETA.git
cd CS495-Real-Time-Truck-ETA
make install          # requires GNU Make (e.g. via winget install GnuWin32.Make)
.venv\Scripts\Activate.ps1
make test
```

Or manually without Make:

```powershell
# Install Python 3.13 from https://www.python.org/downloads/
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Makefile Alternative (Windows — no Make installed)

```powershell
python data_audit.py
python trip_segmentation.py
python eta_model.py
python -m pytest tests/
python -m ruff check .
```

---

## Architecture

### Folder Structure

```
CS495-Real-Time-Truck-ETA/
├── data_audit.py              # read-only data quality check
├── trip_segmentation.py       # split raw pings into discrete trips
├── features.py                # feature engineering (shared by train + inference)
├── eta_model.py               # model training, inference, and persistence
├── metrics.py                 # MAE, RMSE, quantile loss, coverage evaluation
├── sample_trucks_dataset.csv  # raw telematics pings (Git LFS)
├── segmented_trips.csv        # output of trip_segmentation.py (Git LFS)
├── requirements.txt           # pip dependencies
├── Makefile                   # cross-platform build and run commands
├── PLAN.md                    # this file
└── tests/                     # pytest test suite
```

### Data Flow

```
raw telemetry CSV
       │
       ▼
 data_audit.py          ← read-only quality check
       │
       ▼
 trip_segmentation.py   ← split pings → trips → segmented_trips.csv
       │
       ▼
 features.py            ← transform trip rows → model-ready feature vectors
       │
       ▼
 eta_model.py           ← train / load quantile regressor, run inference
       │
       ▼
 metrics.py             ← evaluate MAE / RMSE / quantile loss / coverage
       │
       ▼
 streaming loop         ← stateful per-(VIN, TripId) live prediction
       │
       ▼
 Streamlit UI           ← map + ETA confidence band visualization
```

---

## Tasks & Milestones

### Phase 1 — Data Foundation 🔄
- [x] Write `data_audit.py` — missingness, duplicates, out-of-order pings, per-VIN counts
- [ ] Write `trip_segmentation.py` — detect trip boundaries using time gaps and speed
- [ ] Produce `segmented_trips.csv` — cleaned, trip-labeled dataset

### Phase 2 — Feature Engineering 🔄
- [ ] Implement feature transforms in `features.py`
  - [ ] Elapsed trip time since first ping
  - [ ] Speed statistics: rolling mean, rolling std, max speed so far
  - [ ] Distance remaining (haversine to estimated destination)
  - [ ] Time-of-day encoding (hour, sin/cos cyclical)
  - [ ] Day-of-week encoding
  - [ ] Payload weight flag (sparse feature — present vs absent)
- [ ] Write unit tests for all feature transforms

### Phase 2b — Load State Detection (Change Point Detection) 🔲
- [ ] Create `load_state.py` — change point detection module
  - [ ] Apply PELT algorithm (via `ruptures`) to per-trip speed signal to detect load/unload events
  - [ ] Use `Weight_lbs` readings (where available) as ground-truth labels to validate detected change points
  - [ ] Emit a `LoadState` label per ping: `loaded`, `unloaded`, or `unknown`
  - [ ] Derive `load_change_count` feature (number of state transitions per trip)
  - [ ] Add `LoadState` and `load_change_count` as features in `features.py`
- [ ] Evaluate CPD accuracy against known weight events
- [ ] Write unit tests for change point detection edge cases (flat signal, single ping, no weight data)

### Phase 3 — Modeling 🔲
- [ ] Train/validation/test split (trip-level, not row-level, to prevent leakage)
- [ ] Implement baseline: linear regression on elapsed time
- [ ] Train LightGBM quantile regressor (P10, P50, P90)
- [ ] Train XGBoost quantile regressor for comparison
- [ ] Hyperparameter tuning with cross-validation (learning rate, depth, leaves)
- [ ] Persist best model with `joblib`

### Phase 4 — Evaluation 🔲
- [ ] Compute MAE, RMSE, MAPE on P50 predictions
- [ ] Compute pinball loss on P10 and P90
- [ ] Coverage check: % of actuals inside P10–P90 interval (target ≥ 80%)
- [ ] Compare model versions in `metrics.py`

### Phase 5 — Streaming Inference 🔲
- [ ] Stateful per-(VIN, TripId) feature accumulator
- [ ] Inference loop: emit updated ETA at every new ping
- [ ] Unit tests for state management and edge cases (short trips, GPS gaps)

### Phase 6 — Backtesting & Stability 🔲
- [ ] Replay engine: feed historical trips ping-by-ping
- [ ] Track ETA drift over trip lifetime
- [ ] Apply exponential moving average smoothing
- [ ] Ablation study: measure impact of removing each feature group

### Phase 7 — Demo & Delivery 🔲
- [ ] Build Streamlit dashboard: live map + ETA band per truck
- [ ] Add interactive prediction form (VIN selector, trip playback)
- [ ] Final README polish, usage examples, and attribution
- [ ] Record demo video or prepare live walkthrough

---

## Methods & Models

### Data Processing
- `pandas` for data loading and manipulation
- `pyarrow` for fast CSV / parquet I/O
- Custom trip segmentation using time-gap and speed thresholds
- `ruptures` for PELT change point detection (load state inference)
- `scikit-learn` preprocessing: `StandardScaler`, `OrdinalEncoder`

### Load State Detection

Since `Weight_lbs` is sparse and event-driven (not present on every ping), a change point detection (CPD) algorithm is used to infer truck load state from the speed signal instead.

- **Algorithm:** PELT (Pruned Exact Linear Time) — detects structural breaks in a time series with optimal cost minimization
- **Signal:** per-trip speed sequence (and optionally acceleration derived from consecutive pings)
- **Validation:** when `Weight_lbs` readings are present, use them as ground-truth to tune the penalty parameter
- **Output:** `LoadState` label (`loaded` / `unloaded` / `unknown`) per ping, plus `load_change_count` per trip
- **Library:** `ruptures` (Python change point detection library)
- **Fallback:** if speed signal is too flat or trip is too short, default to `unknown`

### Machine Learning Models

| Model | Role | Library |
|---|---|---|
| Linear Regression | Baseline | scikit-learn |
| LightGBM Quantile | Primary ETA model (P10/P50/P90) | lightgbm |
| XGBoost Quantile | Comparison model | xgboost |

### Evaluation Metrics

- **MAE** — Mean Absolute Error on P50 (primary point estimate)
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **Pinball Loss** — Quantile loss at P10 and P90
- **Coverage** — % of actual arrivals inside P10–P90 interval (target: ≥ 80%)

### Visualization
- `matplotlib` / `seaborn` — EDA and training diagnostics
- `Streamlit` — interactive demo dashboard with map and ETA confidence band

---

## Data Sources

### Primary Dataset

`sample_trucks_dataset.csv` — raw telematics pings (tracked via Git LFS)

| Column | Type | Notes |
|---|---|---|
| `VIN` | string | Unique truck identifier |
| `Timestamp` | datetime | Ping time — used for trip segmentation and elapsed time features |
| `Latitude` | float | GPS latitude |
| `Longitude` | float | GPS longitude |
| `Speed` | float | mph at ping time |
| `Weight_lbs` | float | Sparse / event-driven — not present on every ping |
| `Device_Type` | string | OEM telematics device category |
| `Source` | string | Telematics manufacturer (e.g. Volvo) |

### Derived Dataset

`segmented_trips.csv` — output of `trip_segmentation.py`, adds `TripId` and trip boundary markers to each ping.

### Labels

ETA labels are derived from the data itself: the timestamp of the final ping of each trip is used as the ground-truth arrival time.

---

## Testing Strategy

- All tests live in `tests/` and run with `make test` (or `python -m pytest tests/ -v`)
- **Unit tests** — individual functions in `features.py`, `trip_segmentation.py`, and `metrics.py`
- **Integration tests** — end-to-end pipeline: raw CSV → segmented trips → features → predictions
- **Model validation** — held-out test set (trip-level split); assert MAE < 10 min and coverage ≥ 80%
- **Linting** — `ruff` via `make lint`; enforced in CI
- **Data quality** — `data_audit.py` run as a smoke test before every training run

---

## Timeline

| Week | Dates | Goals |
|---|---|---|
| 1 | Apr 21 – Apr 27 | Data audit ✅, repo setup ✅ |
| 2 | Apr 28 – May 4 | Trip segmentation (`trip_segmentation.py`), unit tests |
| 3 | May 5 – May 11 | Feature engineering (`features.py`), load state detection (`load_state.py` — PELT CPD) |
| 4 | May 12 – May 18 | Validate CPD against weight readings; integrate `LoadState` into feature set |
| 5 | May 19 – May 25 | Baseline model, LightGBM training, initial metrics |
| 6 | May 26 – Jun 1 | XGBoost comparison, hyperparameter tuning, full evaluation |
| 7 | Jun 2 – Jun 8 | Streaming inference loop, backtesting replay engine, state management tests |
| 8 | Jun 9 – Jun 15 | Streamlit dashboard, stability smoothing, ablation study |
| **Final** | **Jun 16 – Jun 17** | **Documentation polish, final README, submission ✅** |

---

## Conventions

- Feature logic lives exclusively in `features.py` — training and inference both import from it
- No hardcoded file paths — use `pathlib.Path` and CLI args
- Tests in `tests/`, run with `make test`
- Do not commit model artifacts (`.pkl`, `.bin`) or untracked large files — use Git LFS for CSVs

---

*Last updated: 2026-04-27 — Final submission deadline: June 17, 2026*
