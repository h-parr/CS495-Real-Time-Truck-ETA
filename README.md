# CS495 - Real Time Truck ETA prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-green)
![ETA](https://img.shields.io/badge/ETA-Quantile%20Regression-orange)
![Telematics](https://img.shields.io/badge/Telematics-Real%20Time-red)

# Project Description

Using telematics to build a real-time model and streaming service to predict estimated time of arrival (ETA) and uncertainty.

# Objectives

• ETA Model: Quantile regression (P10/P50/P90) to predict estimated time of arrival

• Streaming Inference: Real-time state per (ID, TripId) with continuous updates at each telemetry point

• Backtesting: Replay historical trip streams, evaluate stability smoothing, and ablation study

# Dataset

The `data/sample_trucks_dataset.csv` file contains real-time telematics records used for model training and evaluation. Each row represents a single telemetry ping from a truck. The tracked variables are described below:

| Variable | Description |
|---|---|
| `ID` | Unique identifier for each truck |
| `Timestamp` | Date and time of the telemetry ping, used to compute elapsed trip time and derive time-of-day features |
| `Latitude` | GPS latitude coordinate of the truck at the time of the ping |
| `Longitude` | GPS longitude coordinate of the truck at the time of the ping |
| `Speed` | Instantaneous speed of the truck (mph) at the time of the ping |
| `Weight_lbs` | Reported payload weight of the truck in pounds at the time of the ping |

# Tools / Technologies

Tools: Python, LightGBM/CatBoost, DBSCAN/HDBSCAN, H3/geohash spatial indexing, Scikit-learn, Streamlit

# Setup (Makefile Commands)

This project uses a cross-platform `Makefile` to streamline setup and common workflows on macOS/Linux and Windows (with GNU Make installed).

## 1) Create and Populate the Virtual Environment

```bash
make install
```

This will:
- create `.venv` using Python 3.13
- upgrade `pip`
- install packages from `requirements.txt`

For development tools (pytest/coverage/ruff):

```bash
make install-dev
```

## 2) Common Commands

```bash
make help      # list all available targets
make audit     # run data audit script
make segment   # run trip segmentation
make train     # run ETA model training
make test      # run pytest suite
make lint      # run ruff linter
make clean     # remove venv and Python cache files
make rebuild   # clean + install
```

## 3) Optional Maintenance Commands

```bash
make freeze    # write currently installed packages to requirements.txt
```

# Timeline

Minimal implementation roadmap (8–10 weeks)
Build historical trip table → derive trip features

Train ETA quantile regressor (P10/P50/P90)

Build streaming feature state + inference loop

Backtest on replay streams + stability smoothing

Package as a small library + demo UI

Attribution Requirement: Any academic, research, or commercial usage must cite the original repository and authors.
