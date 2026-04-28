# CS495 - Real Time Truck ETA prediction

# Project Description

Using telematics to build a real-time model and streaming service to predict estimated time of arrival (ETA) and uncertainty.

# Objectives

• ETA Model: Quantile regression (P10/P50/P90) to predict estimated time of arrival

• Streaming Inference: Real-time state per (VIN, TripId) with continuous updates at each telemetry point

• Backtesting: Replay historical trip streams, evaluate stability smoothing, and ablation study

# Dataset

The `sample_trucks_dataset.csv` file contains real-time telematics records used for model training and evaluation. Each row represents a single telemetry ping from a truck. The tracked variables are described below:

| Variable | Description |
|---|---|
| `VIN` | Unique identifier for each truck (Vehicle Identification Number / Truck ID) |
| `Timestamp` | Date and time of the telemetry ping, used to compute elapsed trip time and derive time-of-day features |
| `Latitude` | GPS latitude coordinate of the truck at the time of the ping |
| `Longitude` | GPS longitude coordinate of the truck at the time of the ping |
| `Speed` | Instantaneous speed of the truck (mph) at the time of the ping |
| `Weight_lbs` | Reported payload weight of the truck in pounds at the time of the ping |
| `Device_Type` | Type of telematics device installed on the truck (e.g., OEM) |
| `Source` | Manufacturer or data source of the telematics device (e.g., Volvo) |

# Tools / Technologies

Tools: Python, LightGBM/CatBoost, DBSCAN/HDBSCAN, H3/geohash spatial indexing, Scikit-learn, Streamlit

# How to Run

TBD

# Timeline

Minimal implementation roadmap (8–10 weeks)
Build historical trip table → derive trip features

Train ETA quantile regressor (P10/P50/P90)

Build streaming feature state + inference loop

Backtest on replay streams + stability smoothing

Package as a small library + demo UI

Attribution Requirement: Any academic, research, or commercial usage must cite the original repository and authors.
