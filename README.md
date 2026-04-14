# CS495 - Real Time Prediction of Vehicle Final Stop and ETA 

# Project Description

Using telematics to build a real-time model and streaming service to predict final stop locations, final timestamps, and uncertainty.

# Objectives

• Stop Clustering: Build stop database from historical trips using DBSCAN/HDBSCAN on final GPS points
• Destination Classifier: Train over candidate stops with streaming telematics features
• ETA Model: Quantile regression (P10/P50/P90) conditioned on top-k predicted destinations
• Streaming Inference: Real-time state per (VIN, TripId) with continuous updates at each telemetry point
• Backtesting: Replay historical trip streams, evaluate stability smoothing, and ablation study

# Tools / Technologies

Tools: Python, LightGBM/CatBoost, DBSCAN/HDBSCAN, H3/geohash spatial indexing, Scikit-learn, Streamlit

# How to Run

TBD

# Timeline

Minimal implementation roadmap (8–10 weeks)
Build historical trip table → derive final stop points

Cluster final stops into stop IDs (DBSCAN/HDBSCAN)

Train destination classifier (LightGBM/CatBoost)

Train ETA quantile regressor conditioned on top stop(s)

Build streaming feature state + inference loop

Backtest on replay streams + stability smoothing

Package as a small library + demo UI

Attribution Requirement: Any academic, research, or commercial usage must cite the original repository and authors.
