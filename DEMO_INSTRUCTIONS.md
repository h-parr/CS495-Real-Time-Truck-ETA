# 🚚 Truck ETA Demo App - Quick Start

An interactive Streamlit visualization showing real-time truck ETA predictions with animated truck tracking.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install just the demo requirements:

```bash
pip install streamlit plotly scipy pandas numpy joblib lightgbm scikit-learn
```

### 2. Run the Demo

```bash
streamlit run demo_app.py
```

The app will open in your browser (typically http://localhost:8501).

### 3. Explore the Demo

#### Controls (Left Sidebar)
- **Trip Selection**: Choose any trip from the dataset to visualize
- **Animation Speed**: Adjust playback speed (higher = faster)
- **Auto-play**: Enable/disable automatic animation progression

#### Main Display

**🗺️ Map Section**
- Blue line: Route traveled so far
- Gray dashed line: Planned route ahead
- 🚚 Red diamond: Current truck position
- Orange squares: Detected stops
- Green circle: Trip start point
- Red star: Destination

**📍 ETA Prediction**
- Shows three confidence estimates:
  - **P10** (10th percentile): Optimistic - truck likely to arrive before this
  - **P50** (Median): Best estimate for arrival time
  - **P90** (90th percentile): Conservative - accounting for delays
  
- **Color-coded status**:
  - 🟢 Green (≤30 min): On schedule
  - 🟡 Yellow (30-60 min): Check progress
  - 🔴 Red (>60 min): Behind schedule

**📊 Charts**
- **Speed Profile**: Shows speed variations and automatically marks detected stops
- **Weight Profile**: Load weight over time (if available)

**🛑 Detected Stops**
- Automatically identifies stops lasting >1 minute
- Shows stop time, duration, and location
- Expandable details for each stop

## Features Explained

### Models Used
The app uses three pre-trained LightGBM quantile regressors:

| Model | Purpose | File |
|-------|---------|------|
| **P10** | Lower bound (optimistic) | `lgb_q10.pkl` |
| **P50** | Median (best estimate) | `lgb_q50.pkl` |
| **P90** | Upper bound (conservative) | `lgb_q90.pkl` |

**Confidence Interval**: P10 to P90 represents an ~80% confidence band around the median estimate.

### Input Features (12 total)

**Temporal Features**
- `elapsed_s`: Seconds since trip start
- `hour_sin`, `hour_cos`: Hour of day (cyclical encoding)
- `dow`: Day of week

**Speed Features**
- `speed_rolling_mean`: 3-ping rolling average speed
- `speed_rolling_std`: Speed variation
- `speed_cummax`: Maximum speed so far in trip

**Spatial Features**
- `dist_remaining_km`: Distance to destination (haversine)
- `trip_progress`: Fraction of trip completed

**Load Features**
- `has_weight`: Boolean (is weight data available)
- `load_state`: Binary (loaded/unloaded heuristic)
- `load_change_count`: Number of load state transitions

### Stop Detection Algorithm

The app detects stops using a simple heuristic:

1. Identify periods where Speed < 2.0 mph
2. Flag as a stop if duration ≥ 60 seconds
3. Display stop location, start time, and duration

## Data Overview

**Dataset**: `data/segmented_trips.csv`
- **Total rows**: 4,899,013 telemetry pings
- **Unique trips**: 32,931
- **Columns**: VIN, Timestamp, Latitude, Longitude, Speed, Weight_lbs, Device_Type, Source, trip_id
- **Trip lengths**: 
  - Min: 1 ping
  - Max: 21,640 pings
  - Average: 148.8 pings per trip

## Limitations & Notes

1. **Not Real-Time**: Demo uses historical trip data replayed as animation
2. **Navigation**: Uses straight-line (haversine) distances, not actual roads
3. **External Factors**: Weather, traffic, road conditions not considered
4. **Best For**: Regional/medium-distance routes (100+ km typical)
5. **Feature Availability**: Some trips lack weight data; handled gracefully

## Troubleshooting

**"Models not found" error**
```
→ Ensure you ran `make train` first to generate model artifacts
→ Models should be in: models/lgb_q10.pkl, lgb_q50.pkl, lgb_q90.pkl
```

**"Data not found" error**
```
→ Ensure segmented_trips.csv exists in data/ directory
→ If missing, run: python src/trip_segmentation.py
```

**Slow animation**
```
→ Reduce animation_speed slider (higher number = faster, fewer pings shown)
→ Select a shorter trip (fewer total pings)
```

**Chart doesn't load / Import errors**
```
→ Verify all dependencies: pip install -r requirements.txt
→ Restart Streamlit: streamlit run demo_app.py
→ Check Python version: python --version (requires 3.8+)
```

## Example Workflows

### Workflow 1: Find a Long Trip
1. Select different trip IDs in the sidebar
2. Look at the "Trip Duration" metric to find long routes
3. Scrub through the timeline to see how ETA predictions evolve

### Workflow 2: Analyze Stop Patterns
1. Look at the "Stops Detected" counter
2. Expand each stop section to see location and duration
3. Correlate stops with speed profile dips

### Workflow 3: Compare ETA Confidence
1. Watch the bar chart showing P10/P50/P90 estimates
2. Scrub forward in time to see how confidence improves (tighter band)
3. Notice how ETA predictions stabilize as trip progresses

## Technical Details

### How ETA Predictions Work

For any point in the trip, the model:

1. **Extracts 12 features** from historical data up to that point
2. **Runs three models** (P10/P50/P90 quantile regressors)
3. **Outputs time remaining** in seconds (converted to HH:MM format)
4. **Presents confidence interval** from P10 to P90 (80% confidence)

### Performance Metrics (from Run 5 - Full Dataset)

| Metric | LightGBM P50 | Near-Arrival* |
|--------|--------------|---------------|
| Global MAE | 732.2 min | 431.9 min |
| Global RMSE | 2,270.9 min | N/A |
| Within-tolerance rate (±10 min) | 54.25% | N/A |
| Coverage (P10-P90) | 80.5% | N/A |

\* Near-Arrival = when actual ETA ≤ 60 minutes remaining (dispatch-relevant)

## File Structure

```
├── demo_app.py                 # Main Streamlit app (run this!)
├── requirements.txt            # Includes streamlit, plotly, scipy
├── data/
│   └── segmented_trips.csv    # 4.9M telemetry pings
├── models/
│   ├── lgb_q10.pkl            # P10 quantile model
│   ├── lgb_q50.pkl            # P50 median model
│   ├── lgb_q90.pkl            # P90 quantile model
│   └── baseline_hgb.pkl       # Baseline (not used in demo)
└── src/
    ├── features.py             # Feature engineering pipeline
    └── eta_model.py            # Model training (reference)
```

## Questions?

See [README.md](README.md) for project overview and [PLAN.md](PLAN.md) for development history.

---

**Happy truck tracking! 🚚📍**
