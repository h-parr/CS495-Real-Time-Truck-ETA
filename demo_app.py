"""
Truck ETA Prediction Demo - Interactive Visualization
======================================================

An interactive Streamlit app showcasing real-time truck ETA predictions with:
- Animated truck position on an interactive map
- Speed and weight tracking over time
- Stop detection and duration
- Distance remaining and progress
- Quantile regression confidence intervals (P10/P50/P90)
- Current ETA estimates with confidence ranges

Run with: streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ────────────────────────────────────────────────────────────
# Configuration & Setup
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Truck ETA Demo",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .stop-alert {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
    }
    .eta-good {
        color: #2ecc71;
        font-weight: bold;
    }
    .eta-warning {
        color: #f39c12;
        font-weight: bold;
    }
    .eta-danger {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load pre-trained LightGBM quantile models."""
    model_dir = Path(__file__).parent / "models"
    models = {
        'p10': joblib.load(model_dir / "lgb_q10.pkl"),
        'p50': joblib.load(model_dir / "lgb_q50.pkl"),
        'p90': joblib.load(model_dir / "lgb_q90.pkl"),
    }
    return models

@st.cache_data
def load_sample_data():
    """Load sample trip data from segmented_trips.csv."""
    df = pd.read_csv("data/segmented_trips.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def build_features_for_trip(trip_df):
    """Build features for a single trip (simplified version of feature engineering)."""
    from src.features import build_features, FEATURE_COLS
    
    # Apply feature engineering
    trip_with_features = build_features(trip_df.copy())
    
    return trip_with_features, FEATURE_COLS

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in km."""
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def detect_stops(trip_df, speed_threshold=2.0, min_duration=60):
    """Detect stops in the trip where speed is below threshold."""
    stops = []
    trip_df = trip_df.sort_values('Timestamp').reset_index(drop=True)
    
    in_stop = False
    stop_start = None
    
    for idx, row in trip_df.iterrows():
        speed = row['Speed'] if pd.notna(row['Speed']) else 0
        
        if speed < speed_threshold:
            if not in_stop:
                in_stop = True
                stop_start = idx
        else:
            if in_stop:
                stop_end = idx - 1
                if stop_end >= stop_start:
                    stop_duration = (trip_df.iloc[stop_end]['Timestamp'] - 
                                   trip_df.iloc[stop_start]['Timestamp']).total_seconds() / 60
                    if stop_duration >= min_duration:
                        stops.append({
                            'start_idx': stop_start,
                            'end_idx': stop_end,
                            'start_time': trip_df.iloc[stop_start]['Timestamp'],
                            'duration_min': stop_duration,
                            'lat': trip_df.iloc[stop_start]['Latitude'],
                            'lon': trip_df.iloc[stop_start]['Longitude'],
                        })
                in_stop = False
    
    return stops

def format_time_remaining(minutes):
    """Format minutes to HH:MM format."""
    if pd.isna(minutes) or minutes < 0:
        return "N/A"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"

# ────────────────────────────────────────────────────────────
# Main App
# ────────────────────────────────────────────────────────────

st.title("🚚 Truck ETA Prediction Demo")
st.markdown("Real-time truck tracking with quantile regression-based ETA confidence intervals")

# Load data and models
try:
    models = load_models()
    all_data = load_sample_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Demo Controls")
    
    # Trip selection
    all_trips = sorted(all_data['trip_id'].unique())
    selected_trip = st.selectbox(
        "Select a trip to visualize:",
        all_trips,
        index=0,
        help="Choose from available trips in the dataset"
    )
    
    # Animation speed
    animation_speed = st.slider(
        "Animation speed (frames to skip):",
        min_value=1,
        max_value=20,
        value=5,
        help="Higher = faster animation"
    )
    
    # Auto-play toggle
    auto_play = st.checkbox("Auto-play animation", value=True)
    
    st.divider()
    st.subheader("📊 Model Info")
    st.markdown("""
    **Models Used:**
    - **P10**: 10th percentile (optimistic)
    - **P50**: Median (best estimate)
    - **P90**: 90th percentile (conservative)
    
    **Features:**
    - Elapsed time, rolling speed stats
    - Distance remaining, time encoding
    - Weight flag, trip progress
    - Load state indicators
    """)

# Extract selected trip
trip_data = all_data[all_data['trip_id'] == selected_trip].copy()
trip_data = trip_data.sort_values('Timestamp').reset_index(drop=True)

if len(trip_data) == 0:
    st.error("Selected trip not found.")
    st.stop()

# Compute trip statistics
trip_start_time = trip_data.iloc[0]['Timestamp']
trip_end_time = trip_data.iloc[-1]['Timestamp']
trip_duration_hours = (trip_end_time - trip_start_time).total_seconds() / 3600

# Calculate total distance
total_distance_km = 0
for i in range(1, len(trip_data)):
    dist = haversine_distance(
        trip_data.iloc[i-1]['Latitude'], trip_data.iloc[i-1]['Longitude'],
        trip_data.iloc[i]['Latitude'], trip_data.iloc[i]['Longitude']
    )
    total_distance_km += dist

# Detect stops
stops = detect_stops(trip_data)

# Build features
try:
    trip_with_features, feature_cols = build_features_for_trip(trip_data)
except Exception as e:
    st.warning(f"Feature building had issues: {e}. Using available features.")
    trip_with_features = trip_data
    feature_cols = []

# ────────────────────────────────────────────────────────────
# Top Summary Cards
# ────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Trip Duration",
        f"{trip_duration_hours:.1f}h",
        f"{len(trip_data)} pings"
    )

with col2:
    st.metric(
        "Total Distance",
        f"{total_distance_km:.1f} km",
        f"{total_distance_km / trip_duration_hours:.1f} km/h avg"
    )

with col3:
    st.metric(
        "Stops Detected",
        len(stops),
        f"{sum(s['duration_min'] for s in stops):.0f} min total" if stops else "No stops"
    )

with col4:
    avg_weight = trip_data['Weight_lbs'].mean()
    st.metric(
        "Load",
        f"{avg_weight:,.0f} lbs" if pd.notna(avg_weight) else "No data",
        "(avg weight)"
    )

with col5:
    st.metric(
        "Trip ID",
        selected_trip[:20] + "..." if len(selected_trip) > 20 else selected_trip,
        f"Start: {trip_start_time.strftime('%H:%M')}"
    )

st.divider()

# ────────────────────────────────────────────────────────────
# Main Animation Section
# ────────────────────────────────────────────────────────────

st.subheader("🗺️ Truck Route Animation")

# Animation slider
animation_frame = st.slider(
    "Timeline Progress:",
    min_value=0,
    max_value=len(trip_data) - 1,
    value=0,
    step=animation_speed if not auto_play else 1,
    help="Drag to see different points in the trip"
)

# Get current position and data
current_ping = trip_data.iloc[animation_frame]
current_idx = animation_frame

# Calculate cumulative distance up to current point
cumulative_distance = 0
for i in range(1, current_idx + 1):
    dist = haversine_distance(
        trip_data.iloc[i-1]['Latitude'], trip_data.iloc[i-1]['Longitude'],
        trip_data.iloc[i]['Latitude'], trip_data.iloc[i]['Longitude']
    )
    cumulative_distance += dist

# Estimate remaining distance (simple linear extrapolation)
if current_idx > 0:
    distance_remaining = max(0, total_distance_km - cumulative_distance)
    # Estimate ETA from distance and average speed
    elapsed_time = (current_ping['Timestamp'] - trip_start_time).total_seconds() / 3600
    if elapsed_time > 0:
        avg_speed = cumulative_distance / elapsed_time
        eta_minutes_naive = (distance_remaining / avg_speed * 60) if avg_speed > 0 else 0
    else:
        eta_minutes_naive = 0
else:
    distance_remaining = total_distance_km
    eta_minutes_naive = 0

# Make predictions if features available
eta_predictions = {'p10': None, 'p50': None, 'p90': None}
try:
    if feature_cols and len(trip_with_features) > current_idx:
        features_at_idx = trip_with_features.iloc[current_idx][feature_cols].values.reshape(1, -1)
        eta_predictions['p10'] = models['p10'].predict(features_at_idx)[0] / 60  # Convert to minutes
        eta_predictions['p50'] = models['p50'].predict(features_at_idx)[0] / 60
        eta_predictions['p90'] = models['p90'].predict(features_at_idx)[0] / 60
except Exception as e:
    st.warning(f"Could not make ETA predictions: {e}")

# Create map visualization
fig_map = go.Figure()

# Add route line
fig_map.add_trace(go.Scattergeo(
    lon=trip_data['Longitude'][:current_idx+1],
    lat=trip_data['Latitude'][:current_idx+1],
    mode='lines',
    line=dict(color='blue', width=2),
    name='Route traveled',
    hoverinfo='skip'
))

# Add remaining route (light gray)
if current_idx < len(trip_data) - 1:
    fig_map.add_trace(go.Scattergeo(
        lon=trip_data['Longitude'][current_idx:],
        lat=trip_data['Latitude'][current_idx:],
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dash'),
        name='Planned route',
        hoverinfo='skip'
    ))

# Add stops as markers
if stops:
    stop_lats = [s['lat'] for s in stops if s['start_idx'] <= current_idx]
    stop_lons = [s['lon'] for s in stops if s['start_idx'] <= current_idx]
    stop_times = [f"{s['duration_min']:.0f} min" for s in stops if s['start_idx'] <= current_idx]
    
    fig_map.add_trace(go.Scattergeo(
        lon=stop_lons,
        lat=stop_lats,
        mode='markers',
        marker=dict(size=10, color='orange', symbol='square'),
        name='Stops',
        text=stop_times,
        hovertemplate='<b>Stop</b><br>Duration: %{text}<extra></extra>'
    ))

# Add current position
fig_map.add_trace(go.Scattergeo(
    lon=[current_ping['Longitude']],
    lat=[current_ping['Latitude']],
    mode='markers+text',
    marker=dict(size=15, color='red', symbol='diamond'),
    name='Current location',
    text=['🚚'],
    textposition='top center',
    hovertemplate=f"<b>Current Position</b><br>Lat: {current_ping['Latitude']:.4f}<br>Lon: {current_ping['Longitude']:.4f}<br>Speed: {current_ping['Speed']:.1f} mph<extra></extra>"
))

# Add start point
fig_map.add_trace(go.Scattergeo(
    lon=[trip_data.iloc[0]['Longitude']],
    lat=[trip_data.iloc[0]['Latitude']],
    mode='markers',
    marker=dict(size=12, color='green', symbol='circle'),
    name='Start',
    hoverinfo='skip'
))

# Add end point
fig_map.add_trace(go.Scattergeo(
    lon=[trip_data.iloc[-1]['Longitude']],
    lat=[trip_data.iloc[-1]['Latitude']],
    mode='markers',
    marker=dict(size=12, color='darkred', symbol='star'),
    name='Destination',
    hoverinfo='skip'
))

fig_map.update_layout(
    title=f"Trip Progress: {current_idx+1}/{len(trip_data)} pings ({current_idx/len(trip_data)*100:.1f}%)",
    geo=dict(
        projection_type='mercator',
        showland=True,
        landcolor='rgb(243, 243, 243)',
    ),
    hovermode='closest',
    height=500,
    showlegend=True,
)

st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────
# ETA & Metrics Section
# ────────────────────────────────────────────────────────────

col_eta, col_progress = st.columns([2, 1])

with col_eta:
    st.subheader("📍 ETA Prediction")
    
    # Display ETA confidence interval
    if eta_predictions['p50'] is not None:
        p10_time = format_time_remaining(eta_predictions['p10'])
        p50_time = format_time_remaining(eta_predictions['p50'])
        p90_time = format_time_remaining(eta_predictions['p90'])
        
        # Color code ETA confidence
        if eta_predictions['p50'] <= 30:
            confidence_class = "eta-good"
            status = "✅ On Schedule"
        elif eta_predictions['p50'] <= 60:
            confidence_class = "eta-warning"
            status = "⚠️ Check Progress"
        else:
            confidence_class = "eta-danger"
            status = "❌ Behind Schedule"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Arrival Time Estimate</h3>
            <div style="font-size: 24px; margin: 10px 0;">
                <span class="{confidence_class}">
                    {p50_time} (±{max(eta_predictions['p90'] - eta_predictions['p50'], eta_predictions['p50'] - eta_predictions['p10']):.0f} min)
                </span>
            </div>
            <p style="margin: 0; color: #666;">80% confident arriving between {p10_time} and {p90_time}</p>
            <p style="margin: 10px 0 0 0; font-size: 16px;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ETA timeline chart
        confidence_data = {
            'Scenario': ['10th %ile', 'Median', '90th %ile'],
            'Minutes': [
                eta_predictions['p10'],
                eta_predictions['p50'],
                eta_predictions['p90']
            ],
            'Color': ['green', 'blue', 'orange']
        }
        
        fig_eta = px.bar(
            confidence_data,
            x='Scenario',
            y='Minutes',
            color='Scenario',
            color_discrete_map={'10th %ile': 'green', 'Median': 'blue', '90th %ile': 'orange'},
            title='ETA Quantile Estimates',
            labels={'Minutes': 'Time to Arrival (minutes)'},
            height=300
        )
        fig_eta.update_traces(showlegend=False)
        st.plotly_chart(fig_eta, use_container_width=True)
    else:
        st.info("ETA predictions not available for this frame")

with col_progress:
    st.subheader("📊 Trip Progress")
    
    progress_pct = (current_idx / len(trip_data)) * 100
    st.metric("Distance Completed", f"{cumulative_distance:.1f} km", f"{progress_pct:.1f}%")
    st.metric("Distance Remaining", f"{distance_remaining:.1f} km", f"{100-progress_pct:.1f}%")
    
    # Progress bar
    st.progress(progress_pct / 100)

st.divider()

# ────────────────────────────────────────────────────────────
# Time Series Charts
# ────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Speed Over Time")
    
    fig_speed = go.Figure()
    
    # Speed line
    fig_speed.add_trace(go.Scatter(
        x=trip_data.index[:current_idx+1],
        y=trip_data['Speed'][:current_idx+1],
        mode='lines',
        name='Speed',
        line=dict(color='blue', width=2),
        fill='tozeroy'
    ))
    
    # Mark stops
    for stop in stops:
        if stop['start_idx'] <= current_idx:
            fig_speed.add_vline(
                x=stop['start_idx'],
                line_dash='dash',
                line_color='orange',
                annotation_text='Stop',
                annotation_position='top'
            )
    
    # Current position
    fig_speed.add_vline(x=current_idx, line_dash='solid', line_color='red')
    
    fig_speed.update_layout(
        title='Speed Profile',
        xaxis_title='Ping Index',
        yaxis_title='Speed (mph)',
        hovermode='x unified',
        height=350
    )
    
    st.plotly_chart(fig_speed, use_container_width=True)

with chart_col2:
    st.subheader("⚖️ Weight Over Time")
    
    fig_weight = go.Figure()
    
    weight_data = trip_data['Weight_lbs'][:current_idx+1]
    
    if weight_data.notna().sum() > 0:
        fig_weight.add_trace(go.Scatter(
            x=trip_data.index[:current_idx+1],
            y=weight_data,
            mode='lines',
            name='Weight',
            line=dict(color='purple', width=2),
            fill='tozeroy'
        ))
        
        fig_weight.add_hline(
            y=weight_data.mean(),
            line_dash='dash',
            line_color='gray',
            annotation_text=f'Avg: {weight_data.mean():,.0f} lbs',
            annotation_position='right'
        )
    else:
        fig_weight.add_annotation(
            text='No weight data available',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False
        )
    
    fig_weight.update_layout(
        title='Load Weight Profile',
        xaxis_title='Ping Index',
        yaxis_title='Weight (lbs)',
        hovermode='x unified',
        height=350
    )
    
    st.plotly_chart(fig_weight, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────
# Stop Details
# ────────────────────────────────────────────────────────────

if stops:
    st.subheader("🛑 Detected Stops")
    
    relevant_stops = [s for s in stops if s['start_idx'] <= current_idx]
    
    if relevant_stops:
        for i, stop in enumerate(relevant_stops, 1):
            with st.expander(f"Stop #{i}: {stop['start_time'].strftime('%H:%M')} - Duration: {stop['duration_min']:.0f} min"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start Time", stop['start_time'].strftime('%H:%M:%S'))
                with col2:
                    st.metric("Duration", f"{stop['duration_min']:.1f} min")
                with col3:
                    st.metric("Location", f"{stop['lat']:.4f}, {stop['lon']:.4f}")

# ────────────────────────────────────────────────────────────
# Footer & Info
# ────────────────────────────────────────────────────────────

st.divider()

with st.expander("ℹ️ How This Demo Works"):
    st.markdown("""
    ### ETA Prediction Model
    
    This demo uses **quantile regression** with LightGBM to predict truck arrival times:
    
    - **P10 (10th percentile)**: Optimistic scenario - 1 in 10 chance of arriving earlier
    - **P50 (Median)**: Best estimate - most likely arrival time
    - **P90 (90th percentile)**: Conservative scenario - 1 in 10 chance of arriving later
    
    ### Features Used
    - **Temporal**: Elapsed time, hour of day, day of week
    - **Spatial**: Current location, distance to destination, haversine distance
    - **Vehicle**: Speed statistics (rolling mean, std, cummax), load state
    - **Progress**: Trip progress fraction, number of stops so far
    
    ### Limitations
    - Uses actual trip data (not real-time predictions)
    - Navigation routes are simplified (straight-line distances)
    - Weather, traffic, and other external factors not considered
    - Works best for regional/medium-distance routes
    
    ### Try This
    - Select different trips to see how characteristics vary
    - Scroll through the timeline to watch ETA predictions change
    - Look for patterns in speed/weight during different phases
    """)

st.markdown("""
---
<div style="text-align: center; color: #666; margin-top: 20px;">
    <p><strong>CS495 Real-Time Truck ETA Prediction</strong></p>
    <p>Model: LightGBM Quantile Regression | Dataset: Segmented Truck Telemetry</p>
</div>
""", unsafe_allow_html=True)
