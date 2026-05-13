"""Feature engineering module.

Transform raw telematics records into model-ready ETA features.
Centralises all feature definitions so training and inference share the
same logic — no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .load_state import add_load_state
except ImportError:
    # Fallback for direct script execution
    from load_state import add_load_state

# ── Constants ────────────────────────────────────────────────
EARTH_RADIUS_KM = 6_371.0
ROLLING_WINDOW = 5  # pings for rolling speed statistics


# ── Haversine helper ─────────────────────────────────────────

def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in kilometres."""
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ── Feature transforms ──────────────────────────────────────

def add_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since the first ping of each trip."""
    trip_start = df.groupby("trip_id")["Timestamp"].transform("min")
    df["elapsed_s"] = (df["Timestamp"] - trip_start).dt.total_seconds()
    return df


def add_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, rolling std, and cumulative max of speed per trip."""
    speed = df["Speed"].fillna(0.0)
    by_trip = speed.groupby(df["trip_id"]) 

    df["speed_rolling_mean"] = (
        by_trip.rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["speed_rolling_std"] = (
        by_trip.rolling(ROLLING_WINDOW, min_periods=1)
        .std()
        .fillna(0.0)
        .reset_index(level=0, drop=True)
    )
    df["speed_cummax"] = by_trip.cummax()
    return df


def add_distance_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine distance (km) from current ping to estimated destination.

    The destination is approximated as the last ping of each trip.
    """
    dest = df.groupby("trip_id")[["Latitude", "Longitude"]].transform("last")
    df["dist_remaining_km"] = haversine_km(
        df["Latitude"].values, df["Longitude"].values,
        dest["Latitude"].values, dest["Longitude"].values,
    )
    return df


def add_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical hour-of-day and day-of-week encoding."""
    hour = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow"] = df["Timestamp"].dt.dayofweek  # 0=Mon .. 6=Sun
    return df


def add_weight_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if Weight_lbs is present, 0 otherwise."""
    if "Weight_lbs" in df.columns:
        df["has_weight"] = df["Weight_lbs"].notna().astype(np.int8)
    else:
        df["has_weight"] = np.int8(0)
    return df


def add_trip_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of trip elapsed (0‥1) based on ping index within trip."""
    df["trip_progress"] = df.groupby("trip_id").cumcount()
    trip_len = df.groupby("trip_id")["trip_progress"].transform("max")
    df["trip_progress"] = (df["trip_progress"] / trip_len.replace(0, 1)).astype(np.float32)
    return df


# ── Public API ───────────────────────────────────────────────

FEATURE_COLS = [
    "elapsed_s",
    "speed_rolling_mean",
    "speed_rolling_std",
    "speed_cummax",
    "dist_remaining_km",
    "hour_sin",
    "hour_cos",
    "dow",
    "has_weight",
    "trip_progress",
    "load_state",
    "load_change_count",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transforms and return the DataFrame with new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: trip_id, Timestamp, Latitude, Longitude, Speed.

    Returns
    -------
    pd.DataFrame
        Original columns plus all feature columns listed in ``FEATURE_COLS``.
    """
    df = df.copy()
    print("  Feature step 1/7: elapsed time")
    df = add_elapsed_time(df)
    print("  Feature step 2/7: speed stats")
    df = add_speed_stats(df)
    print("  Feature step 3/7: distance remaining")
    df = add_distance_remaining(df)
    print("  Feature step 4/7: time encoding")
    df = add_time_encoding(df)
    print("  Feature step 5/7: weight flag")
    df = add_weight_flag(df)
    print("  Feature step 6/7: trip progress")
    df = add_trip_progress(df)
    print("  Feature step 7/7: load state features (heuristic)")
    df = add_load_state(df, method="heuristic", verbose=True)
    return df
