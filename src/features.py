"""Feature engineering module.

Transform raw telematics records into model-ready ETA features.
Centralises all feature definitions so training and inference share the
same logic — no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    trip_start = df.groupby("TripId")["Timestamp"].transform("min")
    df["elapsed_s"] = (df["Timestamp"] - trip_start).dt.total_seconds()
    return df


def add_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean, rolling std, and cumulative max of speed per trip."""
    g = df.groupby("TripId")

    df["speed_rolling_mean"] = g["Speed"].transform(
        lambda s: s.fillna(0.0).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["speed_rolling_std"] = g["Speed"].transform(
        lambda s: s.fillna(0.0).rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0.0)
    )
    df["speed_cummax"] = g["Speed"].transform(
        lambda s: s.fillna(0.0).cummax()
    )
    return df


def add_distance_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """Haversine distance (km) from current ping to estimated destination.

    The destination is approximated as the last ping of each trip.
    """
    dest = df.groupby("TripId")[["Latitude", "Longitude"]].transform("last")
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
    df["trip_progress"] = df.groupby("TripId").cumcount()
    trip_len = df.groupby("TripId")["trip_progress"].transform("max")
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
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transforms and return the DataFrame with new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: TripId, Timestamp, Latitude, Longitude, Speed.

    Returns
    -------
    pd.DataFrame
        Original columns plus all feature columns listed in ``FEATURE_COLS``.
    """
    df = df.copy()
    df = add_elapsed_time(df)
    df = add_speed_stats(df)
    df = add_distance_remaining(df)
    df = add_time_encoding(df)
    df = add_weight_flag(df)
    df = add_trip_progress(df)
    return df
