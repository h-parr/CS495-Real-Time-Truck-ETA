"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    add_distance_remaining,
    add_elapsed_time,
    add_speed_stats,
    add_time_encoding,
    add_weight_flag,
    build_features,
    haversine_km,
)


def _trip_df(n: int = 10) -> pd.DataFrame:
    """Minimal trip DataFrame for testing."""
    return pd.DataFrame({
        "TripId": "T1",
        "Timestamp": pd.date_range("2026-04-01 08:00", periods=n, freq="min"),
        "Latitude": np.linspace(40.0, 40.1, n),
        "Longitude": np.linspace(-74.0, -73.9, n),
        "Speed": np.linspace(0, 60, n),
        "Weight_lbs": [np.nan] * (n // 2) + [45000.0] * (n - n // 2),
    })


class TestHaversine:
    def test_same_point_zero(self):
        d = haversine_km(np.array([40.0]), np.array([-74.0]),
                         np.array([40.0]), np.array([-74.0]))
        assert d[0] == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        # NYC to LA ≈ 3,944 km
        d = haversine_km(np.array([40.7128]), np.array([-74.0060]),
                         np.array([34.0522]), np.array([-118.2437]))
        assert 3900 < d[0] < 4000


class TestElapsedTime:
    def test_starts_at_zero(self):
        df = add_elapsed_time(_trip_df())
        assert df["elapsed_s"].iloc[0] == 0.0

    def test_monotonic(self):
        df = add_elapsed_time(_trip_df())
        assert df["elapsed_s"].is_monotonic_increasing


class TestSpeedStats:
    def test_columns_exist(self):
        df = add_speed_stats(_trip_df())
        for col in ["speed_rolling_mean", "speed_rolling_std", "speed_cummax"]:
            assert col in df.columns

    def test_cummax_non_decreasing(self):
        df = add_speed_stats(_trip_df())
        assert df["speed_cummax"].is_monotonic_increasing or (df["speed_cummax"].diff().dropna() >= 0).all()


class TestDistanceRemaining:
    def test_last_ping_zero(self):
        df = add_distance_remaining(_trip_df())
        assert df["dist_remaining_km"].iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_first_greater_than_last(self):
        df = add_distance_remaining(_trip_df())
        assert df["dist_remaining_km"].iloc[0] > df["dist_remaining_km"].iloc[-1]


class TestTimeEncoding:
    def test_sin_cos_range(self):
        df = add_time_encoding(_trip_df())
        assert df["hour_sin"].between(-1, 1).all()
        assert df["hour_cos"].between(-1, 1).all()

    def test_dow_range(self):
        df = add_time_encoding(_trip_df())
        assert df["dow"].between(0, 6).all()


class TestWeightFlag:
    def test_flag_values(self):
        df = add_weight_flag(_trip_df())
        assert set(df["has_weight"].unique()).issubset({0, 1})

    def test_no_weight_column(self):
        df = _trip_df().drop(columns=["Weight_lbs"])
        df = add_weight_flag(df)
        assert (df["has_weight"] == 0).all()


class TestBuildFeatures:
    def test_all_feature_cols_present(self):
        df = build_features(_trip_df())
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_no_nans_in_features(self):
        df = build_features(_trip_df())
        for col in FEATURE_COLS:
            assert not df[col].isna().any(), f"NaN found in {col}"
