"""Tests for trip segmentation."""

import pandas as pd

from src.trip_segmentation import segment_trips, MIN_TRIP_PINGS


def _make_pings(vin: str, timestamps: list[str], speeds: list[float]) -> pd.DataFrame:
    """Helper to build a small telemetry DataFrame."""
    return pd.DataFrame({
        "VIN": vin,
        "Timestamp": pd.to_datetime(timestamps),
        "Latitude": 40.0,
        "Longitude": -74.0,
        "Speed": speeds,
    })


class TestSegmentTrips:
    def test_single_continuous_trip(self):
        """Pings close together → one trip."""
        ts = [f"2026-04-01 10:{m:02d}:00" for m in range(10)]
        speeds = [50.0] * 10
        df = _make_pings("VIN_A", ts, speeds)
        out = segment_trips(df)

        assert out["TripId"].nunique() == 1
        assert len(out) == 10

    def test_time_gap_splits_trip(self):
        """A >30-minute gap should create two trips."""
        ts = [f"2026-04-01 10:{m:02d}:00" for m in range(6)]
        ts += [f"2026-04-01 11:{m:02d}:00" for m in range(6)]  # 54-min gap
        speeds = [60.0] * 12
        df = _make_pings("VIN_B", ts, speeds)
        out = segment_trips(df)

        assert out["TripId"].nunique() == 2

    def test_short_trip_filtered(self):
        """Trips with fewer than MIN_TRIP_PINGS pings are dropped."""
        ts_short = [f"2026-04-01 10:{m:02d}:00" for m in range(MIN_TRIP_PINGS - 1)]
        ts_long = [f"2026-04-01 12:{m:02d}:00" for m in range(MIN_TRIP_PINGS + 5)]
        speeds_short = [60.0] * (MIN_TRIP_PINGS - 1)
        speeds_long = [60.0] * (MIN_TRIP_PINGS + 5)

        df = pd.concat([
            _make_pings("VIN_C", ts_short, speeds_short),
            _make_pings("VIN_C", ts_long, speeds_long),
        ], ignore_index=True).sort_values(["VIN", "Timestamp"]).reset_index(drop=True)

        out = segment_trips(df)
        assert out["TripId"].nunique() == 1
        assert len(out) == MIN_TRIP_PINGS + 5

    def test_multiple_vins(self):
        """Each VIN gets independent trip numbering."""
        ts = [f"2026-04-01 10:{m:02d}:00" for m in range(6)]
        speeds = [50.0] * 6
        df = pd.concat([
            _make_pings("VIN_X", ts, speeds),
            _make_pings("VIN_Y", ts, speeds),
        ], ignore_index=True)
        out = segment_trips(df)

        assert out["TripId"].nunique() == 2
        assert all("VIN_X" in tid for tid in out[out["VIN"] == "VIN_X"]["TripId"])
        assert all("VIN_Y" in tid for tid in out[out["VIN"] == "VIN_Y"]["TripId"])
