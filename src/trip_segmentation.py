"""Trip segmentation module.

Split continuous telemetry streams into discrete trips per VIN using
time-gap and speed-based heuristics.  Produces a trip-labeled CSV that
downstream feature engineering and model code can consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Tuneable thresholds ──────────────────────────────────────
TIME_GAP_MINUTES: float = 30.0   # gap between pings that signals a new trip
STOP_SPEED_MPH: float = 2.0      # speed below which the truck is "stopped"
MIN_TRIP_PINGS: int = 5          # discard trips shorter than this


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read raw telemetry CSV and do minimal type coercion."""
    df = pd.read_csv(
        csv_path,
        parse_dates=["Timestamp"],
        dtype={
            "VIN": str,
            "Latitude": np.float64,
            "Longitude": np.float64,
            "Speed": np.float64,
            "Device_Type": str,
            "Source": str,
        },
    )
    df.columns = df.columns.str.strip()
    df.sort_values(["VIN", "Timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def segment_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a ``TripId`` to every row based on time gaps and speed.

    Algorithm
    ---------
    Within each VIN (sorted by Timestamp):
    1. Compute the time delta between consecutive pings.
    2. Start a new trip whenever:
       a. The delta exceeds ``TIME_GAP_MINUTES``, **or**
       b. The truck was stopped (speed < ``STOP_SPEED_MPH``) for the
          previous ping AND is now moving.
    3. Label each row with ``VIN_NNNN`` trip id.
    4. Drop trips with fewer than ``MIN_TRIP_PINGS`` pings.
    """
    df = df.copy()
    trip_ids: list[str] = []
    trip_counter = 0

    for vin, grp in df.groupby("VIN", sort=False):
        grp = grp.sort_values("Timestamp")
        deltas = grp["Timestamp"].diff()
        speeds = grp["Speed"].fillna(0.0)

        prev_stopped = speeds.shift(1, fill_value=0.0) < STOP_SPEED_MPH
        now_moving = speeds >= STOP_SPEED_MPH
        gap_exceeded = deltas > pd.Timedelta(minutes=TIME_GAP_MINUTES)

        new_trip = gap_exceeded | (prev_stopped & now_moving)
        # The very first ping of each VIN always starts a trip.
        new_trip.iloc[0] = True

        local_trip_num = new_trip.cumsum()
        for _, sub in grp.groupby(local_trip_num, sort=False):
            trip_counter += 1
            tid = f"{vin}_{trip_counter:04d}"
            trip_ids.extend([tid] * len(sub))

    df["TripId"] = trip_ids

    # Filter short trips
    trip_sizes = df.groupby("TripId")["TripId"].transform("count")
    df = df[trip_sizes >= MIN_TRIP_PINGS].reset_index(drop=True)

    return df


def main(input_csv: str | Path, output_csv: str | Path) -> None:
    """End-to-end: load → segment → write."""
    df = load_raw(input_csv)
    print(f"Loaded {len(df):,} rows, {df['VIN'].nunique()} VINs")

    df = segment_trips(df)
    n_trips = df["TripId"].nunique()
    print(f"Segmented into {n_trips:,} trips ({len(df):,} rows kept)")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Segment raw telemetry into discrete trips.")
    parser.add_argument("--input", default=str(_root / "data" / "sample_trucks_dataset.csv"))
    parser.add_argument("--output", default=str(_root / "data" / "segmented_trips.csv"))
    args = parser.parse_args()
    main(args.input, args.output)
