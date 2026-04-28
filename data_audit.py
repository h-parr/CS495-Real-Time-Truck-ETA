"""Data audit module goals:

- Run a read-only data quality check on the raw telemetry CSV.
- Report missingness, validity issues, duplicate pings, and per-VIN row counts.
- Identify sparse or event-driven features (e.g. Weight_lbs) so downstream code
  handles them appropriately rather than dropping rows.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

REQUIRED_COLUMNS = [
    "VIN", "Timestamp", "Latitude", "Longitude",
    "Speed", "Weight_lbs", "Device_Type", "Source",
]


def run_audit(csv_path: str) -> None:
    """Stream through the CSV and print a data quality report.

    Does not modify the input file in any way.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows = 0
    missing: Counter = Counter()
    rows_per_vin: Counter = Counter()
    weight_present_per_vin: Counter = Counter()
    out_of_order_per_vin: Counter = Counter()

    bad_ts = 0
    bad_lat = 0
    bad_lon = 0
    missing_speed = 0
    negative_speed = 0
    invalid_speed = 0
    adjacent_dup_pings = 0

    last_ts_by_vin: dict[str, datetime] = {}
    prev_ping_key = None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        normalized_cols = [c.strip() for c in (reader.fieldnames or [])]
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in normalized_cols]

        for raw in reader:
            row = {
                (k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                for k, v in raw.items()
            }
            rows += 1

            for col in REQUIRED_COLUMNS:
                if not row.get(col):
                    missing[col] += 1

            vin = row.get("VIN", "")
            ts_s = row.get("Timestamp", "")
            lat_s = row.get("Latitude", "")
            lon_s = row.get("Longitude", "")
            spd_s = row.get("Speed", "")
            wt_s = row.get("Weight_lbs", "")

            if vin:
                rows_per_vin[vin] += 1
            if vin and wt_s:
                weight_present_per_vin[vin] += 1

            # Adjacent duplicate ping check
            key = (vin, ts_s, lat_s, lon_s)
            if key == prev_ping_key:
                adjacent_dup_pings += 1
            prev_ping_key = key

            # Timestamp parse + per-VIN ordering
            ts = None
            if ts_s:
                try:
                    ts = datetime.strptime(ts_s, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    bad_ts += 1
            else:
                bad_ts += 1

            if vin and ts is not None:
                prev_ts = last_ts_by_vin.get(vin)
                if prev_ts is not None and ts < prev_ts:
                    out_of_order_per_vin[vin] += 1
                last_ts_by_vin[vin] = ts

            # Coordinate range checks
            try:
                if float(lat_s) < -90 or float(lat_s) > 90:
                    bad_lat += 1
            except (TypeError, ValueError):
                bad_lat += 1

            try:
                if float(lon_s) < -180 or float(lon_s) > 180:
                    bad_lon += 1
            except (TypeError, ValueError):
                bad_lon += 1

            # Speed checks
            if not spd_s:
                missing_speed += 1
            else:
                try:
                    if float(spd_s) < 0:
                        negative_speed += 1
                except (TypeError, ValueError):
                    invalid_speed += 1

    missing_pct = {
        c: round(missing[c] / rows * 100, 3) if rows else 0.0
        for c in REQUIRED_COLUMNS
    }

    # Per-VIN weight coverage
    weight_presence = sorted(
        [
            (vin, n, weight_present_per_vin.get(vin, 0),
             round(weight_present_per_vin.get(vin, 0) / n * 100, 3) if n else 0.0)
            for vin, n in rows_per_vin.items()
        ],
        key=lambda x: x[3],
    )

    print("=" * 60)
    print("DATA QUALITY AUDIT")
    print("=" * 60)
    print(f"  Total rows:                   {rows:,}")
    print(f"  VIN count:                    {len(rows_per_vin)}")
    print()
    print("--- Missing Columns ---")
    if missing_cols:
        for c in missing_cols:
            print(f"  MISSING COLUMN: {c}")
    else:
        print("  None")
    print()
    print("--- Missingness per Column ---")
    for col, pct in missing_pct.items():
        flag = " <-- NOTE" if pct > 5 else ""
        print(f"  {col:<20} {missing[col]:>9,} rows  ({pct:.3f}%){flag}")
    print()
    print("--- Validity Issues ---")
    print(f"  Bad timestamps:               {bad_ts:,}")
    print(f"  Invalid latitude values:      {bad_lat:,}")
    print(f"  Invalid longitude values:     {bad_lon:,}")
    print(f"  Missing speed rows:           {missing_speed:,}  ({missing_pct['Speed']:.3f}%)")
    print(f"  Negative speed rows:          {negative_speed:,}")
    print(f"  Unparseable speed rows:       {invalid_speed:,}")
    print(f"  Adjacent duplicate pings:     {adjacent_dup_pings:,}")
    print(f"  VINs with out-of-order ts:    {len(out_of_order_per_vin)}")
    print()
    print("--- Rows per VIN (top 10 / bottom 10) ---")
    for vin, n in rows_per_vin.most_common(10):
        print(f"  {vin:<15} {n:>9,} rows")
    print("  ...")
    for vin, n in sorted(rows_per_vin.items(), key=lambda x: x[1])[:10]:
        print(f"  {vin:<15} {n:>9,} rows")
    print()
    print("--- Weight_lbs Coverage per VIN ---")
    print(f"  (overall: {round((rows - missing['Weight_lbs']) / rows * 100, 3):.3f}% non-missing)")
    print("  Lowest coverage VINs:")
    for vin, total, present, pct in weight_presence[:10]:
        print(f"    {vin:<15}  {present:>9,}/{total:>9,}  ({pct:.3f}%)")
    print("  Highest coverage VINs:")
    for vin, total, present, pct in reversed(weight_presence[-10:]):
        print(f"    {vin:<15}  {present:>9,}/{total:>9,}  ({pct:.3f}%)")
    print()
    spread = weight_presence[-1][3] - weight_presence[0][3]
    if missing_pct["Weight_lbs"] > 10 and spread > 30:
        print("  INTERPRETATION: Weight_lbs appears event-driven or source-dependent.")
        print(f"  Coverage ranges from {weight_presence[0][3]:.2f}% to "
              f"{weight_presence[-1][3]:.2f}% across VINs.")
        print("  Do NOT drop rows solely because Weight_lbs is missing.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    _default_csv = str(Path(__file__).parent / "sample_trucks_dataset.csv")

    parser = argparse.ArgumentParser(description="Run data quality audit on truck telemetry CSV.")
    parser.add_argument("--input", default=_default_csv, help="Path to source CSV")
    args = parser.parse_args()

    run_audit(args.input)
