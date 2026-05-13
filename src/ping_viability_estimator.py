"""Estimate ping throughput, Google API call volume, and budget viability.

Usage example:
python src/ping_viability_estimator.py \
  --trucks 250 \
  --ping-interval-sec 30 \
  --active-hours-per-day 12 \
  --google-refresh-ratio 0.15 \
  --price-per-1000 5.0 \
  --monthly-budget 2000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

ROUTES_API_QPM_LIMIT = 3000.0
DAYS_PER_MONTH = 30.0


@dataclass
class ViabilityResult:
    pings_per_month: float
    google_calls_per_month: float
    google_calls_per_minute: float
    quota_utilization_pct: float
    estimated_monthly_cost: float
    budget_utilization_pct: float
    quota_ok: bool
    budget_ok: bool


def compute_viability(
    trucks: int,
    ping_interval_sec: float,
    active_hours_per_day: float,
    google_refresh_ratio: float,
    price_per_1000: float,
    monthly_budget: float,
) -> ViabilityResult:
    """Compute scale and cost viability for Google-route-assisted ETA."""
    pings_per_day = trucks * (active_hours_per_day * 3600.0 / ping_interval_sec)
    pings_per_month = pings_per_day * DAYS_PER_MONTH

    google_calls_per_month = pings_per_month * google_refresh_ratio
    total_active_minutes_per_month = active_hours_per_day * 60.0 * DAYS_PER_MONTH
    google_calls_per_minute = google_calls_per_month / max(total_active_minutes_per_month, 1.0)

    quota_utilization_pct = (google_calls_per_minute / ROUTES_API_QPM_LIMIT) * 100.0
    estimated_monthly_cost = (google_calls_per_month / 1000.0) * price_per_1000
    budget_utilization_pct = (estimated_monthly_cost / max(monthly_budget, 1e-9)) * 100.0

    return ViabilityResult(
        pings_per_month=pings_per_month,
        google_calls_per_month=google_calls_per_month,
        google_calls_per_minute=google_calls_per_minute,
        quota_utilization_pct=quota_utilization_pct,
        estimated_monthly_cost=estimated_monthly_cost,
        budget_utilization_pct=budget_utilization_pct,
        quota_ok=google_calls_per_minute <= ROUTES_API_QPM_LIMIT,
        budget_ok=estimated_monthly_cost <= monthly_budget,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Google API viability for ETA project")
    parser.add_argument("--trucks", type=int, required=True, help="Number of active trucks")
    parser.add_argument("--ping-interval-sec", type=float, required=True, help="Ping interval in seconds")
    parser.add_argument("--active-hours-per-day", type=float, default=12.0, help="Active fleet hours per day")
    parser.add_argument(
        "--google-refresh-ratio",
        type=float,
        default=0.1,
        help="Fraction of pings that call Google (0.0 to 1.0)",
    )
    parser.add_argument(
        "--price-per-1000",
        type=float,
        default=5.0,
        help="Assumed Google price per 1000 calls (set from your SKU/plan)",
    )
    parser.add_argument("--monthly-budget", type=float, default=2000.0, help="Monthly budget in USD")
    args = parser.parse_args()

    if args.ping_interval_sec <= 0:
        raise ValueError("--ping-interval-sec must be > 0")
    if not (0.0 <= args.google_refresh_ratio <= 1.0):
        raise ValueError("--google-refresh-ratio must be between 0 and 1")

    result = compute_viability(
        trucks=args.trucks,
        ping_interval_sec=args.ping_interval_sec,
        active_hours_per_day=args.active_hours_per_day,
        google_refresh_ratio=args.google_refresh_ratio,
        price_per_1000=args.price_per_1000,
        monthly_budget=args.monthly_budget,
    )

    print("=== ETA Ping / Google API Viability ===")
    print(f"Trucks: {args.trucks}")
    print(f"Ping interval (sec): {args.ping_interval_sec}")
    print(f"Active hours/day: {args.active_hours_per_day}")
    print(f"Google refresh ratio: {args.google_refresh_ratio:.2%}")
    print()
    print(f"Pings/month: {result.pings_per_month:,.0f}")
    print(f"Google calls/month: {result.google_calls_per_month:,.0f}")
    print(f"Google calls/min (active window): {result.google_calls_per_minute:,.2f}")
    print(f"Quota utilization vs 3,000 QPM: {result.quota_utilization_pct:.2f}%")
    print()
    print(f"Estimated monthly Google cost: ${result.estimated_monthly_cost:,.2f}")
    print(f"Budget utilization: {result.budget_utilization_pct:.2f}%")
    print()
    print(f"Quota viable: {'YES' if result.quota_ok else 'NO'}")
    print(f"Budget viable: {'YES' if result.budget_ok else 'NO'}")


if __name__ == "__main__":
    main()
