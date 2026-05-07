# PLAN2.md

## Real-Time Truck ETA Next Steps (as of 2026-05-06)

This document focuses on the immediate next implementation steps after the current baseline pipeline.

> Status note: API integration and Google Maps usage in this document are ideas/prototypes for planning only. They are not committed production requirements yet.

## 1) What Is Already Implemented (Baseline)

- Trip segmentation and feature pipeline are in place.
- Current feature set includes elapsed time, rolling speed stats, distance remaining, time encoding, day-of-week, weight presence flag, and trip progress.
- Baseline + LightGBM quantile models are trained and evaluated.
- Test suite is passing.

## 2) Phase 2b Focus: Load State Detection (Next Priority)

### Goal
Infer whether the truck is likely loaded or unloaded during a trip, then inject that state into ETA prediction features.

### Deliverables
- Create `src/load_state.py`.
- Implement change point detection on per-trip speed signal:
  - PELT (primary).
  - Binary Segmentation (comparison baseline).
- Use `Weight_lbs` (when available) as weak ground truth for event alignment.
- Output per-ping state label: `loaded`, `unloaded`, `unknown`.
- Create derived feature: `load_change_count` per trip.
- Integrate into feature builder and model training.

### What are PELT and BinSeg?
- **PELT (Pruned Exact Linear Time)**: a change-point detection algorithm that finds breakpoints in a time series by minimizing a total objective:
  - `total_cost = fit_error_within_segments + penalty_per_change * number_of_changes`
  In plain terms, it searches for the best set of regime changes (for example, unloaded vs loaded driving behavior) while penalizing too many breakpoints so it does not overfit noise.
- **BinSeg (Binary Segmentation)**: a greedy change-point method that repeatedly splits the signal at the strongest detected breakpoint, then recurses on each segment. It is usually faster and simpler, but can be less globally optimal than PELT.
- **How this helps ETA**: both methods can identify behavior shifts in speed patterns that may correspond to load/unload events, traffic regime changes, or stop-go phases.

#### How PELT works (simple view)
1. Build a 1D signal per trip (start with speed, optionally add acceleration).
2. Choose a segment cost (for example, piecewise-constant mean or variance change).
3. Sweep forward through the series and compute the best segmentation up to each time index using dynamic programming.
4. Prune candidate breakpoints that cannot be optimal later (this is the "pruned" part that makes it efficient).
5. Return the final set of change points and map segments to states (`loaded`, `unloaded`, `unknown`).

Practical tuning notes:
- Higher penalty -> fewer change points (more conservative, less sensitive).
- Lower penalty -> more change points (more sensitive, higher false-positive risk).
- Tune penalty against `Weight_lbs`-aligned events and report both F1 and detection lag.

### Evaluation
- Compare PELT vs BinSeg with:
  - Event F1 score.
  - Median detection lag (pings from true event).
- Add tests for edge cases:
  - Flat speed signal.
  - Short trips / single-ping trips.
  - Missing weight readings.

## 3) Feature Engineering Items Not Yet Implemented

### High-value additions
- `acceleration_mphps`: first difference of speed by trip over elapsed seconds.
- `trip_duration_estimate_min`: rolling estimate of remaining trip duration from recent pace.
- `stop_go_ratio`: ratio of low-speed pings (for congestion-like behavior).
- `bearing_change_rate`: proxy for route complexity / urban turns.
- `load_state` and `load_change_count` (from Phase 2b).

### Trip Category Features (New)
Add a trip category so models can learn different dynamics by haul length and operating pattern.

- `trip_category_short`: local/urban trips (example: expected total duration < 2 hours).
- `trip_category_medium`: regional trips (example: 2 to 8 hours).
- `trip_category_long`: long-haul/interstate trips (example: > 8 hours).
- `trip_category_stop_go`: optional operational category for high stop density routes.

### Trip Length-Based Categorization (Detailed)
You can categorize trips by either planned/observed distance or expected total duration. In practice, keeping both is useful because road class and congestion can make time and distance disagree.

Suggested initial distance bands (tune from your dataset percentiles):
- **Short-haul**: 0 to 150 km.
- **Medium-haul**: 150 to 500 km.
- **Long-haul**: > 500 km.

Suggested initial duration bands (if distance is noisy or unavailable):
- **Short-haul**: < 2 hours total trip duration.
- **Medium-haul**: 2 to 8 hours.
- **Long-haul**: > 8 hours.

How to use this in prediction:
- Train one global model with one-hot trip-category features as the first step.
- Then test a mixture strategy: one specialized model per category (short/medium/long).
- Route each live trip to a category model using current best estimate of total trip length.
- Re-evaluate category every N pings (for example every 5 pings) so early misclassification can self-correct.

Why this can improve ETA:
- Short-haul trips are more affected by stops, intersections, and local congestion; long-haul trips are more stable and highway-dominant.
- Error distributions are usually different across haul lengths, so one model may underfit one regime while overfitting another.
- Category-specific calibration can improve both MAE and interval coverage.

Practical safeguards:
- Add hysteresis when changing category mid-trip to avoid oscillation near boundaries.
- Keep an `unknown` category for cold-start trips with too little signal.
- Report metrics by category (MAE, RMSE, coverage) to confirm the categorization adds value.

Implementation notes:
- Start with short/medium/long based on historical trip duration or planned route distance.
- If true duration is unknown early in trip, use a provisional category from remaining distance + current pace and allow it to update as the trip progresses.
- Encode categories with one-hot features (for tree models this is still useful for interpretability and analysis).
- Evaluate per-category MAE to verify whether segmentation improves predictive quality.

### Categorical Encoding Plan (One-Hot Encoding)
Current model mostly uses numeric features. If categorical fields like `Device_Type` or `Source` are available and stable, encode them as follows:

**One-hot encoding definition:** one-hot encoding converts one categorical column into multiple binary columns (0/1), one per category value. For example, `trip_category` with values `{short, medium, long}` becomes three columns: `trip_category_short`, `trip_category_medium`, `trip_category_long`. Each row has exactly one `1` and the rest `0`.

- One-hot encode low-cardinality categories (`Device_Type`, `Source`) during training.
- Persist category vocabulary with the model so train/inference schemas match.
- Add unknown bucket handling for unseen categories in production.
- If cardinality becomes high, use frequency encoding or target encoding instead of full one-hot.

Recommended implementation approach:
- Use a `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for categorical columns.
- Keep numeric features untouched in parallel.
- Save the full preprocessing + model pipeline together with `joblib`.

## 4) Company-Facing API Design (Proposed)

This section is exploratory design only (not a committed build target yet).

### Endpoint set
- `POST /v1/eta/predict`
  - Predict ETA remaining from current GPS ping and destination.
- `POST /v1/eta/predict/batch`
  - Batch predictions for fleet updates.
- `GET /v1/health`
  - Liveness/readiness check.
- `POST /v1/trips/close`
  - Optional endpoint to close and archive trip state.

### Core request example

```json
{
  "vin": "1HGCM82633A123456",
  "trip_id": "TRIP-2026-05-06-001",
  "timestamp": "2026-05-06T15:04:00Z",
  "latitude": 41.8781,
  "longitude": -87.6298,
  "destination_latitude": 39.7392,
  "destination_longitude": -104.9903,
  "speed_mph": 55.0,
  "use_google_routes": false
}
```

### Core response example

```json
{
  "vin": "1HGCM82633A123456",
  "trip_id": "TRIP-2026-05-06-001",
  "eta_remaining_min": 1058.24,
  "distance_remaining_km": 1484.51,
  "method": "physics_speed_haversine",
  "google_route_eta_min": null,
  "blended_eta_min": 1058.24,
  "as_of": "2026-05-06T15:04:00Z"
}
```

### Prediction strategy
- At every ping, compute local ETA via distance/speed and model features.
- Optionally refresh route ETA from Google at lower cadence (for traffic awareness).
- Blend ETA values when both are available:
  - Example: `blended_eta = 0.7 * model_p50 + 0.3 * google_eta`.

## 5) Google API Use for ETA: Viability Analysis

This section is feasibility analysis only. Google API integration is an optional idea for calibration, not a dependency for core project completion.

> Guidance note: all limits, pricing references, and ping calculations below are planning estimates and example scenarios. Treat them as directional guidance, not a fixed implementation commitment.

### Recommended API choice
- Prefer Routes API (`ComputeRoutes`) over Distance Matrix (Legacy).
- Reason: Distance Matrix endpoint is in Legacy status and should not be the long-term integration path.

### Limits and implications
From Google documentation (checked 2026-05-06):
- Routes API `ComputeRoutes` limit: 3,000 QPM.
- Routes API `ComputeRouteMatrix` limit: 3,000 EPM.
- Distance Matrix (Legacy): billed per element and legacy status.

This means a hard throughput ceiling of:
- 3,000 requests/minute = 50 requests/second if every ping hits Google.

### Free vs paid usage (in ping terms)

Assumption for this table: `1 ping -> 1 Google ComputeRoutes request`.

These values are shown as possible planning scenarios only.

| Scenario | Included/allowed calls | Equivalent pings/month (if 1 ping = 1 request) | Notes |
|---|---:|---:|---|
| New account trial credit | `$300` credit (not a fixed call count) | depends on SKU price | Converts to calls based on active SKU pricing |
| Subscription: Starter plan | `50,000` monthly calls (plan total) | up to `50,000` pings/month | Routes coverage in Starter can vary by included products; verify in console before relying on this for Routes |
| Subscription: Essentials plan | `100,000` monthly calls | up to `100,000` pings/month | Includes Routes products per current pricing page |
| Subscription: Pro plan | `250,000` monthly calls | up to `250,000` pings/month | Includes Routes products per current pricing page |
| No free calls left (pay-as-you-go) | No monthly free bucket | unlimited by budget, billed per request/element | Still constrained by quota: 3,000 QPM (ComputeRoutes), 3,000 EPM (RouteMatrix) |

Operational interpretation:
- "How many pings for free" = the included monthly call bucket for your active plan/SKU.
- "How many without free" = as many as you are willing to pay for, but never above minute-level quota limits.

### How many pings can we use?
If you call Google on every ping, max fleet throughput is approximately:

- `max_trucks = (3000 * ping_interval_seconds) / 60`

Examples:
- 10-second pings -> about 500 trucks max.
- 30-second pings -> about 1,500 trucks max.
- 60-second pings -> about 3,000 trucks max.

These are quota ceilings only, not cost-optimized operation.

### Cost viability guidance
Google pricing pages now include plan-based and SKU-based billing that can change. Use the pricing calculator for exact monthly cost. Practical policy:

- Do not call Google per ping.
- Keep ML model as the per-ping predictor.
- Refresh Google route ETA every 5 to 15 minutes per active trip, or on route deviation only.
- Cache route responses by `(trip_id, rounded_origin, destination)` with TTL.

Rule-of-thumb viability:
- Viable for this project if Google calls are throttled to <= 10% to 20% of pings.
- Not viable at scale if every ping triggers a route call.

### Google API request examples and exact use cases

The request payloads below are example patterns to communicate possible integration ideas.

Use case A (single truck ETA refresh every 5 to 15 minutes): `ComputeRoutes`

```http
POST https://routes.googleapis.com/directions/v2:computeRoutes
X-Goog-Api-Key: YOUR_API_KEY
X-Goog-FieldMask: routes.duration,routes.distanceMeters
Content-Type: application/json

{
  "origin": {
    "location": {
      "latLng": {"latitude": 41.8781, "longitude": -87.6298}
    }
  },
  "destination": {
    "location": {
      "latLng": {"latitude": 39.7392, "longitude": -104.9903}
    }
  },
  "travelMode": "DRIVE",
  "routingPreference": "TRAFFIC_AWARE"
}
```

How it is used in this project:
- Read `routes.duration` as traffic-aware route ETA.
- Blend with model ETA (`P50`) instead of replacing model output.

Use case B (fleet dispatch board, many trucks to one depot): `ComputeRouteMatrix`

```http
POST https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix
X-Goog-Api-Key: YOUR_API_KEY
X-Goog-FieldMask: originIndex,destinationIndex,duration,distanceMeters,status
Content-Type: application/json

{
  "origins": [
    {"waypoint": {"location": {"latLng": {"latitude": 41.88, "longitude": -87.63}}}},
    {"waypoint": {"location": {"latLng": {"latitude": 41.50, "longitude": -88.10}}}}
  ],
  "destinations": [
    {"waypoint": {"location": {"latLng": {"latitude": 39.7392, "longitude": -104.9903}}}}
  ],
  "travelMode": "DRIVE",
  "routingPreference": "TRAFFIC_AWARE"
}
```

How it is used in this project:
- Batch-update ETA priors for multiple active trucks.
- Keep per-ping prediction local in the ML model; refresh matrix calls at lower cadence to control cost.

## 6) Runnable Artifacts Added

### A) Endpoint prototype server
File: `src/eta_endpoint_server.py`

What it does:
- Runs a local HTTP server.
- Implements `POST /v1/eta/predict`.
- Computes ETA from GPS and speed.
- Optionally calls Google Routes API if `GOOGLE_MAPS_API_KEY` is set and request asks for it.

Run:

```powershell
python src/eta_endpoint_server.py --port 8080
```

Example request (PowerShell):

```powershell
$body = @{
  vin = "1HGCM82633A123456"
  trip_id = "TRIP-2026-05-06-001"
  timestamp = "2026-05-06T15:04:00Z"
  latitude = 41.8781
  longitude = -87.6298
  destination_latitude = 39.7392
  destination_longitude = -104.9903
  speed_mph = 55
  use_google_routes = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/v1/eta/predict" -Method Post -ContentType "application/json" -Body $body
```

### B) Ping and quota viability estimator
File: `src/ping_viability_estimator.py`

What it does:
- Estimates monthly ping count and Google call count.
- Checks whether Google calls exceed 3,000 QPM quota.
- Computes budget usage from an adjustable price input.

Run example:

```powershell
python src/ping_viability_estimator.py --trucks 250 --ping-interval-sec 30 --active-hours-per-day 12 --google-refresh-ratio 0.15 --price-per-1000 5.0 --monthly-budget 2000
```

## 7) Implementation Order (Recommended)

1. Build `src/load_state.py` with PELT + BinSeg and tests.
2. Add new engineered features (especially acceleration and duration estimate).
3. Introduce one-hot encoding pipeline for categorical telematics fields.
4. Retrain and compare HGB vs LightGBM vs XGBoost.
5. Integrate production API service around the trained model.
6. Use Google Routes as periodic calibration, not per-ping dependency.

## 8) Exit Criteria for This Phase

- Load-state features integrated and tested.
- Coverage improved to >= 80% while maintaining or reducing MAE.
- API endpoint returns ETA reliably from GPS in a local integration test.
- Google call policy documented and constrained by quota + budget rules.
