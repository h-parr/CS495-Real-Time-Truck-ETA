"""Minimal ETA endpoint prototype for GPS-based ETA prediction.

This server provides a company-facing endpoint:
  POST /v1/eta/predict

It computes ETA from current GPS location, destination, and speed.
Optionally, it can query Google Routes API when requested and when
GOOGLE_MAPS_API_KEY is available in environment variables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EARTH_RADIUS_KM = 6371.0
KM_PER_MILE = 1.60934
GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two lat/lon points."""
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def local_eta_minutes(distance_km: float, speed_mph: float) -> float:
    """Return ETA in minutes using distance/speed with a safe minimum speed."""
    speed_kmph = max(speed_mph, 1.0) * KM_PER_MILE
    hours = distance_km / speed_kmph
    return hours * 60.0


def call_google_routes_eta_minutes(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> float | None:
    """Return Google Routes duration in minutes, or None on failure."""
    body = {
        "origin": {
            "location": {
                "latLng": {"latitude": origin_lat, "longitude": origin_lon}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": dest_lat, "longitude": dest_lon}
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }

    request_data = json.dumps(body).encode("utf-8")
    req = Request(
        GOOGLE_ROUTES_URL,
        data=request_data,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.duration",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

    routes = payload.get("routes", [])
    if not routes:
        return None

    duration_text = routes[0].get("duration", "")
    if not duration_text.endswith("s"):
        return None

    try:
        duration_seconds = float(duration_text[:-1])
    except ValueError:
        return None

    return duration_seconds / 60.0


class EtaHandler(BaseHTTPRequestHandler):
    """HTTP handler for ETA prediction endpoint."""

    server_version = "EtaPrototype/0.1"

    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (HTTP method name)
        if self.path == "/v1/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 (HTTP method name)
        if self.path != "/v1/eta/predict":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._send_json(400, {"error": "Request body is required"})
            return

        try:
            body = self.rfile.read(length)
            req = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Invalid JSON"})
            return

        required = [
            "vin",
            "trip_id",
            "latitude",
            "longitude",
            "destination_latitude",
            "destination_longitude",
            "speed_mph",
        ]
        missing = [key for key in required if key not in req]
        if missing:
            self._send_json(400, {"error": f"Missing fields: {', '.join(missing)}"})
            return

        try:
            lat = float(req["latitude"])
            lon = float(req["longitude"])
            dest_lat = float(req["destination_latitude"])
            dest_lon = float(req["destination_longitude"])
            speed_mph = float(req["speed_mph"])
        except (TypeError, ValueError):
            self._send_json(400, {"error": "latitude/longitude/speed_mph must be numeric"})
            return

        distance_km = haversine_km(lat, lon, dest_lat, dest_lon)
        eta_local = local_eta_minutes(distance_km, speed_mph)

        use_google = bool(req.get("use_google_routes", False))
        google_eta_min = None
        if use_google:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
            if api_key:
                google_eta_min = call_google_routes_eta_minutes(
                    api_key,
                    lat,
                    lon,
                    dest_lat,
                    dest_lon,
                )

        if google_eta_min is not None:
            blended_eta = (0.7 * eta_local) + (0.3 * google_eta_min)
            method = "local_plus_google_blend"
        else:
            blended_eta = eta_local
            method = "physics_speed_haversine"

        response = {
            "vin": str(req["vin"]),
            "trip_id": str(req["trip_id"]),
            "distance_remaining_km": round(distance_km, 3),
            "eta_remaining_min": round(eta_local, 3),
            "google_route_eta_min": None if google_eta_min is None else round(google_eta_min, 3),
            "blended_eta_min": round(blended_eta, 3),
            "method": method,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }

        self._send_json(200, response)

    def log_message(self, fmt: str, *args: object) -> None:
        """Keep server logs concise and useful for local demos."""
        print("[eta-api]", fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETA prototype HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), EtaHandler)
    print(f"ETA endpoint running on http://{args.host}:{args.port}")
    print("POST /v1/eta/predict and GET /v1/health are available")
    server.serve_forever()


if __name__ == "__main__":
    main()
