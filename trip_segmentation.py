"""Trip segmentation module goals:

- Split continuous telemetry streams into discrete trips.
- Identify trip start/end boundaries using time gaps, speed, and location signals.
- Produce trip-level outputs that downstream ETA feature and model code can consume.
"""
