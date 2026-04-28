"""Feature engineering module goals:

- Transform raw telematics records into model-ready ETA features.
- Derive time, movement, and route-context signals from sequential pings.
- Centralize feature definitions so training and inference use the same logic.
"""

