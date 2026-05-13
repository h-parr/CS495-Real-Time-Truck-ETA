"""Load state detection from speed signals.

Primary method uses a fast rolling-mean heuristic. Optional CPD methods
(PELT and Binary Segmentation) are available for offline comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


# ── Constants ────────────────────────────────────────────────

MIN_TRIP_LENGTH = 5       # minimum pings per trip to apply CPD methods
CPD_PENALTY_PELT = 1.0    # PELT penalty (lower = more sensitive)
CPD_PENALTY_BINSEG = 1.0  # Binary Segmentation penalty
SPEED_THRESHOLD = 10.0    # mph threshold for loaded/unloaded classification
MIN_SEGMENT_LENGTH = 2    # minimum pings per segment


# ── Change Point Detection (CPD) ─────────────────────────────

def detect_changepoints_pelt(
    speed: np.ndarray,
    penalty: float = CPD_PENALTY_PELT,
    min_size: int = 2,
) -> np.ndarray:
    """Detect change points using PELT algorithm.

    Parameters
    ----------
    speed : np.ndarray
        Speed values (mph) for a single trip, sorted by timestamp.
    penalty : float
        Penalty parameter for PELT cost function. Higher = fewer changepoints.
    min_size : int
        Minimum segment length (in pings).

    Returns
    -------
    np.ndarray
        Array of change point indices (1-based; 0 means no changepoints).
        E.g., [5, 12] means segments [0:5), [5:12), [12:end).
    """
    if not HAS_RUPTURES:
        return np.array([])

    if len(speed) < MIN_TRIP_LENGTH:
        return np.array([])

    # Normalize speed signal
    speed_clean = np.nan_to_num(speed, nan=0.0)
    if speed_clean.std() < 1e-6:
        # Flat signal — no changepoints
        return np.array([])

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(
        speed_clean.reshape(-1, 1)
    )
    changepoints = algo.predict(pen=penalty)
    # ruptures returns 1-based indices; convert to 0-based
    return np.array(changepoints[:-1])  # Exclude final boundary


def detect_changepoints_binseg(
    speed: np.ndarray,
    n_bkps: int | None = None,
    min_size: int = 2,
) -> np.ndarray:
    """Detect change points using Binary Segmentation algorithm.

    Parameters
    ----------
    speed : np.ndarray
        Speed values (mph) for a single trip.
    n_bkps : int or None
        Number of breakpoints to find. If None, estimate from PELT result.
    min_size : int
        Minimum segment length.

    Returns
    -------
    np.ndarray
        Array of change point indices (0-based).
    """
    if not HAS_RUPTURES:
        return np.array([])

    if len(speed) < MIN_TRIP_LENGTH:
        return np.array([])

    speed_clean = np.nan_to_num(speed, nan=0.0)
    if speed_clean.std() < 1e-6:
        return np.array([])

    # Estimate n_bkps if not provided
    if n_bkps is None:
        pelt_cps = detect_changepoints_pelt(speed, penalty=CPD_PENALTY_PELT)
        n_bkps = max(1, len(pelt_cps))

    algo = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(
        speed_clean.reshape(-1, 1)
    )
    changepoints = algo.predict(n_bkps=n_bkps)
    return np.array(changepoints[:-1])  # Exclude final boundary


# ── Load State Inference ─────────────────────────────────────


def infer_load_state_heuristic(speed: np.ndarray, window: int = 3) -> np.ndarray:
    """Fast heuristic load state inference based on rolling mean speed.
    
    Uses a rolling window mean to smooth the speed signal and classify
    each ping as loaded (low speed) or unloaded (high speed).
    Much faster than PELT for large datasets.
    """
    speed_clean = np.nan_to_num(speed, nan=0.0)
    
    if len(speed_clean) < window:
        load_state = np.where(speed_clean < SPEED_THRESHOLD, 1, 0)
        return load_state.astype(np.int8)
    
    rolling_mean = pd.Series(speed_clean).rolling(window=window, center=True, min_periods=1).mean().values
    load_state = np.where(rolling_mean < SPEED_THRESHOLD, 1, 0)
    return load_state.astype(np.int8)


def infer_load_state_from_segments(
    speed: np.ndarray,
    changepoints: np.ndarray,
    threshold: float = SPEED_THRESHOLD,
) -> np.ndarray:
    """Assign load state (loaded=1, unloaded=0, unknown=-1) to each ping.

    Algorithm:
    1. Split speed signal at detected changepoints.
    2. Compute mean speed in each segment.
    3. High mean speed (≥ threshold) → unloaded; Low → loaded.
    4. Assign state to all pings within that segment.

    Parameters
    ----------
    speed : np.ndarray
        Speed values (mph) for a single trip.
    changepoints : np.ndarray
        Change point indices (0-based).
    threshold : float
        Speed threshold (mph) to distinguish loaded from unloaded.

    Returns
    -------
    np.ndarray
        Load state per ping: 1 = loaded, 0 = unloaded, -1 = unknown.
    """
    load_state = np.full(len(speed), -1, dtype=np.int8)  # default: unknown

    speed_clean = np.nan_to_num(speed, nan=0.0)

    # Build segments (convert boundaries to int)
    boundaries = np.concatenate([[0], changepoints, [len(speed)]])
    boundaries = np.unique(boundaries).astype(int)

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        segment_speed = speed_clean[start_idx:end_idx]

        if len(segment_speed) > 0:
            mean_speed = np.mean(segment_speed)
            # Loaded = low speed; Unloaded = high speed
            state = 0 if mean_speed >= threshold else 1
            load_state[start_idx:end_idx] = state

    return load_state


def count_load_changes(load_state: np.ndarray) -> int:
    """Count number of load state transitions (loaded ↔ unloaded) per trip.

    Transitions to/from 'unknown' (-1) are not counted.

    Parameters
    ----------
    load_state : np.ndarray
        Load state per ping (1=loaded, 0=unloaded, -1=unknown).

    Returns
    -------
    int
        Number of state transitions (excluding unknown).
    """
    valid_state = load_state[load_state != -1]
    if len(valid_state) < 2:
        return 0
    transitions = np.sum(np.diff(valid_state) != 0)
    return int(transitions)


# ── Validation Against Weight Data ──────────────────────────

def evaluate_cpd_against_weight(
    load_state: np.ndarray,
    weight_lbs: np.ndarray | None,
    weight_threshold: float = 10000.0,
) -> dict[str, float]:
    """Evaluate CPD accuracy using Weight_lbs as ground truth.

    When Weight_lbs is present, use it to derive the true load state:
    weight > threshold → loaded (1); weight ≤ threshold → unloaded (0).

    Metrics: precision, recall, F1 score.

    Parameters
    ----------
    load_state : np.ndarray
        Inferred load state (1=loaded, 0=unloaded, -1=unknown).
    weight_lbs : np.ndarray or None
        Raw weight readings (may contain NaN).
    weight_threshold : float
        Weight (lbs) above which truck is considered loaded.

    Returns
    -------
    dict
        {precision, recall, f1, n_compared} or empty dict if insufficient data.
    """
    if weight_lbs is None:
        return {}

    weight_clean = np.array(weight_lbs, dtype=np.float64)
    valid_idx = ~np.isnan(weight_clean)

    if valid_idx.sum() < 5:
        # Not enough ground truth data
        return {}

    true_state = (weight_clean[valid_idx] > weight_threshold).astype(np.int8)
    pred_state = load_state[valid_idx]

    # Filter out unknown predictions
    valid_pred = pred_state != -1
    true_state = true_state[valid_pred]
    pred_state = pred_state[valid_pred]

    if len(pred_state) < 2:
        return {}

    # Compute metrics
    tp = np.sum((pred_state == 1) & (true_state == 1))
    fp = np.sum((pred_state == 1) & (true_state == 0))
    fn = np.sum((pred_state == 0) & (true_state == 1))
    tn = np.sum((pred_state == 0) & (true_state == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_compared": len(pred_state),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compare_cpd_methods(
    speed: np.ndarray,
    weight_lbs: np.ndarray | None = None,
) -> dict[str, dict]:
    """Compare PELT vs Binary Segmentation on a single trip.

    Parameters
    ----------
    speed : np.ndarray
        Speed values for the trip.
    weight_lbs : np.ndarray or None
        Weight values for validation.

    Returns
    -------
    dict
        {
            'pelt': {...changepoints, load_state, evaluation...},
            'binseg': {...},
        }
    """
    result = {}

    # PELT
    pelt_cps = detect_changepoints_pelt(speed)
    pelt_state = infer_load_state_from_segments(speed, pelt_cps)
    pelt_eval = evaluate_cpd_against_weight(pelt_state, weight_lbs)
    result['pelt'] = {
        'changepoints': pelt_cps,
        'load_state': pelt_state,
        'evaluation': pelt_eval,
        'n_changepoints': len(pelt_cps),
    }

    # Binary Segmentation
    binseg_cps = detect_changepoints_binseg(speed, n_bkps=len(pelt_cps))
    binseg_state = infer_load_state_from_segments(speed, binseg_cps)
    binseg_eval = evaluate_cpd_against_weight(binseg_state, weight_lbs)
    result['binseg'] = {
        'changepoints': binseg_cps,
        'load_state': binseg_state,
        'evaluation': binseg_eval,
        'n_changepoints': len(binseg_cps),
    }

    return result


# ── DataFrame API ───────────────────────────────────────────

def add_load_state(df: pd.DataFrame, method: str = "heuristic", verbose: bool = False) -> pd.DataFrame:
    """Add LoadState and load_change_count columns to a trip DataFrame.

    Default method is 'heuristic' (fast rolling mean-based), 'pelt' for slower but
    more accurate change point detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: trip_id, Speed, and optionally Weight_lbs.
    method : str
        CPD method: 'heuristic', 'pelt', or 'binseg'.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: load_state, load_change_count.
    """
    if method not in {"heuristic", "pelt", "binseg"}:
        raise ValueError("method must be one of: 'heuristic', 'pelt', 'binseg'")

    if not HAS_RUPTURES and method in {"pelt", "binseg"}:
        # CPD methods require ruptures. Keep behavior explicit.
        raise ImportError("ruptures is required when method is 'pelt' or 'binseg'")

    df = df.copy()
    load_states = []
    change_counts = []

    n_trips = df["trip_id"].nunique()
    trip_idx = 0

    for _, grp in df.groupby("trip_id", sort=False):
        trip_idx += 1
        if verbose and trip_idx % 100 == 0:
            print(f"  Processing trip {trip_idx}/{n_trips}")

        trip_df = grp.sort_values("Timestamp").reset_index(drop=True)
        speed = trip_df["Speed"].values

        if method == "heuristic":
            state = infer_load_state_heuristic(speed)
        else:
            if len(speed) >= MIN_TRIP_LENGTH:
                if method == "pelt":
                    cps = detect_changepoints_pelt(speed)
                else:  # binseg
                    pelt_cps = detect_changepoints_pelt(speed)
                    cps = detect_changepoints_binseg(speed, n_bkps=len(pelt_cps))
            else:
                cps = np.array([])

            state = infer_load_state_from_segments(speed, cps)

        load_states.extend(state)

        n_changes = count_load_changes(state)
        change_counts.extend([n_changes] * len(trip_df))

    df["load_state"] = load_states
    df["load_change_count"] = change_counts
    return df
