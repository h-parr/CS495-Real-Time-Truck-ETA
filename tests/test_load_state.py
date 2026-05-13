"""Tests for load state detection via change point detection."""

import numpy as np
import pandas as pd
import pytest

from src.load_state import (
    add_load_state,
    compare_cpd_methods,
    count_load_changes,
    detect_changepoints_binseg,
    detect_changepoints_pelt,
    evaluate_cpd_against_weight,
    infer_load_state_from_segments,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def flat_speed():
    """Flat speed signal (no structure)."""
    return np.array([20.0] * 20)


@pytest.fixture
def two_segment_speed():
    """Two distinct speed segments: [slow, fast]."""
    return np.concatenate([
        np.full(10, 5.0),   # slow (loaded)
        np.full(10, 30.0),  # fast (unloaded)
    ])


@pytest.fixture
def multi_segment_speed():
    """Multiple alternating speed segments."""
    return np.concatenate([
        np.full(8, 5.0),    # slow
        np.full(8, 35.0),   # fast
        np.full(8, 8.0),    # slow
        np.full(8, 25.0),   # fast
    ])


@pytest.fixture
def short_speed():
    """Short trip (< MIN_TRIP_LENGTH)."""
    return np.array([20.0, 21.0, 22.0])


@pytest.fixture
def trip_with_weights():
    """Trip DataFrame with Weight_lbs."""
    n = 20
    return pd.DataFrame({
        'trip_id': 'T1',
        'Timestamp': pd.date_range('2026-04-01 08:00', periods=n, freq='min'),
        'Latitude': np.linspace(40.0, 40.1, n),
        'Longitude': np.linspace(-74.0, -73.9, n),
        'Speed': np.concatenate([np.full(10, 5.0), np.full(10, 30.0)]),
        'Weight_lbs': np.concatenate([
            np.full(10, 45000.0),  # loaded
            np.full(10, np.nan),   # no weight (assume unloaded)
        ]),
    })


# ── Test CPD Detection ───────────────────────────────────────

class TestDetectChangepoints:
    def test_pelt_flat_signal(self, flat_speed):
        """PELT should detect no changepoints in flat signal."""
        cps = detect_changepoints_pelt(flat_speed)
        assert len(cps) == 0

    def test_pelt_two_segment(self, two_segment_speed):
        """PELT should detect one changepoint in two-segment signal."""
        # Lower penalty to make it more sensitive for testing
        cps = detect_changepoints_pelt(two_segment_speed, penalty=0.5)
        # Should have at least one changepoint near index 10
        assert len(cps) > 0, f"Expected changepoints but got {cps}"
        assert any(8 <= cp <= 12 for cp in cps)

    def test_pelt_short_signal(self, short_speed):
        """PELT should return empty array for short signals."""
        cps = detect_changepoints_pelt(short_speed)
        assert len(cps) == 0

    def test_pelt_with_nan(self):
        """PELT should handle NaN values gracefully."""
        speed = np.array([5.0, np.nan, 5.0, 30.0, 30.0, 30.0] * 3)
        cps = detect_changepoints_pelt(speed)
        # Should not crash; may or may not detect changepoints
        assert isinstance(cps, np.ndarray)

    def test_binseg_flat_signal(self, flat_speed):
        """BinSeg should detect no changepoints in flat signal."""
        cps = detect_changepoints_binseg(flat_speed)
        assert len(cps) == 0

    def test_binseg_two_segment(self, two_segment_speed):
        """BinSeg should detect changepoints in two-segment signal."""
        # Explicitly ask for 1 changepoint
        cps = detect_changepoints_binseg(two_segment_speed, n_bkps=1)
        assert len(cps) > 0, f"Expected changepoints but got {cps}"

    def test_binseg_short_signal(self, short_speed):
        """BinSeg should return empty array for short signals."""
        cps = detect_changepoints_binseg(short_speed)
        assert len(cps) == 0

    def test_binseg_respects_n_bkps(self, multi_segment_speed):
        """BinSeg should find at most n_bkps changepoints."""
        cps = detect_changepoints_binseg(multi_segment_speed, n_bkps=2)
        assert len(cps) <= 2


# ── Test Load State Inference ────────────────────────────────

class TestInferLoadState:
    def test_single_segment_low_speed(self):
        """Single segment with low mean speed → loaded."""
        speed = np.full(10, 5.0)
        cps = np.array([])
        state = infer_load_state_from_segments(speed, cps, threshold=10.0)
        assert np.all(state == 1)  # all loaded

    def test_single_segment_high_speed(self):
        """Single segment with high mean speed → unloaded."""
        speed = np.full(10, 30.0)
        cps = np.array([])
        state = infer_load_state_from_segments(speed, cps, threshold=10.0)
        assert np.all(state == 0)  # all unloaded

    def test_two_segments(self):
        """Two segments: loaded then unloaded."""
        speed = np.concatenate([np.full(10, 5.0), np.full(10, 30.0)])
        cps = np.array([10])  # changepoint at index 10
        state = infer_load_state_from_segments(speed, cps, threshold=10.0)
        assert np.all(state[:10] == 1)  # first segment loaded
        assert np.all(state[10:] == 0)  # second segment unloaded

    def test_with_nan(self):
        """Handle NaN values in speed signal."""
        speed = np.array([5.0, np.nan, 5.0, np.nan] * 5)
        cps = np.array([])
        state = infer_load_state_from_segments(speed, cps)
        assert state.dtype == np.int8
        assert len(state) == len(speed)
        # NaNs become 0, so low speed
        assert np.all(state == 1)


class TestCountLoadChanges:
    def test_no_changes(self):
        """Constant load state → zero transitions."""
        state = np.array([1, 1, 1, 1, 1])
        n_changes = count_load_changes(state)
        assert n_changes == 0

    def test_single_transition(self):
        """One transition: loaded → unloaded."""
        state = np.array([1, 1, 1, 0, 0, 0])
        n_changes = count_load_changes(state)
        assert n_changes == 1

    def test_multiple_transitions(self):
        """Multiple transitions."""
        state = np.array([1, 1, 0, 0, 1, 1])
        n_changes = count_load_changes(state)
        assert n_changes == 2

    def test_ignore_unknown(self):
        """Unknown state (-1) should not create transitions."""
        state = np.array([1, 1, -1, 0, 0])
        n_changes = count_load_changes(state)
        # Only transition between 1 and 0 (ignoring -1)
        assert n_changes == 1

    def test_all_unknown(self):
        """All unknown → zero transitions."""
        state = np.array([-1, -1, -1])
        n_changes = count_load_changes(state)
        assert n_changes == 0

    def test_too_short(self):
        """Single element → zero transitions."""
        state = np.array([1])
        n_changes = count_load_changes(state)
        assert n_changes == 0


# ── Test Validation Against Weight ──────────────────────────

class TestValidateCPDAgainstWeight:
    def test_perfect_agreement(self):
        """Perfect match between predicted and weight-derived state."""
        pred_state = np.array([1, 1, 1, 0, 0, 0])
        weight = np.array([45000.0, 45000.0, 45000.0, 5000.0, 5000.0, 5000.0])
        metrics = evaluate_cpd_against_weight(pred_state, weight, weight_threshold=10000.0)
        assert metrics.get('f1', 0) == pytest.approx(1.0)

    def test_no_weight_data(self):
        """Returns empty dict when weight data is missing."""
        pred_state = np.array([1, 1, 1, 0, 0, 0])
        metrics = evaluate_cpd_against_weight(pred_state, None)
        assert metrics == {}

    def test_insufficient_weight_data(self):
        """Returns empty dict when < 5 weight readings."""
        pred_state = np.array([1, 1, 1, 0, 0, 0])
        weight = np.array([45000.0, np.nan, np.nan, np.nan, np.nan, np.nan])
        metrics = evaluate_cpd_against_weight(pred_state, weight)
        assert metrics == {}

    def test_unknown_predictions_filtered(self):
        """Unknown predictions (-1) should be filtered out."""
        pred_state = np.array([1, 1, -1, 0, 0])
        weight = np.array([45000.0, 45000.0, 45000.0, 5000.0, 5000.0])
        metrics = evaluate_cpd_against_weight(pred_state, weight)
        assert metrics['n_compared'] == 4  # -1 is excluded

    def test_compute_precision_recall_f1(self):
        """Compute metrics correctly."""
        pred_state = np.array([1, 1, 0, 0, 0, 1])
        weight = np.array([45000.0, 45000.0, 5000.0, 50000.0, 3000.0, 48000.0])
        metrics = evaluate_cpd_against_weight(pred_state, weight, weight_threshold=10000.0)
        # Predictions: [1, 1, 0, 0, 0, 1], True: [1, 1, 0, 1, 0, 1]
        # TP=3 (indices 0,1,5), FP=0, FN=1 (index 3)
        assert metrics['tp'] == 3
        assert metrics['fp'] == 0
        assert metrics['fn'] == 1


# ── Test Method Comparison ───────────────────────────────────

class TestComparisonCPDMethods:
    def test_returns_both_methods(self, two_segment_speed):
        """Result should contain both 'pelt' and 'binseg'."""
        result = compare_cpd_methods(two_segment_speed)
        assert 'pelt' in result
        assert 'binseg' in result

    def test_both_methods_return_results(self, two_segment_speed):
        """Both methods should return load_state arrays."""
        result = compare_cpd_methods(two_segment_speed)
        assert 'load_state' in result['pelt']
        assert 'load_state' in result['binseg']
        assert len(result['pelt']['load_state']) == len(two_segment_speed)
        assert len(result['binseg']['load_state']) == len(two_segment_speed)

    def test_with_weight_validation(self, trip_with_weights):
        """Compare methods with weight validation."""
        speed = trip_with_weights['Speed'].values
        weight = trip_with_weights['Weight_lbs'].values
        result = compare_cpd_methods(speed, weight)

        # Both should have evaluation results
        assert 'evaluation' in result['pelt']
        assert 'evaluation' in result['binseg']


# ── Test DataFrame API ───────────────────────────────────────

class TestAddLoadState:
    def test_basic_add_load_state(self, trip_with_weights):
        """add_load_state should add required columns."""
        df = add_load_state(trip_with_weights, method='pelt')
        assert 'load_state' in df.columns
        assert 'load_change_count' in df.columns
        assert len(df) == len(trip_with_weights)

    def test_load_state_values(self, trip_with_weights):
        """Load state values should be in {-1, 0, 1}."""
        df = add_load_state(trip_with_weights, method='pelt')
        valid_states = {-1, 0, 1}
        assert set(df['load_state'].unique()).issubset(valid_states)

    def test_load_change_count_scalar(self, trip_with_weights):
        """load_change_count should be consistent per trip."""
        df = add_load_state(trip_with_weights, method='pelt')
        assert len(set(df['load_change_count'])) == 1  # All same per trip

    def test_binseg_method(self, trip_with_weights):
        """Method parameter should accept 'binseg'."""
        df = add_load_state(trip_with_weights, method='binseg')
        assert 'load_state' in df.columns
        assert 'load_change_count' in df.columns

    def test_multiple_trips(self):
        """Test with multiple trips."""
        df = pd.DataFrame({
            'trip_id': ['T1'] * 10 + ['T2'] * 10,
            'Timestamp': pd.date_range('2026-04-01', periods=20, freq='min'),
            'Speed': np.concatenate([
                np.full(10, 20.0),
                np.concatenate([np.full(5, 5.0), np.full(5, 30.0)])
            ]),
            'Weight_lbs': np.nan,
        })
        df_out = add_load_state(df)
        assert 'load_state' in df_out.columns
        # Trip 1 should have 0 changes (constant speed)
        assert df_out[df_out['trip_id'] == 'T1']['load_change_count'].iloc[0] == 0


# ── Edge Cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_length_speed(self):
        """Handle empty speed array."""
        speed = np.array([])
        cps = detect_changepoints_pelt(speed)
        assert len(cps) == 0

    def test_single_value_speed(self):
        """Handle single value."""
        speed = np.array([20.0])
        cps = detect_changepoints_pelt(speed)
        assert len(cps) == 0

    def test_all_nan_speed(self):
        """Handle all-NaN speed."""
        speed = np.full(10, np.nan)
        cps = detect_changepoints_pelt(speed)
        # Should handle gracefully (returns empty or flat detection)
        assert isinstance(cps, np.ndarray)

    def test_alternating_values(self):
        """Alternating speed values."""
        speed = np.array([5.0, 30.0] * 10)
        state = infer_load_state_from_segments(speed, np.array([]))
        # Mean is 17.5, threshold is 10.0, so 17.5 >= 10.0 → unloaded (0)
        assert np.all(state == 0)
