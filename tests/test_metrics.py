"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.metrics import coverage, evaluate, mae, mape, pinball_loss, rmse


class TestMAE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_known(self):
        assert mae(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(3.5)


class TestRMSE:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known(self):
        assert rmse(np.array([0.0]), np.array([3.0])) == pytest.approx(3.0)


class TestMAPE:
    def test_zero_true_excluded(self):
        result = mape(np.array([0.0, 10.0]), np.array([5.0, 12.0]))
        # Only 10→12 counts: |10-12|/10 = 0.2 → 20%
        assert result == pytest.approx(20.0)


class TestPinball:
    def test_over_prediction_p90(self):
        # If y_true < y_pred at q=0.9, penalty is (1-0.9)*|err|
        loss = pinball_loss(np.array([5.0]), np.array([10.0]), 0.9)
        assert loss == pytest.approx(0.1 * 5.0)

    def test_under_prediction_p10(self):
        # If y_true > y_pred at q=0.1, penalty is 0.1*|err|
        loss = pinball_loss(np.array([10.0]), np.array([5.0]), 0.1)
        assert loss == pytest.approx(0.1 * 5.0)


class TestCoverage:
    def test_all_inside(self):
        y = np.array([2.0, 3.0, 4.0])
        assert coverage(y, np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 5.0])) == 1.0

    def test_none_inside(self):
        y = np.array([10.0, 20.0])
        assert coverage(y, np.array([0.0, 0.0]), np.array([5.0, 5.0])) == 0.0

    def test_partial(self):
        y = np.array([3.0, 10.0])
        cov = coverage(y, np.array([1.0, 1.0]), np.array([5.0, 5.0]))
        assert cov == pytest.approx(0.5)


class TestEvaluate:
    def test_basic_report(self):
        y = np.array([10.0, 20.0, 30.0])
        p50 = np.array([11.0, 19.0, 31.0])
        report = evaluate(y, p50)
        assert "mae" in report
        assert "rmse" in report
        assert "mape" in report

    def test_with_quantiles(self):
        y = np.array([10.0, 20.0, 30.0])
        p50 = y.copy()
        p10 = y - 5
        p90 = y + 5
        report = evaluate(y, p50, p10, p90)
        assert report["coverage_p10_p90"] == 1.0
