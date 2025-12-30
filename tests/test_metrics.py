"""Tests for metrics module."""

import numpy as np
import pytest

from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k, compute_rank


class TestRecallAtK:
    def test_target_in_predictions(self):
        predictions = np.array([1, 2, 3, 4, 5])
        assert recall_at_k(predictions, 3) == 1.0

    def test_target_not_in_predictions(self):
        predictions = np.array([1, 2, 3, 4, 5])
        assert recall_at_k(predictions, 10) == 0.0

    def test_target_at_first_position(self):
        predictions = np.array([5, 2, 3, 4, 1])
        assert recall_at_k(predictions, 5) == 1.0


class TestMRRAtK:
    def test_target_at_first_position(self):
        predictions = np.array([5, 2, 3, 4, 1])
        assert mrr_at_k(predictions, 5) == 1.0

    def test_target_at_third_position(self):
        predictions = np.array([1, 2, 5, 4, 3])
        assert mrr_at_k(predictions, 5) == pytest.approx(1.0 / 3.0)

    def test_target_not_in_predictions(self):
        predictions = np.array([1, 2, 3, 4, 5])
        assert mrr_at_k(predictions, 10) == 0.0


class TestNDCGAtK:
    def test_target_at_first_position(self):
        predictions = np.array([5, 2, 3, 4, 1])
        # NDCG = 1/log2(1+1) = 1
        assert ndcg_at_k(predictions, 5) == 1.0

    def test_target_at_second_position(self):
        predictions = np.array([1, 5, 3, 4, 2])
        # NDCG = 1/log2(2+1) = 1/log2(3)
        expected = 1.0 / np.log2(3)
        assert ndcg_at_k(predictions, 5) == pytest.approx(expected)


class TestComputeRank:
    def test_standard_mode(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        # target_idx=1 has score 0.5, one score (0.9) is higher
        assert compute_rank(scores, target_idx=1, mode='standard') == 2

    def test_conservative_mode(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        # target_idx=1 has score 0.5, two scores (0.5, 0.9) are >= 0.5
        assert compute_rank(scores, target_idx=1, mode='conservative') == 2

    def test_top_ranked(self):
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        assert compute_rank(scores, target_idx=3, mode='standard') == 1
        assert compute_rank(scores, target_idx=3, mode='conservative') == 1
