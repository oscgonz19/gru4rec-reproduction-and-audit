"""Tests for baseline models."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import PopularityBaseline, MarkovBaseline


@pytest.fixture
def sample_data():
    """Create sample session data for testing."""
    data = {
        'SessionId': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'ItemId': [10, 20, 30, 10, 40, 20, 30, 40, 50],
        'Time': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    return pd.DataFrame(data)


class TestPopularityBaseline:
    def test_fit(self, sample_data):
        model = PopularityBaseline()
        model.fit(sample_data)

        assert model.item_counts is not None
        assert model.top_items is not None
        # Items 10, 20, 30, 40 appear twice each, 50 appears once
        assert len(model.top_items) == 5

    def test_predict(self, sample_data):
        model = PopularityBaseline()
        model.fit(sample_data)

        predictions = model.predict([10, 20], k=3)
        assert len(predictions) == 3
        # Should return most popular items

    def test_predict_before_fit_raises(self):
        model = PopularityBaseline()
        with pytest.raises(ValueError):
            model.predict([1, 2, 3])


class TestMarkovBaseline:
    def test_fit(self, sample_data):
        model = MarkovBaseline()
        model.fit(sample_data)

        assert model.transitions is not None
        # After 10, we see 20 (session 1) and 40 (session 2)
        assert 20 in model.transitions[10]
        assert 40 in model.transitions[10]

    def test_predict_with_known_transition(self, sample_data):
        model = MarkovBaseline()
        model.fit(sample_data)

        # After seeing item 10, should predict 20 or 40
        predictions = model.predict([10], k=5)
        assert len(predictions) > 0
        assert 20 in predictions or 40 in predictions

    def test_predict_unknown_item_falls_back_to_popularity(self, sample_data):
        model = MarkovBaseline()
        model.fit(sample_data)

        # Item 999 not in training data, should fall back to popularity
        predictions = model.predict([999], k=3)
        assert len(predictions) == 3

    def test_predict_before_fit_raises(self):
        model = MarkovBaseline()
        with pytest.raises(ValueError):
            model.predict([1, 2, 3])
