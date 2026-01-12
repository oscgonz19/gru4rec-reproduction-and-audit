"""
Popularity baseline for session-based recommendations.

Recommends the most popular items regardless of session context.
This is the simplest baseline and establishes a lower bound.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional

# Handle both package and direct imports
try:
    from ..metrics import recall_at_k, mrr_at_k
except ImportError:
    from metrics import recall_at_k, mrr_at_k


class PopularityBaseline:
    """Popularity-based recommendation baseline.

    Always recommends the globally most popular items.
    """

    def __init__(self):
        self.item_counts: Optional[Counter] = None
        self.top_items: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame, item_key: str = 'ItemId') -> 'PopularityBaseline':
        """Fit the model by counting item occurrences.

        Args:
            train_df: Training data with item interactions.
            item_key: Column name for item ID.

        Returns:
            Self for chaining.
        """
        self.item_counts = Counter(train_df[item_key])
        # Pre-compute top items sorted by popularity
        self.top_items = np.array([item for item, _ in self.item_counts.most_common()])
        return self

    def predict(self, session: list, k: int = 20) -> np.ndarray:
        """Predict next items for a session.

        Args:
            session: List of item IDs in the session (ignored for popularity).
            k: Number of items to recommend.

        Returns:
            Array of top-k item IDs.
        """
        if self.top_items is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.top_items[:k]

    def evaluate(
        self,
        test_df: pd.DataFrame,
        k: list[int] = [5, 10, 20],
        session_key: str = 'SessionId',
        item_key: str = 'ItemId'
    ) -> dict:
        """Evaluate the model on test data.

        Uses next-item prediction: for each position in a session,
        predict the next item using the history so far.

        Args:
            test_df: Test data.
            k: List of cutoff values.
            session_key: Column name for session ID.
            item_key: Column name for item ID.

        Returns:
            Dictionary with Recall@K and MRR@K for each k.
        """
        if self.top_items is None:
            raise ValueError("Model not fitted. Call fit() first.")

        results = {f'Recall@{kk}': 0.0 for kk in k}
        results.update({f'MRR@{kk}': 0.0 for kk in k})
        n_predictions = 0

        # Group by session
        for session_id, group in test_df.groupby(session_key):
            items = group[item_key].values

            # For each position (except first), predict next item
            for i in range(len(items) - 1):
                target = items[i + 1]
                predictions = self.predict(items[:i + 1], k=max(k))

                for kk in k:
                    results[f'Recall@{kk}'] += recall_at_k(predictions[:kk], target)
                    results[f'MRR@{kk}'] += mrr_at_k(predictions[:kk], target)

                n_predictions += 1

        # Average
        for key in results:
            results[key] /= n_predictions if n_predictions > 0 else 1

        results['n_predictions'] = n_predictions
        return results
