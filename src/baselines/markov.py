"""
First-order Markov chain baseline for session-based recommendations.

Predicts the next item based on transition probabilities from the last item.
Falls back to popularity when the last item has no transitions.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Optional

from ..metrics import recall_at_k, mrr_at_k


class MarkovBaseline:
    """First-order Markov chain recommendation baseline.

    Models P(next_item | current_item) using transition counts.
    """

    def __init__(self, alpha: float = 0.0):
        """Initialize Markov baseline.

        Args:
            alpha: Smoothing parameter for popularity fallback.
                   0 = pure Markov, 1 = pure popularity.
        """
        self.alpha = alpha
        self.transitions: Optional[dict] = None
        self.popularity: Optional[Counter] = None
        self.top_items: Optional[np.ndarray] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        session_key: str = 'SessionId',
        item_key: str = 'ItemId',
        time_key: str = 'Time'
    ) -> 'MarkovBaseline':
        """Fit the model by counting transitions.

        Args:
            train_df: Training data.
            session_key: Column name for session ID.
            item_key: Column name for item ID.
            time_key: Column name for timestamp.

        Returns:
            Self for chaining.
        """
        # Sort by session and time
        df = train_df.sort_values([session_key, time_key])

        # Count transitions
        self.transitions = defaultdict(Counter)

        for _, group in df.groupby(session_key):
            items = group[item_key].values
            for i in range(len(items) - 1):
                self.transitions[items[i]][items[i + 1]] += 1

        # Also compute popularity for fallback
        self.popularity = Counter(train_df[item_key])
        self.top_items = np.array([item for item, _ in self.popularity.most_common()])

        return self

    def predict(self, session: list, k: int = 20) -> np.ndarray:
        """Predict next items for a session.

        Args:
            session: List of item IDs in the session.
            k: Number of items to recommend.

        Returns:
            Array of top-k item IDs.
        """
        if self.transitions is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if not session:
            return self.top_items[:k]

        last_item = session[-1]
        transition_counts = self.transitions.get(last_item, Counter())

        if not transition_counts:
            # Fallback to popularity
            return self.top_items[:k]

        # Get top items by transition probability
        # Blend with popularity if alpha > 0
        if self.alpha > 0:
            scores = {}
            all_items = set(transition_counts.keys()) | set(self.top_items[:k * 2])

            max_trans = max(transition_counts.values()) if transition_counts else 1
            max_pop = max(self.popularity.values())

            for item in all_items:
                trans_score = transition_counts.get(item, 0) / max_trans
                pop_score = self.popularity.get(item, 0) / max_pop
                scores[item] = (1 - self.alpha) * trans_score + self.alpha * pop_score

            sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return np.array(sorted_items[:k])
        else:
            sorted_items = [item for item, _ in transition_counts.most_common(k)]
            # Pad with popular items if not enough transitions
            if len(sorted_items) < k:
                seen = set(sorted_items)
                for item in self.top_items:
                    if item not in seen:
                        sorted_items.append(item)
                        if len(sorted_items) >= k:
                            break
            return np.array(sorted_items[:k])

    def evaluate(
        self,
        test_df: pd.DataFrame,
        k: list[int] = [5, 10, 20],
        session_key: str = 'SessionId',
        item_key: str = 'ItemId',
        time_key: str = 'Time'
    ) -> dict:
        """Evaluate the model on test data.

        Args:
            test_df: Test data.
            k: List of cutoff values.
            session_key: Column name for session ID.
            item_key: Column name for item ID.
            time_key: Column name for timestamp.

        Returns:
            Dictionary with Recall@K and MRR@K for each k.
        """
        if self.transitions is None:
            raise ValueError("Model not fitted. Call fit() first.")

        results = {f'Recall@{kk}': 0.0 for kk in k}
        results.update({f'MRR@{kk}': 0.0 for kk in k})
        n_predictions = 0

        # Sort and group by session
        df = test_df.sort_values([session_key, time_key])

        for session_id, group in df.groupby(session_key):
            items = group[item_key].values

            for i in range(len(items) - 1):
                target = items[i + 1]
                history = list(items[:i + 1])
                predictions = self.predict(history, k=max(k))

                for kk in k:
                    results[f'Recall@{kk}'] += recall_at_k(predictions[:kk], target)
                    results[f'MRR@{kk}'] += mrr_at_k(predictions[:kk], target)

                n_predictions += 1

        # Average
        for key in results:
            if key != 'n_predictions':
                results[key] /= n_predictions if n_predictions > 0 else 1

        results['n_predictions'] = n_predictions
        return results
