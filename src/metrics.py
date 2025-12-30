"""
Evaluation metrics for session-based recommendations.

Implements Recall@K and MRR@K with consistent definitions
matching the official GRU4Rec evaluation protocol.
"""

import numpy as np
from typing import Union


def recall_at_k(predictions: np.ndarray, target: Union[int, np.ndarray]) -> float:
    """Compute Recall@K for a single prediction.

    Recall@K = 1 if target is in top-K predictions, 0 otherwise.

    Args:
        predictions: Array of predicted item IDs (top-K).
        target: Ground truth item ID(s).

    Returns:
        1.0 if target in predictions, 0.0 otherwise.
    """
    if isinstance(target, (int, np.integer)):
        return 1.0 if target in predictions else 0.0
    else:
        # Multiple targets: return fraction found
        found = sum(1 for t in target if t in predictions)
        return found / len(target)


def mrr_at_k(predictions: np.ndarray, target: Union[int, np.ndarray]) -> float:
    """Compute Mean Reciprocal Rank at K.

    MRR@K = 1/rank if target is in top-K, 0 otherwise.
    Rank is 1-indexed (first position = rank 1).

    Args:
        predictions: Array of predicted item IDs (top-K).
        target: Ground truth item ID.

    Returns:
        Reciprocal rank if found, 0.0 otherwise.
    """
    if isinstance(target, (int, np.integer)):
        try:
            idx = np.where(predictions == target)[0]
            if len(idx) > 0:
                return 1.0 / (idx[0] + 1)
            return 0.0
        except (ValueError, IndexError):
            return 0.0
    else:
        # Multiple targets: return average MRR
        mrr_sum = 0.0
        for t in target:
            mrr_sum += mrr_at_k(predictions, t)
        return mrr_sum / len(target)


def ndcg_at_k(predictions: np.ndarray, target: Union[int, np.ndarray]) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    For single-target next-item prediction, NDCG@K = 1/log2(rank+1) if found.

    Args:
        predictions: Array of predicted item IDs (top-K).
        target: Ground truth item ID.

    Returns:
        NDCG score.
    """
    if isinstance(target, (int, np.integer)):
        try:
            idx = np.where(predictions == target)[0]
            if len(idx) > 0:
                return 1.0 / np.log2(idx[0] + 2)  # +2 because log2(1)=0
            return 0.0
        except (ValueError, IndexError):
            return 0.0
    else:
        # Multiple targets: compute DCG/IDCG
        dcg = 0.0
        for i, pred in enumerate(predictions):
            if pred in target:
                dcg += 1.0 / np.log2(i + 2)

        # IDCG: best case is all targets at top
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(target), len(predictions))))

        return dcg / idcg if idcg > 0 else 0.0


def compute_rank(scores: np.ndarray, target_idx: int, mode: str = 'conservative') -> int:
    """Compute rank of target item given scores.

    This matches the GRU4Rec evaluation protocol.

    Args:
        scores: Array of scores for all items.
        target_idx: Index of the target item.
        mode: Ranking mode:
            - 'standard': rank = count(score > target_score) + 1
            - 'conservative': rank = count(score >= target_score)
            - 'median': rank = count(score > target) + 0.5*count(score == target) + 0.5

    Returns:
        Rank of target item (1-indexed).
    """
    target_score = scores[target_idx]

    if mode == 'standard':
        return int((scores > target_score).sum() + 1)
    elif mode == 'conservative':
        return int((scores >= target_score).sum())
    elif mode == 'median':
        above = (scores > target_score).sum()
        tied = (scores == target_score).sum()
        return int(above + 0.5 * (tied - 1) + 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def metrics_from_ranks(
    ranks: np.ndarray,
    cutoffs: list[int] = [5, 10, 20]
) -> dict:
    """Compute metrics from precomputed ranks.

    Args:
        ranks: Array of ranks for each prediction.
        cutoffs: List of K values.

    Returns:
        Dictionary with Recall@K and MRR@K.
    """
    results = {}

    for k in cutoffs:
        in_top_k = ranks <= k
        results[f'Recall@{k}'] = in_top_k.mean()
        results[f'MRR@{k}'] = np.where(in_top_k, 1.0 / ranks, 0.0).mean()

    return results
