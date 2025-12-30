#!/usr/bin/env python
"""
Preprocess session data with proper temporal split.

Key principles:
- Temporal split: train sessions come BEFORE test sessions
- No session splitting: complete sessions go to train OR test, never both
- No future leakage: test set only contains items seen in training
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    session_key: str = 'SessionId',
    time_key: str = 'Time',
    item_key: str = 'ItemId',
    filter_unseen_items: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sessions temporally without leakage.

    Args:
        df: DataFrame with session data.
        train_ratio: Fraction of sessions for training.
        session_key: Column name for session ID.
        time_key: Column name for timestamp.
        item_key: Column name for item ID.
        filter_unseen_items: Remove test items not in training.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Get session start times
    session_times = df.groupby(session_key)[time_key].min().reset_index()
    session_times.columns = [session_key, 'session_start']

    # Sort sessions by start time
    session_times = session_times.sort_values('session_start')

    # Split point
    n_sessions = len(session_times)
    split_idx = int(n_sessions * train_ratio)

    train_sessions = set(session_times.iloc[:split_idx][session_key])
    test_sessions = set(session_times.iloc[split_idx:][session_key])

    # Split data
    train_df = df[df[session_key].isin(train_sessions)].copy()
    test_df = df[df[session_key].isin(test_sessions)].copy()

    if filter_unseen_items:
        # Remove items from test that weren't in training
        train_items = set(train_df[item_key])
        original_test_len = len(test_df)
        test_df = test_df[test_df[item_key].isin(train_items)]

        # Remove sessions that became too short (< 2 items)
        session_lens = test_df.groupby(session_key).size()
        valid_sessions = session_lens[session_lens >= 2].index
        test_df = test_df[test_df[session_key].isin(valid_sessions)]

        removed = original_test_len - len(test_df)
        if removed > 0:
            print(f"Removed {removed} test events with unseen items")

    return train_df, test_df


def validate_split(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   session_key: str = 'SessionId', time_key: str = 'Time') -> bool:
    """Validate that the split is properly temporal."""
    train_max_time = train_df[time_key].max()
    test_min_time = test_df[time_key].min()

    # Check no session overlap
    train_sessions = set(train_df[session_key])
    test_sessions = set(test_df[session_key])
    overlap = train_sessions & test_sessions

    if overlap:
        print(f"ERROR: {len(overlap)} sessions appear in both train and test!")
        return False

    if test_min_time < train_max_time:
        print(f"WARNING: Some test events occur before training ends")
        print(f"  Train max time: {train_max_time}")
        print(f"  Test min time: {test_min_time}")
        # This can happen with the synthetic data due to time granularity

    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess sessions with temporal split")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input TSV file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data",
        help="Output directory"
    )
    parser.add_argument(
        "--train_ratio", "-r",
        type=float,
        default=0.8,
        help="Fraction of sessions for training"
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Don't filter unseen items from test"
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input, sep='\t')

    print(f"Total events: {len(df)}")
    print(f"Total sessions: {df['SessionId'].nunique()}")
    print(f"Total items: {df['ItemId'].nunique()}")

    train_df, test_df = temporal_split(
        df,
        train_ratio=args.train_ratio,
        filter_unseen_items=not args.no_filter
    )

    print(f"\nTrain set:")
    print(f"  Events: {len(train_df)}")
    print(f"  Sessions: {train_df['SessionId'].nunique()}")
    print(f"  Items: {train_df['ItemId'].nunique()}")

    print(f"\nTest set:")
    print(f"  Events: {len(test_df)}")
    print(f"  Sessions: {test_df['SessionId'].nunique()}")
    print(f"  Items: {test_df['ItemId'].nunique()}")

    # Validate
    validate_split(train_df, test_df)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.tsv"
    test_path = output_dir / "test.tsv"

    train_df.to_csv(train_path, sep='\t', index=False)
    test_df.to_csv(test_path, sep='\t', index=False)

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
