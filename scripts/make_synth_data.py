#!/usr/bin/env python
"""
Generate synthetic session data for testing and CI.

Creates a realistic session dataset with:
- Power-law item popularity distribution
- Variable session lengths (2-20 items)
- Temporal ordering of sessions
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_sessions(
    n_sessions: int = 1000,
    n_items: int = 500,
    min_session_len: int = 2,
    max_session_len: int = 20,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic session data.

    Args:
        n_sessions: Number of sessions to generate.
        n_items: Number of unique items.
        min_session_len: Minimum items per session.
        max_session_len: Maximum items per session.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [SessionId, ItemId, Time].
    """
    np.random.seed(seed)

    # Power-law popularity distribution (Zipf-like)
    item_weights = 1.0 / np.arange(1, n_items + 1) ** 0.8
    item_weights /= item_weights.sum()

    sessions = []
    current_time = 1000000000  # Base timestamp

    for session_id in range(n_sessions):
        # Session length follows a distribution biased toward shorter sessions
        session_len = np.random.randint(min_session_len, max_session_len + 1)

        # Sample items for this session (with replacement for realistic behavior)
        items = np.random.choice(n_items, size=session_len, p=item_weights)

        # Generate timestamps within session (small increments)
        for i, item_id in enumerate(items):
            sessions.append({
                'SessionId': session_id,
                'ItemId': int(item_id),
                'Time': current_time + i * 10  # 10 second intervals within session
            })

        # Gap between sessions (1-60 minutes)
        current_time += session_len * 10 + np.random.randint(60, 3600)

    df = pd.DataFrame(sessions)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic session data")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synth_sessions.tsv",
        help="Output file path (TSV format)"
    )
    parser.add_argument(
        "--n_sessions", "-n",
        type=int,
        default=1000,
        help="Number of sessions to generate"
    )
    parser.add_argument(
        "--n_items", "-i",
        type=int,
        default=500,
        help="Number of unique items"
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=2,
        help="Minimum session length"
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Maximum session length"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print(f"Generating synthetic data:")
    print(f"  Sessions: {args.n_sessions}")
    print(f"  Items: {args.n_items}")
    print(f"  Session length: {args.min_len}-{args.max_len}")

    df = generate_synthetic_sessions(
        n_sessions=args.n_sessions,
        n_items=args.n_items,
        min_session_len=args.min_len,
        max_session_len=args.max_len,
        seed=args.seed
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as TSV
    df.to_csv(output_path, sep='\t', index=False)

    print(f"\nGenerated {len(df)} events across {args.n_sessions} sessions")
    print(f"Unique items: {df['ItemId'].nunique()}")
    print(f"Avg session length: {len(df) / args.n_sessions:.1f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
