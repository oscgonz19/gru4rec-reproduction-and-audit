#!/usr/bin/env python
"""
Fetch the official GRU4Rec PyTorch implementation.

This script clones the official repository to vendor/ directory.
The vendor/ directory is gitignored to avoid redistributing the code.

Original implementation by Balázs Hidasi:
https://github.com/hidasib/GRU4Rec_PyTorch_Official
"""

import subprocess
import sys
from pathlib import Path


OFFICIAL_REPO = "https://github.com/hidasib/GRU4Rec_PyTorch_Official.git"
VENDOR_DIR = Path(__file__).parent.parent / "vendor"
GRU4REC_DIR = VENDOR_DIR / "GRU4Rec_PyTorch_Official"


def fetch_official(force: bool = False) -> Path:
    """Clone or update the official GRU4Rec repository.

    Args:
        force: If True, remove existing and re-clone.

    Returns:
        Path to the cloned repository.
    """
    VENDOR_DIR.mkdir(exist_ok=True)

    if GRU4REC_DIR.exists():
        if force:
            print(f"Removing existing {GRU4REC_DIR}")
            import shutil
            shutil.rmtree(GRU4REC_DIR)
        else:
            print(f"Official repo already exists at {GRU4REC_DIR}")
            print("Use --force to re-clone")
            return GRU4REC_DIR

    print(f"Cloning {OFFICIAL_REPO}")
    print(f"Destination: {GRU4REC_DIR}")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO, str(GRU4REC_DIR)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error cloning repository: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Successfully cloned official GRU4Rec implementation")
    print("\nAttribution:")
    print("  Original implementation by Balázs Hidasi")
    print("  Paper: 'Session-based Recommendations with Recurrent Neural Networks' (ICLR 2016)")
    print("  License: Free for research and education; contact author for commercial use")

    return GRU4REC_DIR


def get_gru4rec_path() -> Path:
    """Get path to GRU4Rec, fetching if necessary."""
    if not GRU4REC_DIR.exists():
        return fetch_official()
    return GRU4REC_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch official GRU4Rec implementation")
    parser.add_argument("--force", action="store_true", help="Force re-clone even if exists")
    args = parser.parse_args()

    fetch_official(force=args.force)
