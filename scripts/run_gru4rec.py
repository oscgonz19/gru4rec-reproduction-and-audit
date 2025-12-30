#!/usr/bin/env python
"""
Wrapper script to run the official GRU4Rec implementation.

This script provides a simplified interface to the vendor's run.py,
handling path setup and common configurations.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

# Add vendor to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "GRU4Rec_PyTorch_Official"


def check_vendor():
    """Ensure vendor directory exists."""
    if not VENDOR_DIR.exists():
        print("Official GRU4Rec not found. Fetching...")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "fetch_official.py")], check=True)


def train(args):
    """Train a GRU4Rec model."""
    check_vendor()

    # Convert to absolute paths since we run from vendor directory
    data_path = Path(args.data).resolve()
    model_path = Path(args.model).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(VENDOR_DIR / "run.py"),
        str(data_path),
        "-ps", f"layers={args.layers},batch_size={args.batch_size},n_epochs={args.epochs},loss={args.loss}",
        "-d", args.device,
        "-s", str(model_path)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(VENDOR_DIR))

    if result.returncode == 0:
        print(f"\nModel saved to: {model_path}")

        # Save training config
        config = {
            'data': str(data_path),
            'layers': args.layers,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'loss': args.loss,
            'device': args.device
        }
        config_path = model_path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_path}")

    return result.returncode


def evaluate(args):
    """Evaluate a trained model."""
    check_vendor()

    # Convert to absolute paths since we run from vendor directory
    model_path = Path(args.model).resolve()
    test_path = Path(args.test).resolve()

    cmd = [
        sys.executable,
        str(VENDOR_DIR / "run.py"),
        str(model_path),
        "-l",  # load model
        "-t", str(test_path),
        "-m", *[str(c) for c in args.cutoffs],
        "-e", args.mode,
        "-d", args.device
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(VENDOR_DIR), capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse and save results
    if result.returncode == 0:
        results = {'model': str(model_path), 'test': str(test_path), 'output': result.stdout}
        results_path = model_path.with_suffix('.results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run GRU4Rec (wrapper)")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', '-d', type=str, required=True, help='Training data TSV')
    train_parser.add_argument('--model', '-m', type=str, default='results/model.pt', help='Output model path')
    train_parser.add_argument('--layers', '-l', type=int, default=100, help='GRU hidden size')
    train_parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    train_parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--loss', type=str, default='cross-entropy',
                              choices=['cross-entropy', 'bpr-max'], help='Loss function')
    train_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda:0)')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('--model', '-m', type=str, required=True, help='Model path')
    eval_parser.add_argument('--test', '-t', type=str, required=True, help='Test data TSV')
    eval_parser.add_argument('--cutoffs', '-c', type=int, nargs='+', default=[5, 10, 20],
                             help='Cutoff values for metrics')
    eval_parser.add_argument('--mode', type=str, default='conservative',
                             choices=['standard', 'conservative', 'median'], help='Ranking mode')
    eval_parser.add_argument('--device', type=str, default='cpu', help='Device')

    args = parser.parse_args()

    if args.command == 'train':
        sys.exit(train(args))
    elif args.command == 'eval':
        sys.exit(evaluate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
