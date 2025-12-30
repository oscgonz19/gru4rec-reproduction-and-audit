#!/usr/bin/env python
"""
Generate comparison reports for the reproduction study.

Creates visualizations comparing GRU4Rec vs baselines.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load all result files from a directory.

    Args:
        results_dir: Path to results directory.

    Returns:
        Dictionary mapping model name to results.
    """
    results = {}

    for json_file in results_dir.glob("*.results.json"):
        model_name = json_file.stem.replace('.results', '')
        with open(json_file) as f:
            results[model_name] = json.load(f)

    return results


def parse_gru4rec_output(output: str) -> dict:
    """Parse GRU4Rec evaluation output to extract metrics.

    Args:
        output: Raw output string from GRU4Rec evaluation.

    Returns:
        Dictionary with parsed metrics.
    """
    metrics = {}

    for line in output.split('\n'):
        line = line.strip()
        # Look for lines like "Recall@20: 0.1234" or "MRR@20: 0.0567"
        if '@' in line and ':' in line:
            try:
                metric, value = line.split(':')
                metrics[metric.strip()] = float(value.strip())
            except (ValueError, IndexError):
                continue

    return metrics


def plot_comparison(
    results: dict,
    cutoffs: list[int] = [5, 10, 20],
    output_path: Optional[Path] = None
) -> None:
    """Create comparison bar chart.

    Args:
        results: Dictionary mapping model name to metrics dict.
        cutoffs: List of K values to plot.
        output_path: Path to save the figure. If None, displays.
    """
    models = list(results.keys())
    n_models = len(models)
    n_cutoffs = len(cutoffs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n_cutoffs)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Recall plot
    ax1 = axes[0]
    for i, model in enumerate(models):
        model_results = results[model]
        recalls = [model_results.get(f'Recall@{k}', 0) for k in cutoffs]
        ax1.bar(x + i * width, recalls, width, label=model, color=colors[i])

    ax1.set_xlabel('Cutoff K')
    ax1.set_ylabel('Recall@K')
    ax1.set_title('Recall Comparison')
    ax1.set_xticks(x + width * (n_models - 1) / 2)
    ax1.set_xticklabels([f'@{k}' for k in cutoffs])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # MRR plot
    ax2 = axes[1]
    for i, model in enumerate(models):
        model_results = results[model]
        mrrs = [model_results.get(f'MRR@{k}', 0) for k in cutoffs]
        ax2.bar(x + i * width, mrrs, width, label=model, color=colors[i])

    ax2.set_xlabel('Cutoff K')
    ax2.set_ylabel('MRR@K')
    ax2.set_title('MRR Comparison')
    ax2.set_xticks(x + width * (n_models - 1) / 2)
    ax2.set_xticklabels([f'@{k}' for k in cutoffs])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    else:
        plt.show()


def generate_markdown_table(results: dict, cutoffs: list[int] = [5, 10, 20]) -> str:
    """Generate a markdown table comparing results.

    Args:
        results: Dictionary mapping model name to metrics dict.
        cutoffs: List of K values.

    Returns:
        Markdown table string.
    """
    models = list(results.keys())

    # Header
    header = "| Model |"
    for k in cutoffs:
        header += f" Recall@{k} | MRR@{k} |"

    separator = "|" + "---|" * (1 + 2 * len(cutoffs))

    # Rows
    rows = []
    for model in models:
        row = f"| {model} |"
        for k in cutoffs:
            recall = results[model].get(f'Recall@{k}', 0)
            mrr = results[model].get(f'MRR@{k}', 0)
            row += f" {recall:.4f} | {mrr:.4f} |"
        rows.append(row)

    return "\n".join([header, separator] + rows)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument(
        "--results_dir", "-r",
        type=str,
        default="results",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/comparison.png",
        help="Output path for comparison plot"
    )
    parser.add_argument(
        "--cutoffs", "-k",
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help="Cutoff values for metrics"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No result files found in {results_dir}")
        print("Run evaluations first to generate results.")
        return

    print(f"Found results for: {list(results.keys())}")

    # Generate plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, cutoffs=args.cutoffs, output_path=output_path)

    # Print markdown table
    print("\n## Results Comparison\n")
    print(generate_markdown_table(results, cutoffs=args.cutoffs))


if __name__ == "__main__":
    main()
