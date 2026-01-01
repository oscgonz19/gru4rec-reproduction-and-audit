"""
Visualization module for GRU4Rec Reproduction Study.

Creates publication-quality plots for storytelling and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Style configuration for consistent, professional plots
STYLE_CONFIG = {
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 10,
    'legend.frameon': False,
}

# Color palette (colorblind-friendly)
COLORS = {
    'popularity': '#2ecc71',  # Green
    'markov': '#3498db',      # Blue
    'gru4rec': '#e74c3c',     # Red
    'neutral': '#95a5a6',     # Gray
    'highlight': '#f39c12',   # Orange
    'background': '#ecf0f1',  # Light gray
}


def apply_style():
    """Apply consistent style to all plots."""
    plt.rcParams.update(STYLE_CONFIG)


def save_figure(fig, path: Path, formats: List[str] = ['png', 'svg']):
    """Save figure in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'),
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
    plt.close(fig)


# =============================================================================
# DATA EXPLORATION PLOTS
# =============================================================================

def plot_session_length_distribution(df: pd.DataFrame,
                                      output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot the distribution of session lengths.

    Shows how many items users typically interact with per session.
    """
    apply_style()

    session_lengths = df.groupby('SessionId').size()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with KDE-like smooth line
    counts, bins, patches = ax.hist(session_lengths, bins=30,
                                     color=COLORS['neutral'],
                                     edgecolor='white',
                                     alpha=0.7)

    # Add mean and median lines
    mean_len = session_lengths.mean()
    median_len = session_lengths.median()

    ax.axvline(mean_len, color=COLORS['gru4rec'], linestyle='--',
               linewidth=2, label=f'Mean: {mean_len:.1f}')
    ax.axvline(median_len, color=COLORS['markov'], linestyle=':',
               linewidth=2, label=f'Median: {median_len:.1f}')

    ax.set_xlabel('Session Length (number of items)')
    ax.set_ylabel('Number of Sessions')
    ax.set_title('Distribution of Session Lengths\nHow many items do users view per session?')
    ax.legend()

    # Add annotation
    ax.annotate(f'Total Sessions: {len(session_lengths):,}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                fontsize=10, color=COLORS['neutral'])

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_item_popularity_distribution(df: pd.DataFrame,
                                       output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot item popularity distribution (long-tail).

    Visualizes the power-law distribution typical in e-commerce.
    """
    apply_style()

    item_counts = df['ItemId'].value_counts().sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Linear scale
    ax1.bar(range(len(item_counts)), item_counts.values,
            color=COLORS['popularity'], alpha=0.7, width=1.0)
    ax1.set_xlabel('Item Rank')
    ax1.set_ylabel('Number of Interactions')
    ax1.set_title('Item Popularity Distribution\n(Linear Scale)')

    # Highlight top 20%
    top_20_idx = int(len(item_counts) * 0.2)
    top_20_share = item_counts.iloc[:top_20_idx].sum() / item_counts.sum() * 100
    ax1.axvline(top_20_idx, color=COLORS['gru4rec'], linestyle='--', linewidth=2)
    ax1.annotate(f'Top 20% items\n= {top_20_share:.0f}% of interactions',
                 xy=(top_20_idx, item_counts.max() * 0.8),
                 xytext=(top_20_idx + len(item_counts) * 0.1, item_counts.max() * 0.8),
                 arrowprops=dict(arrowstyle='->', color=COLORS['gru4rec']),
                 fontsize=10, color=COLORS['gru4rec'])

    # Right: Log-log scale (power law)
    ax2.loglog(range(1, len(item_counts) + 1), item_counts.values,
               'o', color=COLORS['markov'], alpha=0.5, markersize=4)
    ax2.set_xlabel('Item Rank (log)')
    ax2.set_ylabel('Number of Interactions (log)')
    ax2.set_title('Long-Tail Distribution\n(Log-Log Scale)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_temporal_pattern(df: pd.DataFrame,
                          output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot temporal patterns in the data.
    """
    apply_style()

    df = df.copy()
    df['Time_dt'] = pd.to_datetime(df['Time'], unit='s')
    df['Hour'] = df['Time_dt'].dt.hour

    hourly_counts = df.groupby('Hour').size()

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(hourly_counts.index, hourly_counts.values,
                  color=COLORS['markov'], alpha=0.7, edgecolor='white')

    # Highlight peak hours
    peak_hour = hourly_counts.idxmax()
    bars[peak_hour].set_color(COLORS['highlight'])

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Interactions')
    ax.set_title('Interaction Volume by Hour\nWhen are users most active?')
    ax.set_xticks(range(0, 24, 2))

    if output_path:
        save_figure(fig, output_path)

    return fig


# =============================================================================
# MODEL COMPARISON PLOTS
# =============================================================================

def plot_model_comparison(results: Dict[str, Dict[str, float]],
                          output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a comprehensive model comparison visualization.

    Args:
        results: Dict with model names as keys, metric dicts as values
                 e.g., {'Popularity': {'Recall@20': 0.35, 'MRR@20': 0.12}, ...}
    """
    apply_style()

    models = list(results.keys())
    metrics = list(results[models[0]].keys())

    # Create figure with subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    model_colors = [COLORS.get(m.lower(), COLORS['neutral']) for m in models]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[m][metric] for m in models]

        bars = ax.bar(models, values, color=model_colors, alpha=0.8, edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5),
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

        ax.set_ylabel(metric)
        ax.set_title(metric.replace('@', ' @ '))
        ax.set_ylim(0, max(values) * 1.2)

    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_recall_at_k_curves(results: Dict[str, Dict[str, float]],
                            output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot Recall@K curves for different models.

    Shows how recall improves as K increases.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    model_styles = {
        'Popularity': {'color': COLORS['popularity'], 'marker': 'o', 'linestyle': '-'},
        'Markov': {'color': COLORS['markov'], 'marker': 's', 'linestyle': '--'},
        'GRU4Rec': {'color': COLORS['gru4rec'], 'marker': '^', 'linestyle': '-'},
    }

    for model, metrics in results.items():
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith('Recall@')}
        if not recall_metrics:
            continue

        ks = sorted([int(k.split('@')[1]) for k in recall_metrics.keys()])
        recalls = [metrics[f'Recall@{k}'] for k in ks]

        style = model_styles.get(model, {'color': COLORS['neutral'], 'marker': 'x', 'linestyle': ':'})
        ax.plot(ks, recalls, label=model, linewidth=2.5, markersize=10, **style)

    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('Recall@K')
    ax.set_title('Recall@K Comparison\nHow often is the target item in top-K recommendations?')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    # Add interpretation annotation
    ax.annotate('Higher is better ↑',
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10, color=COLORS['neutral'],
                style='italic')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_mrr_at_k_curves(results: Dict[str, Dict[str, float]],
                         output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot MRR@K curves for different models.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    model_styles = {
        'Popularity': {'color': COLORS['popularity'], 'marker': 'o', 'linestyle': '-'},
        'Markov': {'color': COLORS['markov'], 'marker': 's', 'linestyle': '--'},
        'GRU4Rec': {'color': COLORS['gru4rec'], 'marker': '^', 'linestyle': '-'},
    }

    for model, metrics in results.items():
        mrr_metrics = {k: v for k, v in metrics.items() if k.startswith('MRR@')}
        if not mrr_metrics:
            continue

        ks = sorted([int(k.split('@')[1]) for k in mrr_metrics.keys()])
        mrrs = [metrics[f'MRR@{k}'] for k in ks]

        style = model_styles.get(model, {'color': COLORS['neutral'], 'marker': 'x', 'linestyle': ':'})
        ax.plot(ks, mrrs, label=model, linewidth=2.5, markersize=10, **style)

    ax.set_xlabel('K (Number of Recommendations)')
    ax.set_ylabel('MRR@K')
    ax.set_title('Mean Reciprocal Rank Comparison\nHow high does the target item rank?')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    ax.annotate('Higher is better ↑',
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10, color=COLORS['neutral'],
                style='italic')

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_training_curve(losses: List[float],
                        output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot training loss curve.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(losses) + 1)

    ax.plot(epochs, losses, color=COLORS['gru4rec'], linewidth=2.5, marker='o', markersize=8)
    ax.fill_between(epochs, losses, alpha=0.2, color=COLORS['gru4rec'])

    # Annotate improvement
    if len(losses) > 1:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        ax.annotate(f'↓ {improvement:.1f}% improvement',
                   xy=(len(losses), losses[-1]),
                   xytext=(len(losses) - 0.5, losses[-1] + (losses[0] - losses[-1]) * 0.3),
                   fontsize=12, color=COLORS['gru4rec'], fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gru4rec']))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('GRU4Rec Training Progress\nLoss decreasing indicates learning')
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

    if output_path:
        save_figure(fig, output_path)

    return fig


# =============================================================================
# STORYTELLING PLOTS
# =============================================================================

def plot_problem_statement(output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a visual representation of the anonymous user problem.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'The Anonymous User Problem',
            fontsize=20, fontweight='bold', ha='center')

    # User icons (circles)
    # Anonymous users (70-80%)
    np.random.seed(42)
    for i in range(16):
        x = 1 + (i % 4) * 0.8
        y = 6 + (i // 4) * 0.8
        circle = plt.Circle((x, y), 0.3, color=COLORS['neutral'], alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, '?', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # Known users (20-30%)
    for i in range(4):
        x = 6 + (i % 2) * 0.8
        y = 7 + (i // 2) * 0.8
        circle = plt.Circle((x, y), 0.3, color=COLORS['popularity'], alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, '✓', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    # Labels
    ax.text(2.5, 5, '70-80%\nAnonymous', ha='center', fontsize=14, color=COLORS['neutral'], fontweight='bold')
    ax.text(6.5, 5, '20-30%\nKnown', ha='center', fontsize=14, color=COLORS['popularity'], fontweight='bold')

    # Problem box
    rect = mpatches.FancyBboxPatch((0.5, 1), 9, 2.5,
                                     boxstyle='round,pad=0.1',
                                     facecolor=COLORS['background'],
                                     edgecolor=COLORS['gru4rec'],
                                     linewidth=2)
    ax.add_patch(rect)

    ax.text(5, 2.8, 'Traditional recommenders FAIL for anonymous users:',
            ha='center', fontsize=12, fontweight='bold', color=COLORS['gru4rec'])
    ax.text(5, 2.0, '✗ Require weeks of user history    ✗ Need login/cookies    ✗ Depend on explicit ratings',
            ha='center', fontsize=11)
    ax.text(5, 1.3, '→ Lost revenue: 20-40% of potential conversions',
            ha='center', fontsize=11, style='italic', color=COLORS['gru4rec'])

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_solution_comparison(output_path: Optional[Path] = None) -> plt.Figure:
    """
    Visual comparison of traditional vs session-based recommendations.
    """
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Traditional approach
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Traditional Approach', fontsize=16, fontweight='bold', color=COLORS['neutral'])

    # User history (long)
    for i, item in enumerate(['Shoes', 'Shirt', 'Bag', '...', 'Phone']):
        ax1.text(1 + i * 1.5, 7, item, fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor=COLORS['background']))
    ax1.text(5, 5.5, '← Weeks/months of history →', ha='center', fontsize=11, color=COLORS['neutral'])

    # Arrow
    ax1.annotate('', xy=(5, 3.5), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=2))

    # Simple recommendation
    ax1.text(5, 2.5, 'Recommend: More shoes', fontsize=14, ha='center',
             bbox=dict(boxstyle='round', facecolor=COLORS['background']))

    # Right: Session-based approach
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Session-Based (GRU4Rec)', fontsize=16, fontweight='bold', color=COLORS['gru4rec'])

    # Current session only
    session_items = ['Running', 'Shoes', 'Socks', 'Water']
    for i, item in enumerate(session_items):
        ax2.text(2 + i * 1.5, 7, item, fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='#fadbd8'))
        if i < len(session_items) - 1:
            ax2.annotate('', xy=(2.8 + i * 1.5, 7), xytext=(2.2 + i * 1.5, 7),
                        arrowprops=dict(arrowstyle='->', color=COLORS['markov'], lw=1.5))

    ax2.text(5, 5.5, '← Current session only →', ha='center', fontsize=11, color=COLORS['gru4rec'])

    # GRU processing
    rect = mpatches.FancyBboxPatch((2, 4), 6, 1,
                                     boxstyle='round',
                                     facecolor=COLORS['gru4rec'],
                                     alpha=0.2)
    ax2.add_patch(rect)
    ax2.text(5, 4.5, 'GRU Neural Network', ha='center', fontsize=12, color=COLORS['gru4rec'])

    # Arrow
    ax2.annotate('', xy=(5, 3), xytext=(5, 3.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['gru4rec'], lw=2))

    # Smart recommendation
    ax2.text(5, 2, 'Recommend: Running gear, fitness accessories', fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='#fadbd8'))
    ax2.text(5, 1.2, '✓ Understands sequential intent', ha='center', fontsize=10,
             color=COLORS['gru4rec'], style='italic')

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_evaluation_protocol(output_path: Optional[Path] = None) -> plt.Figure:
    """
    Visualize the difference between sampled and full ranking evaluation.
    """
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sampled evaluation (misleading)
    ax1.set_title('Sampled Evaluation (Misleading)', fontsize=14, fontweight='bold', color=COLORS['highlight'])

    items_sampled = ['Target', 'Neg 1', 'Neg 2', '...', 'Neg 100']
    colors_sampled = [COLORS['gru4rec']] + [COLORS['neutral']] * 4
    y_pos = range(len(items_sampled))

    ax1.barh(y_pos, [0.9, 0.3, 0.25, 0.2, 0.1], color=colors_sampled, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(items_sampled)
    ax1.set_xlabel('Score')
    ax1.invert_yaxis()

    ax1.annotate('Recall@20 ≈ 80%\n(Inflated!)', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, color=COLORS['highlight'],
                bbox=dict(boxstyle='round', facecolor='#fef9e7'))

    # Full ranking (realistic)
    ax2.set_title('Full Ranking (Realistic)', fontsize=14, fontweight='bold', color=COLORS['popularity'])

    # Simulate 10000 items
    np.random.seed(42)
    n_items = 50  # Representing many items
    scores = np.random.exponential(0.3, n_items)
    scores = np.sort(scores)[::-1]

    colors_full = [COLORS['neutral']] * n_items
    target_pos = 15  # Target at position 15
    colors_full[target_pos] = COLORS['gru4rec']

    ax2.barh(range(n_items), scores, color=colors_full, alpha=0.7, height=0.8)
    ax2.set_ylabel('Item Rank (among ALL items)')
    ax2.set_xlabel('Score')
    ax2.invert_yaxis()

    ax2.annotate('Target item', xy=(scores[target_pos], target_pos),
                xytext=(scores[target_pos] + 0.3, target_pos),
                arrowprops=dict(arrowstyle='->', color=COLORS['gru4rec']),
                fontsize=10, color=COLORS['gru4rec'])

    ax2.annotate('Recall@20 ≈ 35%\n(Realistic)', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12, color=COLORS['popularity'],
                bbox=dict(boxstyle='round', facecolor='#eafaf1'))

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_architecture_diagram(output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a visual diagram of the GRU4Rec architecture.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(6, 11.5, 'GRU4Rec Architecture', fontsize=18, fontweight='bold', ha='center')

    # Input session
    ax.text(6, 10.5, 'Input Session: [item₁, item₂, item₃, ...]',
            fontsize=12, ha='center', style='italic')

    # Embedding layer
    rect1 = mpatches.FancyBboxPatch((2, 8.5), 8, 1.2,
                                      boxstyle='round,pad=0.1',
                                      facecolor=COLORS['markov'],
                                      alpha=0.3,
                                      edgecolor=COLORS['markov'],
                                      linewidth=2)
    ax.add_patch(rect1)
    ax.text(6, 9.1, 'Embedding Layer', fontsize=14, fontweight='bold', ha='center', color=COLORS['markov'])
    ax.text(6, 8.7, 'Items → Dense Vectors', fontsize=10, ha='center')

    # Arrow
    ax.annotate('', xy=(6, 7.8), xytext=(6, 8.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # GRU layers
    rect2 = mpatches.FancyBboxPatch((2, 5.5), 8, 2,
                                      boxstyle='round,pad=0.1',
                                      facecolor=COLORS['gru4rec'],
                                      alpha=0.3,
                                      edgecolor=COLORS['gru4rec'],
                                      linewidth=2)
    ax.add_patch(rect2)
    ax.text(6, 7.1, 'GRU Layers', fontsize=14, fontweight='bold', ha='center', color=COLORS['gru4rec'])
    ax.text(6, 6.5, 'hₜ = GRU(hₜ₋₁, xₜ)', fontsize=12, ha='center', family='monospace')
    ax.text(6, 5.9, 'Learns sequential patterns', fontsize=10, ha='center')

    # Arrow
    ax.annotate('', xy=(6, 4.8), xytext=(6, 5.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output layer
    rect3 = mpatches.FancyBboxPatch((2, 3.3), 8, 1.2,
                                      boxstyle='round,pad=0.1',
                                      facecolor=COLORS['popularity'],
                                      alpha=0.3,
                                      edgecolor=COLORS['popularity'],
                                      linewidth=2)
    ax.add_patch(rect3)
    ax.text(6, 3.9, 'Output Layer', fontsize=14, fontweight='bold', ha='center', color=COLORS['popularity'])
    ax.text(6, 3.5, 'Score for each item in catalog', fontsize=10, ha='center')

    # Arrow
    ax.annotate('', xy=(6, 2.6), xytext=(6, 3.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Predictions
    ax.text(6, 2.2, 'Top-K Recommendations', fontsize=14, fontweight='bold', ha='center')
    ax.text(6, 1.7, '[item₄₂, item₁₇, item₈₉, ...]', fontsize=12, ha='center', family='monospace')

    # Side annotations
    ax.text(11, 9.1, 'd = 100-500', fontsize=9, color=COLORS['neutral'])
    ax.text(11, 6.5, 'h = 100-1000', fontsize=9, color=COLORS['neutral'])
    ax.text(11, 3.9, '|V| = catalog size', fontsize=9, color=COLORS['neutral'])

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_pipeline_overview(output_path: Optional[Path] = None) -> plt.Figure:
    """
    Create a visual overview of the entire pipeline.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(7, 5.5, 'Reproduction Pipeline', fontsize=18, fontweight='bold', ha='center')

    # Pipeline steps
    steps = [
        ('1. Fetch', 'Clone official\nGRU4Rec', COLORS['neutral']),
        ('2. Data', 'Generate\nsynthetic', COLORS['markov']),
        ('3. Split', 'Temporal\ntrain/test', COLORS['markov']),
        ('4. Train', 'Baselines +\nGRU4Rec', COLORS['gru4rec']),
        ('5. Evaluate', 'Full ranking\nmetrics', COLORS['popularity']),
    ]

    for i, (title, desc, color) in enumerate(steps):
        x = 1.5 + i * 2.5

        # Box
        rect = mpatches.FancyBboxPatch((x - 0.8, 2), 1.6, 2.2,
                                         boxstyle='round,pad=0.1',
                                         facecolor=color,
                                         alpha=0.3,
                                         edgecolor=color,
                                         linewidth=2)
        ax.add_patch(rect)

        ax.text(x, 3.7, title, fontsize=12, fontweight='bold', ha='center', color=color)
        ax.text(x, 2.8, desc, fontsize=10, ha='center', va='center')

        # Arrow to next
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.1, 3.1), xytext=(x + 0.9, 3.1),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Command annotations
    commands = ['make fetch', 'make synth_data', 'make preprocess', 'make train', 'make eval']
    for i, cmd in enumerate(commands):
        x = 1.5 + i * 2.5
        ax.text(x, 1.5, cmd, fontsize=9, ha='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['background'], edgecolor='none'))

    if output_path:
        save_figure(fig, output_path)

    return fig


def create_all_visualizations(data_path: Path,
                               results: Dict[str, Dict[str, float]],
                               output_dir: Path,
                               training_losses: Optional[List[float]] = None):
    """
    Generate all visualizations and save to output directory.

    Args:
        data_path: Path to the session data TSV file
        results: Model evaluation results
        output_dir: Directory to save figures
        training_losses: Optional list of training losses per epoch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations in {output_dir}/")

    # Load data
    df = pd.read_csv(data_path, sep='\t')

    # Data exploration
    print("  → Session length distribution...")
    plot_session_length_distribution(df, output_dir / 'session_lengths')

    print("  → Item popularity distribution...")
    plot_item_popularity_distribution(df, output_dir / 'item_popularity')

    # Model comparison
    print("  → Model comparison...")
    plot_model_comparison(results, output_dir / 'model_comparison')

    print("  → Recall@K curves...")
    plot_recall_at_k_curves(results, output_dir / 'recall_curves')

    print("  → MRR@K curves...")
    plot_mrr_at_k_curves(results, output_dir / 'mrr_curves')

    # Training curve
    if training_losses:
        print("  → Training curve...")
        plot_training_curve(training_losses, output_dir / 'training_curve')

    # Storytelling
    print("  → Problem statement...")
    plot_problem_statement(output_dir / 'problem_statement')

    print("  → Solution comparison...")
    plot_solution_comparison(output_dir / 'solution_comparison')

    print("  → Evaluation protocol...")
    plot_evaluation_protocol(output_dir / 'evaluation_protocol')

    print("  → Architecture diagram...")
    plot_architecture_diagram(output_dir / 'architecture')

    print("  → Pipeline overview...")
    plot_pipeline_overview(output_dir / 'pipeline')

    print(f"\nDone! Generated {len(list(output_dir.glob('*.png')))} visualizations.")
