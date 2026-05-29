"""
Visualization Module

This module provides comprehensive visualization functions for
time-series analysis, fragments, clustering, and forecasting results.

All plots are designed to be:
- Clear and readable
- Properly labeled in Ukrainian
- Suitable for inclusion in reports
- Saved to results/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple


# Set default style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def plot_full_series(
    df: pd.DataFrame,
    value_column: str = 'Close',
    title: str = 'Повний часовий ряд',
    save_path: Optional[str] = None
) -> None:
    """
    Plot the complete time series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date and value columns
    value_column : str, default 'Close'
        Name of the column to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure. If None, displays the plot.

    Examples
    --------
    >>> plot_full_series(df, save_path='results/figures/full_series.png')
    """

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df['Date'], df[value_column], linewidth=1.5, color='steelblue')

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Ціна закриття (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_raw_fragments(
    X_raw: np.ndarray,
    metadata: pd.DataFrame,
    fragment_indices: Optional[List[int]] = None,
    max_fragments: int = 10,
    title: str = 'Приклади сирих 3-тижневих фрагментів',
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple raw fragments to visualize their patterns.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw fragment matrix (number_of_fragments, 21)
    metadata : pd.DataFrame
        Fragment metadata
    fragment_indices : list of int, optional
        Specific fragment indices to plot. If None, plots first max_fragments.
    max_fragments : int, default 10
        Maximum number of fragments to plot if fragment_indices not specified
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_raw_fragments(X_raw, metadata, fragment_indices=[0, 10, 20, 30])
    """

    if fragment_indices is None:
        n_fragments = min(max_fragments, len(X_raw))
        fragment_indices = list(range(n_fragments))

    fig, ax = plt.subplots(figsize=(12, 6))

    days = np.arange(1, 22)

    for idx in fragment_indices:
        if idx >= len(X_raw):
            continue

        fragment = X_raw[idx]
        start_date = metadata.iloc[idx]['start_date']
        label = f"Фрагмент {idx} ({pd.to_datetime(start_date).strftime('%Y-%m-%d')})"

        ax.plot(days, fragment, marker='o', markersize=3, linewidth=1.5,
                alpha=0.7, label=label)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('День у фрагменті', fontsize=12)
    ax.set_ylabel('Ціна закриття (USD)', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_normalized_fragments(
    X_norm: np.ndarray,
    metadata: pd.DataFrame,
    fragment_indices: Optional[List[int]] = None,
    max_fragments: int = 10,
    title: str = 'Приклади нормалізованих фрагментів',
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple normalized fragments to visualize their patterns.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized fragment matrix (number_of_fragments, 21)
    metadata : pd.DataFrame
        Fragment metadata
    fragment_indices : list of int, optional
        Specific fragment indices to plot
    max_fragments : int, default 10
        Maximum number of fragments to plot if fragment_indices not specified
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_normalized_fragments(X_norm, metadata, max_fragments=15)
    """

    if fragment_indices is None:
        n_fragments = min(max_fragments, len(X_norm))
        fragment_indices = list(range(n_fragments))

    fig, ax = plt.subplots(figsize=(12, 6))

    days = np.arange(1, 22)

    for idx in fragment_indices:
        if idx >= len(X_norm):
            continue

        fragment = X_norm[idx]
        start_date = metadata.iloc[idx]['start_date']
        label = f"Фрагмент {idx} ({pd.to_datetime(start_date).strftime('%Y-%m-%d')})"

        ax.plot(days, fragment, marker='o', markersize=3, linewidth=1.5,
                alpha=0.7, label=label)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('День у фрагменті', fontsize=12)
    ax.set_ylabel('Нормалізоване значення', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_raw_vs_normalized_fragment(
    X_raw: np.ndarray,
    X_norm: np.ndarray,
    metadata: pd.DataFrame,
    fragment_index: int = 0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot raw and normalized versions of the same fragment side by side.

    This visualization demonstrates the normalization process and shows
    how absolute price values are mapped to [0, 1] range while preserving
    the behavioral pattern.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw fragment matrix
    X_norm : np.ndarray
        Normalized fragment matrix
    metadata : pd.DataFrame
        Fragment metadata
    fragment_index : int, default 0
        Index of the fragment to visualize
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_raw_vs_normalized_fragment(X_raw, X_norm, metadata, fragment_index=5)
    """

    if fragment_index >= len(X_raw):
        raise ValueError(f"Fragment index {fragment_index} out of range")

    raw_fragment = X_raw[fragment_index]
    norm_fragment = X_norm[fragment_index]
    start_date = pd.to_datetime(metadata.iloc[fragment_index]['start_date'])
    end_date = pd.to_datetime(metadata.iloc[fragment_index]['end_date'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    days = np.arange(1, 22)

    # Raw fragment
    axes[0].plot(days, raw_fragment, marker='o', linewidth=2, color='steelblue')
    axes[0].set_title(f'Сирий фрагмент {fragment_index}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('День у фрагменті', fontsize=11)
    axes[0].set_ylabel('Ціна закриття (USD)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.98, f'Період: {start_date.strftime("%Y-%m-%d")} до {end_date.strftime("%Y-%m-%d")}',
                 transform=axes[0].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Normalized fragment
    axes[1].plot(days, norm_fragment, marker='o', linewidth=2, color='darkorange')
    axes[1].set_title(f'Нормалізований фрагмент {fragment_index}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('День у фрагменті', fontsize=11)
    axes[1].set_ylabel('Нормалізоване значення', fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.98, f'Min: {np.min(raw_fragment):.2f}\nMax: {np.max(raw_fragment):.2f}',
                 transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_fragments_grid(
    X_norm: np.ndarray,
    metadata: pd.DataFrame,
    n_samples: int = 20,
    title: str = 'Випадкова вибірка нормалізованих фрагментів',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a grid of normalized fragments for visual inspection.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized fragment matrix
    metadata : pd.DataFrame
        Fragment metadata
    n_samples : int, default 20
        Number of fragments to display in grid
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_fragments_grid(X_norm, metadata, n_samples=16)
    """

    n_fragments = len(X_norm)
    n_samples = min(n_samples, n_fragments)

    # Random sample
    indices = np.random.choice(n_fragments, size=n_samples, replace=False)
    indices = sorted(indices)

    # Calculate grid dimensions
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    days = np.arange(1, 22)

    for i, idx in enumerate(indices):
        ax = axes[i]
        fragment = X_norm[idx]
        start_date = pd.to_datetime(metadata.iloc[idx]['start_date'])

        ax.plot(days, fragment, linewidth=1.5, color='steelblue')
        ax.set_title(f'#{idx}\n{start_date.strftime("%Y-%m-%d")}', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cluster_fragments(
    X_norm: np.ndarray,
    cluster_ids: np.ndarray,
    weight_vector: np.ndarray,
    target_cluster_id: int,
    neuron_coords: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot all fragments in a specific cluster with the SOM weight vector overlay.

    This visualization shows how similar fragments are grouped together
    and displays the learned cluster pattern (weight vector).

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized fragment matrix (number_of_fragments, 21)
    cluster_ids : np.ndarray
        Array of cluster IDs for each fragment
    weight_vector : np.ndarray
        SOM weight vector for the target cluster (length 21)
    target_cluster_id : int
        ID of the cluster to visualize
    neuron_coords : tuple of (int, int), optional
        (x, y) coordinates of the neuron in SOM grid
    title : str, optional
        Custom plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_cluster_fragments(X_norm, cluster_ids, weight_matrix[5],
    ...                        target_cluster_id=5, neuron_coords=(0, 1))
    """

    # Find fragments in this cluster
    fragment_indices = np.where(cluster_ids == target_cluster_id)[0]
    n_fragments = len(fragment_indices)

    if n_fragments == 0:
        print(f"Кластер {target_cluster_id} порожній - немає фрагментів для візуалізації")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    days = np.arange(1, 22)

    # Plot individual fragments with thin transparent lines
    for idx in fragment_indices:
        fragment = X_norm[idx]
        ax.plot(days, fragment, linewidth=0.8, alpha=0.3, color='steelblue')

    # Overlay the weight vector with thick highlighted line
    ax.plot(days, weight_vector, linewidth=3.5, color='darkred',
            label='Ваговий вектор (типовий патерн)', zorder=10)

    # Title
    if title is None:
        coord_str = f" [{neuron_coords[0]}, {neuron_coords[1]}]" if neuron_coords else ""
        title = f'Кластер {target_cluster_id}{coord_str}: {n_fragments} фрагментів'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('День у фрагменті', fontsize=12)
    ax.set_ylabel('Нормалізоване значення', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add info text
    info_text = f'{n_fragments} фрагментів у кластері'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_clusters(
    X_norm: np.ndarray,
    cluster_ids: np.ndarray,
    weight_matrix: np.ndarray,
    cluster_list: List[int],
    neuron_mapping: Optional[dict] = None,
    title: str = 'Порівняння кластерів',
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple clusters in a grid layout for comparison.

    Each subplot shows fragments in one cluster with its weight vector.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized fragment matrix
    cluster_ids : np.ndarray
        Array of cluster IDs for each fragment
    weight_matrix : np.ndarray
        Matrix of all SOM weight vectors (n_clusters, 21)
    cluster_list : list of int
        List of cluster IDs to visualize
    neuron_mapping : dict, optional
        Mapping from cluster_id to (x, y) neuron coordinates
    title : str
        Overall figure title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_multiple_clusters(X_norm, cluster_ids, weight_matrix,
    ...                        cluster_list=[0, 5, 10, 15, 20, 25])
    """

    n_clusters_to_plot = len(cluster_list)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = int(np.ceil(n_clusters_to_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_clusters_to_plot > 1 else [axes]

    days = np.arange(1, 22)

    for i, cluster_id in enumerate(cluster_list):
        ax = axes[i]

        # Find fragments in this cluster
        fragment_indices = np.where(cluster_ids == cluster_id)[0]
        n_fragments = len(fragment_indices)

        # Plot fragments
        for idx in fragment_indices:
            fragment = X_norm[idx]
            ax.plot(days, fragment, linewidth=0.7, alpha=0.25, color='steelblue')

        # Plot weight vector
        if cluster_id < len(weight_matrix):
            weight_vector = weight_matrix[cluster_id]
            ax.plot(days, weight_vector, linewidth=2.5, color='darkred',
                   label='Ваговий вектор', zorder=10)

        # Title with coordinates if available
        coord_str = ""
        if neuron_mapping and cluster_id in neuron_mapping:
            x, y = neuron_mapping[cluster_id]
            coord_str = f" [{x}, {y}]"

        ax.set_title(f'Кластер {cluster_id}{coord_str}\n{n_fragments} фрагментів',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('День', fontsize=9)
        ax.set_ylabel('Норм. значення', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if i == 0:  # Legend only on first subplot
            ax.legend(loc='best', fontsize=8)

    # Hide unused subplots
    for i in range(n_clusters_to_plot, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_som_heatmap(
    cluster_ids: np.ndarray,
    som_grid_x: int = 5,
    som_grid_y: int = 6,
    title: str = 'Карта щільності кластерів SOM',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a heatmap showing the number of fragments in each SOM neuron.

    This visualization shows which parts of the SOM are most activated
    and helps understand the distribution of patterns.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Array of cluster IDs for each fragment
    som_grid_x : int, default 5
        SOM grid width
    som_grid_y : int, default 6
        SOM grid height
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_som_heatmap(cluster_ids, som_grid_x=5, som_grid_y=6)
    """

    # Count fragments per cluster
    n_clusters = som_grid_x * som_grid_y
    cluster_sizes = np.bincount(cluster_ids, minlength=n_clusters)

    # Reshape to grid
    heatmap = np.zeros((som_grid_y, som_grid_x))
    for cluster_id, size in enumerate(cluster_sizes):
        x = cluster_id % som_grid_x
        y = cluster_id // som_grid_x
        if y < som_grid_y:
            heatmap[y, x] = size

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Кількість фрагментів', fontsize=12)

    # Add text annotations
    for y in range(som_grid_y):
        for x in range(som_grid_x):
            cluster_id = y * som_grid_x + x
            text = ax.text(x, y, f'{int(heatmap[y, x])}\n(ID: {cluster_id})',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X координата', fontsize=12)
    ax.set_ylabel('Y координата', fontsize=12)
    ax.set_xticks(np.arange(som_grid_x))
    ax.set_yticks(np.arange(som_grid_y))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_target_cluster_match(
    experimental_prefix_norm: np.ndarray,
    target_weight_vector: np.ndarray,
    cluster_id: int,
    neuron_coords: Optional[Tuple[int, int]] = None,
    distance: Optional[float] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot experimental prefix vs target cluster weight vector prefix.

    This visualization shows how well the experimental prefix matches
    the selected cluster pattern (first 14 values).

    Parameters
    ----------
    experimental_prefix_norm : np.ndarray
        Normalized experimental prefix (14 values)
    target_weight_vector : np.ndarray
        Target cluster weight vector (21 values)
    cluster_id : int
        ID of the target cluster
    neuron_coords : tuple of (int, int), optional
        (x, y) coordinates of the neuron
    distance : float, optional
        Distance between prefix and target
    title : str, optional
        Custom plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_target_cluster_match(prefix_norm, target_weight_vector,
    ...                           cluster_id=15, distance=0.0234)
    """

    fig, ax = plt.subplots(figsize=(12, 7))

    days_prefix = np.arange(1, 15)
    days_full = np.arange(1, 22)

    # Plot full weight vector (faded)
    ax.plot(days_full, target_weight_vector, linewidth=2, alpha=0.3,
            color='darkred', linestyle='--', label='Повний ваговий вектор')

    # Plot weight vector prefix (highlighted)
    ax.plot(days_prefix, target_weight_vector[:14], linewidth=3,
            color='darkred', label=f'Префікс вагового вектора (кластер {cluster_id})')

    # Plot experimental prefix
    ax.plot(days_prefix, experimental_prefix_norm, linewidth=3,
            color='steelblue', marker='o', markersize=5,
            label='Експериментальний префікс')

    # Mark the forecasting point
    ax.axvline(x=14.5, color='orange', linestyle=':', linewidth=2.5,
               label='Точка прогнозування', alpha=0.8)

    # Title
    if title is None:
        coord_str = f" [{neuron_coords[0]}, {neuron_coords[1]}]" if neuron_coords else ""
        title = f'Відповідність цільового кластера {cluster_id}{coord_str}'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('День у фрагменті', fontsize=12)
    ax.set_ylabel('Нормалізоване значення', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add info text
    info_lines = [f'Цільовий кластер: {cluster_id}']
    if neuron_coords:
        info_lines.append(f'Координати нейрона: {neuron_coords}')
    if distance is not None:
        info_lines.append(f'Відстань: {distance:.6f}')

    ax.text(0.02, 0.98, '\n'.join(info_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_normalized_forecast(
    experimental_prefix_norm: np.ndarray,
    forecast_norm: np.ndarray,
    target_weight_vector: np.ndarray,
    cluster_id: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot normalized forecast in the context of the full pattern.

    Shows the experimental prefix, target weight vector, and normalized forecast.

    Parameters
    ----------
    experimental_prefix_norm : np.ndarray
        Normalized experimental prefix (14 values)
    forecast_norm : np.ndarray
        Normalized forecast (7 values)
    target_weight_vector : np.ndarray
        Target cluster weight vector (21 values)
    cluster_id : int
        ID of the target cluster
    title : str, optional
        Custom plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_normalized_forecast(prefix_norm, forecast_norm,
    ...                          target_weight_vector, cluster_id=15)
    """

    fig, ax = plt.subplots(figsize=(12, 7))

    days_prefix = np.arange(1, 15)
    days_forecast = np.arange(15, 22)
    days_full = np.arange(1, 22)

    # Plot target weight vector (reference pattern)
    ax.plot(days_full, target_weight_vector, linewidth=2.5, alpha=0.4,
            color='gray', linestyle='--', label=f'Ваговий вектор кластера {cluster_id}')

    # Plot experimental prefix
    ax.plot(days_prefix, experimental_prefix_norm, linewidth=3,
            color='steelblue', marker='o', markersize=5,
            label='Експериментальний префікс (відомі дані)')

    # Plot normalized forecast
    ax.plot(days_forecast, forecast_norm, linewidth=3,
            color='darkorange', marker='s', markersize=6,
            label='Нормалізований прогноз')

    # Mark the forecasting point
    ax.axvline(x=14.5, color='red', linestyle=':', linewidth=2.5,
               label='Точка прогнозування', alpha=0.8)

    # Fill areas
    ax.fill_between(days_prefix, 0, experimental_prefix_norm,
                    alpha=0.2, color='steelblue')
    ax.fill_between(days_forecast, 0, forecast_norm,
                    alpha=0.2, color='darkorange')

    if title is None:
        title = 'Нормалізований прогноз на основі цільового кластера'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('День у фрагменті', fontsize=12)
    ax.set_ylabel('Нормалізоване значення', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_real_forecast_vs_actual(
    experimental_prefix_raw: np.ndarray,
    forecast_real: np.ndarray,
    actual_postfix: np.ndarray,
    prefix_dates: pd.Series,
    postfix_dates: pd.Series,
    metrics: Optional[dict] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot real-value forecast vs actual values.

    This is the main result visualization showing forecast quality.

    Parameters
    ----------
    experimental_prefix_raw : np.ndarray
        Raw experimental prefix (14 values)
    forecast_real : np.ndarray
        Real-value forecast (7 values)
    actual_postfix : np.ndarray
        Actual postfix values (7 values)
    prefix_dates : pd.Series
        Dates for prefix period
    postfix_dates : pd.Series
        Dates for postfix period
    metrics : dict, optional
        Dictionary with MAE, RMSE, MAPE
    title : str, optional
        Custom plot title
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_real_forecast_vs_actual(prefix_raw, forecast_real, actual_postfix,
    ...                              prefix_dates, postfix_dates, metrics)
    """

    fig, ax = plt.subplots(figsize=(14, 7))

    # Combine dates for continuous x-axis
    all_dates = pd.concat([prefix_dates, postfix_dates])
    prefix_values = experimental_prefix_raw
    postfix_values_actual = actual_postfix
    postfix_values_forecast = forecast_real

    # Plot experimental prefix (known data)
    ax.plot(prefix_dates, prefix_values, linewidth=3, color='steelblue',
            marker='o', markersize=5, label='Експериментальний префікс (відомі дані)')

    # Plot actual postfix
    ax.plot(postfix_dates, postfix_values_actual, linewidth=3, color='green',
            marker='o', markersize=6, label='Фактичні значення')

    # Plot forecast
    ax.plot(postfix_dates, postfix_values_forecast, linewidth=3, color='darkorange',
            marker='s', markersize=6, linestyle='--', label='Прогноз')

    # Connect prefix to forecast
    connection_dates = [prefix_dates.iloc[-1], postfix_dates.iloc[0]]
    connection_values = [prefix_values[-1], postfix_values_forecast[0]]
    ax.plot(connection_dates, connection_values, linewidth=2,
            color='darkorange', linestyle=':', alpha=0.5)

    # Mark forecasting point
    forecast_start_date = postfix_dates.iloc[0]
    ax.axvline(x=forecast_start_date, color='red', linestyle=':', linewidth=2.5,
               label='Точка прогнозування', alpha=0.8)

    # Fill areas
    ax.fill_between(postfix_dates, postfix_values_actual, postfix_values_forecast,
                    alpha=0.3, color='gray', label='Помилка прогнозу')

    if title is None:
        title = 'Прогноз проти фактичних значень'

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Ціна закриття (USD)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add metrics text box if provided
    if metrics:
        metrics_text = (
            f"MAE: {metrics['MAE']:.2f}\n"
            f"RMSE: {metrics['RMSE']:.2f}\n"
            f"MAPE: {metrics['MAPE']:.2f}%"
        )
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_forecast_comparison_grid(
    experimental_prefix_raw: np.ndarray,
    forecast_real: np.ndarray,
    actual_postfix: np.ndarray,
    prefix_norm: np.ndarray,
    forecast_norm: np.ndarray,
    target_weight_vector: np.ndarray,
    prefix_dates: pd.Series,
    postfix_dates: pd.Series,
    cluster_id: int,
    metrics: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive grid showing all forecast aspects.

    Combines normalized and real-value visualizations in one figure.

    Parameters
    ----------
    experimental_prefix_raw : np.ndarray
        Raw experimental prefix
    forecast_real : np.ndarray
        Real-value forecast
    actual_postfix : np.ndarray
        Actual postfix values
    prefix_norm : np.ndarray
        Normalized prefix
    forecast_norm : np.ndarray
        Normalized forecast
    target_weight_vector : np.ndarray
        Target weight vector
    prefix_dates : pd.Series
        Prefix dates
    postfix_dates : pd.Series
        Postfix dates
    cluster_id : int
        Target cluster ID
    metrics : dict
        Forecast metrics
    save_path : str, optional
        Path to save the figure

    Examples
    --------
    >>> plot_forecast_comparison_grid(prefix_raw, forecast_real, actual,
    ...                               prefix_norm, forecast_norm, weight_vector,
    ...                               dates_prefix, dates_postfix, cluster_id=15,
    ...                               metrics=metrics)
    """

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Top plot: Normalized space
    ax1 = axes[0]
    days_prefix = np.arange(1, 15)
    days_forecast = np.arange(15, 22)
    days_full = np.arange(1, 22)

    ax1.plot(days_full, target_weight_vector, linewidth=2, alpha=0.3,
            color='gray', linestyle='--', label=f'Ваговий вектор кластера {cluster_id}')
    ax1.plot(days_prefix, prefix_norm, linewidth=3, color='steelblue',
            marker='o', markersize=5, label='Нормалізований префікс')
    ax1.plot(days_forecast, forecast_norm, linewidth=3, color='darkorange',
            marker='s', markersize=6, label='Нормалізований прогноз')
    ax1.axvline(x=14.5, color='red', linestyle=':', linewidth=2, alpha=0.8)

    ax1.set_title('Нормалізований простір [0, 1]', fontsize=14, fontweight='bold')
    ax1.set_xlabel('День у фрагменті', fontsize=11)
    ax1.set_ylabel('Нормалізоване значення', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Real price space
    ax2 = axes[1]
    all_dates = pd.concat([prefix_dates, postfix_dates])

    ax2.plot(prefix_dates, experimental_prefix_raw, linewidth=3,
            color='steelblue', marker='o', markersize=5,
            label='Префікс (відомі дані)')
    ax2.plot(postfix_dates, actual_postfix, linewidth=3,
            color='green', marker='o', markersize=6,
            label='Фактичні значення')
    ax2.plot(postfix_dates, forecast_real, linewidth=3,
            color='darkorange', marker='s', markersize=6,
            linestyle='--', label='Прогноз')
    ax2.fill_between(postfix_dates, actual_postfix, forecast_real,
                     alpha=0.3, color='gray')

    forecast_start = postfix_dates.iloc[0]
    ax2.axvline(x=forecast_start, color='red', linestyle=':', linewidth=2, alpha=0.8)

    ax2.set_title('Реальний ціновий простір', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Дата', fontsize=11)
    ax2.set_ylabel('Ціна закриття (USD)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Add metrics
    metrics_text = (
        f"Метрики якості:\n"
        f"MAE: {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"MAPE: {metrics['MAPE']:.2f}%"
    )
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Порівняння прогнозу: нормалізований vs реальний простір',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Збережено: {save_path}")
    else:
        plt.show()

    plt.close()
