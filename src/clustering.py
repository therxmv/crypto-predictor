"""
Clustering Module

This module implements Self-Organizing Map (SOM) / Kohonen neural network
clustering for time-series fragment analysis.

The SOM learns typical behavioral patterns from normalized 21-day fragments
and groups similar patterns into clusters. Each cluster is represented by
a weight vector that captures the typical shape of fragments in that cluster.

Key features:
- Uses MiniSom library for SOM implementation
- Trains on normalized fragments (values in [0, 1])
- Produces ~30 clusters (5x6 grid)
- Reproducible results with fixed random seed
- Extracts weight vectors for forecasting
"""

import numpy as np
from minisom import MiniSom
from typing import Tuple, Dict, List
import warnings


def train_som(
    X_norm: np.ndarray,
    som_grid_x: int = 5,
    som_grid_y: int = 6,
    input_len: int = 21,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    num_iterations: int = 10000,
    random_seed: int = 42
) -> MiniSom:
    """
    Train a Self-Organizing Map (SOM) on normalized time-series fragments.

    The SOM learns to organize fragments into a 2D grid where similar
    patterns are placed close to each other. Each neuron in the grid
    develops a weight vector that represents a typical pattern.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized fragment matrix of shape (number_of_fragments, 21)
        Values should be in [0, 1] range
    som_grid_x : int, default 5
        Number of neurons in X dimension (columns)
    som_grid_y : int, default 6
        Number of neurons in Y dimension (rows)
    input_len : int, default 21
        Length of input vectors (must be 21 for this project)
    sigma : float, default 1.0
        Spread of the neighborhood function (Gaussian)
    learning_rate : float, default 0.5
        Initial learning rate
    num_iterations : int, default 10000
        Number of training iterations
    random_seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    MiniSom
        Trained SOM object

    Raises
    ------
    ValueError
        If input dimensions are incorrect or parameters are invalid

    Examples
    --------
    >>> som = train_som(X_norm, som_grid_x=5, som_grid_y=6)
    >>> print(f"Trained SOM with {som_grid_x * som_grid_y} neurons")
    Trained SOM with 30 neurons

    Notes
    -----
    Grid size: 5x6 = 30 possible clusters
    This provides enough granularity to capture different behavioral patterns
    while keeping the number of clusters manageable.

    The SOM training is unsupervised - it automatically discovers patterns
    in the data without any labels or target values.
    """

    # Input validation
    if X_norm.ndim != 2:
        raise ValueError(
            f"X_norm must be 2-dimensional, got shape {X_norm.shape}"
        )

    if X_norm.shape[1] != input_len:
        raise ValueError(
            f"Input vectors must have length {input_len}, "
            f"got {X_norm.shape[1]}"
        )

    if input_len != 21:
        raise ValueError(
            f"Input length must be 21 for this project, got {input_len}"
        )

    if X_norm.size == 0:
        raise ValueError("X_norm is empty, cannot train SOM")

    n_fragments = X_norm.shape[0]
    n_clusters = som_grid_x * som_grid_y

    print(f"Тренування SOM:")
    print(f"  - Розмір сітки: {som_grid_x} x {som_grid_y} = {n_clusters} нейронів")
    print(f"  - Кількість фрагментів: {n_fragments}")
    print(f"  - Довжина вхідного вектора: {input_len}")
    print(f"  - Кількість ітерацій: {num_iterations}")
    print(f"  - Випадкове зерно: {random_seed}")

    # Initialize SOM
    som = MiniSom(
        x=som_grid_x,
        y=som_grid_y,
        input_len=input_len,
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=random_seed
    )

    # Random initialization of weights
    som.random_weights_init(X_norm)

    # Train SOM
    print("  - Тренування SOM...")
    som.train_random(
        data=X_norm,
        num_iteration=num_iterations,
        verbose=False
    )

    print("  ✓ Тренування завершено")

    return som


def assign_clusters(
    som: MiniSom,
    X_norm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign each fragment to its winning neuron (cluster).

    For each input fragment, finds the best matching unit (BMU) in the SOM
    and returns the cluster assignment.

    Parameters
    ----------
    som : MiniSom
        Trained SOM object
    X_norm : np.ndarray
        Normalized fragment matrix of shape (number_of_fragments, 21)

    Returns
    -------
    cluster_ids : np.ndarray
        1D array of cluster IDs for each fragment
        Cluster ID is computed as: y * grid_x + x
    neuron_coords : np.ndarray
        2D array of shape (number_of_fragments, 2) with (x, y) coordinates
        of winning neurons
    distances : np.ndarray
        1D array of distances from each fragment to its winning neuron

    Examples
    --------
    >>> cluster_ids, neuron_coords, distances = assign_clusters(som, X_norm)
    >>> print(f"Fragment 0 assigned to cluster {cluster_ids[0]}")
    >>> print(f"Winning neuron at position {neuron_coords[0]}")
    """

    n_fragments = X_norm.shape[0]
    som_grid_x = som.get_weights().shape[0]

    cluster_ids = np.zeros(n_fragments, dtype=int)
    neuron_coords = np.zeros((n_fragments, 2), dtype=int)
    distances = np.zeros(n_fragments, dtype=float)

    for i in range(n_fragments):
        fragment = X_norm[i]

        # Find winning neuron (Best Matching Unit)
        winner = som.winner(fragment)
        x, y = winner

        # Store coordinates
        neuron_coords[i] = [x, y]

        # Compute cluster ID: linearize 2D coordinates
        cluster_id = y * som_grid_x + x
        cluster_ids[i] = cluster_id

        # Calculate distance to winning neuron
        winner_weights = som.get_weights()[x, y]
        distance = np.linalg.norm(fragment - winner_weights)
        distances[i] = distance

    return cluster_ids, neuron_coords, distances


def get_som_weight_vectors(som: MiniSom) -> Tuple[np.ndarray, Dict]:
    """
    Extract weight vectors from trained SOM.

    Each neuron in the SOM has a weight vector that represents the
    typical pattern of fragments assigned to that neuron.

    Parameters
    ----------
    som : MiniSom
        Trained SOM object

    Returns
    -------
    weight_matrix : np.ndarray
        Array of shape (n_clusters, 21) where each row is a weight vector
        n_clusters = som_grid_x * som_grid_y
    neuron_mapping : dict
        Dictionary mapping cluster_id -> (x, y) neuron coordinates

    Examples
    --------
    >>> weight_matrix, neuron_mapping = get_som_weight_vectors(som)
    >>> print(weight_matrix.shape)
    (30, 21)
    >>> print(neuron_mapping[0])  # Coordinates of cluster 0
    (0, 0)
    """

    weights = som.get_weights()  # Shape: (grid_x, grid_y, input_len)
    som_grid_x, som_grid_y, input_len = weights.shape

    n_clusters = som_grid_x * som_grid_y

    # Reshape to (n_clusters, input_len)
    weight_matrix = np.zeros((n_clusters, input_len))
    neuron_mapping = {}

    cluster_id = 0
    for y in range(som_grid_y):
        for x in range(som_grid_x):
            weight_matrix[cluster_id] = weights[x, y]
            neuron_mapping[cluster_id] = (x, y)
            cluster_id += 1

    return weight_matrix, neuron_mapping


def get_cluster_statistics(
    cluster_ids: np.ndarray,
    n_clusters: int
) -> Dict:
    """
    Calculate statistics about cluster assignments.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Array of cluster IDs for each fragment
    n_clusters : int
        Total number of clusters (neurons)

    Returns
    -------
    dict
        Dictionary with statistics:
        - cluster_sizes: array of fragment counts per cluster
        - empty_clusters: list of cluster IDs with no fragments
        - largest_cluster: ID of cluster with most fragments
        - smallest_nonempty_cluster: ID of smallest non-empty cluster

    Examples
    --------
    >>> stats = get_cluster_statistics(cluster_ids, n_clusters=30)
    >>> print(f"Largest cluster: {stats['largest_cluster']} "
    ...       f"with {stats['cluster_sizes'][stats['largest_cluster']]} fragments")
    """

    cluster_sizes = np.bincount(cluster_ids, minlength=n_clusters)

    empty_clusters = list(np.where(cluster_sizes == 0)[0])
    nonempty_clusters = np.where(cluster_sizes > 0)[0]

    largest_cluster = int(np.argmax(cluster_sizes))

    if len(nonempty_clusters) > 0:
        nonempty_sizes = cluster_sizes[nonempty_clusters]
        smallest_nonempty_cluster = int(nonempty_clusters[np.argmin(nonempty_sizes)])
    else:
        smallest_nonempty_cluster = None

    return {
        'cluster_sizes': cluster_sizes,
        'empty_clusters': empty_clusters,
        'largest_cluster': largest_cluster,
        'smallest_nonempty_cluster': smallest_nonempty_cluster,
        'n_empty': len(empty_clusters),
        'n_nonempty': len(nonempty_clusters)
    }


def get_fragments_in_cluster(
    cluster_ids: np.ndarray,
    target_cluster_id: int
) -> np.ndarray:
    """
    Get indices of all fragments assigned to a specific cluster.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Array of cluster IDs for each fragment
    target_cluster_id : int
        The cluster ID to query

    Returns
    -------
    np.ndarray
        Array of fragment indices belonging to the target cluster

    Examples
    --------
    >>> fragment_indices = get_fragments_in_cluster(cluster_ids, target_cluster_id=5)
    >>> print(f"Cluster 5 contains {len(fragment_indices)} fragments")
    """

    indices = np.where(cluster_ids == target_cluster_id)[0]
    return indices


def print_cluster_summary(
    som: MiniSom,
    cluster_ids: np.ndarray,
    weight_matrix: np.ndarray
) -> None:
    """
    Print a summary of SOM clustering results.

    Parameters
    ----------
    som : MiniSom
        Trained SOM object
    cluster_ids : np.ndarray
        Array of cluster IDs
    weight_matrix : np.ndarray
        SOM weight vectors

    Examples
    --------
    >>> print_cluster_summary(som, cluster_ids, weight_matrix)
    """

    weights = som.get_weights()
    som_grid_x, som_grid_y, input_len = weights.shape
    n_clusters = som_grid_x * som_grid_y
    n_fragments = len(cluster_ids)

    stats = get_cluster_statistics(cluster_ids, n_clusters)

    print("\n" + "="*60)
    print("ПІДСУМОК КЛАСТЕРИЗАЦІЇ SOM")
    print("="*60)
    print(f"Розмір сітки SOM: {som_grid_x} x {som_grid_y}")
    print(f"Загальна кількість кластерів: {n_clusters}")
    print(f"Загальна кількість фрагментів: {n_fragments}")
    print(f"Непусті кластери: {stats['n_nonempty']}")
    print(f"Пусті кластери: {stats['n_empty']}")
    print(f"\nНайбільший кластер: {stats['largest_cluster']} "
          f"({stats['cluster_sizes'][stats['largest_cluster']]} фрагментів)")

    if stats['smallest_nonempty_cluster'] is not None:
        print(f"Найменший непустий кластер: {stats['smallest_nonempty_cluster']} "
              f"({stats['cluster_sizes'][stats['smallest_nonempty_cluster']]} фрагментів)")

    print(f"\nРозмір матриці вагових векторів: {weight_matrix.shape}")
    print(f"Довжина кожного вагового вектора: {weight_matrix.shape[1]}")
    print("="*60 + "\n")
