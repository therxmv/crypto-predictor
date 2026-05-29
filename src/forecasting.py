"""
Forecasting Module

This module implements the forecasting logic based on SOM clustering.

The forecasting approach:
1. Take the last 21 days as experimental image
2. Split into 14-day prefix (known) + 7-day postfix (to predict)
3. Normalize the experimental prefix
4. Find the SOM weight vector whose prefix (first 14 values) best matches
5. Use the postfix of that weight vector (last 7 values) as normalized forecast
6. Denormalize forecast to real price scale using prefix min/max

This approach finds historical patterns similar to the current situation
and assumes the future will follow a similar pattern.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from src.normalization import normalize_single_fragment, denormalize_values


def find_target_cluster_by_prefix(
    experimental_prefix_raw: np.ndarray,
    weight_matrix: np.ndarray,
    neuron_mapping: Optional[Dict] = None,
    distance_metric: str = 'euclidean'
) -> Dict:
    """
    Find the SOM cluster whose prefix best matches the experimental prefix.

    This function implements the core forecasting logic:
    1. Normalize the experimental prefix (14 values)
    2. Compare with the first 14 values of each SOM weight vector
    3. Find the minimum-distance weight vector
    4. Return information about the target cluster

    Parameters
    ----------
    experimental_prefix_raw : np.ndarray
        Raw experimental prefix (14 values) before forecasting point
    weight_matrix : np.ndarray
        Matrix of SOM weight vectors, shape (n_clusters, 21)
    neuron_mapping : dict, optional
        Mapping from cluster_id to (x, y) neuron coordinates
    distance_metric : str, default 'euclidean'
        Distance metric to use ('euclidean' is recommended)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cluster_id': ID of the best matching cluster
        - 'neuron_coords': (x, y) coordinates if available, else None
        - 'distance': Euclidean distance to the target cluster
        - 'target_weight_vector': Full 21-value weight vector of target cluster
        - 'prefix_norm': Normalized experimental prefix
        - 'prefix_min': Minimum value of experimental prefix
        - 'prefix_max': Maximum value of experimental prefix

    Raises
    ------
    ValueError
        If experimental prefix length is not 14 or weight vectors are invalid

    Examples
    --------
    >>> result = find_target_cluster_by_prefix(prefix_raw, weight_matrix)
    >>> print(f"Target cluster: {result['cluster_id']}")
    >>> print(f"Distance: {result['distance']:.4f}")

    Notes
    -----
    Critical: This function uses ONLY the experimental prefix (14 values).
    The actual postfix (next 7 days) is NOT used for selecting the cluster.
    This prevents data leakage and ensures the forecast is truly predictive.
    """

    # Validation
    if len(experimental_prefix_raw) != 14:
        raise ValueError(
            f"Experimental prefix must have exactly 14 values, "
            f"got {len(experimental_prefix_raw)}"
        )

    if weight_matrix.ndim != 2 or weight_matrix.shape[1] != 21:
        raise ValueError(
            f"Weight matrix must have shape (n_clusters, 21), "
            f"got {weight_matrix.shape}"
        )

    n_clusters = weight_matrix.shape[0]

    print(f"\nПошук цільового кластера:")
    print(f"  - Кількість кластерів для порівняння: {n_clusters}")
    print(f"  - Довжина експериментального префіксу: {len(experimental_prefix_raw)}")

    # Step 1: Normalize experimental prefix
    prefix_norm, prefix_min, prefix_max = normalize_single_fragment(
        experimental_prefix_raw
    )

    print(f"  - Префікс нормалізовано: min={prefix_min:.2f}, max={prefix_max:.2f}")

    # Step 2: Extract first 14 values (prefix) from each weight vector
    weight_prefixes = weight_matrix[:, :14]  # Shape: (n_clusters, 14)

    # Step 3: Calculate distances from experimental prefix to each weight prefix
    if distance_metric == 'euclidean':
        # Euclidean distance: sqrt(sum((x - y)^2))
        distances = np.zeros(n_clusters)
        for i in range(n_clusters):
            distances[i] = np.linalg.norm(prefix_norm - weight_prefixes[i])
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # Step 4: Find cluster with minimum distance
    target_cluster_id = int(np.argmin(distances))
    min_distance = float(distances[target_cluster_id])

    # Step 5: Extract target weight vector
    target_weight_vector = weight_matrix[target_cluster_id]

    # Step 6: Get neuron coordinates if available
    neuron_coords = None
    if neuron_mapping and target_cluster_id in neuron_mapping:
        neuron_coords = neuron_mapping[target_cluster_id]

    print(f"  - Цільовий кластер знайдено: {target_cluster_id}")
    print(f"  - Відстань до цільового кластера: {min_distance:.6f}")
    if neuron_coords:
        print(f"  - Координати нейрона: {neuron_coords}")

    return {
        'cluster_id': target_cluster_id,
        'neuron_coords': neuron_coords,
        'distance': min_distance,
        'target_weight_vector': target_weight_vector,
        'prefix_norm': prefix_norm,
        'prefix_min': prefix_min,
        'prefix_max': prefix_max
    }


def build_normalized_forecast(
    target_weight_vector: np.ndarray,
    prefix_length: int = 14
) -> np.ndarray:
    """
    Build normalized forecast from the target weight vector.

    The forecast is simply the postfix (last 7 values) of the target
    weight vector, which represents the typical continuation pattern
    for fragments in that cluster.

    Parameters
    ----------
    target_weight_vector : np.ndarray
        Weight vector of the target cluster (length 21)
    prefix_length : int, default 14
        Length of the prefix (forecast starts after this)

    Returns
    -------
    np.ndarray
        Normalized forecast (7 values in [0, 1] range)

    Raises
    ------
    ValueError
        If weight vector length is not 21

    Examples
    --------
    >>> forecast_norm = build_normalized_forecast(target_weight_vector)
    >>> print(len(forecast_norm))
    7

    Notes
    -----
    The normalized forecast captures the behavioral pattern from the cluster
    but is still in [0, 1] range. It must be denormalized to get real prices.
    """

    if len(target_weight_vector) != 21:
        raise ValueError(
            f"Target weight vector must have length 21, "
            f"got {len(target_weight_vector)}"
        )

    if prefix_length != 14:
        raise ValueError(
            f"Prefix length must be 14, got {prefix_length}"
        )

    # Extract last 7 values as forecast
    forecast_norm = target_weight_vector[prefix_length:]

    if len(forecast_norm) != 7:
        raise ValueError(
            f"Forecast must have length 7, got {len(forecast_norm)}"
        )

    print(f"\nПобудова прогнозу:")
    print(f"  - Довжина нормалізованого прогнозу: {len(forecast_norm)}")
    print(f"  - Діапазон значень: [{np.min(forecast_norm):.4f}, {np.max(forecast_norm):.4f}]")

    return forecast_norm


def build_real_forecast(
    forecast_norm: np.ndarray,
    prefix_min: float,
    prefix_max: float
) -> np.ndarray:
    """
    Convert normalized forecast to real price values.

    Uses the scale from the experimental prefix to map the normalized
    behavioral pattern to the current price range.

    Parameters
    ----------
    forecast_norm : np.ndarray
        Normalized forecast (7 values)
    prefix_min : float
        Minimum value from experimental prefix
    prefix_max : float
        Maximum value from experimental prefix

    Returns
    -------
    np.ndarray
        Real-value forecast (7 values in original price scale)

    Examples
    --------
    >>> forecast_real = build_real_forecast(forecast_norm, 50000, 52000)
    >>> print(forecast_real)
    [50500. 51000. 51500. 51200. 51800. 51600. 51900.]

    Notes
    -----
    Denormalization formula:
        x_real = x_norm * (prefix_max - prefix_min) + prefix_min

    This maps the [0, 1] pattern to the current price scale defined by
    the experimental prefix range.
    """

    if len(forecast_norm) != 7:
        raise ValueError(
            f"Forecast must have length 7, got {len(forecast_norm)}"
        )

    forecast_real = denormalize_values(forecast_norm, prefix_min, prefix_max)

    print(f"\nДенормалізація прогнозу:")
    print(f"  - Використано масштаб префіксу: [{prefix_min:.2f}, {prefix_max:.2f}]")
    print(f"  - Діапазон реальних цін прогнозу: [{np.min(forecast_real):.2f}, {np.max(forecast_real):.2f}]")

    return forecast_real


def forecast_next_week(
    experimental_prefix_raw: np.ndarray,
    weight_matrix: np.ndarray,
    neuron_mapping: Optional[Dict] = None
) -> Dict:
    """
    Complete forecasting pipeline: find target cluster and build forecast.

    This is a convenience function that combines all forecasting steps:
    1. Find target cluster by prefix matching
    2. Build normalized forecast from target weight vector
    3. Denormalize forecast to real price values

    Parameters
    ----------
    experimental_prefix_raw : np.ndarray
        Raw experimental prefix (14 values)
    weight_matrix : np.ndarray
        Matrix of SOM weight vectors (n_clusters, 21)
    neuron_mapping : dict, optional
        Mapping from cluster_id to neuron coordinates

    Returns
    -------
    dict
        Complete forecasting results with keys:
        - 'cluster_id': Target cluster ID
        - 'neuron_coords': Neuron coordinates
        - 'distance': Distance to target cluster
        - 'target_weight_vector': Target weight vector (21 values)
        - 'prefix_norm': Normalized prefix
        - 'prefix_min': Prefix minimum
        - 'prefix_max': Prefix maximum
        - 'forecast_norm': Normalized forecast (7 values)
        - 'forecast_real': Real-value forecast (7 values)

    Examples
    --------
    >>> result = forecast_next_week(prefix_raw, weight_matrix, neuron_mapping)
    >>> print(f"Target cluster: {result['cluster_id']}")
    >>> print(f"Forecast: {result['forecast_real']}")
    """

    print("="*70)
    print("ПРОГНОЗУВАННЯ НАСТУПНИХ 7 ДНІВ")
    print("="*70)

    # Step 1: Find target cluster
    target_info = find_target_cluster_by_prefix(
        experimental_prefix_raw=experimental_prefix_raw,
        weight_matrix=weight_matrix,
        neuron_mapping=neuron_mapping
    )

    # Step 2: Build normalized forecast
    forecast_norm = build_normalized_forecast(
        target_weight_vector=target_info['target_weight_vector']
    )

    # Step 3: Denormalize forecast
    forecast_real = build_real_forecast(
        forecast_norm=forecast_norm,
        prefix_min=target_info['prefix_min'],
        prefix_max=target_info['prefix_max']
    )

    # Combine results
    result = {
        **target_info,
        'forecast_norm': forecast_norm,
        'forecast_real': forecast_real
    }

    print("="*70)
    print("ПРОГНОЗУВАННЯ ЗАВЕРШЕНО")
    print("="*70 + "\n")

    return result


def validate_forecast_logic(
    experimental_prefix_length: int,
    forecast_length: int,
    weight_vector_length: int
) -> bool:
    """
    Validate that forecasting parameters are correct.

    Parameters
    ----------
    experimental_prefix_length : int
        Length of experimental prefix
    forecast_length : int
        Length of forecast horizon
    weight_vector_length : int
        Length of SOM weight vectors

    Returns
    -------
    bool
        True if all parameters are valid

    Raises
    ------
    ValueError
        If any parameter is incorrect
    """

    if experimental_prefix_length != 14:
        raise ValueError(
            f"Experimental prefix must be 14 days, got {experimental_prefix_length}"
        )

    if forecast_length != 7:
        raise ValueError(
            f"Forecast horizon must be 7 days, got {forecast_length}"
        )

    if weight_vector_length != 21:
        raise ValueError(
            f"Weight vectors must have length 21, got {weight_vector_length}"
        )

    if experimental_prefix_length + forecast_length != weight_vector_length:
        raise ValueError(
            f"Prefix ({experimental_prefix_length}) + forecast ({forecast_length}) "
            f"must equal weight vector length ({weight_vector_length})"
        )

    return True
