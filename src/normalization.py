"""
Normalization Module

This module provides functionality for per-fragment normalization
and denormalization of time-series fragments.

Key principles:
- Each fragment is normalized INDEPENDENTLY using its own min and max
- Global normalization across all fragments is NOT used
- Normalization maps values to [0, 1] range
- Denormalization reconstructs original scale

This approach allows comparing behavioral patterns across different
price scales (e.g., Bitcoin at $500 vs $100,000).
"""

import numpy as np
from typing import Tuple


def normalize_fragments(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each fragment independently to [0, 1] range.

    Each fragment (row) is normalized using its own minimum and maximum values.
    This allows comparing behavioral patterns regardless of absolute price scale.

    Formula for each fragment:
        x_norm = (x_i - x_min) / (x_max - x_min)

    where x_min and x_max are computed for each fragment independently.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw fragment matrix of shape (number_of_fragments, 21)
        Each row is one 21-day fragment with raw price values

    Returns
    -------
    X_norm : np.ndarray
        Normalized fragment matrix of shape (number_of_fragments, 21)
        Each row contains values in [0, 1] range
    x_mins : np.ndarray
        Array of minimum values for each fragment, shape (number_of_fragments,)
    x_maxs : np.ndarray
        Array of maximum values for each fragment, shape (number_of_fragments,)

    Raises
    ------
    ValueError
        If input array is empty or has wrong dimensions

    Examples
    --------
    >>> X_raw = np.array([[100, 110, 105], [50000, 51000, 50500]])
    >>> X_norm, x_mins, x_maxs = normalize_fragments(X_raw)
    >>> print(X_norm)
    [[0.  1.  0.5]
     [0.  1.  0.5]]
    >>> print(x_mins)
    [100. 50000.]
    >>> print(x_maxs)
    [110. 51000.]

    Notes
    -----
    Edge case handling:
    - If x_max == x_min (constant fragment), normalized values are set to 0.5
    - This represents a neutral flat pattern
    - Avoids division by zero
    """

    # Input validation
    if X_raw.size == 0:
        raise ValueError("Input array X_raw is empty")

    if X_raw.ndim != 2:
        raise ValueError(
            f"Input array must be 2-dimensional (number_of_fragments, 21), "
            f"got shape {X_raw.shape}"
        )

    if X_raw.shape[1] != 21:
        raise ValueError(
            f"Each fragment must have exactly 21 values, "
            f"got {X_raw.shape[1]} values per fragment"
        )

    number_of_fragments = X_raw.shape[0]

    # Initialize output arrays
    X_norm = np.zeros_like(X_raw, dtype=np.float64)
    x_mins = np.zeros(number_of_fragments, dtype=np.float64)
    x_maxs = np.zeros(number_of_fragments, dtype=np.float64)

    # Normalize each fragment independently
    for i in range(number_of_fragments):
        fragment = X_raw[i, :]

        x_min = np.min(fragment)
        x_max = np.max(fragment)

        x_mins[i] = x_min
        x_maxs[i] = x_max

        # Handle edge case: constant fragment (x_max == x_min)
        if x_max == x_min:
            # Fill with 0.5 to represent neutral flat pattern
            X_norm[i, :] = 0.5
        else:
            # Standard normalization
            X_norm[i, :] = (fragment - x_min) / (x_max - x_min)

    return X_norm, x_mins, x_maxs


def denormalize_values(
    x_norm: np.ndarray,
    x_min: float,
    x_max: float
) -> np.ndarray:
    """
    Denormalize values from [0, 1] range back to original scale.

    This function reconstructs real values from normalized values
    using the inverse transformation.

    Formula:
        x_real = x_norm * (x_max - x_min) + x_min

    Parameters
    ----------
    x_norm : np.ndarray
        Normalized values in [0, 1] range
        Can be 1D array or any shape
    x_min : float
        Minimum value of the original scale
    x_max : float
        Maximum value of the original scale

    Returns
    -------
    np.ndarray
        Denormalized values in original scale, same shape as input

    Raises
    ------
    ValueError
        If x_max < x_min or if inputs are invalid

    Examples
    --------
    >>> x_norm = np.array([0.0, 0.5, 1.0])
    >>> x_real = denormalize_values(x_norm, x_min=100, x_max=110)
    >>> print(x_real)
    [100. 105. 110.]

    Notes
    -----
    For forecasting, this function is typically used with:
    - x_norm: normalized forecast from SOM weight vector
    - x_min, x_max: min/max from experimental prefix

    This maps the cluster's behavioral pattern to the current price scale.
    """

    # Input validation
    if x_max < x_min:
        raise ValueError(
            f"x_max must be >= x_min, got x_min={x_min}, x_max={x_max}"
        )

    # Handle edge case: constant scale (x_max == x_min)
    if x_max == x_min:
        # All denormalized values equal to x_min
        return np.full_like(x_norm, x_min, dtype=np.float64)

    # Standard denormalization
    x_real = x_norm * (x_max - x_min) + x_min

    return x_real


def normalize_single_fragment(
    fragment: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize a single fragment and return normalized values with scale parameters.

    This is a convenience function for normalizing individual fragments,
    such as the experimental prefix.

    Parameters
    ----------
    fragment : np.ndarray
        1D array of raw values

    Returns
    -------
    fragment_norm : np.ndarray
        Normalized values in [0, 1] range
    x_min : float
        Minimum value of the fragment
    x_max : float
        Maximum value of the fragment

    Examples
    --------
    >>> prefix = np.array([100, 110, 105, 108, 112, 115, 113, 117, 120, 118, 122, 125, 123, 127])
    >>> prefix_norm, x_min, x_max = normalize_single_fragment(prefix)
    >>> print(f"Min: {x_min}, Max: {x_max}")
    Min: 100, Max: 127
    >>> print(prefix_norm[:3])
    [0.         0.37037037 0.18518519]
    """

    if fragment.ndim != 1:
        raise ValueError(
            f"Fragment must be 1-dimensional, got shape {fragment.shape}"
        )

    x_min = float(np.min(fragment))
    x_max = float(np.max(fragment))

    # Handle constant fragment
    if x_max == x_min:
        fragment_norm = np.full_like(fragment, 0.5, dtype=np.float64)
    else:
        fragment_norm = (fragment - x_min) / (x_max - x_min)

    return fragment_norm, x_min, x_max


def validate_normalization(
    X_raw: np.ndarray,
    X_norm: np.ndarray,
    x_mins: np.ndarray,
    x_maxs: np.ndarray
) -> bool:
    """
    Validate normalization results.

    Checks:
    1. X_norm has same shape as X_raw
    2. All normalized values are in [0, 1] range (with tolerance)
    3. x_mins and x_maxs have correct length
    4. Denormalization reconstructs original values

    Parameters
    ----------
    X_raw : np.ndarray
        Original raw fragment matrix
    X_norm : np.ndarray
        Normalized fragment matrix
    x_mins : np.ndarray
        Array of minimum values
    x_maxs : np.ndarray
        Array of maximum values

    Returns
    -------
    bool
        True if all validation checks pass

    Raises
    ------
    AssertionError
        If any validation check fails
    """

    number_of_fragments = X_raw.shape[0]

    # Check 1: Same shape
    assert X_norm.shape == X_raw.shape, \
        f"Shape mismatch: X_raw {X_raw.shape} vs X_norm {X_norm.shape}"

    # Check 2: Values in [0, 1] range (with small tolerance for floating point)
    tolerance = 1e-10
    assert np.all(X_norm >= -tolerance) and np.all(X_norm <= 1 + tolerance), \
        f"Normalized values outside [0, 1] range. Min: {np.min(X_norm)}, Max: {np.max(X_norm)}"

    # Check 3: Correct array lengths
    assert len(x_mins) == number_of_fragments, \
        f"x_mins length {len(x_mins)} != number_of_fragments {number_of_fragments}"

    assert len(x_maxs) == number_of_fragments, \
        f"x_maxs length {len(x_maxs)} != number_of_fragments {number_of_fragments}"

    # Check 4: Denormalization reconstruction (sample check)
    for i in range(min(5, number_of_fragments)):  # Check first 5 fragments
        reconstructed = denormalize_values(X_norm[i], x_mins[i], x_maxs[i])
        if not np.allclose(reconstructed, X_raw[i], rtol=1e-10):
            raise AssertionError(
                f"Denormalization failed for fragment {i}. "
                f"Max difference: {np.max(np.abs(reconstructed - X_raw[i]))}"
            )

    return True
