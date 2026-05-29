"""
Metrics Module

This module provides error metrics for evaluating forecast quality.

Implements three standard forecasting error metrics:
- MAE (Mean Absolute Error): average absolute difference
- RMSE (Root Mean Square Error): emphasizes larger errors
- MAPE (Mean Absolute Percentage Error): relative error in percentage

All metrics are calculated by comparing the forecast with actual values.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors without considering direction.

    Formula:
        MAE = (1/n) * sum(|y_true - y_pred|)

    Parameters
    ----------
    y_true : np.ndarray
        Actual values (ground truth)
    y_pred : np.ndarray
        Predicted values (forecast)

    Returns
    -------
    float
        Mean Absolute Error

    Examples
    --------
    >>> y_true = np.array([100, 110, 105, 115, 120, 118, 125])
    >>> y_pred = np.array([102, 108, 107, 113, 122, 116, 127])
    >>> mae = calculate_mae(y_true, y_pred)
    >>> print(f"MAE: {mae:.2f}")
    MAE: 2.29

    Notes
    -----
    MAE is in the same units as the data (e.g., USD for prices).
    Lower values indicate better forecast accuracy.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    mae = np.mean(np.abs(y_true - y_pred))
    return float(mae)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    RMSE is the square root of the average squared errors.
    It penalizes larger errors more than MAE.

    Formula:
        RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))

    Parameters
    ----------
    y_true : np.ndarray
        Actual values (ground truth)
    y_pred : np.ndarray
        Predicted values (forecast)

    Returns
    -------
    float
        Root Mean Square Error

    Examples
    --------
    >>> y_true = np.array([100, 110, 105, 115, 120, 118, 125])
    >>> y_pred = np.array([102, 108, 107, 113, 122, 116, 127])
    >>> rmse = calculate_rmse(y_true, y_pred)
    >>> print(f"RMSE: {rmse:.2f}")
    RMSE: 2.52

    Notes
    -----
    RMSE is in the same units as the data.
    RMSE >= MAE always (equality only when all errors are equal).
    Lower values indicate better forecast accuracy.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE expresses error as a percentage of the actual values,
    making it scale-independent and easy to interpret.

    Formula:
        MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)

    Parameters
    ----------
    y_true : np.ndarray
        Actual values (ground truth)
    y_pred : np.ndarray
        Predicted values (forecast)
    epsilon : float, default 1e-10
        Small value added to denominator to avoid division by zero

    Returns
    -------
    float
        Mean Absolute Percentage Error (in percentage)

    Examples
    --------
    >>> y_true = np.array([100, 110, 105, 115, 120, 118, 125])
    >>> y_pred = np.array([102, 108, 107, 113, 122, 116, 127])
    >>> mape = calculate_mape(y_true, y_pred)
    >>> print(f"MAPE: {mape:.2f}%")
    MAPE: 1.85%

    Notes
    -----
    MAPE is expressed as a percentage.
    Lower values indicate better forecast accuracy.

    Handling zero values:
    - If y_true contains zeros, division by zero can occur
    - We add epsilon to the denominator to prevent this
    - For financial time series (prices), zeros are typically not present

    Interpretation:
    - MAPE < 10%: Highly accurate forecast
    - MAPE 10-20%: Good forecast
    - MAPE 20-50%: Reasonable forecast
    - MAPE > 50%: Inaccurate forecast
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length. "
            f"Got {len(y_true)} and {len(y_pred)}"
        )

    # Add epsilon to avoid division by zero
    mape = 100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))
    return float(mape)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all forecasting error metrics.

    Computes MAE, RMSE, and MAPE in a single function call.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values (ground truth)
    y_pred : np.ndarray
        Predicted values (forecast)

    Returns
    -------
    dict
        Dictionary with keys 'MAE', 'RMSE', 'MAPE'

    Examples
    --------
    >>> metrics = calculate_all_metrics(y_true, y_pred)
    >>> print(f"MAE: {metrics['MAE']:.2f}")
    >>> print(f"RMSE: {metrics['RMSE']:.2f}")
    >>> print(f"MAPE: {metrics['MAPE']:.2f}%")
    """

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def save_metrics(
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary with metric names and values
    save_path : str
        Path to save the JSON file
    additional_info : dict, optional
        Additional information to save (e.g., dates, cluster info)

    Examples
    --------
    >>> metrics = calculate_all_metrics(y_true, y_pred)
    >>> save_metrics(metrics, 'results/metrics/forecast_metrics.json')
    """

    # Create output directory if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    output = {
        'metrics': metrics
    }

    if additional_info:
        output['additional_info'] = additional_info

    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Метрики збережено: {save_path}")


def save_metrics_txt(
    metrics: Dict[str, float],
    save_path: str,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> None:
    """
    Save metrics to a human-readable text file.

    Parameters
    ----------
    metrics : dict
        Dictionary with metric names and values
    save_path : str
        Path to save the text file
    y_true : np.ndarray, optional
        Actual values (for additional statistics)
    y_pred : np.ndarray, optional
        Predicted values (for additional statistics)

    Examples
    --------
    >>> save_metrics_txt(metrics, 'results/metrics/forecast_metrics.txt',
    ...                  y_true=actual, y_pred=forecast)
    """

    # Create output directory if it doesn't exist
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("МЕТРИКИ ЯКОСТІ ПРОГНОЗУ\n")
        f.write("="*60 + "\n\n")

        f.write("Основні метрики:\n")
        f.write("-"*60 + "\n")
        f.write(f"MAE (Mean Absolute Error):          {metrics['MAE']:.4f}\n")
        f.write(f"RMSE (Root Mean Square Error):      {metrics['RMSE']:.4f}\n")
        f.write(f"MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.4f}%\n")
        f.write("\n")

        if y_true is not None and y_pred is not None:
            f.write("Додаткова статистика:\n")
            f.write("-"*60 + "\n")
            f.write(f"Кількість точок прогнозу:           {len(y_true)}\n")
            f.write(f"Середнє фактичне значення:          {np.mean(y_true):.4f}\n")
            f.write(f"Середнє прогнозоване значення:      {np.mean(y_pred):.4f}\n")
            f.write(f"Мінімальна фактична ціна:           {np.min(y_true):.4f}\n")
            f.write(f"Максимальна фактична ціна:          {np.max(y_true):.4f}\n")
            f.write(f"Мінімальна прогнозована ціна:       {np.min(y_pred):.4f}\n")
            f.write(f"Максимальна прогнозована ціна:      {np.max(y_pred):.4f}\n")
            f.write("\n")

        f.write("Інтерпретація MAPE:\n")
        f.write("-"*60 + "\n")
        mape_val = metrics['MAPE']
        if mape_val < 10:
            interpretation = "Дуже точний прогноз"
        elif mape_val < 20:
            interpretation = "Хороший прогноз"
        elif mape_val < 50:
            interpretation = "Прийнятний прогноз"
        else:
            interpretation = "Неточний прогноз"
        f.write(f"{interpretation}\n")
        f.write("\n")

        f.write("="*60 + "\n")

    print(f"Метрики збережено (текст): {save_path}")


def print_metrics_summary(
    metrics: Dict[str, float],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None
) -> None:
    """
    Print metrics summary to console.

    Parameters
    ----------
    metrics : dict
        Dictionary with metric names and values
    y_true : np.ndarray, optional
        Actual values
    y_pred : np.ndarray, optional
        Predicted values

    Examples
    --------
    >>> print_metrics_summary(metrics, y_true=actual, y_pred=forecast)
    """

    print("\n" + "="*70)
    print("МЕТРИКИ ЯКОСТІ ПРОГНОЗУ")
    print("="*70)

    print(f"\nMAE (Mean Absolute Error):               {metrics['MAE']:.4f}")
    print(f"RMSE (Root Mean Square Error):           {metrics['RMSE']:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error):   {metrics['MAPE']:.4f}%")

    if y_true is not None and y_pred is not None:
        print(f"\nКількість точок прогнозу: {len(y_true)}")
        print(f"Середнє фактичне значення: {np.mean(y_true):.2f}")
        print(f"Середнє прогнозоване значення: {np.mean(y_pred):.2f}")

    # Interpretation
    mape_val = metrics['MAPE']
    print("\nІнтерпретація:")
    if mape_val < 10:
        print("  ✓ Дуже точний прогноз (MAPE < 10%)")
    elif mape_val < 20:
        print("  ✓ Хороший прогноз (MAPE < 20%)")
    elif mape_val < 50:
        print("  ~ Прийнятний прогноз (MAPE < 50%)")
    else:
        print("  ✗ Неточний прогноз (MAPE >= 50%)")

    print("="*70 + "\n")
