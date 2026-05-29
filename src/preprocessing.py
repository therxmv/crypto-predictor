"""
Data Preprocessing Module

This module provides functionality for preparing time-series data
and creating 21-day overlapping fragments with 7-day step.

Key responsibilities:
- Clean and prepare time-series data
- Handle missing values and duplicates
- Optionally align fragments to weekdays (Monday)
- Create 3-week (21-day) overlapping fragments
- Generate fragment metadata
- Support train/experimental split for forecasting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def prepare_time_series(
    df: pd.DataFrame,
    align_to_monday: bool = True
) -> pd.DataFrame:
    """
    Prepare time-series data for fragment creation.

    This function performs the following operations:
    1. Keeps only required columns: Date, Close
    2. Removes rows with missing Date
    3. Removes rows with missing Close
    4. Ensures Close is numeric
    5. Removes duplicate dates
    6. Sorts by date
    7. Validates minimum length requirement
    8. Optionally aligns to Monday start

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least Date and Close columns
    align_to_monday : bool, default True
        If True, start from the first Monday in the series.
        This ensures all fragments start and end on the same weekday.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with Date and Close columns, sorted by date

    Raises
    ------
    ValueError
        If the DataFrame doesn't have enough rows (minimum 28 required)
        or if required columns are missing

    Notes
    -----
    Minimum length requirement:
    - Absolute minimum: 28 rows (allows 2 fragments: one for training, one for experimental)
    - Recommended: several years of daily data for robust analysis

    The 28-row minimum comes from:
    - Fragment length: 21 days
    - Step: 7 days
    - Need at least 2 fragments: 21 + 7 = 28
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # 1. Keep only required columns
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Date' and 'Close' columns")

    df = df[['Date', 'Close']].copy()

    # 2. Remove rows with missing Date
    df = df.dropna(subset=['Date'])

    # 3. Remove rows with missing Close
    df = df.dropna(subset=['Close'])

    # 4. Ensure Close is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])

    # 5. Remove duplicate dates
    df = df.drop_duplicates(subset=['Date'], keep='first')

    # 6. Sort by date
    df = df.sort_values('Date').reset_index(drop=True)

    # 7. Validate minimum length
    if len(df) < 28:
        raise ValueError(
            f"Insufficient data: {len(df)} rows found, but at least 28 rows required.\n"
            f"Minimum requirement: 28 rows (21-day fragment + 7-day step).\n"
            f"Recommended: several years of daily observations."
        )

    # 8. Optional weekday alignment
    if align_to_monday:
        df = align_to_weekday(df, weekday=0)  # 0 = Monday

    return df


def align_to_weekday(
    df: pd.DataFrame,
    weekday: int = 0
) -> pd.DataFrame:
    """
    Align time series to start from a specific weekday.

    This ensures that all 21-day fragments start and end on the same weekday,
    which is recommended by the laboratory task requirements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date column
    weekday : int, default 0
        Target weekday (0=Monday, 1=Tuesday, ..., 6=Sunday)

    Returns
    -------
    pd.DataFrame
        DataFrame starting from the first occurrence of the target weekday

    Notes
    -----
    Since the fragment step is 7 days, once the first fragment is aligned,
    all following fragments will maintain the same weekday alignment.
    """

    if df.empty:
        return df

    # Find the first occurrence of the target weekday
    weekday_mask = df['Date'].dt.weekday == weekday
    first_weekday_idx = weekday_mask.idxmax() if weekday_mask.any() else 0

    # Start from that date
    df_aligned = df.loc[first_weekday_idx:].reset_index(drop=True)

    return df_aligned


def create_fragments(
    df: pd.DataFrame,
    value_column: str = "Close",
    fragment_length: int = 21,
    step: int = 7
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create overlapping time-series fragments.

    Each fragment represents 3 weeks (21 days) of consecutive observations.
    Fragments overlap by shifting the start position by 7 days (1 week).

    Parameters
    ----------
    df : pd.DataFrame
        Clean time-series DataFrame with Date and value columns
    value_column : str, default "Close"
        Name of the column containing values to fragment
    fragment_length : int, default 21
        Length of each fragment in days (must be 21 for this project)
    step : int, default 7
        Step size between consecutive fragments in days (must be 7 for this project)

    Returns
    -------
    X_raw : np.ndarray
        Raw fragment matrix of shape (number_of_fragments, 21)
        Each row is one 21-day fragment
    metadata : pd.DataFrame
        Fragment metadata with columns:
        - fragment_index: Sequential fragment number
        - start_date: First date in fragment
        - end_date: Last date in fragment
        - start_position: Start row index in original DataFrame
        - end_position: End row index in original DataFrame
        - x_min: Minimum value in fragment
        - x_max: Maximum value in fragment

    Raises
    ------
    ValueError
        If insufficient data for creating at least one fragment

    Examples
    --------
    >>> df = prepare_time_series(raw_df)
    >>> X_raw, metadata = create_fragments(df)
    >>> print(X_raw.shape)
    (519, 21)
    >>> print(metadata.columns)
    Index(['fragment_index', 'start_date', 'end_date', 'start_position',
           'end_position', 'x_min', 'x_max'], dtype='object')

    Notes
    -----
    Fragment creation logic:
    - Fragment 0: values[0:21]   (days 0-20)
    - Fragment 1: values[7:28]   (days 7-27)
    - Fragment 2: values[14:35]  (days 14-34)
    - ...
    - Fragment k: values[k*7 : k*7+21]

    Stop when: start_position + fragment_length > len(values)
    """

    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame")

    values = df[value_column].values
    dates = df['Date'].values
    n = len(values)

    # Validate fragment parameters
    if fragment_length != 21:
        raise ValueError(
            f"Fragment length must be 21 days, got {fragment_length}.\n"
            f"This is a core project requirement."
        )

    if step != 7:
        raise ValueError(
            f"Step must be 7 days, got {step}.\n"
            f"This is a core project requirement."
        )

    if n < fragment_length:
        raise ValueError(
            f"Insufficient data: {n} rows available, "
            f"but at least {fragment_length} rows required for one fragment."
        )

    # Calculate number of fragments
    fragments = []
    metadata_records = []

    start_pos = 0
    fragment_idx = 0

    while start_pos + fragment_length <= n:
        end_pos = start_pos + fragment_length - 1

        # Extract fragment values
        fragment_values = values[start_pos : start_pos + fragment_length]

        # Create metadata
        metadata_record = {
            'fragment_index': fragment_idx,
            'start_date': dates[start_pos],
            'end_date': dates[end_pos],
            'start_position': start_pos,
            'end_position': end_pos,
            'x_min': float(np.min(fragment_values)),
            'x_max': float(np.max(fragment_values))
        }

        fragments.append(fragment_values)
        metadata_records.append(metadata_record)

        # Move to next fragment
        start_pos += step
        fragment_idx += 1

    if len(fragments) == 0:
        raise ValueError(
            f"Unable to create any fragments. "
            f"Need at least {fragment_length} consecutive rows."
        )

    # Convert to numpy array
    X_raw = np.array(fragments)

    # Convert metadata to DataFrame
    metadata = pd.DataFrame(metadata_records)

    return X_raw, metadata


def split_train_experimental_series(
    df: pd.DataFrame,
    prefix_length: int = 14,
    forecast_horizon: int = 7
) -> Dict:
    """
    Split time series into training and experimental parts for forecasting.

    The last 21 days are used as the experimental image:
    - First 14 days (prefix): known data before forecasting point
    - Last 7 days (postfix): actual future values for evaluation

    Training data includes all observations before the actual postfix begins.

    Parameters
    ----------
    df : pd.DataFrame
        Clean time-series DataFrame with Date and Close columns
    prefix_length : int, default 14
        Length of experimental prefix (known part)
    forecast_horizon : int, default 7
        Length of forecast horizon (future part to predict)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'train_df': Training DataFrame (excludes actual postfix)
        - 'experimental_full': Full 21-day experimental DataFrame
        - 'experimental_prefix': First 14 days of experimental period
        - 'actual_postfix': Last 7 days of experimental period (ground truth)
        - 'forecast_start_date': First date of forecast period
        - 'forecast_end_date': Last date of forecast period

    Raises
    ------
    ValueError
        If insufficient data for creating experimental image

    Notes
    -----
    Data leakage prevention:
    - Training fragments must NOT include actual postfix dates
    - Actual postfix is used ONLY for final evaluation
    - Forecasting model sees ONLY the 14-day prefix

    Example split for 3640 rows:
    - train_df: rows 0-3626 (all data before last 7 days)
    - experimental_full: rows 3619-3639 (last 21 days)
    - experimental_prefix: rows 3619-3632 (days 1-14 of experimental)
    - actual_postfix: rows 3633-3639 (days 15-21 of experimental)
    """

    experimental_full_length = prefix_length + forecast_horizon

    if len(df) < experimental_full_length:
        raise ValueError(
            f"Insufficient data: {len(df)} rows available, "
            f"but at least {experimental_full_length} rows required "
            f"({prefix_length} prefix + {forecast_horizon} forecast)."
        )

    # Extract last 21 days as experimental image
    experimental_full = df.iloc[-experimental_full_length:].copy().reset_index(drop=True)

    # Split experimental image into prefix and postfix
    experimental_prefix = experimental_full.iloc[:prefix_length].copy()
    actual_postfix = experimental_full.iloc[prefix_length:].copy()

    # Training data: all rows before actual postfix begins
    # The postfix starts at position: len(df) - forecast_horizon
    train_end_position = len(df) - forecast_horizon
    train_df = df.iloc[:train_end_position].copy().reset_index(drop=True)

    # Extract forecast dates
    forecast_start_date = actual_postfix['Date'].iloc[0]
    forecast_end_date = actual_postfix['Date'].iloc[-1]

    return {
        'train_df': train_df,
        'experimental_full': experimental_full,
        'experimental_prefix': experimental_prefix,
        'actual_postfix': actual_postfix,
        'forecast_start_date': forecast_start_date,
        'forecast_end_date': forecast_end_date
    }
