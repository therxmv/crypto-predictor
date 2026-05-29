"""
Data Loading Module

This module provides functionality for loading local CSV files containing
financial time-series data.

The module follows strict project constraints:
- Only local CSV files are allowed
- No internet data downloading
- No external API calls
- No yfinance usage
"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_local_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load financial time-series data from a local CSV file.

    This function performs comprehensive validation and preprocessing:
    1. Validates that the file exists
    2. Validates that the file is not empty
    3. Reads the CSV file
    4. Validates required columns (Date, Close)
    5. Parses Date column as datetime
    6. Converts Close column to numeric
    7. Sorts rows by date in ascending order
    8. Returns a clean DataFrame

    Parameters
    ----------
    file_path : str or Path
        Path to the local CSV file containing financial data

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with at least Date and Close columns,
        sorted by date in ascending order

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the file is empty, required columns are missing,
        or data cannot be parsed correctly

    Examples
    --------
    >>> df = load_local_csv('data/raw/BTC-USD.csv')
    >>> df.head()
           Date    Close
    0  2014-09-17  465.864014
    1  2014-09-18  457.334015
    """

    file_path = Path(file_path)

    # 1. Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please ensure the CSV file exists at the specified path."
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path exists but is not a file: {file_path}\n"
            f"Please provide a path to a CSV file."
        )

    # 2. Read CSV file
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"The CSV file is empty: {file_path}\n"
            f"Please provide a file with valid data."
        )
    except Exception as e:
        raise ValueError(
            f"Failed to read CSV file: {file_path}\n"
            f"Error: {str(e)}"
        )

    # 3. Validate file is not empty
    if df.empty:
        raise ValueError(
            f"The CSV file contains no data rows: {file_path}\n"
            f"Please provide a file with at least one data row."
        )

    # 4. Validate required columns exist
    if 'Date' not in df.columns:
        raise ValueError(
            f"Required column 'Date' not found in CSV file.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please ensure the CSV file contains a 'Date' column."
        )

    if 'Close' not in df.columns:
        raise ValueError(
            f"Required column 'Close' not found in CSV file.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please ensure the CSV file contains a 'Close' column."
        )

    # 5. Parse Date column as datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        raise ValueError(
            f"Failed to parse 'Date' column as datetime.\n"
            f"Error: {str(e)}\n"
            f"Please ensure Date column contains valid date values."
        )

    # 6. Convert Close column to numeric
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    except Exception as e:
        raise ValueError(
            f"Failed to convert 'Close' column to numeric.\n"
            f"Error: {str(e)}"
        )

    # Check if Close conversion resulted in all NaN values
    if df['Close'].isna().all():
        raise ValueError(
            f"All values in 'Close' column are invalid or missing.\n"
            f"Please ensure Close column contains valid numeric price data."
        )

    # 7. Sort rows by Date in ascending order
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def validate_input_columns(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame contains required columns for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate

    Raises
    ------
    ValueError
        If required columns are missing or invalid
    """

    if df.empty:
        raise ValueError("DataFrame is empty")

    if 'Date' not in df.columns:
        raise ValueError(
            f"Required column 'Date' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if 'Close' not in df.columns:
        raise ValueError(
            f"Required column 'Close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        raise ValueError("'Date' column must be datetime type")

    if not pd.api.types.is_numeric_dtype(df['Close']):
        raise ValueError("'Close' column must be numeric type")
