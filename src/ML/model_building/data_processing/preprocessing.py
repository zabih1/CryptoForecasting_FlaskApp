"""
This script loads raw cryptocurrency market data from a CSV file, processes it 
by adding new features (price range, close-to-open difference, and target variable), 
and saves the processed data as a new CSV file.
"""

import os
import pandas as pd

# ===================== Load Raw Data =====================
def load_raw_data(base_path, symbol, interval):
    """
    Load raw market data from a CSV file.

    Parameters:
        base_path (str): Directory where the raw data CSV file is stored.
        symbol (str): Cryptocurrency pair (e.g., 'BTCUSDT').
        interval (str): Time interval for candlestick data (e.g., '1d', '1h').

    Returns:
        pd.DataFrame: A DataFrame containing the loaded market data.
    """
    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)

# ===================== Process Data =====================
def process_data(df):
    """
    Process raw market data by converting columns to numeric values,
    adding new features, and creating a target variable.

    Parameters:
        df (pd.DataFrame): The raw market data DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame with new features and target variable.
    """
    df = df.apply(pd.to_numeric, errors='coerce')
    
    df['price_range'] = df['high'] - df['low']
    df['close_to_open'] = df['close'] - df['open']
    df['target'] = df['close'].shift(-1)  # Next day's closing price as target

    df = df.dropna()
    
    return df

# ===================== Save Processed Data =====================
def save_processed_data(df, base_path, symbol, interval):
    """
    Save the processed market data to a CSV file.

    Parameters:
        df (pd.DataFrame): The processed market data DataFrame.
        base_path (str): Directory where the processed data CSV file will be saved.
        symbol (str): Cryptocurrency pair (e.g., 'BTCUSDT').
        interval (str): Time interval for candlestick data (e.g., '1d', '1h').

    Returns:
        str: The file path where the processed CSV is saved.
    """
    filename = f"{symbol.lower()}_{interval}_processed.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
