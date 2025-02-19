"""
This script loads raw cryptocurrency market data from a CSV file, processes it 
by performing data cleaning and feature engineering (calculating average price, 
price change, and the next period's closing price as the target), and saves the 
processed data as a new CSV file.
"""

import os
import pandas as pd

# ===================== Load Raw Data =====================
def load_raw_data(base_path, symbol, interval):

    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)

# ===================== Process Data =====================
def process_data(df):

    columns_to_drop = ["ignore", "timestamp", "close_time"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
    
    numeric_columns = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]
    # Only process columns that actually exist in the DataFrame
    existing_numeric_columns = [col for col in numeric_columns if col in df.columns]
    df[existing_numeric_columns] = df[existing_numeric_columns].apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with any missing values
    df = df.dropna()
    
    if not df.empty:
        # Create new features
        df["average_price"] = (df["high"] + df["low"]) / 2
        df["price_change"] = df["close"] - df["open"]
        df["target_close"] = df["close"].shift(-1)  # Next period's closing price
        
        # Drop any rows with missing values (which may be introduced by shifting)
        df = df.dropna()
        
        # Drop columns that are no longer needed
        df = df.drop(columns=["close"], errors="ignore")
    
    # Reset the index
    df = df.reset_index(drop=True)
    
    return df

# ===================== Save Processed Data =====================
def save_processed_data(df, base_path, symbol, interval):

    filename = f"{symbol.lower()}_{interval}_processed.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

