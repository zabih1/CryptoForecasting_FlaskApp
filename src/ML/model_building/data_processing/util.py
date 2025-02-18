"""
This script fetches historical cryptocurrency price data from the Binance API 
and saves it as a CSV file. The data includes open, high, low, close prices, 
volume, and trading details for a specified symbol and interval.
"""

import pandas as pd
import requests
import os

# ===================== Data Fetching Function =====================

def get_data(symbol, interval='1d', start_date=None, end_date=None, limit=1000):
    

    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    
    if start_date:
        url += f'&startTime={int(pd.Timestamp(start_date).timestamp() * 1000)}'
    if end_date:
        url += f'&endTime={int(pd.Timestamp(end_date).timestamp() * 1000)}'
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f'Error fetching data: {response.status_code} - {response.text}')
    
    data = response.json()
    
    if not data:
        raise Exception("No data returned from Binance API.")

# =============== Convert API response to DataFrame =================

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['close_time', 'ignore'], inplace=True)
    
    return df

# ===================== CSV Saving Function =======================

def save_to_csv(df, symbol, interval, base_path):
    """
    Save the DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing market data.
        symbol (str): Cryptocurrency pair (e.g., 'BTCUSDT').
        interval (str): Time interval for candlestick data (e.g., '1d', '1h').
        base_path (str): Directory where the CSV file will be saved.
    
    Returns:
        str: The file path where the CSV is saved.
    """
    filename = f"{symbol.lower()}_{interval}.csv"
    path = os.path.join(base_path, filename)
    os.makedirs(base_path, exist_ok=True)
    df.to_csv(path, index=False)
    return path
